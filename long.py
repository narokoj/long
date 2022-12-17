from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch
from coqpit import Coqpit
from torch import nn



from .attentions import init_attn
from .common_layers import Prenet


from TTS.tts.layers.align_tts.mdn import MDNBlock
from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.duration_predictor import DurationPredictor
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import generate_path, maximum_path, sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.io import load_fsspec


@dataclass
class AlignTTSArgs(Coqpit):


    num_chars: int = None
    out_channels: int = 80
    hidden_channels: int = 256
    hidden_channels_dp: int = 256
    encoder_type: str = "fftransformer"
    encoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    length_scale: float = 1.0
    num_speakers: int = 0
    use_speaker_embedding: bool = False
    use_d_vector_file: bool = False
    d_vector_dim: int = 0


class AlignTTS(BaseTTS):

    # pylint: disable=dangerous-default-value

    def __init__(
        self,
        config: "AlignTTSConfig",
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
    ):

        super().__init__(config, ap, tokenizer, speaker_manager)
        self.speaker_manager = speaker_manager
        self.phase = -1
        self.length_scale = (
            float(config.model_args.length_scale)
            if isinstance(config.model_args.length_scale, int)
            else config.model_args.length_scale
        )

        self.emb = nn.Embedding(self.config.model_args.num_chars, self.config.model_args.hidden_channels)

        self.embedded_speaker_dim = 0
        self.init_multispeaker(config)

        self.pos_encoder = PositionalEncoding(config.model_args.hidden_channels)
        self.encoder = Encoder(
            config.model_args.hidden_channels,
            config.model_args.hidden_channels,
            config.model_args.encoder_type,
            config.model_args.encoder_params,
            self.embedded_speaker_dim,
        )
        self.decoder = Decoder(
            config.model_args.out_channels,
            config.model_args.hidden_channels,
            config.model_args.decoder_type,
            config.model_args.decoder_params,
        )
        self.duration_predictor = DurationPredictor(config.model_args.hidden_channels_dp)

        self.mod_layer = nn.Conv1d(config.model_args.hidden_channels, config.model_args.hidden_channels, 1)

        self.mdn_block = MDNBlock(config.model_args.hidden_channels, 2 * config.model_args.out_channels)

        if self.embedded_speaker_dim > 0 and self.embedded_speaker_dim != config.model_args.hidden_channels:
            self.proj_g = nn.Conv1d(self.embedded_speaker_dim, config.model_args.hidden_channels, 1)

    @staticmethod
    def compute_log_probs(mu, log_sigma, y):
        # pylint: disable=protected-access, c-extension-no-member
        y = y.transpose(1, 2).unsqueeze(1)  # [B, 1, T1, D]
        mu = mu.transpose(1, 2).unsqueeze(2)  # [B, T2, 1, D]
        log_sigma = log_sigma.transpose(1, 2).unsqueeze(2)  # [B, T2, 1, D]
        expanded_y, expanded_mu = torch.broadcast_tensors(y, mu)
        exponential = -0.5 * torch.mean(
            torch._C._nn.mse_loss(expanded_y, expanded_mu, 0) / torch.pow(log_sigma.exp(), 2), dim=-1
        )  # B, L, T
        logp = exponential - 0.5 * log_sigma.mean(dim=-1)
        return logp

    def compute_align_path(self, mu, log_sigma, y, x_mask, y_mask):
        # find the max alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        log_p = self.compute_log_probs(mu, log_sigma, y)
        # [B, T_en, T_dec]
        attn = maximum_path(log_p, attn_mask.squeeze(1)).unsqueeze(1)
        dr_mas = torch.sum(attn, -1)
        return dr_mas.squeeze(1), log_p

    @staticmethod
    def generate_attn(dr, x_mask, y_mask=None):
        # compute decode mask from the durations
        if y_mask is None:
            y_lengths = dr.sum(1).long()
            y_lengths[y_lengths < 1] = 1
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(dr.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(dr, attn_mask.squeeze(1)).to(dr.dtype)
        return attn

    def expand_encoder_outputs(self, en, dr, x_mask, y_mask):
        
        attn = self.generate_attn(dr, x_mask, y_mask)
        o_en_ex = torch.matmul(attn.squeeze(1).transpose(1, 2), en.transpose(1, 2)).transpose(1, 2)
        return o_en_ex, attn

    def format_durations(self, o_dr_log, x_mask):
        o_dr = (torch.exp(o_dr_log) - 1) * x_mask * self.length_scale
        o_dr[o_dr < 1] = 1.0
        o_dr = torch.round(o_dr)
        return o_dr

    @staticmethod
    def _concat_speaker_embedding(o_en, g):
        g_exp = g.expand(-1, -1, o_en.size(-1))  # [B, C, T_en]
        o_en = torch.cat([o_en, g_exp], 1)
        return o_en

    def _sum_speaker_embedding(self, x, g):
        # project g to decoder dim.
        if hasattr(self, "proj_g"):
            g = self.proj_g(g)

        return x + g

    def _forward_encoder(self, x, x_lengths, g=None):
        if hasattr(self, "emb_g"):
            g = nn.functional.normalize(self.speaker_embedding(g))  # [B, C, 1]

        if g is not None:
            g = g.unsqueeze(-1)

        # [B, T, C]
        x_emb = self.emb(x)
        # [B, C, T]
        x_emb = torch.transpose(x_emb, 1, -1)

        # compute sequence masks
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).to(x.dtype)

        # encoder pass
        o_en = self.encoder(x_emb, x_mask)

        # speaker conditioning for duration predictor
        if g is not None:
            o_en_dp = self._concat_speaker_embedding(o_en, g)
        else:
            o_en_dp = o_en
        return o_en, o_en_dp, x_mask, g

    def _forward_decoder(self, o_en, o_en_dp, dr, x_mask, y_lengths, g):
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en_dp.dtype)
        # expand o_en with durations
        o_en_ex, attn = self.expand_encoder_outputs(o_en, dr, x_mask, y_mask)
        # positional encoding
        if hasattr(self, "pos_encoder"):
            o_en_ex = self.pos_encoder(o_en_ex, y_mask)
        # speaker embedding
        if g is not None:
            o_en_ex = self._sum_speaker_embedding(o_en_ex, g)
        # decoder pass
        o_de = self.decoder(o_en_ex, y_mask, g=g)
        return o_de, attn.transpose(1, 2)

    def _forward_mdn(self, o_en, y, y_lengths, x_mask):
        # MAS potentials and alignment
        mu, log_sigma = self.mdn_block(o_en)
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        dr_mas, logp = self.compute_align_path(mu, log_sigma, y, x_mask, y_mask)
        return dr_mas, mu, log_sigma, logp

    def forward(
        self, x, x_lengths, y, y_lengths, aux_input={"d_vectors": None}, phase=None
    ):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - y_lengths: :math:`[B]`
            - dr: :math:`[B, T_max]`
            - g: :math:`[B, C]`
        """
        y = y.transpose(1, 2)
        g = aux_input["d_vectors"] if "d_vectors" in aux_input else None
        o_de, o_dr_log, dr_mas_log, attn, mu, log_sigma, logp = None, None, None, None, None, None, None
        if phase == 0:
            # train encoder and MDN
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            dr_mas, mu, log_sigma, logp = self._forward_mdn(o_en, y, y_lengths, x_mask)
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en_dp.dtype)
            attn = self.generate_attn(dr_mas, x_mask, y_mask)
        elif phase == 1:
            # train decoder
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            dr_mas, _, _, _ = self._forward_mdn(o_en, y, y_lengths, x_mask)
            o_de, attn = self._forward_decoder(o_en.detach(), o_en_dp.detach(), dr_mas.detach(), x_mask, y_lengths, g=g)
        elif phase == 2:
            # train the whole except duration predictor
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            dr_mas, mu, log_sigma, logp = self._forward_mdn(o_en, y, y_lengths, x_mask)
            o_de, attn = self._forward_decoder(o_en, o_en_dp, dr_mas, x_mask, y_lengths, g=g)
        elif phase == 3:
            # train duration predictor
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            o_dr_log = self.duration_predictor(x, x_mask)
            dr_mas, mu, log_sigma, logp = self._forward_mdn(o_en, y, y_lengths, x_mask)
            o_de, attn = self._forward_decoder(o_en, o_en_dp, dr_mas, x_mask, y_lengths, g=g)
            o_dr_log = o_dr_log.squeeze(1)
        else:
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            o_dr_log = self.duration_predictor(o_en_dp.detach(), x_mask)
            dr_mas, mu, log_sigma, logp = self._forward_mdn(o_en, y, y_lengths, x_mask)
            o_de, attn = self._forward_decoder(o_en, o_en_dp, dr_mas, x_mask, y_lengths, g=g)
            o_dr_log = o_dr_log.squeeze(1)
        dr_mas_log = torch.log(dr_mas + 1).squeeze(1)
        outputs = {
            "model_outputs": o_de.transpose(1, 2),
            "alignments": attn,
            "durations_log": o_dr_log,
            "durations_mas_log": dr_mas_log,
            "mu": mu,
            "log_sigma": log_sigma,
            "logp": logp,
        }
        return outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"d_vectors": None}):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - g: :math:`[B, C]`
        """
        g = aux_input["d_vectors"] if "d_vectors" in aux_input else None
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)
        # pad input to prevent dropping the last word
        # x = torch.nn.functional.pad(x, pad=(0, 5), mode='constant', value=0)
        o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
        # o_dr_log = self.duration_predictor(x, x_mask)
        o_dr_log = self.duration_predictor(o_en_dp, x_mask)
        # duration predictor pass
        o_dr = self.format_durations(o_dr_log, x_mask).squeeze(1)
        y_lengths = o_dr.sum(1)
        o_de, attn = self._forward_decoder(o_en, o_en_dp, o_dr, x_mask, y_lengths, g=g)
        outputs = {"model_outputs": o_de.transpose(1, 2), "alignments": attn}
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]

        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}
        outputs = self.forward(text_input, text_lengths, mel_input, mel_lengths, aux_input, self.phase)
        loss_dict = criterion(
            outputs["logp"],
            outputs["model_outputs"],
            mel_input,
            mel_lengths,
            outputs["durations_log"],
            outputs["durations_mas_log"],
            text_lengths,
            phase=self.phase,
        )

        return outputs, loss_dict

    def _create_logs(self, batch, outputs, ap):  # pylint: disable=no-self-use
        model_outputs = outputs["model_outputs"]
        alignments = outputs["alignments"]
        mel_input = batch["mel_input"]

        pred_spec = model_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        # Sample audio
        train_audio = ap.inv_melspectrogram(pred_spec.T)
        return figures, {"audio": train_audio}

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ) -> None:  # pylint: disable=no-self-use
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training

    def get_criterion(self):
        from TTS.tts.layers.losses import AlignTTSLoss  # pylint: disable=import-outside-toplevel

        return AlignTTSLoss(self.config)

    @staticmethod
    def _set_phase(config, global_step):
        """Decide AlignTTS training phase"""
        if isinstance(config.phase_start_steps, list):
            vals = [i < global_step for i in config.phase_start_steps]
            if not True in vals:
                phase = 0
            else:
                phase = (
                    len(config.phase_start_steps)
                    - [i < global_step for i in config.phase_start_steps][::-1].index(True)
                    - 1
                )
        else:
            phase = None
        return phase

    def on_epoch_start(self, trainer):
        """Set AlignTTS training phase on epoch start."""
        self.phase = self._set_phase(trainer.config, trainer.total_steps_done)

    @staticmethod
    def init_from_config(config: "AlignTTSConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (AlignTTSConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        return AlignTTS(new_config, ap, tokenizer, speaker_manager)

"""--------------------------------------------------------------------------------------------"""


# coding: utf-8
# adapted from https://github.com/r9y9/tacotron_pytorch


class BatchNormConv1d(nn.Module):
    r"""A wrapper for Conv1d with BatchNorm. It sets the activation
    function between Conv and BatchNorm layers. BatchNorm layer
    is initialized with the TF default values for momentum and eps.

    Args:
        in_channels: size of each input sample
        out_channels: size of each output samples
        kernel_size: kernel size of conv filters
        stride: stride of conv filters
        padding: padding of conv filters
        activation: activation function set b/w Conv1d and BatchNorm

    Shapes:
        - input: (B, D)
        - output: (B, D)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None):

        super().__init__()
        self.padding = padding
        self.padder = nn.ConstantPad1d(padding, 0)
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False
        )
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3)
        self.activation = activation
        # self.init_layers()

    def init_layers(self):
        if isinstance(self.activation, torch.nn.ReLU):
            w_gain = "relu"
        elif isinstance(self.activation, torch.nn.Tanh):
            w_gain = "tanh"
        elif self.activation is None:
            w_gain = "linear"
        else:
            raise RuntimeError("Unknown activation function")
        torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=torch.nn.init.calculate_gain(w_gain))

    def forward(self, x):
        x = self.padder(x)
        x = self.conv1d(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Highway(nn.Module):
    r"""Highway layers as explained in https://arxiv.org/abs/1505.00387

    Args:
        in_features (int): size of each input sample
        out_feature (int): size of each output sample

    Shapes:
        - input: (B, *, H_in)
        - output: (B, *, H_out)
    """

    # TODO: Try GLU layer
    def __init__(self, in_features, out_feature):
        super().__init__()
        self.H = nn.Linear(in_features, out_feature)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_features, out_feature)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(self.H.weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self.T.weight, gain=torch.nn.init.calculate_gain("sigmoid"))

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
    - 1-d convolution banks
    - Highway networks + residual connections
    - Bidirectional gated recurrent units

    Args:
        in_features (int): sample size
        K (int): max filter size in conv bank
        projections (list): conv channel sizes for conv projections
        num_highways (int): number of highways layers

    Shapes:
        - input: (B, C, T_in)
        - output: (B, T_in, C*2)
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        in_features,
        K=16,
        conv_bank_features=128,
        conv_projections=[128, 128],
        highway_features=128,
        gru_features=128,
        num_highways=4,
    ):
        super().__init__()
        self.in_features = in_features
        self.conv_bank_features = conv_bank_features
        self.highway_features = highway_features
        self.gru_features = gru_features
        self.conv_projections = conv_projections
        self.relu = nn.ReLU()
        # list of conv1d bank with filter size k=1...K
        # TODO: try dilational layers instead
        self.conv1d_banks = nn.ModuleList(
            [
                BatchNormConv1d(
                    in_features,
                    conv_bank_features,
                    kernel_size=k,
                    stride=1,
                    padding=[(k - 1) // 2, k // 2],
                    activation=self.relu,
                )
                for k in range(1, K + 1)
            ]
        )
        # max pooling of conv bank, with padding
        # TODO: try average pooling OR larger kernel size
        out_features = [K * conv_bank_features] + conv_projections[:-1]
        activations = [self.relu] * (len(conv_projections) - 1)
        activations += [None]
        # setup conv1d projection layers
        layer_set = []
        for (in_size, out_size, ac) in zip(out_features, conv_projections, activations):
            layer = BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1, padding=[1, 1], activation=ac)
            layer_set.append(layer)
        self.conv1d_projections = nn.ModuleList(layer_set)
        # setup Highway layers
        if self.highway_features != conv_projections[-1]:
            self.pre_highway = nn.Linear(conv_projections[-1], highway_features, bias=False)
        self.highways = nn.ModuleList([Highway(highway_features, highway_features) for _ in range(num_highways)])
        # bi-directional GPU layer
        self.gru = nn.GRU(gru_features, gru_features, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        # (B, in_features, T_in)
        x = inputs
        # (B, hid_features*K, T_in)
        # Concat conv1d bank outputs
        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        assert x.size(1) == self.conv_bank_features * len(self.conv1d_banks)
        for conv1d in self.conv1d_projections:
            x = conv1d(x)
        x += inputs
        x = x.transpose(1, 2)
        if self.highway_features != self.conv_projections[-1]:
            x = self.pre_highway(x)
        # Residual connection
        # TODO: try residual scaling as in Deep Voice 3
        # TODO: try plain residual layers
        for highway in self.highways:
            x = highway(x)
        # (B, T_in, hid_features*2)
        # TODO: replace GRU with convolution as in Deep Voice 3
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)
        return outputs


class EncoderCBHG(nn.Module):
    r"""CBHG module with Encoder specific arguments"""

    def __init__(self):
        super().__init__()
        self.cbhg = CBHG(
            128,
            K=16,
            conv_bank_features=128,
            conv_projections=[128, 128],
            highway_features=128,
            gru_features=128,
            num_highways=4,
        )

    def forward(self, x):
        return self.cbhg(x)


class Encoder(nn.Module):
    r"""Stack Prenet and CBHG module for encoder
    Args:
        inputs (FloatTensor): embedding features

    Shapes:
        - inputs: (B, T, D_in)
        - outputs: (B, T, 128 * 2)
    """

    def __init__(self, in_features):
        super().__init__()
        self.prenet = Prenet(in_features, out_features=[256, 128])
        self.cbhg = EncoderCBHG()

    def forward(self, inputs):
        # B x T x prenet_dim
        outputs = self.prenet(inputs)
        outputs = self.cbhg(outputs.transpose(1, 2))
        return outputs


class PostCBHG(nn.Module):
    def __init__(self, mel_dim):
        super().__init__()
        self.cbhg = CBHG(
            mel_dim,
            K=8,
            conv_bank_features=128,
            conv_projections=[256, mel_dim],
            highway_features=128,
            gru_features=128,
            num_highways=4,
        )

    def forward(self, x):
        return self.cbhg(x)


class Decoder(nn.Module):
    """Tacotron decoder.

    Args:
        in_channels (int): number of input channels.
        frame_channels (int): number of feature frame channels.
        r (int): number of outputs per time step (reduction rate).
        memory_size (int): size of the past window. if <= 0 memory_size = r
        attn_type (string): type of attention used in decoder.
        attn_windowing (bool): if true, define an attention window centered to maximum
            attention response. It provides more robust attention alignment especially
            at interence time.
        attn_norm (string): attention normalization function. 'sigmoid' or 'softmax'.
        prenet_type (string): 'original' or 'bn'.
        prenet_dropout (float): prenet dropout rate.
        forward_attn (bool): if true, use forward attention method. https://arxiv.org/abs/1807.06736
        trans_agent (bool): if true, use transition agent. https://arxiv.org/abs/1807.06736
        forward_attn_mask (bool): if true, mask attention values smaller than a threshold.
        location_attn (bool): if true, use location sensitive attention.
        attn_K (int): number of attention heads for GravesAttention.
        separate_stopnet (bool): if true, detach stopnet input to prevent gradient flow.
        d_vector_dim (int): size of speaker embedding vector, for multi-speaker training.
        max_decoder_steps (int): Maximum number of steps allowed for the decoder. Defaults to 500.
    """

    # Pylint gets confused by PyTorch conventions here
    # pylint: disable=attribute-defined-outside-init

    def __init__(
        self,
        in_channels,
        frame_channels,
        r,
        memory_size,
        attn_type,
        attn_windowing,
        attn_norm,
        prenet_type,
        prenet_dropout,
        forward_attn,
        trans_agent,
        forward_attn_mask,
        location_attn,
        attn_K,
        separate_stopnet,
        max_decoder_steps,
    ):
        super().__init__()
        self.r_init = r
        self.r = r
        self.in_channels = in_channels
        self.max_decoder_steps = max_decoder_steps
        self.use_memory_queue = memory_size > 0
        self.memory_size = memory_size if memory_size > 0 else r
        self.frame_channels = frame_channels
        self.separate_stopnet = separate_stopnet
        self.query_dim = 256
        # memory -> |Prenet| -> processed_memory
        prenet_dim = frame_channels * self.memory_size if self.use_memory_queue else frame_channels
        self.prenet = Prenet(prenet_dim, prenet_type, prenet_dropout, out_features=[256, 128])
        # processed_inputs, processed_memory -> |Attention| -> Attention, attention, RNN_State
        # attention_rnn generates queries for the attention mechanism
        self.attention_rnn = nn.GRUCell(in_channels + 128, self.query_dim)
        self.attention = init_attn(
            attn_type=attn_type,
            query_dim=self.query_dim,
            embedding_dim=in_channels,
            attention_dim=128,
            location_attention=location_attn,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
            windowing=attn_windowing,
            norm=attn_norm,
            forward_attn=forward_attn,
            trans_agent=trans_agent,
            forward_attn_mask=forward_attn_mask,
            attn_K=attn_K,
        )
        # (processed_memory | attention context) -> |Linear| -> decoder_RNN_input
        self.project_to_decoder_in = nn.Linear(256 + in_channels, 256)
        # decoder_RNN_input -> |RNN| -> RNN_state
        self.decoder_rnns = nn.ModuleList([nn.GRUCell(256, 256) for _ in range(2)])
        # RNN_state -> |Linear| -> mel_spec
        self.proj_to_mel = nn.Linear(256, frame_channels * self.r_init)
        # learn init values instead of zero init.
        self.stopnet = StopNet(256 + frame_channels * self.r_init)

    def set_r(self, new_r):
        self.r = new_r

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        # Grouping multiple frames if necessary
        if memory.size(-1) == self.frame_channels:
            memory = memory.view(memory.shape[0], memory.size(1) // self.r, -1)
        # Time first (T_decoder, B, frame_channels)
        memory = memory.transpose(0, 1)
        return memory

    def _init_states(self, inputs):
        """
        Initialization of decoder states
        """
        B = inputs.size(0)
        # go frame as zeros matrix
        if self.use_memory_queue:
            self.memory_input = torch.zeros(1, device=inputs.device).repeat(B, self.frame_channels * self.memory_size)
        else:
            self.memory_input = torch.zeros(1, device=inputs.device).repeat(B, self.frame_channels)
        # decoder states
        self.attention_rnn_hidden = torch.zeros(1, device=inputs.device).repeat(B, 256)
        self.decoder_rnn_hiddens = [
            torch.zeros(1, device=inputs.device).repeat(B, 256) for idx in range(len(self.decoder_rnns))
        ]
        self.context_vec = inputs.data.new(B, self.in_channels).zero_()
        # cache attention inputs
        self.processed_inputs = self.attention.preprocess_inputs(inputs)

    def _parse_outputs(self, outputs, attentions, stop_tokens):
        # Back to batch first
        attentions = torch.stack(attentions).transpose(0, 1)
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        outputs = outputs.view(outputs.size(0), -1, self.frame_channels)
        outputs = outputs.transpose(1, 2)
        return outputs, attentions, stop_tokens

    def decode(self, inputs, mask=None):
        # Prenet
        processed_memory = self.prenet(self.memory_input)
        # Attention RNN
        self.attention_rnn_hidden = self.attention_rnn(
            torch.cat((processed_memory, self.context_vec), -1), self.attention_rnn_hidden
        )
        self.context_vec = self.attention(self.attention_rnn_hidden, inputs, self.processed_inputs, mask)
        # Concat RNN output and attention context vector
        decoder_input = self.project_to_decoder_in(torch.cat((self.attention_rnn_hidden, self.context_vec), -1))

        # Pass through the decoder RNNs
        for idx, decoder_rnn in enumerate(self.decoder_rnns):
            self.decoder_rnn_hiddens[idx] = decoder_rnn(decoder_input, self.decoder_rnn_hiddens[idx])
            # Residual connection
            decoder_input = self.decoder_rnn_hiddens[idx] + decoder_input
        decoder_output = decoder_input

        # predict mel vectors from decoder vectors
        output = self.proj_to_mel(decoder_output)
        # output = torch.sigmoid(output)
        # predict stop token
        stopnet_input = torch.cat([decoder_output, output], -1)
        if self.separate_stopnet:
            stop_token = self.stopnet(stopnet_input.detach())
        else:
            stop_token = self.stopnet(stopnet_input)
        output = output[:, : self.r * self.frame_channels]
        return output, stop_token, self.attention.attention_weights

    def _update_memory_input(self, new_memory):
        if self.use_memory_queue:
            if self.memory_size > self.r:
                # memory queue size is larger than number of frames per decoder iter
                self.memory_input = torch.cat(
                    [new_memory, self.memory_input[:, : (self.memory_size - self.r) * self.frame_channels].clone()],
                    dim=-1,
                )
            else:
                # memory queue size smaller than number of frames per decoder iter
                self.memory_input = new_memory[:, : self.memory_size * self.frame_channels]
        else:
            # use only the last frame prediction
            # assert new_memory.shape[-1] == self.r * self.frame_channels
            self.memory_input = new_memory[:, self.frame_channels * (self.r - 1) :]

    def forward(self, inputs, memory, mask):
        """
        Args:
            inputs: Encoder outputs.
            memory: Decoder memory (autoregression. If None (at eval-time),
              decoder outputs are used as decoder inputs. If None, it uses the last
              output as the input.
            mask: Attention mask for sequence padding.

        Shapes:
            - inputs: (B, T, D_out_enc)
            - memory: (B, T_mel, D_mel)
        """
        # Run greedy decoding if memory is None
        memory = self._reshape_memory(memory)
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        self.attention.init_states(inputs)
        while len(outputs) < memory.size(0):
            if t > 0:
                new_memory = memory[t - 1]
                self._update_memory_input(new_memory)

            output, stop_token, attention = self.decode(inputs, mask)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token.squeeze(1)]
            t += 1
        return self._parse_outputs(outputs, attentions, stop_tokens)

    def inference(self, inputs):
        """
        Args:
            inputs: encoder outputs.
        Shapes:
            - inputs: batch x time x encoder_out_dim
        """
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        self.attention.init_states(inputs)
        while True:
            if t > 0:
                new_memory = outputs[-1]
                self._update_memory_input(new_memory)
            output, stop_token, attention = self.decode(inputs, None)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            t += 1
            if t > inputs.shape[1] / 4 and (stop_token > 0.6 or attention[:, -1].item() > 0.6):
                break
            if t > self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break
        return self._parse_outputs(outputs, attentions, stop_tokens)


class StopNet(nn.Module):
    r"""Stopnet signalling decoder to stop inference.
    Args:
        in_features (int): feature dimension of input.
    """

    def __init__(self, in_features):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=torch.nn.init.calculate_gain("linear"))

    def forward(self, inputs):
        outputs = self.dropout(inputs)
        outputs = self.linear(outputs)
        return outputs

"""--------------------------------------------------------------------------------------------"""
import os
import sys
import threading
import time

import pytest

import audbackend
import audeer
import audformat.testing

import audb


class SlowFileSystem(audbackend.FileSystem):
    r"""Emulate a slow file system.

    Introduces a short delay when getting a file from the backend.
    This ensures that timeouts are reached in the tests.

    """
    def _get_file(self, *args):
        time.sleep(0.1)
        super()._get_file(*args)


audbackend.register(
    'slow-file-system',
    SlowFileSystem,
)


class CrashFileSystem(audbackend.FileSystem):
    r"""Emulate a file system that crashes.

    Raises an exception when getting a file from the backend.

    """
    def _get_file(self, *args):
        assert any([os.path.exists(path) for path in DB_LOCK_PATHS])
        raise RuntimeError()


audbackend.register(
    'crash-file-system',
    CrashFileSystem,
)


os.environ['AUDB_CACHE_ROOT'] = pytest.CACHE_ROOT
os.environ['AUDB_SHARED_CACHE_ROOT'] = pytest.SHARED_CACHE_ROOT


@pytest.fixture(
    scope='function',
    autouse=True,
)
def fixture_ensure_lock_file_deleted():
    assert not any([os.path.exists(path) for path in DB_LOCK_PATHS])
    yield
    assert not any([os.path.exists(path) for path in DB_LOCK_PATHS])


@pytest.fixture(
    scope='function',
    autouse=True,
)
def fixture_set_repositories(request):
    audb.config.REPOSITORIES = [
        audb.Repository(
            name=pytest.REPOSITORY_NAME,
            host=pytest.FILE_SYSTEM_HOST,
            backend=request.param,
        ),
    ]


DB_NAME = f'test_lock-{pytest.ID}'
DB_ROOT = os.path.join(pytest.ROOT, 'db')
DB_VERSIONS = ['1.0.0', '2.0.0']

DB_LOCK_PATHS = []
for version in DB_VERSIONS:
    DB_LOCK_PATHS.append(
        audeer.path(
            pytest.CACHE_ROOT,
            DB_NAME,
            version,
            '.lock',
        )
    )
    DB_LOCK_PATHS.append(
        audeer.path(
            pytest.CACHE_ROOT,
            DB_NAME,
            version,
            audb.Flavor().short_id,
            '.lock',
        )
    )


def clear_root(root: str):
    audeer.rmdir(root)


@pytest.fixture(
    scope='function',
    autouse=True,
)
def fixture_remove_db_from_cache():
    root = audeer.path(pytest.CACHE_ROOT, DB_NAME)
    clear_root(root)


@pytest.fixture(
    scope='module',
    autouse=True,
)
def fixture_publish_db():

    audb.config.REPOSITORIES = pytest.REPOSITORIES

    clear_root(DB_ROOT)
    clear_root(pytest.FILE_SYSTEM_HOST)

    # create db

    db = audformat.testing.create_db(minimal=True)
    db.name = DB_NAME
    db.schemes['scheme'] = audformat.Scheme()
    audformat.testing.add_table(
        db,
        'table',
        'filewise',
        num_files=[0, 1, 2],
    )
    db.save(DB_ROOT)
    audformat.testing.create_audio_files(db)

    # publish 1.0.0

    audb.publish(
        DB_ROOT,
        DB_VERSIONS[0],
        pytest.PUBLISH_REPOSITORY,
        verbose=False,
    )

    # publish 2.0.0

    audformat.testing.add_table(
        db,
        'empty',
        'filewise',
        num_files=0,
    )
    db.save(DB_ROOT)
    audb.publish(
        DB_ROOT,
        DB_VERSIONS[1],
        pytest.PUBLISH_REPOSITORY,
        previous_version=DB_VERSIONS[0],
        verbose=False,
    )

    yield

    clear_root(DB_ROOT)
    clear_root(pytest.FILE_SYSTEM_HOST)


def load_deps():
    return audb.dependencies(
        DB_NAME,
        version=DB_VERSIONS[0],
    )


@pytest.mark.parametrize(
    'fixture_set_repositories',
    ['slow-file-system'],
    indirect=True,
)
@pytest.mark.parametrize(
    'multiprocessing',
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    'num_workers',
    [
        10,
    ]
)
def test_lock_dependencies(fixture_set_repositories, multiprocessing,
                           num_workers):

    # avoid
    # AttributeError: module pytest has no attribute CACHE_ROOT
    # when multiprocessing=True on Windows and macOS
    if multiprocessing and sys.platform in ['win32', 'darwin']:
        return

    result = audeer.run_tasks(
        load_deps,
        [([], {})] * num_workers,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
    )

    assert len(result) == num_workers


def load_header():
    return audb.info.header(
        DB_NAME,
        version=DB_VERSIONS[0],
    )


@pytest.mark.parametrize(
    'fixture_set_repositories',
    ['slow-file-system'],
    indirect=True,
)
@pytest.mark.parametrize(
    'multiprocessing',
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    'num_workers',
    [
        10,
    ]
)
def test_lock_header(fixture_set_repositories, multiprocessing, num_workers):

    # avoid
    # AttributeError: module pytest has no attribute CACHE_ROOT
    # when multiprocessing=True on Windows and macOS
    if multiprocessing and sys.platform in ['win32', 'darwin']:
        return

    result = audeer.run_tasks(
        load_header,
        [([], {})] * num_workers,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
    )

    assert len(result) == num_workers


def load_db(timeout):
    return audb.load(
        DB_NAME,
        version=DB_VERSIONS[0],
        timeout=timeout,
        verbose=False,
    )


@pytest.mark.parametrize(
    'fixture_set_repositories',
    ['slow-file-system'],
    indirect=True,
)
@pytest.mark.parametrize(
    'multiprocessing',
    [
        False,
        True,
    ]
)
@pytest.mark.parametrize(
    'num_workers, timeout, expected',
    [
        (2, -1, 2),
        (2, 9999, 2),
        (2, 0, 1),
    ]
)
def test_lock_load(fixture_set_repositories, multiprocessing, num_workers,
                   timeout, expected):

    # avoid
    # AttributeError: module pytest has no attribute CACHE_ROOT
    # when multiprocessing=True on Windows and macOS
    if multiprocessing and sys.platform in ['win32', 'darwin']:
        return

    warns = not multiprocessing and num_workers != expected
    with pytest.warns(
            UserWarning if warns else None,
            match=audb.core.define.TIMEOUT_MSG,
    ):
        result = audeer.run_tasks(
            load_db,
            [([timeout], {})] * num_workers,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
        )
    result = [x for x in result if x is not None]

    assert len(result) == expected


@pytest.mark.parametrize(
    'fixture_set_repositories',
    ['crash-file-system'],
    indirect=True,
)
def test_lock_load_crash(fixture_set_repositories):

    with pytest.raises(RuntimeError):
        load_db(-1)


@pytest.mark.parametrize(
    'fixture_set_repositories',
    ['file-system'],
    indirect=True,
)
def test_lock_load_from_cached_versions(fixture_set_repositories):

    # ensure immediate timeout if cache folder is locked
    cached_version_timeout = audb.core.define.CACHED_VERSIONS_TIMEOUT
    audb.core.define.CACHED_VERSIONS_TIMEOUT = 0

    # load version 1.0.0
    db_v1 = audb.load(
        DB_NAME,
        version=DB_VERSIONS[0],
        verbose=False,
    )

    # load new files added in version 2.0.0
    audb.load(
        DB_NAME,
        version=DB_VERSIONS[1],
        tables='empty',
        verbose=False,
    )

    # switch to crash backend to ensure remaining files
    # must be copied from version 1.0.0
    audb.config.REPOSITORIES = [
        audb.Repository(
            name=pytest.REPOSITORY_NAME,
            host=pytest.FILE_SYSTEM_HOST,
            backend='crash-file-system',
        ),
    ]

    # lock cache folder of version 1.0.0
    def lock_v1():
        with audb.core.lock.FolderLock(db_v1.root):
            event.wait()

    event = threading.Event()
    thread = threading.Thread(target=lock_v1)
    thread.start()

    # -> loading missing table from cache fails
    with pytest.raises(RuntimeError):
        audb.load(
            DB_NAME,
            version=DB_VERSIONS[1],
            tables='table',
            only_metadata=True,
            verbose=False,
        )

    # release cache folder of version 1.0.0
    event.set()
    thread.join()

    # -> loading missing table from cache succeeds
    audb.load(
        DB_NAME,
        version=DB_VERSIONS[1],
        tables='table',
        only_metadata=True,
        verbose=False,
    )

    # lock cache folder of version 1.0.0
    event.clear()
    thread = threading.Thread(target=lock_v1)
    thread.start()

    # -> loading missing media from cache fails
    with pytest.raises(RuntimeError):
        audb.load(
            DB_NAME,
            version=DB_VERSIONS[1],
            verbose=False,
        )

    # release cache folder of version 1.0.0
    event.set()
    thread.join()

    # -> loading missing media from cache succeeds
    audb.load(
        DB_NAME,
        version=DB_VERSIONS[1],
        verbose=False,
    )

    # reset timeout
    audb.core.define.CACHED_VERSIONS_TIMEOUT = cached_version_timeout


def load_media(timeout):
    return audb.load_media(
        DB_NAME,
        'audio/001.wav',
        version=DB_VERSIONS[0],
        timeout=timeout,
        verbose=False,
    )


@pytest.mark.parametrize(
    'fixture_set_repositories',
    ['slow-file-system'],
    indirect=True,
)
@pytest.mark.parametrize(
    'multiprocessing',
    [
        False,
        True,
    ]
)
@pytest.mark.parametrize(
    'num_workers, timeout, expected',
    [
        (2, -1, 2),
        (2, 9999, 2),
        (2, 0, 1),
    ]
)
def test_lock_load_media(fixture_set_repositories, multiprocessing,
                         num_workers, timeout, expected):

    # avoid
    # AttributeError: module pytest has no attribute CACHE_ROOT
    # when multiprocessing=True on Windows and macOS
    if multiprocessing and sys.platform in ['win32', 'darwin']:
        return

    warns = not multiprocessing and num_workers != expected
    with pytest.warns(
            UserWarning if warns else None,
            match=audb.core.define.TIMEOUT_MSG,
    ):
        result = audeer.run_tasks(
            load_media,
            [([timeout], {})] * num_workers,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
        )
    result = [x for x in result if x is not None]

    assert len(result) == expected


def load_table():
    return audb.load_table(
        DB_NAME,
        'table',
        version=DB_VERSIONS[0],
        verbose=False,
    )


@pytest.mark.parametrize(
    'fixture_set_repositories',
    ['slow-file-system'],
    indirect=True,
)
@pytest.mark.parametrize(
    'multiprocessing',
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    'num_workers',
    [
        10,
    ]
)
def test_lock_load_table(fixture_set_repositories, multiprocessing,
                         num_workers):

    # avoid
    # AttributeError: module pytest has no attribute CACHE_ROOT
    # when multiprocessing=True on Windows and macOS
    if multiprocessing and sys.platform in ['win32', 'darwin']:
        return

    result = audeer.run_tasks(
        load_table,
        [([], {})] * num_workers,
        num_workers=num_workers,
        multiprocessing=multiprocessing,
    )

    assert len(result) == num_workers
"""-------------------------------------------------------------------------------------------------------------------------------------"""
import math
import os
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import TTSDataset, _parse_sample
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.vits.networks import PosteriorEncoder, ResidualCouplingBlocks, TextEncoder
from TTS.tts.layers.vits.stochastic_duration_predictor import StochasticDurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import generate_path, maximum_path, rand_segments, segment, sequence_mask
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.characters import BaseCharacters, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results

##############################
# IO / Feature extraction
##############################

# pylint: disable=global-statement
hann_window = {}
mel_basis = {}


def load_audio(file_path):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, sr = torchaudio.load(file_path)
    assert (x > 1).sum() + (x < -1).sum() == 0
    return x, sr


def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _db_to_amp(x, C=1):
    return torch.exp(x) / C


def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output


def db_to_amp(magnitudes):
    output = _db_to_amp(magnitudes)
    return output


def wav_to_spec(y, n_fft, hop_length, win_length, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`

    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel = amp_to_db(mel)
    return mel


def wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = amp_to_db(spec)
    return spec


##############################
# DATASET
##############################


class VitsDataset(TTSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = self.tokenizer.characters.pad_id

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_text = item["text"]

        wav, _ = load_audio(item["audio_file"])
        wav_filename = os.path.basename(item["audio_file"])

        token_ids = self.get_token_ids(idx, item["text"])

        # after phonemization the text length may change
        # this is a shameful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix
        if len(token_ids) > self.max_text_len or wav.shape[1] < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.__getitem__(self.rescue_item_idx)

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
        }

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = os.path.getsize(wav_file) / 16 * 8  # assuming 16bit audio
            lens.append(audio_len)
        return lens

    def collate_fn(self, batch):
        """
        Return Shapes:
            - tokens: :math:`[B, T]`
            - token_lens :math:`[B]`
            - token_rel_lens :math:`[B]`
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speaker_names: :math:`[B]`
            - language_names: :math:`[B]`
            - audiofile_paths: :math:`[B]`
            - raw_texts: :math:`[B]`
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in batch["wav"]]), dim=0, descending=True
        )

        max_text_len = max([len(x) for x in batch["token_ids"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        token_padded = token_padded.zero_() + self.pad_id
        wav_padded = wav_padded.zero_() + self.pad_id
        for i in range(len(ids_sorted_decreasing)):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)

            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,  # (B x T)
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
        }


##############################
# MODEL DEFINITION
##############################


@dataclass
class VitsArgs(Coqpit):
    """VITS model arguments.

    Args:

        num_chars (int):
            Number of characters in the vocabulary. Defaults to 100.

        out_channels (int):
            Number of output channels of the decoder. Defaults to 513.

        spec_segment_size (int):
            Decoder input segment size. Defaults to 32 `(32 * hoplength = waveform length)`.

        hidden_channels (int):
            Number of hidden channels of the model. Defaults to 192.

        hidden_channels_ffn_text_encoder (int):
            Number of hidden channels of the feed-forward layers of the text encoder transformer. Defaults to 256.

        num_heads_text_encoder (int):
            Number of attention heads of the text encoder transformer. Defaults to 2.

        num_layers_text_encoder (int):
            Number of transformer layers in the text encoder. Defaults to 6.

        kernel_size_text_encoder (int):
            Kernel size of the text encoder transformer FFN layers. Defaults to 3.

        dropout_p_text_encoder (float):
            Dropout rate of the text encoder. Defaults to 0.1.

        dropout_p_duration_predictor (float):
            Dropout rate of the duration predictor. Defaults to 0.1.

        kernel_size_posterior_encoder (int):
            Kernel size of the posterior encoder's WaveNet layers. Defaults to 5.

        dilatation_posterior_encoder (int):
            Dilation rate of the posterior encoder's WaveNet layers. Defaults to 1.

        num_layers_posterior_encoder (int):
            Number of posterior encoder's WaveNet layers. Defaults to 16.

        kernel_size_flow (int):
            Kernel size of the Residual Coupling layers of the flow network. Defaults to 5.

        dilatation_flow (int):
            Dilation rate of the Residual Coupling WaveNet layers of the flow network. Defaults to 1.

        num_layers_flow (int):
            Number of Residual Coupling WaveNet layers of the flow network. Defaults to 6.

        resblock_type_decoder (str):
            Type of the residual block in the decoder network. Defaults to "1".

        resblock_kernel_sizes_decoder (List[int]):
            Kernel sizes of the residual blocks in the decoder network. Defaults to `[3, 7, 11]`.

        resblock_dilation_sizes_decoder (List[List[int]]):
            Dilation sizes of the residual blocks in the decoder network. Defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`.

        upsample_rates_decoder (List[int]):
            Upsampling rates for each concecutive upsampling layer in the decoder network. The multiply of these
            values must be equal to the kop length used for computing spectrograms. Defaults to `[8, 8, 2, 2]`.

        upsample_initial_channel_decoder (int):
            Number of hidden channels of the first upsampling convolution layer of the decoder network. Defaults to 512.

        upsample_kernel_sizes_decoder (List[int]):
            Kernel sizes for each upsampling layer of the decoder network. Defaults to `[16, 16, 4, 4]`.

        use_sdp (bool):
            Use Stochastic Duration Predictor. Defaults to True.

        noise_scale (float):
            Noise scale used for the sample noise tensor in training. Defaults to 1.0.

        inference_noise_scale (float):
            Noise scale used for the sample noise tensor in inference. Defaults to 0.667.

        length_scale (float):
            Scale factor for the predicted duration values. Smaller values result faster speech. Defaults to 1.

        noise_scale_dp (float):
            Noise scale used by the Stochastic Duration Predictor sample noise in training. Defaults to 1.0.

        inference_noise_scale_dp (float):
            Noise scale for the Stochastic Duration Predictor in inference. Defaults to 0.8.

        max_inference_len (int):
            Maximum inference length to limit the memory use. Defaults to None.

        init_discriminator (bool):
            Initialize the disciminator network if set True. Set False for inference. Defaults to True.

        use_spectral_norm_disriminator (bool):
            Use spectral normalization over weight norm in the discriminator. Defaults to False.

        use_speaker_embedding (bool):
            Enable/Disable speaker embedding for multi-speaker models. Defaults to False.

        num_speakers (int):
            Number of speakers for the speaker embedding layer. Defaults to 0.

        speakers_file (str):
            Path to the speaker mapping file for the Speaker Manager. Defaults to None.

        speaker_embedding_channels (int):
            Number of speaker embedding channels. Defaults to 256.

        use_d_vector_file (bool):
            Enable/Disable the use of d-vectors for multi-speaker training. Defaults to False.

        d_vector_file (str):
            Path to the file including pre-computed speaker embeddings. Defaults to None.

        d_vector_dim (int):
            Number of d-vector channels. Defaults to 0.

        detach_dp_input (bool):
            Detach duration predictor's input from the network for stopping the gradients. Defaults to True.

        use_language_embedding (bool):
            Enable/Disable language embedding for multilingual models. Defaults to False.

        embedded_language_dim (int):
            Number of language embedding channels. Defaults to 4.

        num_languages (int):
            Number of languages for the language embedding layer. Defaults to 0.

        language_ids_file (str):
            Path to the language mapping file for the Language Manager. Defaults to None.

        use_speaker_encoder_as_loss (bool):
            Enable/Disable Speaker Consistency Loss (SCL). Defaults to False.

        speaker_encoder_config_path (str):
            Path to the file speaker encoder config file, to use for SCL. Defaults to "".

        speaker_encoder_model_path (str):
            Path to the file speaker encoder checkpoint file, to use for SCL. Defaults to "".

        condition_dp_on_speaker (bool):
            Condition the duration predictor on the speaker embedding. Defaults to True.

        freeze_encoder (bool):
            Freeze the encoder weigths during training. Defaults to False.

        freeze_DP (bool):
            Freeze the duration predictor weigths during training. Defaults to False.

        freeze_PE (bool):
            Freeze the posterior encoder weigths during training. Defaults to False.

        freeze_flow_encoder (bool):
            Freeze the flow encoder weigths during training. Defaults to False.

        freeze_waveform_decoder (bool):
            Freeze the waveform decoder weigths during training. Defaults to False.
    """

    num_chars: int = 100
    out_channels: int = 513
    spec_segment_size: int = 32
    hidden_channels: int = 192
    hidden_channels_ffn_text_encoder: int = 768
    num_heads_text_encoder: int = 2
    num_layers_text_encoder: int = 6
    kernel_size_text_encoder: int = 3
    dropout_p_text_encoder: float = 0.1
    dropout_p_duration_predictor: float = 0.5
    kernel_size_posterior_encoder: int = 5
    dilation_rate_posterior_encoder: int = 1
    num_layers_posterior_encoder: int = 16
    kernel_size_flow: int = 5
    dilation_rate_flow: int = 1
    num_layers_flow: int = 4
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    use_sdp: bool = True
    noise_scale: float = 1.0
    inference_noise_scale: float = 0.667
    length_scale: float = 1
    noise_scale_dp: float = 1.0
    inference_noise_scale_dp: float = 1.0
    max_inference_len: int = None
    init_discriminator: bool = True
    use_spectral_norm_disriminator: bool = False
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: str = None
    speaker_embedding_channels: int = 256
    use_d_vector_file: bool = False
    d_vector_dim: int = 0
    detach_dp_input: bool = True
    use_language_embedding: bool = False
    embedded_language_dim: int = 4
    num_languages: int = 0
    language_ids_file: str = None
    use_speaker_encoder_as_loss: bool = False
    speaker_encoder_config_path: str = ""
    speaker_encoder_model_path: str = ""
    condition_dp_on_speaker: bool = True
    freeze_encoder: bool = False
    freeze_DP: bool = False
    freeze_PE: bool = False
    freeze_flow_decoder: bool = False
    freeze_waveform_decoder: bool = False


class Vits(BaseTTS):
    """VITS TTS model

    Paper::
        https://arxiv.org/pdf/2106.06103.pdf

    Paper Abstract::
        Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel
        sampling have been proposed, but their sample quality does not match that of two-stage TTS systems.
        In this work, we present a parallel endto-end TTS method that generates more natural sounding audio than
        current two-stage models. Our method adopts variational inference augmented with normalizing flows and
        an adversarial training process, which improves the expressive power of generative modeling. We also propose a
        stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the
        uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the
        natural one-to-many relationship in which a text input can be spoken in multiple ways
        with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS)
        on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly
        available TTS systems and achieves a MOS comparable to ground truth.

    Check :class:`TTS.tts.configs.vits_config.VitsConfig` for class arguments.

    Examples:
        >>> from TTS.tts.configs.vits_config import VitsConfig
        >>> from TTS.tts.models.vits import Vits
        >>> config = VitsConfig()
        >>> model = Vits(config)
    """

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
    ):

        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)

        self.init_multispeaker(config)
        self.init_multilingual(config)

        self.length_scale = self.args.length_scale
        self.noise_scale = self.args.noise_scale
        self.inference_noise_scale = self.args.inference_noise_scale
        self.inference_noise_scale_dp = self.args.inference_noise_scale_dp
        self.noise_scale_dp = self.args.noise_scale_dp
        self.max_inference_len = self.args.max_inference_len
        self.spec_segment_size = self.args.spec_segment_size

        self.text_encoder = TextEncoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )

        self.posterior_encoder = PosteriorEncoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_posterior_encoder,
            dilation_rate=self.args.dilation_rate_posterior_encoder,
            num_layers=self.args.num_layers_posterior_encoder,
            cond_channels=self.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )

        if self.args.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                self.args.hidden_channels,
                192,
                3,
                self.args.dropout_p_duration_predictor,
                4,
                cond_channels=self.embedded_speaker_dim if self.args.condition_dp_on_speaker else 0,
                language_emb_dim=self.embedded_language_dim,
            )
        else:
            self.duration_predictor = DurationPredictor(
                self.args.hidden_channels,
                256,
                3,
                self.args.dropout_p_duration_predictor,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=self.embedded_language_dim,
            )

        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(use_spectral_norm=self.args.use_spectral_norm_disriminator)

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.embedded_speaker_dim = 0
        self.num_speakers = self.args.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

        # TODO: make this a function
        if self.args.use_speaker_encoder_as_loss:
            if self.speaker_manager.encoder is None and (
                not self.args.speaker_encoder_model_path or not self.args.speaker_encoder_config_path
            ):
                raise RuntimeError(
                    " [!] To use the speaker consistency loss (SCL) you need to specify speaker_encoder_model_path and speaker_encoder_config_path !!"
                )

            self.speaker_manager.encoder.eval()
            print(" > External Speaker Encoder Loaded !!")

            if (
                hasattr(self.speaker_manager.encoder, "audio_config")
                and self.config.audio["sample_rate"] != self.speaker_manager.encoder.audio_config["sample_rate"]
            ):
                self.audio_transform = torchaudio.transforms.Resample(
                    orig_freq=self.audio_config["sample_rate"],
                    new_freq=self.speaker_manager.encoder.audio_config["sample_rate"],
                )
            # pylint: disable=W0101,W0105
            self.audio_transform = torchaudio.transforms.Resample(
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.speaker_manager.encoder.audio_config["sample_rate"],
            )

    def _init_speaker_embedding(self):
        # pylint: disable=attribute-defined-outside-init
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.args.speaker_embedding_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

    def _init_d_vector(self):
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.embedded_speaker_dim = self.args.d_vector_dim

    def init_multilingual(self, config: Coqpit):
        """Initialize multilingual modules of a model.

        Args:
            config (Coqpit): Model configuration.
        """
        if self.args.language_ids_file is not None:
            self.language_manager = LanguageManager(language_ids_file_path=config.language_ids_file)

        if self.args.use_language_embedding and self.language_manager:
            print(" > initialization of language-embedding layers.")
            self.num_languages = self.language_manager.num_languages
            self.embedded_language_dim = self.args.embedded_language_dim
            self.emb_l = nn.Embedding(self.num_languages, self.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l.weight)
        else:
            self.embedded_language_dim = 0

    def get_aux_input(self, aux_input: Dict):
        sid, g, lid = self._set_cond_input(aux_input)
        return {"speaker_ids": sid, "style_wav": None, "d_vectors": g, "language_ids": lid}

    def _freeze_layers(self):
        if self.args.freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if hasattr(self, "emb_l"):
                for param in self.emb_l.parameters():
                    param.requires_grad = False

        if self.args.freeze_PE:
            for param in self.posterior_encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_DP:
            for param in self.duration_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_flow_decoder:
            for param in self.flow.parameters():
                param.requires_grad = False

        if self.args.freeze_waveform_decoder:
            for param in self.waveform_decoder.parameters():
                param.requires_grad = False

    @staticmethod
    def _set_cond_input(aux_input: Dict):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid = None, None, None
        if "speaker_ids" in aux_input and aux_input["speaker_ids"] is not None:
            sid = aux_input["speaker_ids"]
            if sid.ndim == 0:
                sid = sid.unsqueeze_(0)
        if "d_vectors" in aux_input and aux_input["d_vectors"] is not None:
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if "language_ids" in aux_input and aux_input["language_ids"] is not None:
            lid = aux_input["language_ids"]
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)

        return sid, g, lid

    def _set_speaker_input(self, aux_input: Dict):
        d_vectors = aux_input.get("d_vectors", None)
        speaker_ids = aux_input.get("speaker_ids", None)

        if d_vectors is not None and speaker_ids is not None:
            raise ValueError("[!] Cannot use d-vectors and speaker-ids together.")

        if speaker_ids is not None and not hasattr(self, "emb_g"):
            raise ValueError("[!] Cannot use speaker-ids without enabling speaker embedding.")

        g = speaker_ids if speaker_ids is not None else d_vectors
        return g

    def forward_mas(self, outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g, lang_emb):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        if self.args.use_sdp:
            loss_duration = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                attn_durations,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn

    def forward(  # pylint: disable=dangerous-default-value
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None},
    ) -> Dict:
        """Forward pass of the model.

        Args:
            x (torch.tensor): Batch of input character sequence IDs.
            x_lengths (torch.tensor): Batch of input character sequence lengths.
            y (torch.tensor): Batch of input spectrograms.
            y_lengths (torch.tensor): Batch of input spectrogram lengths.
            waveform (torch.tensor): Batch of ground truth waveforms per sample.
            aux_input (dict, optional): Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.

        Returns:
            Dict: model outputs keyed by the output name.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - y: :math:`[B, C, T_spec]`
            - y_lengths: :math:`[B]`
            - waveform: :math:`[B, 1, T_wav]`
            - d_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
            - language_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
            - m_q: :math:`[B, C, T_dec]`
            - logs_q: :math:`[B, C, T_dec]`
            - waveform_seg: :math:`[B, 1, spec_seg_size * hop_length]`
            - gt_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            - syn_spk_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
        """
        outputs = {}
        sid, g, lid = self._set_cond_input(aux_input)
        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        # duration predictor
        outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g, lang_emb=lang_emb)

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)
        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            self.args.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            # pylint: disable=W0105
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_manager.encoder.forward(wavs_batch, l2_norm=True)

            # split generated and GT speaker embeddings
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
                "slice_ids": slice_ids,
            }
        )
        return outputs

    @staticmethod
    def _set_x_lengths(x, aux_input):
        if "x_lengths" in aux_input and aux_input["x_lengths"] is not None:
            return aux_input["x_lengths"]
        return torch.tensor(x.shape[1:2]).to(x.device)

    def inference(
        self, x, aux_input={"x_lengths": None, "d_vectors": None, "speaker_ids": None, "language_ids": None}
    ):  # pylint: disable=dangerous-default-value
        """
        Note:
            To run in batch mode, provide `x_lengths` else model assumes that the batch size is 1.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - d_vectors: :math:`[B, C]`
            - speaker_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
        """
        sid, g, lid = self._set_cond_input(aux_input)
        x_lengths = self._set_x_lengths(x, aux_input)

        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)

        if self.args.use_sdp:
            logw = self.duration_predictor(
                x,
                x_mask,
                g=g if self.args.condition_dp_on_speaker else None,
                reverse=True,
                noise_scale=self.inference_noise_scale_dp,
                lang_emb=lang_emb,
            )
        else:
            logw = self.duration_predictor(
                x, x_mask, g=g if self.args.condition_dp_on_speaker else None, lang_emb=lang_emb
            )

        w = torch.exp(logw) * x_mask * self.length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        outputs = {"model_outputs": o, "alignments": attn.squeeze(1), "z": z, "z_p": z_p, "m_p": m_p, "logs_p": logs_p}
        return outputs

    @torch.no_grad()
    def inference_voice_conversion(
        self, reference_wav, speaker_id=None, d_vector=None, reference_speaker_id=None, reference_d_vector=None
    ):
        """Inference for voice conversion

        Args:
            reference_wav (Tensor): Reference wavform. Tensor of shape [B, T]
            speaker_id (Tensor): speaker_id of the target speaker. Tensor of shape [B]
            d_vector (Tensor): d_vector embedding of target speaker. Tensor of shape `[B, C]`
            reference_speaker_id (Tensor): speaker_id of the reference_wav speaker. Tensor of shape [B]
            reference_d_vector (Tensor): d_vector embedding of the reference_wav speaker. Tensor of shape `[B, C]`
        """
        # compute spectrograms
        y = wav_to_spec(
            reference_wav,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False,
        ).transpose(1, 2)
        y_lengths = torch.tensor([y.size(-1)]).to(y.device)
        speaker_cond_src = reference_speaker_id if reference_speaker_id is not None else reference_d_vector
        speaker_cond_tgt = speaker_id if speaker_id is not None else d_vector
        # print(y.shape, y_lengths.shape)
        wav, _, _ = self.voice_conversion(y, y_lengths, speaker_cond_src, speaker_cond_tgt)
        return wav

    def voice_conversion(self, y, y_lengths, speaker_cond_src, speaker_cond_tgt):
        """Forward pass for voice conversion

        TODO: create an end-point for voice conversion

        Args:
            y (Tensor): Reference spectrograms. Tensor of shape [B, T, C]
            y_lengths (Tensor): Length of each reference spectrogram. Tensor of shape [B]
            speaker_cond_src (Tensor): Reference speaker ID. Tensor of shape [B,]
            speaker_cond_tgt (Tensor): Target speaker ID. Tensor of shape [B,]
        """
        assert self.num_speakers > 0, "num_speakers have to be larger than 0."
        # speaker embedding
        if self.args.use_speaker_embedding and not self.args.use_d_vector_file:
            g_src = self.emb_g(speaker_cond_src).unsqueeze(-1)
            g_tgt = self.emb_g(speaker_cond_tgt).unsqueeze(-1)
        elif not self.args.use_speaker_embedding and self.args.use_d_vector_file:
            g_src = F.normalize(speaker_cond_src).unsqueeze(-1)
            g_tgt = F.normalize(speaker_cond_tgt).unsqueeze(-1)
        else:
            raise RuntimeError(" [!] Voice conversion is only supported on multi-speaker models.")

        z, _, _, y_mask = self.posterior_encoder(y.transpose(1, 2), y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """

        self._freeze_layers()

        mel_lens = batch["mel_lens"]

        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lenghts = batch["token_lens"]
            spec = batch["spec"]
            spec_lens = batch["spec_lens"]

            d_vectors = batch["d_vectors"]
            speaker_ids = batch["speaker_ids"]
            language_ids = batch["language_ids"]
            waveform = batch["waveform"]

            # generator pass
            outputs = self.forward(
                tokens,
                token_lenghts,
                spec,
                spec_lens,
                waveform,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores and features
            scores_disc_fake, _, scores_disc_real, _ = self.disc(
                outputs["model_outputs"].detach(), outputs["waveform_seg"]
            )

            # compute loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_real,
                    scores_disc_fake,
                )
            return outputs, loss_dict

        if optimizer_idx == 1:
            mel = batch["mel"]

            # compute melspec segment
            with autocast(enabled=False):
                mel_slice = segment(
                    mel.float(), self.model_outputs_cache["slice_ids"], self.spec_segment_size, pad_short=True
                )
                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            # compute discriminator scores and features
            scores_disc_fake, feats_disc_fake, _, feats_disc_real = self.disc(
                self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    mel_slice_hat=mel_slice.float(),
                    mel_slice=mel_slice_hat.float(),
                    z_p=self.model_outputs_cache["z_p"].float(),
                    logs_q=self.model_outputs_cache["logs_q"].float(),
                    m_p=self.model_outputs_cache["m_p"].float(),
                    logs_p=self.model_outputs_cache["logs_p"].float(),
                    z_len=mel_lens,
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    loss_duration=self.model_outputs_cache["loss_duration"],
                    use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                    gt_spk_emb=self.model_outputs_cache["gt_spk_emb"],
                    syn_spk_emb=self.model_outputs_cache["syn_spk_emb"],
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def _log(self, ap, batch, outputs, name_prefix="train"):  # pylint: disable=unused-argument,no-self-use
        y_hat = outputs[1]["model_outputs"]
        y = outputs[1]["waveform_seg"]
        figures = plot_results(y_hat, y, ap, name_prefix)
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios = {f"{name_prefix}/audio": sample_voice}

        alignments = outputs[1]["alignments"]
        align_img = alignments[0].data.cpu().numpy().T

        figures.update(
            {
                "alignment": plot_alignment(align_img, output_fig=False),
            }
        )
        return figures, audios

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ):  # pylint: disable=no-self-use
        """Create visualizations and waveform examples.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            ap (AudioProcessor): audio processor used at training.
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previoud training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        figures, audios = self._log(self.ap, batch, outputs, "train")
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._log(self.ap, batch, outputs, "eval")
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    def get_aux_input_from_test_sentences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name = None, None, None, None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker_name = sentence_info
            elif len(sentence_info) == 3:
                text, speaker_name, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker_name, style_wav, language_name = sentence_info
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embeddings()
                else:
                    d_vector = self.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.ids[speaker_name]

        # get language id
        if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.ids[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": language_id,
            "language_name": language_name,
        }

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(s_info)
            wav, alignment, _, _ = synthesis(
                self,
                aux_inputs["text"],
                self.config,
                "cuda" in str(next(self.parameters()).device),
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                language_id=aux_inputs["language_id"],
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()
            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment.T, output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        logger.test_figures(steps, outputs["figures"])

    def format_batch(self, batch: Dict) -> Dict:
        """Compute speaker, langugage IDs and d_vector for the batch if necessary."""
        speaker_ids = None
        language_ids = None
        d_vectors = None

        # get numerical speaker ids from speaker names
        if self.speaker_manager is not None and self.speaker_manager.ids and self.args.use_speaker_embedding:
            speaker_ids = [self.speaker_manager.ids[sn] for sn in batch["speaker_names"]]

        if speaker_ids is not None:
            speaker_ids = torch.LongTensor(speaker_ids)
            batch["speaker_ids"] = speaker_ids

        # get d_vectors from audio file names
        if self.speaker_manager is not None and self.speaker_manager.embeddings and self.args.use_d_vector_file:
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors = [d_vector_mapping[w]["embedding"] for w in batch["audio_files"]]
            d_vectors = torch.FloatTensor(d_vectors)

        # get language ids from language names
        if self.language_manager is not None and self.language_manager.ids and self.args.use_language_embedding:
            language_ids = [self.language_manager.ids[ln] for ln in batch["language_names"]]

        if language_ids is not None:
            language_ids = torch.LongTensor(language_ids)

        batch["language_ids"] = language_ids
        batch["d_vectors"] = d_vectors
        batch["speaker_ids"] = speaker_ids
        return batch

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        ac = self.config.audio

        # compute spectrograms
        batch["spec"] = wav_to_spec(batch["waveform"], ac.fft_size, ac.hop_length, ac.win_length, center=False)
        batch["mel"] = spec_to_mel(
            spec=batch["spec"],
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
        )
        assert batch["spec"].shape[2] == batch["mel"].shape[2], f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"

        # compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()
        assert (batch["spec_lens"] - batch["mel_lens"]).sum() == 0

        # zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        return batch

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = VitsDataset(
                samples=samples,
                # batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                verbose=verbose,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)

            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=False,  # shuffle is done in the dataset.
                drop_last=False,  # setting this False might cause issues in AMP training.
                sampler=sampler,
                collate_fn=dataset.collate_fn,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
            )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        # select generator parameters
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_disc, self.config.lr_gen]

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in
        `train_step()`"""
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            VitsDiscriminatorLoss,
            VitsGeneratorLoss,
        )

        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        # compat band-aid for the pre-trained models to not use the encoder baked into the model
        # TODO: consider baking the speaker encoder into the model and call it from there.
        # as it is probably easier for model distribution.
        state["model"] = {k: v for k, v in state["model"].items() if "speaker_encoder" not in k}
        # handle fine-tuning from a checkpoint with additional speakers
        if hasattr(self, "emb_g") and state["model"]["emb_g.weight"].shape != self.emb_g.weight.shape:
            num_new_speakers = self.emb_g.weight.shape[0] - state["model"]["emb_g.weight"].shape[0]
            print(f" > Loading checkpoint with {num_new_speakers} additional speakers.")
            emb_g = state["model"]["emb_g.weight"]
            new_row = torch.randn(num_new_speakers, emb_g.shape[1])
            emb_g = torch.cat([emb_g, new_row], axis=0)
            state["model"]["emb_g.weight"] = emb_g
        # load the model weights
        self.load_state_dict(state["model"], strict=strict)

        if eval:
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "VitsConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initiate model from config

        Args:
            config (VitsConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()
        assert (
            upsample_rate == config.audio.hop_length
        ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"

        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        language_manager = LanguageManager.init_from_config(config)

        if config.model_args.speaker_encoder_model_path:
            speaker_manager.init_encoder(
                config.model_args.speaker_encoder_model_path, config.model_args.speaker_encoder_config_path
            )
        return Vits(new_config, ap, tokenizer, speaker_manager, language_manager)


##################################
# VITS CHARACTERS
##################################


class VitsCharacters(BaseCharacters):
    """Characters class for VITs model for compatibility with pre-trained models"""

    def __init__(
        self,
        graphemes: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        ipa_characters: str = _phonemes,
    ) -> None:
        if ipa_characters is not None:
            graphemes += ipa_characters
        super().__init__(graphemes, punctuations, pad, None, None, "<BLNK>", is_unique=False, is_sorted=True)

    def _create_vocab(self):
        self._vocab = [self._pad] + list(self._punctuations) + list(self._characters) + [self._blank]
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        # pylint: disable=unnecessary-comprehension
        self._id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    @staticmethod
    def init_from_config(config: Coqpit):
        if config.characters is not None:
            _pad = config.characters["pad"]
            _punctuations = config.characters["punctuations"]
            _letters = config.characters["characters"]
            _letters_ipa = config.characters["phonemes"]
            return (
                VitsCharacters(graphemes=_letters, ipa_characters=_letters_ipa, punctuations=_punctuations, pad=_pad),
                config,
            )
        characters = VitsCharacters()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self._characters,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=None,
            bos=None,
            blank=self._blank,
            is_unique=False,
            is_sorted=True,
        )
"""------------------------------------------------------------------------------------------------------------------------------------------------"""
import torch
from scipy.stats import betabinom
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.tacotron.common_layers import Linear


class LocationLayer(nn.Module):
    """Layers for Location Sensitive Attention

    Args:
        attention_dim (int): number of channels in the input tensor.
        attention_n_filters (int, optional): number of filters in convolution. Defaults to 32.
        attention_kernel_size (int, optional): kernel size of convolution filter. Defaults to 31.
    """

    def __init__(self, attention_dim, attention_n_filters=32, attention_kernel_size=31):
        super().__init__()
        self.location_conv1d = nn.Conv1d(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=(attention_kernel_size - 1) // 2,
            bias=False,
        )
        self.location_dense = Linear(attention_n_filters, attention_dim, bias=False, init_gain="tanh")

    def forward(self, attention_cat):
        """
        Shapes:
            attention_cat: [B, 2, C]
        """
        processed_attention = self.location_conv1d(attention_cat)
        processed_attention = self.location_dense(processed_attention.transpose(1, 2))
        return processed_attention


class GravesAttention(nn.Module):
    """Graves Attention as is ref1 with updates from ref2.
    ref1: https://arxiv.org/abs/1910.10288
    ref2: https://arxiv.org/pdf/1906.01083.pdf

    Args:
        query_dim (int): number of channels in query tensor.
        K (int): number of Gaussian heads to be used for computing attention.
    """

    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, query_dim, K):

        super().__init__()
        self._mask_value = 1e-8
        self.K = K
        # self.attention_alignment = 0.05
        self.eps = 1e-5
        self.J = None
        self.N_a = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True), nn.ReLU(), nn.Linear(query_dim, 3 * K, bias=True)
        )
        self.attention_weights = None
        self.mu_prev = None
        self.init_layers()

    def init_layers(self):
        torch.nn.init.constant_(self.N_a[2].bias[(2 * self.K) : (3 * self.K)], 1.0)  # bias mean
        torch.nn.init.constant_(self.N_a[2].bias[self.K : (2 * self.K)], 10)  # bias std

    def init_states(self, inputs):
        if self.J is None or inputs.shape[1] + 1 > self.J.shape[-1]:
            self.J = torch.arange(0, inputs.shape[1] + 2.0).to(inputs.device) + 0.5
        self.attention_weights = torch.zeros(inputs.shape[0], inputs.shape[1]).to(inputs.device)
        self.mu_prev = torch.zeros(inputs.shape[0], self.K).to(inputs.device)

    # pylint: disable=R0201
    # pylint: disable=unused-argument
    def preprocess_inputs(self, inputs):
        return None

    def forward(self, query, inputs, processed_inputs, mask):
        """
        Shapes:
            query: [B, C_attention_rnn]
            inputs: [B, T_in, C_encoder]
            processed_inputs: place_holder
            mask: [B, T_in]
        """
        gbk_t = self.N_a(query)
        gbk_t = gbk_t.view(gbk_t.size(0), -1, self.K)

        # attention model parameters
        # each B x K
        g_t = gbk_t[:, 0, :]
        b_t = gbk_t[:, 1, :]
        k_t = gbk_t[:, 2, :]

        # dropout to decorrelate attention heads
        g_t = torch.nn.functional.dropout(g_t, p=0.5, training=self.training)

        # attention GMM parameters
        sig_t = torch.nn.functional.softplus(b_t) + self.eps

        mu_t = self.mu_prev + torch.nn.functional.softplus(k_t)
        g_t = torch.softmax(g_t, dim=-1) + self.eps

        j = self.J[: inputs.size(1) + 1]

        # attention weights
        phi_t = g_t.unsqueeze(-1) * (1 / (1 + torch.sigmoid((mu_t.unsqueeze(-1) - j) / sig_t.unsqueeze(-1))))

        # discritize attention weights
        alpha_t = torch.sum(phi_t, 1)
        alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]
        alpha_t[alpha_t == 0] = 1e-8

        # apply masking
        if mask is not None:
            alpha_t.data.masked_fill_(~mask, self._mask_value)

        context = torch.bmm(alpha_t.unsqueeze(1), inputs).squeeze(1)
        self.attention_weights = alpha_t
        self.mu_prev = mu_t
        return context


class OriginalAttention(nn.Module):
    """Bahdanau Attention with various optional modifications.
    - Location sensitive attnetion: https://arxiv.org/abs/1712.05884
    - Forward Attention: https://arxiv.org/abs/1807.06736 + state masking at inference
    - Using sigmoid instead of softmax normalization
    - Attention windowing at inference time

    Note:
        Location Sensitive Attention extends the additive attention mechanism
    to use cumulative attention weights from previous decoder time steps with the current time step features.

        Forward attention computes most probable monotonic alignment. The modified attention probabilities at each
    timestep are computed recursively by the forward algorithm.

        Transition agent in the forward attention explicitly gates the attention mechanism whether to move forward or
    stay at each decoder timestep.

        Attention windowing is a inductive prior that prevents the model from attending to previous and future timesteps
    beyond a certain window.

    Args:
        query_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the vakue tensor. In general, the value tensor is the output of the encoder layer.
        attention_dim (int): number of channels of the inner attention layers.
        location_attention (bool): enable/disable location sensitive attention.
        attention_location_n_filters (int): number of location attention filters.
        attention_location_kernel_size (int): filter size of location attention convolution layer.
        windowing (int): window size for attention windowing. if it is 5, for computing the attention, it only considers the time steps [(t-5), ..., (t+5)] of the input.
        norm (str): normalization method applied to the attention weights. 'softmax' or 'sigmoid'
        forward_attn (bool): enable/disable forward attention.
        trans_agent (bool): enable/disable transition agent in the forward attention.
        forward_attn_mask (int): enable/disable an explicit masking in forward attention. It is useful to set at especially inference time.
    """

    # Pylint gets confused by PyTorch conventions here
    # pylint: disable=attribute-defined-outside-init
    def __init__(
        self,
        query_dim,
        embedding_dim,
        attention_dim,
        location_attention,
        attention_location_n_filters,
        attention_location_kernel_size,
        windowing,
        norm,
        forward_attn,
        trans_agent,
        forward_attn_mask,
    ):
        super().__init__()
        self.query_layer = Linear(query_dim, attention_dim, bias=False, init_gain="tanh")
        self.inputs_layer = Linear(embedding_dim, attention_dim, bias=False, init_gain="tanh")
        self.v = Linear(attention_dim, 1, bias=True)
        if trans_agent:
            self.ta = nn.Linear(query_dim + embedding_dim, 1, bias=True)
        if location_attention:
            self.location_layer = LocationLayer(
                attention_dim,
                attention_location_n_filters,
                attention_location_kernel_size,
            )
        self._mask_value = -float("inf")
        self.windowing = windowing
        self.win_idx = None
        self.norm = norm
        self.forward_attn = forward_attn
        self.trans_agent = trans_agent
        self.forward_attn_mask = forward_attn_mask
        self.location_attention = location_attention

    def init_win_idx(self):
        self.win_idx = -1
        self.win_back = 2
        self.win_front = 6

    def init_forward_attn(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.alpha = torch.cat([torch.ones([B, 1]), torch.zeros([B, T])[:, :-1] + 1e-7], dim=1).to(inputs.device)
        self.u = (0.5 * torch.ones([B, 1])).to(inputs.device)

    def init_location_attention(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights_cum = torch.zeros([B, T], device=inputs.device)

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights = torch.zeros([B, T], device=inputs.device)
        if self.location_attention:
            self.init_location_attention(inputs)
        if self.forward_attn:
            self.init_forward_attn(inputs)
        if self.windowing:
            self.init_win_idx()

    def preprocess_inputs(self, inputs):
        return self.inputs_layer(inputs)

    def update_location_attention(self, alignments):
        self.attention_weights_cum += alignments

    def get_location_attention(self, query, processed_inputs):
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def get_attention(self, query, processed_inputs):
        processed_query = self.query_layer(query.unsqueeze(1))
        energies = self.v(torch.tanh(processed_query + processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def apply_windowing(self, attention, inputs):
        back_win = self.win_idx - self.win_back
        front_win = self.win_idx + self.win_front
        if back_win > 0:
            attention[:, :back_win] = -float("inf")
        if front_win < inputs.shape[1]:
            attention[:, front_win:] = -float("inf")
        # this is a trick to solve a special problem.
        # but it does not hurt.
        if self.win_idx == -1:
            attention[:, 0] = attention.max()
        # Update the window
        self.win_idx = torch.argmax(attention, 1).long()[0].item()
        return attention

    def apply_forward_attention(self, alignment):
        # forward attention
        fwd_shifted_alpha = F.pad(self.alpha[:, :-1].clone().to(alignment.device), (1, 0, 0, 0))
        # compute transition potentials
        alpha = ((1 - self.u) * self.alpha + self.u * fwd_shifted_alpha + 1e-8) * alignment
        # force incremental alignment
        if not self.training and self.forward_attn_mask:
            _, n = fwd_shifted_alpha.max(1)
            val, _ = alpha.max(1)
            for b in range(alignment.shape[0]):
                alpha[b, n[b] + 3 :] = 0
                alpha[b, : (n[b] - 1)] = 0  # ignore all previous states to prevent repetition.
                alpha[b, (n[b] - 2)] = 0.01 * val[b]  # smoothing factor for the prev step
        # renormalize attention weights
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha

    def forward(self, query, inputs, processed_inputs, mask):
        """
        shapes:
            query: [B, C_attn_rnn]
            inputs: [B, T_en, D_en]
            processed_inputs: [B, T_en, D_attn]
            mask: [B, T_en]
        """
        if self.location_attention:
            attention, _ = self.get_location_attention(query, processed_inputs)
        else:
            attention, _ = self.get_attention(query, processed_inputs)
        # apply masking
        if mask is not None:
            attention.data.masked_fill_(~mask, self._mask_value)
        # apply windowing - only in eval mode
        if not self.training and self.windowing:
            attention = self.apply_windowing(attention, inputs)

        # normalize attention values
        if self.norm == "softmax":
            alignment = torch.softmax(attention, dim=-1)
        elif self.norm == "sigmoid":
            alignment = torch.sigmoid(attention) / torch.sigmoid(attention).sum(dim=1, keepdim=True)
        else:
            raise ValueError("Unknown value for attention norm type")

        if self.location_attention:
            self.update_location_attention(alignment)

        # apply forward attention if enabled
        if self.forward_attn:
            alignment = self.apply_forward_attention(alignment)
            self.alpha = alignment

        context = torch.bmm(alignment.unsqueeze(1), inputs)
        context = context.squeeze(1)
        self.attention_weights = alignment

        # compute transition agent
        if self.forward_attn and self.trans_agent:
            ta_input = torch.cat([context, query.squeeze(1)], dim=-1)
            self.u = torch.sigmoid(self.ta(ta_input))
        return context


class MonotonicDynamicConvolutionAttention(nn.Module):
    """Dynamic convolution attention from
    https://arxiv.org/pdf/1910.10288.pdf


    query -> linear -> tanh -> linear ->|
                                        |                                            mask values
                                        v                                              |    |
               atten_w(t-1) -|-> conv1d_dynamic -> linear -|-> tanh -> + -> softmax -> * -> * -> context
                             |-> conv1d_static  -> linear -|           |
                             |-> conv1d_prior   -> log ----------------|

    query: attention rnn output.

    Note:
        Dynamic convolution attention is an alternation of the location senstive attention with
    dynamically computed convolution filters from the previous attention scores and a set of
    constraints to keep the attention alignment diagonal.
        DCA is sensitive to mixed precision training and might cause instable training.

    Args:
        query_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the value tensor.
        static_filter_dim (int): number of channels in the convolution layer computing the static filters.
        static_kernel_size (int): kernel size for the convolution layer computing the static filters.
        dynamic_filter_dim (int): number of channels in the convolution layer computing the dynamic filters.
        dynamic_kernel_size (int): kernel size for the convolution layer computing the dynamic filters.
        prior_filter_len (int, optional): [description]. Defaults to 11 from the paper.
        alpha (float, optional): [description]. Defaults to 0.1 from the paper.
        beta (float, optional): [description]. Defaults to 0.9 from the paper.
    """

    def __init__(
        self,
        query_dim,
        embedding_dim,  # pylint: disable=unused-argument
        attention_dim,
        static_filter_dim,
        static_kernel_size,
        dynamic_filter_dim,
        dynamic_kernel_size,
        prior_filter_len=11,
        alpha=0.1,
        beta=0.9,
    ):
        super().__init__()
        self._mask_value = 1e-8
        self.dynamic_filter_dim = dynamic_filter_dim
        self.dynamic_kernel_size = dynamic_kernel_size
        self.prior_filter_len = prior_filter_len
        self.attention_weights = None
        # setup key and query layers
        self.query_layer = nn.Linear(query_dim, attention_dim)
        self.key_layer = nn.Linear(attention_dim, dynamic_filter_dim * dynamic_kernel_size, bias=False)
        self.static_filter_conv = nn.Conv1d(
            1,
            static_filter_dim,
            static_kernel_size,
            padding=(static_kernel_size - 1) // 2,
            bias=False,
        )
        self.static_filter_layer = nn.Linear(static_filter_dim, attention_dim, bias=False)
        self.dynamic_filter_layer = nn.Linear(dynamic_filter_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

        prior = betabinom.pmf(range(prior_filter_len), prior_filter_len - 1, alpha, beta)
        self.register_buffer("prior", torch.FloatTensor(prior).flip(0))

    # pylint: disable=unused-argument
    def forward(self, query, inputs, processed_inputs, mask):
        """
        query: [B, C_attn_rnn]
        inputs: [B, T_en, D_en]
        processed_inputs: place holder.
        mask: [B, T_en]
        """
        # compute prior filters
        prior_filter = F.conv1d(
            F.pad(self.attention_weights.unsqueeze(1), (self.prior_filter_len - 1, 0)), self.prior.view(1, 1, -1)
        )
        prior_filter = torch.log(prior_filter.clamp_min_(1e-6)).squeeze(1)
        G = self.key_layer(torch.tanh(self.query_layer(query)))
        # compute dynamic filters
        dynamic_filter = F.conv1d(
            self.attention_weights.unsqueeze(0),
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=(self.dynamic_kernel_size - 1) // 2,
            groups=query.size(0),
        )
        dynamic_filter = dynamic_filter.view(query.size(0), self.dynamic_filter_dim, -1).transpose(1, 2)
        # compute static filters
        static_filter = self.static_filter_conv(self.attention_weights.unsqueeze(1)).transpose(1, 2)
        alignment = (
            self.v(
                torch.tanh(self.static_filter_layer(static_filter) + self.dynamic_filter_layer(dynamic_filter))
            ).squeeze(-1)
            + prior_filter
        )
        # compute attention weights
        attention_weights = F.softmax(alignment, dim=-1)
        # apply masking
        if mask is not None:
            attention_weights.data.masked_fill_(~mask, self._mask_value)
        self.attention_weights = attention_weights
        # compute context
        context = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        return context

    def preprocess_inputs(self, inputs):  # pylint: disable=no-self-use
        return None

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights = torch.zeros([B, T], device=inputs.device)
        self.attention_weights[:, 0] = 1.0


def init_attn(
    attn_type,
    query_dim,
    embedding_dim,
    attention_dim,
    location_attention,
    attention_location_n_filters,
    attention_location_kernel_size,
    windowing,
    norm,
    forward_attn,
    trans_agent,
    forward_attn_mask,
    attn_K,
):
    if attn_type == "original":
        return OriginalAttention(
            query_dim,
            embedding_dim,
            attention_dim,
            location_attention,
            attention_location_n_filters,
            attention_location_kernel_size,
            windowing,
            norm,
            forward_attn,
            trans_agent,
            forward_attn_mask,
        )
    if attn_type == "graves":
        return GravesAttention(query_dim, attn_K)
    if attn_type == "dynamic_convolution":
        return MonotonicDynamicConvolutionAttention(
            query_dim,
            embedding_dim,
            attention_dim,
            static_filter_dim=8,
            static_kernel_size=21,
            dynamic_filter_dim=8,
            dynamic_kernel_size=21,
            prior_filter_len=11,
            alpha=0.1,
            beta=0.9,
        )

    raise RuntimeError(" [!] Given Attention Type '{attn_type}' is not exist.")
"""----------------------------------------------------------------------------------"""
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from TTS.tts.utils.visual import plot_spectrogram
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_fsspec
from TTS.vocoder.datasets.wavernn_dataset import WaveRNNDataset
from TTS.vocoder.layers.losses import WaveRNNLoss
from TTS.vocoder.models.base_vocoder import BaseVocoder
from TTS.vocoder.utils.distribution import sample_from_discretized_mix_logistic, sample_from_gaussian


def stream(string, variables):
    sys.stdout.write(f"\r{string}" % variables)


# pylint: disable=abstract-method
# relates https://github.com/pytorch/pytorch/issues/42305
class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    def __init__(self, num_res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    def __init__(
        self,
        feat_dims,
        upsample_scales,
        compute_dims,
        num_res_blocks,
        res_out_dims,
        pad,
        use_aux_net,
    ):
        super().__init__()
        self.total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * self.total_scale
        self.use_aux_net = use_aux_net
        if use_aux_net:
            self.resnet = MelResNet(num_res_blocks, feat_dims, compute_dims, res_out_dims, pad)
            self.resnet_stretch = Stretch2d(self.total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        if self.use_aux_net:
            aux = self.resnet(m).unsqueeze(1)
            aux = self.resnet_stretch(aux)
            aux = aux.squeeze(1)
            aux = aux.transpose(1, 2)
        else:
            aux = None
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent : -self.indent]
        return m.transpose(1, 2), aux


class Upsample(nn.Module):
    def __init__(self, scale, pad, num_res_blocks, feat_dims, compute_dims, res_out_dims, use_aux_net):
        super().__init__()
        self.scale = scale
        self.pad = pad
        self.indent = pad * scale
        self.use_aux_net = use_aux_net
        self.resnet = MelResNet(num_res_blocks, feat_dims, compute_dims, res_out_dims, pad)

    def forward(self, m):
        if self.use_aux_net:
            aux = self.resnet(m)
            aux = torch.nn.functional.interpolate(aux, scale_factor=self.scale, mode="linear", align_corners=True)
            aux = aux.transpose(1, 2)
        else:
            aux = None
        m = torch.nn.functional.interpolate(m, scale_factor=self.scale, mode="linear", align_corners=True)
        m = m[:, :, self.indent : -self.indent]
        m = m * 0.045  # empirically found

        return m.transpose(1, 2), aux


@dataclass
class WavernnArgs(Coqpit):
    """🐸 WaveRNN model arguments.

    rnn_dims (int):
        Number of hidden channels in RNN layers. Defaults to 512.
    fc_dims (int):
        Number of hidden channels in fully-conntected layers. Defaults to 512.
    compute_dims (int):
        Number of hidden channels in the feature ResNet. Defaults to 128.
    res_out_dim (int):
        Number of hidden channels in the feature ResNet output. Defaults to 128.
    num_res_blocks (int):
        Number of residual blocks in the ResNet. Defaults to 10.
    use_aux_net (bool):
        enable/disable the feature ResNet. Defaults to True.
    use_upsample_net (bool):
        enable/ disable the upsampling networl. If False, basic upsampling is used. Defaults to True.
    upsample_factors (list):
        Upsampling factors. The multiply of the values must match the `hop_length`. Defaults to ```[4, 8, 8]```.
    mode (str):
        Output mode of the WaveRNN vocoder. `mold` for Mixture of Logistic Distribution, `gauss` for a single
        Gaussian Distribution and `bits` for quantized bits as the model's output.
    mulaw (bool):
        enable / disable the use of Mulaw quantization for training. Only applicable if `mode == 'bits'`. Defaults
        to `True`.
    pad (int):
            Padding applied to the input feature frames against the convolution layers of the feature network.
            Defaults to 2.
    """

    rnn_dims: int = 512
    fc_dims: int = 512
    compute_dims: int = 128
    res_out_dims: int = 128
    num_res_blocks: int = 10
    use_aux_net: bool = True
    use_upsample_net: bool = True
    upsample_factors: List[int] = field(default_factory=lambda: [4, 8, 8])
    mode: str = "mold"  # mold [string], gauss [string], bits [int]
    mulaw: bool = True  # apply mulaw if mode is bits
    pad: int = 2
    feat_dims: int = 80


class Wavernn(BaseVocoder):
    def __init__(self, config: Coqpit):
        """🐸 WaveRNN model.
        Original paper - https://arxiv.org/abs/1802.08435
        Official implementation - https://github.com/fatchord/WaveRNN

        Args:
            config (Coqpit): [description]

        Raises:
            RuntimeError: [description]

        Examples:
            >>> from TTS.vocoder.configs import WavernnConfig
            >>> config = WavernnConfig()
            >>> model = Wavernn(config)

        Paper Abstract:
            Sequential models achieve state-of-the-art results in audio, visual and textual domains with respect to
            both estimating the data distribution and generating high-quality samples. Efficient sampling for this
            class of models has however remained an elusive problem. With a focus on text-to-speech synthesis, we
            describe a set of general techniques for reducing sampling time while maintaining high output quality.
            We first describe a single-layer recurrent neural network, the WaveRNN, with a dual softmax layer that
            matches the quality of the state-of-the-art WaveNet model. The compact form of the network makes it
            possible to generate 24kHz 16-bit audio 4x faster than real time on a GPU. Second, we apply a weight
            pruning technique to reduce the number of weights in the WaveRNN. We find that, for a constant number of
            parameters, large sparse networks perform better than small dense networks and this relationship holds for
            sparsity levels beyond 96%. The small number of weights in a Sparse WaveRNN makes it possible to sample
            high-fidelity audio on a mobile CPU in real time. Finally, we propose a new generation scheme based on
            subscaling that folds a long sequence into a batch of shorter sequences and allows one to generate multiple
            samples at once. The Subscale WaveRNN produces 16 samples per step without loss of quality and offers an
            orthogonal method for increasing sampling efficiency.
        """
        super().__init__(config)

        if isinstance(self.args.mode, int):
            self.n_classes = 2**self.args.mode
        elif self.args.mode == "mold":
            self.n_classes = 3 * 10
        elif self.args.mode == "gauss":
            self.n_classes = 2
        else:
            raise RuntimeError("Unknown model mode value - ", self.args.mode)

        self.aux_dims = self.args.res_out_dims // 4

        if self.args.use_upsample_net:
            assert (
                np.cumproduct(self.args.upsample_factors)[-1] == config.audio.hop_length
            ), " [!] upsample scales needs to be equal to hop_length"
            self.upsample = UpsampleNetwork(
                self.args.feat_dims,
                self.args.upsample_factors,
                self.args.compute_dims,
                self.args.num_res_blocks,
                self.args.res_out_dims,
                self.args.pad,
                self.args.use_aux_net,
            )
        else:
            self.upsample = Upsample(
                config.audio.hop_length,
                self.args.pad,
                self.args.num_res_blocks,
                self.args.feat_dims,
                self.args.compute_dims,
                self.args.res_out_dims,
                self.args.use_aux_net,
            )
        if self.args.use_aux_net:
            self.I = nn.Linear(self.args.feat_dims + self.aux_dims + 1, self.args.rnn_dims)
            self.rnn1 = nn.GRU(self.args.rnn_dims, self.args.rnn_dims, batch_first=True)
            self.rnn2 = nn.GRU(self.args.rnn_dims + self.aux_dims, self.args.rnn_dims, batch_first=True)
            self.fc1 = nn.Linear(self.args.rnn_dims + self.aux_dims, self.args.fc_dims)
            self.fc2 = nn.Linear(self.args.fc_dims + self.aux_dims, self.args.fc_dims)
            self.fc3 = nn.Linear(self.args.fc_dims, self.n_classes)
        else:
            self.I = nn.Linear(self.args.feat_dims + 1, self.args.rnn_dims)
            self.rnn1 = nn.GRU(self.args.rnn_dims, self.args.rnn_dims, batch_first=True)
            self.rnn2 = nn.GRU(self.args.rnn_dims, self.args.rnn_dims, batch_first=True)
            self.fc1 = nn.Linear(self.args.rnn_dims, self.args.fc_dims)
            self.fc2 = nn.Linear(self.args.fc_dims, self.args.fc_dims)
            self.fc3 = nn.Linear(self.args.fc_dims, self.n_classes)

    def forward(self, x, mels):
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.args.rnn_dims).to(x.device)
        h2 = torch.zeros(1, bsize, self.args.rnn_dims).to(x.device)
        mels, aux = self.upsample(mels)

        if self.args.use_aux_net:
            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0] : aux_idx[1]]
            a2 = aux[:, :, aux_idx[1] : aux_idx[2]]
            a3 = aux[:, :, aux_idx[2] : aux_idx[3]]
            a4 = aux[:, :, aux_idx[3] : aux_idx[4]]

        x = (
            torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
            if self.args.use_aux_net
            else torch.cat([x.unsqueeze(-1), mels], dim=2)
        )
        x = self.I(x)
        res = x
        self.rnn1.flatten_parameters()
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2) if self.args.use_aux_net else x
        self.rnn2.flatten_parameters()
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2) if self.args.use_aux_net else x
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2) if self.args.use_aux_net else x
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def inference(self, mels, batched=None, target=None, overlap=None):

        self.eval()
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            if isinstance(mels, np.ndarray):
                mels = torch.FloatTensor(mels).to(str(next(self.parameters()).device))

            if mels.ndim == 2:
                mels = mels.unsqueeze(0)
            wave_len = (mels.size(-1) - 1) * self.config.audio.hop_length

            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.args.pad, side="both")
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                if aux is not None:
                    aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.args.rnn_dims).type_as(mels)
            h2 = torch.zeros(b_size, self.args.rnn_dims).type_as(mels)
            x = torch.zeros(b_size, 1).type_as(mels)

            if self.args.use_aux_net:
                d = self.aux_dims
                aux_split = [aux[:, :, d * i : d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mels[:, i, :]

                if self.args.use_aux_net:
                    a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1) if self.args.use_aux_net else torch.cat([x, m_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1) if self.args.use_aux_net else x
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1) if self.args.use_aux_net else x
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1) if self.args.use_aux_net else x
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                if self.args.mode == "mold":
                    sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    x = sample.transpose(0, 1).type_as(mels)
                elif self.args.mode == "gauss":
                    sample = sample_from_gaussian(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    x = sample.transpose(0, 1).type_as(mels)
                elif isinstance(self.args.mode, int):
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.0) - 1.0
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError("Unknown model mode value - ", self.args.mode)

                if i % 100 == 0:
                    self.gen_display(i, seq_len, b_size, start)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu()
        if batched:
            output = output.numpy()
            output = output.astype(np.float64)

            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        if self.args.mulaw and isinstance(self.args.mode, int):
            output = AudioProcessor.mulaw_decode(output, self.args.mode)

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.config.audio.hop_length)
        output = output[:wave_len]

        if wave_len > len(fade_out):
            output[-20 * self.config.audio.hop_length :] *= fade_out

        self.train()
        return output

    def gen_display(self, i, seq_len, b_size, start):
        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
        realtime_ratio = gen_rate * 1000 / self.config.audio.sample_rate
        stream(
            "%i/%i -- batch_size: %i -- gen_rate: %.1f kHz -- x_realtime: %.1f  ",
            (i * b_size, seq_len * b_size, b_size, gen_rate, realtime_ratio),
        )

    def fold_with_overlap(self, x, target, overlap):
        """Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()
        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)
        Details:
            x = [[h1, h2, ... hn]]
            Where each h is a vector of conditioning features
            Eg: target=2, overlap=1 with x.size(1)=10
            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        """

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side="after")

        folded = torch.zeros(num_folds, target + 2 * overlap, features).to(x.device)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    @staticmethod
    def get_gru_cell(gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    @staticmethod
    def pad_tensor(x, pad, side="both"):
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == "both" else t + pad
        padded = torch.zeros(b, total, c).to(x.device)
        if side in ("before", "both"):
            padded[:, pad : pad + t, :] = x
        elif side == "after":
            padded[:, :t, :] = x
        return padded

    @staticmethod
    def xfade_and_unfold(y, target, overlap):
        """Applies a crossfade and unfolds into a 1d array.
        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64
        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]
            Apply a gain envelope at both ends of the sequences
            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]
            Stagger and add up the groups of samples:
            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]
        """

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training

    def train_step(self, batch: Dict, criterion: Dict) -> Tuple[Dict, Dict]:
        mels = batch["input"]
        waveform = batch["waveform"]
        waveform_coarse = batch["waveform_coarse"]

        y_hat = self.forward(waveform, mels)
        if isinstance(self.args.mode, int):
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
        else:
            waveform_coarse = waveform_coarse.float()
        waveform_coarse = waveform_coarse.unsqueeze(-1)
        # compute losses
        loss_dict = criterion(y_hat, waveform_coarse)
        return {"model_output": y_hat}, loss_dict

    def eval_step(self, batch: Dict, criterion: Dict) -> Tuple[Dict, Dict]:
        return self.train_step(batch, criterion)

    @torch.no_grad()
    def test(
        self, assets: Dict, test_loader: "DataLoader", output: Dict  # pylint: disable=unused-argument
    ) -> Tuple[Dict, Dict]:
        ap = assets["audio_processor"]
        figures = {}
        audios = {}
        samples = test_loader.dataset.load_test_samples(1)
        for idx, sample in enumerate(samples):
            x = torch.FloatTensor(sample[0])
            x = x.to(next(self.parameters()).device)
            y_hat = self.inference(x, self.config.batched, self.config.target_samples, self.config.overlap_samples)
            x_hat = ap.melspectrogram(y_hat)
            figures.update(
                {
                    f"test_{idx}/ground_truth": plot_spectrogram(x.T),
                    f"test_{idx}/prediction": plot_spectrogram(x_hat.T),
                }
            )
            audios.update({f"test_{idx}/audio": y_hat})
        return figures, audios

    @staticmethod
    def format_batch(batch: Dict) -> Dict:
        waveform = batch[0]
        mels = batch[1]
        waveform_coarse = batch[2]
        return {"input": mels, "waveform": waveform, "waveform_coarse": waveform_coarse}

    def get_data_loader(  # pylint: disable=no-self-use
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: True,
        samples: List,
        verbose: bool,
        num_gpus: int,
    ):
        ap = assets["audio_processor"]
        dataset = WaveRNNDataset(
            ap=ap,
            items=samples,
            seq_len=config.seq_len,
            hop_len=ap.hop_length,
            pad=config.model_args.pad,
            mode=config.model_args.mode,
            mulaw=config.model_args.mulaw,
            is_training=not is_eval,
            verbose=verbose,
        )
        sampler = DistributedSampler(dataset, shuffle=True) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=1 if is_eval else config.batch_size,
            shuffle=num_gpus == 0,
            collate_fn=dataset.collate,
            sampler=sampler,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
            pin_memory=True,
        )
        return loader

    def get_criterion(self):
        # define train functions
        return WaveRNNLoss(self.args.mode)

    @staticmethod
    def init_from_config(config: "WavernnConfig"):
        return Wavernn(config)
"""-----------------------------------------------------------------------------------------"""
# Windows dialog .RC file parser, by Adam Walker.

# This module was adapted from the spambayes project, and is Copyright
# 2003/2004 The Python Software Foundation and is covered by the Python
# Software Foundation license.
"""
This is a parser for Windows .rc files, which are text files which define
dialogs and other Windows UI resources.
"""
__author__ = "Adam Walker"
__version__ = "0.11"

import sys, os, shlex, stat
import pprint
import win32con
import commctrl

_controlMap = {
    "DEFPUSHBUTTON": 0x80,
    "PUSHBUTTON": 0x80,
    "Button": 0x80,
    "GROUPBOX": 0x80,
    "Static": 0x82,
    "CTEXT": 0x82,
    "RTEXT": 0x82,
    "LTEXT": 0x82,
    "LISTBOX": 0x83,
    "SCROLLBAR": 0x84,
    "COMBOBOX": 0x85,
    "EDITTEXT": 0x81,
    "ICON": 0x82,
    "RICHEDIT": "RichEdit20A",
}

# These are "default styles" for certain controls - ie, Visual Studio assumes
# the styles will be applied, and emits a "NOT {STYLE_NAME}" if it is to be
# disabled.  These defaults have been determined by experimentation, so may
# not be completely accurate (most notably, some styles and/or control-types
# may be missing.
_addDefaults = {
    "EDITTEXT": win32con.WS_BORDER | win32con.WS_TABSTOP,
    "GROUPBOX": win32con.BS_GROUPBOX,
    "LTEXT": win32con.SS_LEFT,
    "DEFPUSHBUTTON": win32con.BS_DEFPUSHBUTTON | win32con.WS_TABSTOP,
    "PUSHBUTTON": win32con.WS_TABSTOP,
    "CTEXT": win32con.SS_CENTER,
    "RTEXT": win32con.SS_RIGHT,
    "ICON": win32con.SS_ICON,
    "LISTBOX": win32con.LBS_NOTIFY,
}

defaultControlStyle = win32con.WS_CHILD | win32con.WS_VISIBLE
defaultControlStyleEx = 0


class DialogDef:
    name = ""
    id = 0
    style = 0
    styleEx = None
    caption = ""
    font = "MS Sans Serif"
    fontSize = 8
    x = 0
    y = 0
    w = 0
    h = 0
    template = None

    def __init__(self, n, i):
        self.name = n
        self.id = i
        self.styles = []
        self.stylesEx = []
        self.controls = []
        # print "dialog def for ",self.name, self.id

    def createDialogTemplate(self):
        t = None
        self.template = [
            [
                self.caption,
                (self.x, self.y, self.w, self.h),
                self.style,
                self.styleEx,
                (self.fontSize, self.font),
            ]
        ]
        # Add the controls
        for control in self.controls:
            self.template.append(control.createDialogTemplate())
        return self.template


class ControlDef:
    id = ""
    controlType = ""
    subType = ""
    idNum = 0
    style = defaultControlStyle
    styleEx = defaultControlStyleEx
    label = ""
    x = 0
    y = 0
    w = 0
    h = 0

    def __init__(self):
        self.styles = []
        self.stylesEx = []

    def toString(self):
        s = (
            "<Control id:"
            + self.id
            + " controlType:"
            + self.controlType
            + " subType:"
            + self.subType
            + " idNum:"
            + str(self.idNum)
            + " style:"
            + str(self.style)
            + " styles:"
            + str(self.styles)
            + " label:"
            + self.label
            + " x:"
            + str(self.x)
            + " y:"
            + str(self.y)
            + " w:"
            + str(self.w)
            + " h:"
            + str(self.h)
            + ">"
        )
        return s

    def createDialogTemplate(self):
        ct = self.controlType
        if "CONTROL" == ct:
            ct = self.subType
        if ct in _controlMap:
            ct = _controlMap[ct]
        t = [
            ct,
            self.label,
            self.idNum,
            (self.x, self.y, self.w, self.h),
            self.style,
            self.styleEx,
        ]
        # print t
        return t


class StringDef:
    def __init__(self, id, idNum, value):
        self.id = id
        self.idNum = idNum
        self.value = value

    def __repr__(self):
        return "StringDef(%r, %r, %r)" % (self.id, self.idNum, self.value)


class RCParser:
    next_id = 1001
    dialogs = {}
    _dialogs = {}
    debugEnabled = False
    token = ""

    def __init__(self):
        self.ungot = False
        self.ids = {"IDC_STATIC": -1}
        self.names = {-1: "IDC_STATIC"}
        self.bitmaps = {}
        self.stringTable = {}
        self.icons = {}

    def debug(self, *args):
        if self.debugEnabled:
            print(args)

    def getToken(self):
        if self.ungot:
            self.ungot = False
            self.debug("getToken returns (ungot):", self.token)
            return self.token
        self.token = self.lex.get_token()
        self.debug("getToken returns:", self.token)
        if self.token == "":
            self.token = None
        return self.token

    def ungetToken(self):
        self.ungot = True

    def getCheckToken(self, expected):
        tok = self.getToken()
        assert tok == expected, "Expected token '%s', but got token '%s'!" % (
            expected,
            tok,
        )
        return tok

    def getCommaToken(self):
        return self.getCheckToken(",")

    # Return the *current* token as a number, only consuming a token
    # if it is the negative-sign.
    def currentNumberToken(self):
        mult = 1
        if self.token == "-":
            mult = -1
            self.getToken()
        return int(self.token) * mult

    # Return the *current* token as a string literal (ie, self.token will be a
    # quote.  consumes all tokens until the end of the string
    def currentQuotedString(self):
        # Handle quoted strings - pity shlex doesn't handle it.
        assert self.token.startswith('"'), self.token
        bits = [self.token]
        while 1:
            tok = self.getToken()
            if not tok.startswith('"'):
                self.ungetToken()
                break
            bits.append(tok)
        sval = "".join(bits)[1:-1]  # Remove end quotes.
        # Fixup quotes in the body, and all (some?) quoted characters back
        # to their raw value.
        for i, o in ('""', '"'), ("\\r", "\r"), ("\\n", "\n"), ("\\t", "\t"):
            sval = sval.replace(i, o)
        return sval

    def load(self, rcstream):
        """
        RCParser.loadDialogs(rcFileName) -> None
        Load the dialog information into the parser. Dialog Definations can then be accessed
        using the "dialogs" dictionary member (name->DialogDef). The "ids" member contains the dictionary of id->name.
        The "names" member contains the dictionary of name->id
        """
        self.open(rcstream)
        self.getToken()
        while self.token != None:
            self.parse()
            self.getToken()

    def open(self, rcstream):
        self.lex = shlex.shlex(rcstream)
        self.lex.commenters = "//#"

    def parseH(self, file):
        lex = shlex.shlex(file)
        lex.commenters = "//"
        token = " "
        while token is not None:
            token = lex.get_token()
            if token == "" or token is None:
                token = None
            else:
                if token == "define":
                    n = lex.get_token()
                    i = int(lex.get_token())
                    self.ids[n] = i
                    if i in self.names:
                        # Dupe ID really isn't a problem - most consumers
                        # want to go from name->id, and this is OK.
                        # It means you can't go from id->name though.
                        pass
                        # ignore AppStudio special ones
                        # if not n.startswith("_APS_"):
                        #    print "Duplicate id",i,"for",n,"is", self.names[i]
                    else:
                        self.names[i] = n
                    if self.next_id <= i:
                        self.next_id = i + 1

    def parse(self):
        noid_parsers = {
            "STRINGTABLE": self.parse_stringtable,
        }

        id_parsers = {
            "DIALOG": self.parse_dialog,
            "DIALOGEX": self.parse_dialog,
            #            "TEXTINCLUDE":      self.parse_textinclude,
            "BITMAP": self.parse_bitmap,
            "ICON": self.parse_icon,
        }
        deep = 0
        base_token = self.token
        rp = noid_parsers.get(base_token)
        if rp is not None:
            rp()
        else:
            # Not something we parse that isn't prefixed by an ID
            # See if it is an ID prefixed item - if it is, our token
            # is the resource ID.
            resource_id = self.token
            self.getToken()
            if self.token is None:
                return

            if "BEGIN" == self.token:
                # A 'BEGIN' for a structure we don't understand - skip to the
                # matching 'END'
                deep = 1
                while deep != 0 and self.token is not None:
                    self.getToken()
                    self.debug("Zooming over", self.token)
                    if "BEGIN" == self.token:
                        deep += 1
                    elif "END" == self.token:
                        deep -= 1
            else:
                rp = id_parsers.get(self.token)
                if rp is not None:
                    self.debug("Dispatching '%s'" % (self.token,))
                    rp(resource_id)
                else:
                    # We don't know what the resource type is, but we
                    # have already consumed the next, which can cause problems,
                    # so push it back.
                    self.debug("Skipping top-level '%s'" % base_token)
                    self.ungetToken()

    def addId(self, id_name):
        if id_name in self.ids:
            id = self.ids[id_name]
        else:
            # IDOK, IDCANCEL etc are special - if a real resource has this value
            for n in ["IDOK", "IDCANCEL", "IDYES", "IDNO", "IDABORT"]:
                if id_name == n:
                    v = getattr(win32con, n)
                    self.ids[n] = v
                    self.names[v] = n
                    return v
            id = self.next_id
            self.next_id += 1
            self.ids[id_name] = id
            self.names[id] = id_name
        return id

    def lang(self):
        while (
            self.token[0:4] == "LANG"
            or self.token[0:7] == "SUBLANG"
            or self.token == ","
        ):
            self.getToken()

    def parse_textinclude(self, res_id):
        while self.getToken() != "BEGIN":
            pass
        while 1:
            if self.token == "END":
                break
            s = self.getToken()

    def parse_stringtable(self):
        while self.getToken() != "BEGIN":
            pass
        while 1:
            self.getToken()
            if self.token == "END":
                break
            sid = self.token
            self.getToken()
            sd = StringDef(sid, self.addId(sid), self.currentQuotedString())
            self.stringTable[sid] = sd

    def parse_bitmap(self, name):
        return self.parse_bitmap_or_icon(name, self.bitmaps)

    def parse_icon(self, name):
        return self.parse_bitmap_or_icon(name, self.icons)

    def parse_bitmap_or_icon(self, name, dic):
        self.getToken()
        while not self.token.startswith('"'):
            self.getToken()
        bmf = self.token[1:-1]  # quotes
        dic[name] = bmf

    def parse_dialog(self, name):
        dlg = DialogDef(name, self.addId(name))
        assert len(dlg.controls) == 0
        self._dialogs[name] = dlg
        extras = []
        self.getToken()
        while not self.token.isdigit():
            self.debug("extra", self.token)
            extras.append(self.token)
            self.getToken()
        dlg.x = int(self.token)
        self.getCommaToken()
        self.getToken()  # number
        dlg.y = int(self.token)
        self.getCommaToken()
        self.getToken()  # number
        dlg.w = int(self.token)
        self.getCommaToken()
        self.getToken()  # number
        dlg.h = int(self.token)
        self.getToken()
        while not (self.token == None or self.token == "" or self.token == "END"):
            if self.token == "STYLE":
                self.dialogStyle(dlg)
            elif self.token == "EXSTYLE":
                self.dialogExStyle(dlg)
            elif self.token == "CAPTION":
                self.dialogCaption(dlg)
            elif self.token == "FONT":
                self.dialogFont(dlg)
            elif self.token == "BEGIN":
                self.controls(dlg)
            else:
                break
        self.dialogs[name] = dlg.createDialogTemplate()

    def dialogStyle(self, dlg):
        dlg.style, dlg.styles = self.styles([], win32con.DS_SETFONT)

    def dialogExStyle(self, dlg):
        self.getToken()
        dlg.styleEx, dlg.stylesEx = self.styles([], 0)

    def styles(self, defaults, defaultStyle):
        list = defaults
        style = defaultStyle

        if "STYLE" == self.token:
            self.getToken()
        i = 0
        Not = False
        while (
            (i % 2 == 1 and ("|" == self.token or "NOT" == self.token)) or (i % 2 == 0)
        ) and not self.token == None:
            Not = False
            if "NOT" == self.token:
                Not = True
                self.getToken()
            i += 1
            if self.token != "|":
                if self.token in win32con.__dict__:
                    value = getattr(win32con, self.token)
                else:
                    if self.token in commctrl.__dict__:
                        value = getattr(commctrl, self.token)
                    else:
                        value = 0
                if Not:
                    list.append("NOT " + self.token)
                    self.debug("styles add Not", self.token, value)
                    style &= ~value
                else:
                    list.append(self.token)
                    self.debug("styles add", self.token, value)
                    style |= value
            self.getToken()
        self.debug("style is ", style)

        return style, list

    def dialogCaption(self, dlg):
        if "CAPTION" == self.token:
            self.getToken()
        self.token = self.token[1:-1]
        self.debug("Caption is:", self.token)
        dlg.caption = self.token
        self.getToken()

    def dialogFont(self, dlg):
        if "FONT" == self.token:
            self.getToken()
        dlg.fontSize = int(self.token)
        self.getCommaToken()
        self.getToken()  # Font name
        dlg.font = self.token[1:-1]  # it's quoted
        self.getToken()
        while "BEGIN" != self.token:
            self.getToken()

    def controls(self, dlg):
        if self.token == "BEGIN":
            self.getToken()
        # All controls look vaguely like:
        # TYPE [text, ] Control_id, l, t, r, b [, style]
        # .rc parser documents all control types as:
        # CHECKBOX, COMBOBOX, CONTROL, CTEXT, DEFPUSHBUTTON, EDITTEXT, GROUPBOX,
        # ICON, LISTBOX, LTEXT, PUSHBUTTON, RADIOBUTTON, RTEXT, SCROLLBAR
        without_text = ["EDITTEXT", "COMBOBOX", "LISTBOX", "SCROLLBAR"]
        while self.token != "END":
            control = ControlDef()
            control.controlType = self.token
            self.getToken()
            if control.controlType not in without_text:
                if self.token[0:1] == '"':
                    control.label = self.currentQuotedString()
                # Some funny controls, like icons and picture controls use
                # the "window text" as extra resource ID (ie, the ID of the
                # icon itself).  This may be either a literal, or an ID string.
                elif self.token == "-" or self.token.isdigit():
                    control.label = str(self.currentNumberToken())
                else:
                    # An ID - use the numeric equiv.
                    control.label = str(self.addId(self.token))
                self.getCommaToken()
                self.getToken()
            # Control IDs may be "names" or literal ints
            if self.token == "-" or self.token.isdigit():
                control.id = self.currentNumberToken()
                control.idNum = control.id
            else:
                # name of an ID
                control.id = self.token
                control.idNum = self.addId(control.id)
            self.getCommaToken()

            if control.controlType == "CONTROL":
                self.getToken()
                control.subType = self.token[1:-1]
                thisDefaultStyle = defaultControlStyle | _addDefaults.get(
                    control.subType, 0
                )
                # Styles
                self.getCommaToken()
                self.getToken()
                control.style, control.styles = self.styles([], thisDefaultStyle)
            else:
                thisDefaultStyle = defaultControlStyle | _addDefaults.get(
                    control.controlType, 0
                )
                # incase no style is specified.
                control.style = thisDefaultStyle
            # Rect
            control.x = int(self.getToken())
            self.getCommaToken()
            control.y = int(self.getToken())
            self.getCommaToken()
            control.w = int(self.getToken())
            self.getCommaToken()
            self.getToken()
            control.h = int(self.token)
            self.getToken()
            if self.token == ",":
                self.getToken()
                control.style, control.styles = self.styles([], thisDefaultStyle)
            if self.token == ",":
                self.getToken()
                control.styleEx, control.stylesEx = self.styles(
                    [], defaultControlStyleEx
                )
            # print control.toString()
            dlg.controls.append(control)


def ParseStreams(rc_file, h_file):
    rcp = RCParser()
    if h_file:
        rcp.parseH(h_file)
    try:
        rcp.load(rc_file)
    except:
        lex = getattr(rcp, "lex", None)
        if lex:
            print("ERROR parsing dialogs at line", lex.lineno)
            print("Next 10 tokens are:")
            for i in range(10):
                print(lex.get_token(), end=" ")
            print()
        raise
    return rcp


def Parse(rc_name, h_name=None):
    if h_name:
        h_file = open(h_name, "r")
    else:
        # See if same basename as the .rc
        h_name = rc_name[:-2] + "h"
        try:
            h_file = open(h_name, "r")
        except IOError:
            # See if MSVC default of 'resource.h' in the same dir.
            h_name = os.path.join(os.path.dirname(rc_name), "resource.h")
            try:
                h_file = open(h_name, "r")
            except IOError:
                # .h files are optional anyway
                h_file = None
    rc_file = open(rc_name, "r")
    try:
        return ParseStreams(rc_file, h_file)
    finally:
        if h_file is not None:
            h_file.close()
        rc_file.close()
    return rcp


def GenerateFrozenResource(rc_name, output_name, h_name=None):
    """Converts an .rc windows resource source file into a python source file
    with the same basic public interface as the rest of this module.
    Particularly useful for py2exe or other 'freeze' type solutions,
    where a frozen .py file can be used inplace of a real .rc file.
    """
    rcp = Parse(rc_name, h_name)
    in_stat = os.stat(rc_name)

    out = open(output_name, "wt")
    out.write("#%s\n" % output_name)
    out.write("#This is a generated file. Please edit %s instead.\n" % rc_name)
    out.write("__version__=%r\n" % __version__)
    out.write(
        "_rc_size_=%d\n_rc_mtime_=%d\n"
        % (in_stat[stat.ST_SIZE], in_stat[stat.ST_MTIME])
    )

    out.write("class StringDef:\n")
    out.write("\tdef __init__(self, id, idNum, value):\n")
    out.write("\t\tself.id = id\n")
    out.write("\t\tself.idNum = idNum\n")
    out.write("\t\tself.value = value\n")
    out.write("\tdef __repr__(self):\n")
    out.write(
        '\t\treturn "StringDef(%r, %r, %r)" % (self.id, self.idNum, self.value)\n'
    )

    out.write("class FakeParser:\n")

    for name in "dialogs", "ids", "names", "bitmaps", "icons", "stringTable":
        out.write("\t%s = \\\n" % (name,))
        pprint.pprint(getattr(rcp, name), out)
        out.write("\n")

    out.write("def Parse(s):\n")
    out.write("\treturn FakeParser()\n")
    out.close()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(__doc__)
        print()
        print("See test_win32rcparser.py, and the win32rcparser directory (both")
        print("in the test suite) for an example of this module's usage.")
    else:
        import pprint

        filename = sys.argv[1]
        if "-v" in sys.argv:
            RCParser.debugEnabled = 1
        print("Dumping all resources in '%s'" % filename)
        resources = Parse(filename)
        for id, ddef in resources.dialogs.items():
            print("Dialog %s (%d controls)" % (id, len(ddef)))
            pprint.pprint(ddef)
            print()
        for id, sdef in resources.stringTable.items():
            print("String %s=%r" % (id, sdef.value))
            print()
        for id, sdef in resources.bitmaps.items():
            print("Bitmap %s=%r" % (id, sdef))
            print()
        for id, sdef in resources.icons.items():
            print("Icon %s=%r" % (id, sdef))
            print()
"""-------------------------------------------------"""
# This is a work in progress - see Demos/win32gui_menu.py

# win32gui_struct.py - helpers for working with various win32gui structures.
# As win32gui is "light-weight", it does not define objects for all possible
# win32 structures - in general, "buffer" objects are passed around - it is
# the callers responsibility to pack the buffer in the correct format.
#
# This module defines some helpers for the commonly used structures.
#
# In general, each structure has 3 functions:
#
# buffer, extras = PackSTRUCTURE(items, ...)
# item, ... = UnpackSTRUCTURE(buffer)
# buffer, extras = EmtpySTRUCTURE(...)
#
# 'extras' is always items that must be held along with the buffer, as the
# buffer refers to these object's memory.
# For structures that support a 'mask', this mask is hidden from the user - if
# 'None' is passed, the mask flag will not be set, or on return, None will
# be returned for the value if the mask is not set.
#
# NOTE: I considered making these structures look like real classes, and
# support 'attributes' etc - however, ctypes already has a good structure
# mechanism - I think it makes more sense to support ctype structures
# at the win32gui level, then there will be no need for this module at all.
# XXX - the above makes sense in terms of what is built and passed to
# win32gui (ie, the Pack* functions) - but doesn't make as much sense for
# the Unpack* functions, where the aim is user convenience.

import sys
import win32gui
import win32con
import struct
import array
import commctrl
import pywintypes

is64bit = "64 bit" in sys.version

try:
    from collections import namedtuple

    def _MakeResult(names_str, values):
        names = names_str.split()
        nt = namedtuple(names[0], names[1:])
        return nt(*values)

except ImportError:
    # no namedtuple support - just return the values as a normal tuple.
    def _MakeResult(names_str, values):
        return values


_nmhdr_fmt = "PPi"
if is64bit:
    # When the item past the NMHDR gets aligned (eg, when it is a struct)
    # we need this many bytes padding.
    _nmhdr_align_padding = "xxxx"
else:
    _nmhdr_align_padding = ""

# Encode a string suitable for passing in a win32gui related structure
# If win32gui is built with UNICODE defined (ie, py3k), then functions
# like InsertMenuItem are actually calling InsertMenuItemW etc, so all
# strings will need to be unicode.
if win32gui.UNICODE:

    def _make_text_buffer(text):
        # XXX - at this stage win32gui.UNICODE is only True in py3k,
        # and in py3k is makes sense to reject bytes.
        if not isinstance(text, str):
            raise TypeError("MENUITEMINFO text must be unicode")
        data = (text + "\0").encode("utf-16le")
        return array.array("b", data)

else:

    def _make_text_buffer(text):
        if isinstance(text, str):
            text = text.encode("mbcs")
        return array.array("b", text + "\0")


# make an 'empty' buffer, ready for filling with cch characters.
def _make_empty_text_buffer(cch):
    return _make_text_buffer("\0" * cch)


if sys.version_info < (3, 0):

    def _make_memory(ob):
        return str(buffer(ob))

    def _make_bytes(sval):
        return sval

else:

    def _make_memory(ob):
        return bytes(memoryview(ob))

    def _make_bytes(sval):
        return sval.encode("ascii")


# Generic WM_NOTIFY unpacking
def UnpackWMNOTIFY(lparam):
    format = "PPi"
    buf = win32gui.PyGetMemory(lparam, struct.calcsize(format))
    return _MakeResult("WMNOTIFY hwndFrom idFrom code", struct.unpack(format, buf))


def UnpackNMITEMACTIVATE(lparam):
    format = _nmhdr_fmt + _nmhdr_align_padding
    if is64bit:
        # the struct module doesn't handle this correctly as some of the items
        # are actually structs in structs, which get individually aligned.
        format = format + "iiiiiiixxxxP"
    else:
        format = format + "iiiiiiiP"
    buf = win32gui.PyMakeBuffer(struct.calcsize(format), lparam)
    return _MakeResult(
        "NMITEMACTIVATE hwndFrom idFrom code iItem iSubItem uNewState uOldState uChanged actionx actiony lParam",
        struct.unpack(format, buf),
    )


# MENUITEMINFO struct
# http://msdn.microsoft.com/library/default.asp?url=/library/en-us/winui/WinUI/WindowsUserInterface/Resources/Menus/MenuReference/MenuStructures/MENUITEMINFO.asp
# We use the struct module to pack and unpack strings as MENUITEMINFO
# structures.  We also have special handling for the 'fMask' item in that
# structure to avoid the caller needing to explicitly check validity
# (None is used if the mask excludes/should exclude the value)
_menuiteminfo_fmt = "5i5PiP"


def PackMENUITEMINFO(
    fType=None,
    fState=None,
    wID=None,
    hSubMenu=None,
    hbmpChecked=None,
    hbmpUnchecked=None,
    dwItemData=None,
    text=None,
    hbmpItem=None,
    dwTypeData=None,
):
    # 'extras' are objects the caller must keep a reference to (as their
    # memory is used) for the lifetime of the INFO item.
    extras = []
    # ack - dwItemData and dwTypeData were confused for a while...
    assert (
        dwItemData is None or dwTypeData is None
    ), "sorry - these were confused - you probably want dwItemData"
    # if we are a long way past 209, then we can nuke the above...
    if dwTypeData is not None:
        import warnings

        warnings.warn("PackMENUITEMINFO: please use dwItemData instead of dwTypeData")
    if dwItemData is None:
        dwItemData = dwTypeData or 0

    fMask = 0
    if fType is None:
        fType = 0
    else:
        fMask |= win32con.MIIM_FTYPE
    if fState is None:
        fState = 0
    else:
        fMask |= win32con.MIIM_STATE
    if wID is None:
        wID = 0
    else:
        fMask |= win32con.MIIM_ID
    if hSubMenu is None:
        hSubMenu = 0
    else:
        fMask |= win32con.MIIM_SUBMENU
    if hbmpChecked is None:
        assert hbmpUnchecked is None, "neither or both checkmark bmps must be given"
        hbmpChecked = hbmpUnchecked = 0
    else:
        assert hbmpUnchecked is not None, "neither or both checkmark bmps must be given"
        fMask |= win32con.MIIM_CHECKMARKS
    if dwItemData is None:
        dwItemData = 0
    else:
        fMask |= win32con.MIIM_DATA
    if hbmpItem is None:
        hbmpItem = 0
    else:
        fMask |= win32con.MIIM_BITMAP
    if text is not None:
        fMask |= win32con.MIIM_STRING
        str_buf = _make_text_buffer(text)
        cch = len(text)
        # We are taking address of strbuf - it must not die until windows
        # has finished with our structure.
        lptext = str_buf.buffer_info()[0]
        extras.append(str_buf)
    else:
        lptext = 0
        cch = 0
    # Create the struct.
    # 'P' format does not accept PyHANDLE's !
    item = struct.pack(
        _menuiteminfo_fmt,
        struct.calcsize(_menuiteminfo_fmt),  # cbSize
        fMask,
        fType,
        fState,
        wID,
        int(hSubMenu),
        int(hbmpChecked),
        int(hbmpUnchecked),
        dwItemData,
        lptext,
        cch,
        int(hbmpItem),
    )
    # Now copy the string to a writable buffer, so that the result
    # could be passed to a 'Get' function
    return array.array("b", item), extras


def UnpackMENUITEMINFO(s):
    (
        cb,
        fMask,
        fType,
        fState,
        wID,
        hSubMenu,
        hbmpChecked,
        hbmpUnchecked,
        dwItemData,
        lptext,
        cch,
        hbmpItem,
    ) = struct.unpack(_menuiteminfo_fmt, s)
    assert cb == len(s)
    if fMask & win32con.MIIM_FTYPE == 0:
        fType = None
    if fMask & win32con.MIIM_STATE == 0:
        fState = None
    if fMask & win32con.MIIM_ID == 0:
        wID = None
    if fMask & win32con.MIIM_SUBMENU == 0:
        hSubMenu = None
    if fMask & win32con.MIIM_CHECKMARKS == 0:
        hbmpChecked = hbmpUnchecked = None
    if fMask & win32con.MIIM_DATA == 0:
        dwItemData = None
    if fMask & win32con.MIIM_BITMAP == 0:
        hbmpItem = None
    if fMask & win32con.MIIM_STRING:
        text = win32gui.PyGetString(lptext, cch)
    else:
        text = None
    return _MakeResult(
        "MENUITEMINFO fType fState wID hSubMenu hbmpChecked "
        "hbmpUnchecked dwItemData text hbmpItem",
        (
            fType,
            fState,
            wID,
            hSubMenu,
            hbmpChecked,
            hbmpUnchecked,
            dwItemData,
            text,
            hbmpItem,
        ),
    )


def EmptyMENUITEMINFO(mask=None, text_buf_size=512):
    # text_buf_size is number of *characters* - not necessarily no of bytes.
    extra = []
    if mask is None:
        mask = (
            win32con.MIIM_BITMAP
            | win32con.MIIM_CHECKMARKS
            | win32con.MIIM_DATA
            | win32con.MIIM_FTYPE
            | win32con.MIIM_ID
            | win32con.MIIM_STATE
            | win32con.MIIM_STRING
            | win32con.MIIM_SUBMENU
        )
        # Note: No MIIM_TYPE - this screws win2k/98.

    if mask & win32con.MIIM_STRING:
        text_buffer = _make_empty_text_buffer(text_buf_size)
        extra.append(text_buffer)
        text_addr, _ = text_buffer.buffer_info()
    else:
        text_addr = text_buf_size = 0

    # Now copy the string to a writable buffer, so that the result
    # could be passed to a 'Get' function
    buf = struct.pack(
        _menuiteminfo_fmt,
        struct.calcsize(_menuiteminfo_fmt),  # cbSize
        mask,
        0,  # fType,
        0,  # fState,
        0,  # wID,
        0,  # hSubMenu,
        0,  # hbmpChecked,
        0,  # hbmpUnchecked,
        0,  # dwItemData,
        text_addr,
        text_buf_size,
        0,  # hbmpItem
    )
    return array.array("b", buf), extra


# MENUINFO struct
_menuinfo_fmt = "iiiiPiP"


def PackMENUINFO(
    dwStyle=None,
    cyMax=None,
    hbrBack=None,
    dwContextHelpID=None,
    dwMenuData=None,
    fMask=0,
):
    if dwStyle is None:
        dwStyle = 0
    else:
        fMask |= win32con.MIM_STYLE
    if cyMax is None:
        cyMax = 0
    else:
        fMask |= win32con.MIM_MAXHEIGHT
    if hbrBack is None:
        hbrBack = 0
    else:
        fMask |= win32con.MIM_BACKGROUND
    if dwContextHelpID is None:
        dwContextHelpID = 0
    else:
        fMask |= win32con.MIM_HELPID
    if dwMenuData is None:
        dwMenuData = 0
    else:
        fMask |= win32con.MIM_MENUDATA
    # Create the struct.
    item = struct.pack(
        _menuinfo_fmt,
        struct.calcsize(_menuinfo_fmt),  # cbSize
        fMask,
        dwStyle,
        cyMax,
        hbrBack,
        dwContextHelpID,
        dwMenuData,
    )
    return array.array("b", item)


def UnpackMENUINFO(s):
    (cb, fMask, dwStyle, cyMax, hbrBack, dwContextHelpID, dwMenuData) = struct.unpack(
        _menuinfo_fmt, s
    )
    assert cb == len(s)
    if fMask & win32con.MIM_STYLE == 0:
        dwStyle = None
    if fMask & win32con.MIM_MAXHEIGHT == 0:
        cyMax = None
    if fMask & win32con.MIM_BACKGROUND == 0:
        hbrBack = None
    if fMask & win32con.MIM_HELPID == 0:
        dwContextHelpID = None
    if fMask & win32con.MIM_MENUDATA == 0:
        dwMenuData = None
    return _MakeResult(
        "MENUINFO dwStyle cyMax hbrBack dwContextHelpID dwMenuData",
        (dwStyle, cyMax, hbrBack, dwContextHelpID, dwMenuData),
    )


def EmptyMENUINFO(mask=None):
    if mask is None:
        mask = (
            win32con.MIM_STYLE
            | win32con.MIM_MAXHEIGHT
            | win32con.MIM_BACKGROUND
            | win32con.MIM_HELPID
            | win32con.MIM_MENUDATA
        )

    buf = struct.pack(
        _menuinfo_fmt,
        struct.calcsize(_menuinfo_fmt),  # cbSize
        mask,
        0,  # dwStyle
        0,  # cyMax
        0,  # hbrBack,
        0,  # dwContextHelpID,
        0,  # dwMenuData,
    )
    return array.array("b", buf)


##########################################################################
#
# Tree View structure support - TVITEM, TVINSERTSTRUCT and TVDISPINFO
#
##########################################################################

# XXX - Note that the following implementation of TreeView structures is ripped
# XXX - from the SpamBayes project.  It may not quite work correctly yet - I
# XXX - intend checking them later - but having them is better than not at all!

_tvitem_fmt = "iPiiPiiiiP"
# Helpers for the ugly win32 structure packing/unpacking
# XXX - Note that functions using _GetMaskAndVal run 3x faster if they are
# 'inlined' into the function - see PackLVITEM.  If the profiler points at
# _GetMaskAndVal(), you should nuke it (patches welcome once they have been
# tested)
def _GetMaskAndVal(val, default, mask, flag):
    if val is None:
        return mask, default
    else:
        if flag is not None:
            mask |= flag
        return mask, val


def PackTVINSERTSTRUCT(parent, insertAfter, tvitem):
    tvitem_buf, extra = PackTVITEM(*tvitem)
    tvitem_buf = tvitem_buf.tobytes()
    format = "PP%ds" % len(tvitem_buf)
    return struct.pack(format, parent, insertAfter, tvitem_buf), extra


def PackTVITEM(hitem, state, stateMask, text, image, selimage, citems, param):
    extra = []  # objects we must keep references to
    mask = 0
    mask, hitem = _GetMaskAndVal(hitem, 0, mask, commctrl.TVIF_HANDLE)
    mask, state = _GetMaskAndVal(state, 0, mask, commctrl.TVIF_STATE)
    if not mask & commctrl.TVIF_STATE:
        stateMask = 0
    mask, text = _GetMaskAndVal(text, None, mask, commctrl.TVIF_TEXT)
    mask, image = _GetMaskAndVal(image, 0, mask, commctrl.TVIF_IMAGE)
    mask, selimage = _GetMaskAndVal(selimage, 0, mask, commctrl.TVIF_SELECTEDIMAGE)
    mask, citems = _GetMaskAndVal(citems, 0, mask, commctrl.TVIF_CHILDREN)
    mask, param = _GetMaskAndVal(param, 0, mask, commctrl.TVIF_PARAM)
    if text is None:
        text_addr = text_len = 0
    else:
        text_buffer = _make_text_buffer(text)
        text_len = len(text)
        extra.append(text_buffer)
        text_addr, _ = text_buffer.buffer_info()
    buf = struct.pack(
        _tvitem_fmt,
        mask,
        hitem,
        state,
        stateMask,
        text_addr,
        text_len,  # text
        image,
        selimage,
        citems,
        param,
    )
    return array.array("b", buf), extra


# Make a new buffer suitable for querying hitem's attributes.
def EmptyTVITEM(hitem, mask=None, text_buf_size=512):
    extra = []  # objects we must keep references to
    if mask is None:
        mask = (
            commctrl.TVIF_HANDLE
            | commctrl.TVIF_STATE
            | commctrl.TVIF_TEXT
            | commctrl.TVIF_IMAGE
            | commctrl.TVIF_SELECTEDIMAGE
            | commctrl.TVIF_CHILDREN
            | commctrl.TVIF_PARAM
        )
    if mask & commctrl.TVIF_TEXT:
        text_buffer = _make_empty_text_buffer(text_buf_size)
        extra.append(text_buffer)
        text_addr, _ = text_buffer.buffer_info()
    else:
        text_addr = text_buf_size = 0
    buf = struct.pack(
        _tvitem_fmt, mask, hitem, 0, 0, text_addr, text_buf_size, 0, 0, 0, 0  # text
    )
    return array.array("b", buf), extra


def UnpackTVITEM(buffer):
    (
        item_mask,
        item_hItem,
        item_state,
        item_stateMask,
        item_textptr,
        item_cchText,
        item_image,
        item_selimage,
        item_cChildren,
        item_param,
    ) = struct.unpack(_tvitem_fmt, buffer)
    # ensure only items listed by the mask are valid (except we assume the
    # handle is always valid - some notifications (eg, TVN_ENDLABELEDIT) set a
    # mask that doesn't include the handle, but the docs explicity say it is.)
    if not (item_mask & commctrl.TVIF_TEXT):
        item_textptr = item_cchText = None
    if not (item_mask & commctrl.TVIF_CHILDREN):
        item_cChildren = None
    if not (item_mask & commctrl.TVIF_IMAGE):
        item_image = None
    if not (item_mask & commctrl.TVIF_PARAM):
        item_param = None
    if not (item_mask & commctrl.TVIF_SELECTEDIMAGE):
        item_selimage = None
    if not (item_mask & commctrl.TVIF_STATE):
        item_state = item_stateMask = None

    if item_textptr:
        text = win32gui.PyGetString(item_textptr)
    else:
        text = None
    return _MakeResult(
        "TVITEM item_hItem item_state item_stateMask "
        "text item_image item_selimage item_cChildren item_param",
        (
            item_hItem,
            item_state,
            item_stateMask,
            text,
            item_image,
            item_selimage,
            item_cChildren,
            item_param,
        ),
    )


# Unpack the lparm from a "TVNOTIFY" message
def UnpackTVNOTIFY(lparam):
    item_size = struct.calcsize(_tvitem_fmt)
    format = _nmhdr_fmt + _nmhdr_align_padding
    if is64bit:
        format = format + "ixxxx"
    else:
        format = format + "i"
    format = format + "%ds%ds" % (item_size, item_size)
    buf = win32gui.PyGetMemory(lparam, struct.calcsize(format))
    hwndFrom, id, code, action, buf_old, buf_new = struct.unpack(format, buf)
    item_old = UnpackTVITEM(buf_old)
    item_new = UnpackTVITEM(buf_new)
    return _MakeResult(
        "TVNOTIFY hwndFrom id code action item_old item_new",
        (hwndFrom, id, code, action, item_old, item_new),
    )


def UnpackTVDISPINFO(lparam):
    item_size = struct.calcsize(_tvitem_fmt)
    format = "PPi%ds" % (item_size,)
    buf = win32gui.PyGetMemory(lparam, struct.calcsize(format))
    hwndFrom, id, code, buf_item = struct.unpack(format, buf)
    item = UnpackTVITEM(buf_item)
    return _MakeResult("TVDISPINFO hwndFrom id code item", (hwndFrom, id, code, item))


#
# List view items
_lvitem_fmt = "iiiiiPiiPi"


def PackLVITEM(
    item=None,
    subItem=None,
    state=None,
    stateMask=None,
    text=None,
    image=None,
    param=None,
    indent=None,
):
    extra = []  # objects we must keep references to
    mask = 0
    # _GetMaskAndVal adds quite a bit of overhead to this function.
    if item is None:
        item = 0  # No mask for item
    if subItem is None:
        subItem = 0  # No mask for sibItem
    if state is None:
        state = 0
        stateMask = 0
    else:
        mask |= commctrl.LVIF_STATE
        if stateMask is None:
            stateMask = state

    if image is None:
        image = 0
    else:
        mask |= commctrl.LVIF_IMAGE
    if param is None:
        param = 0
    else:
        mask |= commctrl.LVIF_PARAM
    if indent is None:
        indent = 0
    else:
        mask |= commctrl.LVIF_INDENT

    if text is None:
        text_addr = text_len = 0
    else:
        mask |= commctrl.LVIF_TEXT
        text_buffer = _make_text_buffer(text)
        text_len = len(text)
        extra.append(text_buffer)
        text_addr, _ = text_buffer.buffer_info()
    buf = struct.pack(
        _lvitem_fmt,
        mask,
        item,
        subItem,
        state,
        stateMask,
        text_addr,
        text_len,  # text
        image,
        param,
        indent,
    )
    return array.array("b", buf), extra


def UnpackLVITEM(buffer):
    (
        item_mask,
        item_item,
        item_subItem,
        item_state,
        item_stateMask,
        item_textptr,
        item_cchText,
        item_image,
        item_param,
        item_indent,
    ) = struct.unpack(_lvitem_fmt, buffer)
    # ensure only items listed by the mask are valid
    if not (item_mask & commctrl.LVIF_TEXT):
        item_textptr = item_cchText = None
    if not (item_mask & commctrl.LVIF_IMAGE):
        item_image = None
    if not (item_mask & commctrl.LVIF_PARAM):
        item_param = None
    if not (item_mask & commctrl.LVIF_INDENT):
        item_indent = None
    if not (item_mask & commctrl.LVIF_STATE):
        item_state = item_stateMask = None

    if item_textptr:
        text = win32gui.PyGetString(item_textptr)
    else:
        text = None
    return _MakeResult(
        "LVITEM item_item item_subItem item_state "
        "item_stateMask text item_image item_param item_indent",
        (
            item_item,
            item_subItem,
            item_state,
            item_stateMask,
            text,
            item_image,
            item_param,
            item_indent,
        ),
    )


# Unpack an "LVNOTIFY" message
def UnpackLVDISPINFO(lparam):
    item_size = struct.calcsize(_lvitem_fmt)
    format = _nmhdr_fmt + _nmhdr_align_padding + ("%ds" % (item_size,))
    buf = win32gui.PyGetMemory(lparam, struct.calcsize(format))
    hwndFrom, id, code, buf_item = struct.unpack(format, buf)
    item = UnpackLVITEM(buf_item)
    return _MakeResult("LVDISPINFO hwndFrom id code item", (hwndFrom, id, code, item))


def UnpackLVNOTIFY(lparam):
    format = _nmhdr_fmt + _nmhdr_align_padding + "7i"
    if is64bit:
        format = format + "xxxx"  # point needs padding.
    format = format + "P"
    buf = win32gui.PyGetMemory(lparam, struct.calcsize(format))
    (
        hwndFrom,
        id,
        code,
        item,
        subitem,
        newstate,
        oldstate,
        changed,
        pt_x,
        pt_y,
        lparam,
    ) = struct.unpack(format, buf)
    return _MakeResult(
        "UnpackLVNOTIFY hwndFrom id code item subitem "
        "newstate oldstate changed pt lparam",
        (
            hwndFrom,
            id,
            code,
            item,
            subitem,
            newstate,
            oldstate,
            changed,
            (pt_x, pt_y),
            lparam,
        ),
    )


# Make a new buffer suitable for querying an items attributes.
def EmptyLVITEM(item, subitem, mask=None, text_buf_size=512):
    extra = []  # objects we must keep references to
    if mask is None:
        mask = (
            commctrl.LVIF_IMAGE
            | commctrl.LVIF_INDENT
            | commctrl.LVIF_TEXT
            | commctrl.LVIF_PARAM
            | commctrl.LVIF_STATE
        )
    if mask & commctrl.LVIF_TEXT:
        text_buffer = _make_empty_text_buffer(text_buf_size)
        extra.append(text_buffer)
        text_addr, _ = text_buffer.buffer_info()
    else:
        text_addr = text_buf_size = 0
    buf = struct.pack(
        _lvitem_fmt,
        mask,
        item,
        subitem,
        0,
        0,
        text_addr,
        text_buf_size,  # text
        0,
        0,
        0,
    )
    return array.array("b", buf), extra


# List view column structure
_lvcolumn_fmt = "iiiPiiii"


def PackLVCOLUMN(fmt=None, cx=None, text=None, subItem=None, image=None, order=None):
    extra = []  # objects we must keep references to
    mask = 0
    mask, fmt = _GetMaskAndVal(fmt, 0, mask, commctrl.LVCF_FMT)
    mask, cx = _GetMaskAndVal(cx, 0, mask, commctrl.LVCF_WIDTH)
    mask, text = _GetMaskAndVal(text, None, mask, commctrl.LVCF_TEXT)
    mask, subItem = _GetMaskAndVal(subItem, 0, mask, commctrl.LVCF_SUBITEM)
    mask, image = _GetMaskAndVal(image, 0, mask, commctrl.LVCF_IMAGE)
    mask, order = _GetMaskAndVal(order, 0, mask, commctrl.LVCF_ORDER)
    if text is None:
        text_addr = text_len = 0
    else:
        text_buffer = _make_text_buffer(text)
        extra.append(text_buffer)
        text_addr, _ = text_buffer.buffer_info()
        text_len = len(text)
    buf = struct.pack(
        _lvcolumn_fmt, mask, fmt, cx, text_addr, text_len, subItem, image, order  # text
    )
    return array.array("b", buf), extra


def UnpackLVCOLUMN(lparam):
    mask, fmt, cx, text_addr, text_size, subItem, image, order = struct.unpack(
        _lvcolumn_fmt, lparam
    )
    # ensure only items listed by the mask are valid
    if not (mask & commctrl.LVCF_FMT):
        fmt = None
    if not (mask & commctrl.LVCF_WIDTH):
        cx = None
    if not (mask & commctrl.LVCF_TEXT):
        text_addr = text_size = None
    if not (mask & commctrl.LVCF_SUBITEM):
        subItem = None
    if not (mask & commctrl.LVCF_IMAGE):
        image = None
    if not (mask & commctrl.LVCF_ORDER):
        order = None
    if text_addr:
        text = win32gui.PyGetString(text_addr)
    else:
        text = None
    return _MakeResult(
        "LVCOLUMN fmt cx text subItem image order",
        (fmt, cx, text, subItem, image, order),
    )


# Make a new buffer suitable for querying an items attributes.
def EmptyLVCOLUMN(mask=None, text_buf_size=512):
    extra = []  # objects we must keep references to
    if mask is None:
        mask = (
            commctrl.LVCF_FMT
            | commctrl.LVCF_WIDTH
            | commctrl.LVCF_TEXT
            | commctrl.LVCF_SUBITEM
            | commctrl.LVCF_IMAGE
            | commctrl.LVCF_ORDER
        )
    if mask & commctrl.LVCF_TEXT:
        text_buffer = _make_empty_text_buffer(text_buf_size)
        extra.append(text_buffer)
        text_addr, _ = text_buffer.buffer_info()
    else:
        text_addr = text_buf_size = 0
    buf = struct.pack(
        _lvcolumn_fmt, mask, 0, 0, text_addr, text_buf_size, 0, 0, 0  # text
    )
    return array.array("b", buf), extra


# List view hit-test.
def PackLVHITTEST(pt):
    format = "iiiii"
    buf = struct.pack(format, pt[0], pt[1], 0, 0, 0)
    return array.array("b", buf), None


def UnpackLVHITTEST(buf):
    format = "iiiii"
    x, y, flags, item, subitem = struct.unpack(format, buf)
    return _MakeResult(
        "LVHITTEST pt flags item subitem", ((x, y), flags, item, subitem)
    )


def PackHDITEM(
    cxy=None, text=None, hbm=None, fmt=None, param=None, image=None, order=None
):
    extra = []  # objects we must keep references to
    mask = 0
    mask, cxy = _GetMaskAndVal(cxy, 0, mask, commctrl.HDI_HEIGHT)
    mask, text = _GetMaskAndVal(text, None, mask, commctrl.LVCF_TEXT)
    mask, hbm = _GetMaskAndVal(hbm, 0, mask, commctrl.HDI_BITMAP)
    mask, fmt = _GetMaskAndVal(fmt, 0, mask, commctrl.HDI_FORMAT)
    mask, param = _GetMaskAndVal(param, 0, mask, commctrl.HDI_LPARAM)
    mask, image = _GetMaskAndVal(image, 0, mask, commctrl.HDI_IMAGE)
    mask, order = _GetMaskAndVal(order, 0, mask, commctrl.HDI_ORDER)

    if text is None:
        text_addr = text_len = 0
    else:
        text_buffer = _make_text_buffer(text)
        extra.append(text_buffer)
        text_addr, _ = text_buffer.buffer_info()
        text_len = len(text)

    format = "iiPPiiPiiii"
    buf = struct.pack(
        format, mask, cxy, text_addr, hbm, text_len, fmt, param, image, order, 0, 0
    )
    return array.array("b", buf), extra


# Device notification stuff

# Generic function for packing a DEV_BROADCAST_* structure - generally used
# by the other PackDEV_BROADCAST_* functions in this module.
def PackDEV_BROADCAST(devicetype, rest_fmt, rest_data, extra_data=_make_bytes("")):
    # It seems a requirement is 4 byte alignment, even for the 'BYTE data[1]'
    # field (eg, that would make DEV_BROADCAST_HANDLE 41 bytes, but we must
    # be 44.
    extra_data += _make_bytes("\0" * (4 - len(extra_data) % 4))
    format = "iii" + rest_fmt
    full_size = struct.calcsize(format) + len(extra_data)
    data = (full_size, devicetype, 0) + rest_data
    return struct.pack(format, *data) + extra_data


def PackDEV_BROADCAST_HANDLE(
    handle,
    hdevnotify=0,
    guid=_make_bytes("\0" * 16),
    name_offset=0,
    data=_make_bytes("\0"),
):
    return PackDEV_BROADCAST(
        win32con.DBT_DEVTYP_HANDLE,
        "PP16sl",
        (int(handle), int(hdevnotify), _make_memory(guid), name_offset),
        data,
    )


def PackDEV_BROADCAST_VOLUME(unitmask, flags):
    return PackDEV_BROADCAST(win32con.DBT_DEVTYP_VOLUME, "II", (unitmask, flags))


def PackDEV_BROADCAST_DEVICEINTERFACE(classguid, name=""):
    if win32gui.UNICODE:
        # This really means "is py3k?" - so not accepting bytes is OK
        if not isinstance(name, str):
            raise TypeError("Must provide unicode for the name")
        name = name.encode("utf-16le")
    else:
        # py2k was passed a unicode object - encode as mbcs.
        if isinstance(name, str):
            name = name.encode("mbcs")

    # 16 bytes for the IID followed by \0 term'd string.
    rest_fmt = "16s%ds" % len(name)
    # _make_memory(iid) hoops necessary to get the raw IID bytes.
    rest_data = (_make_memory(pywintypes.IID(classguid)), name)
    return PackDEV_BROADCAST(win32con.DBT_DEVTYP_DEVICEINTERFACE, rest_fmt, rest_data)


# An object returned by UnpackDEV_BROADCAST.
class DEV_BROADCAST_INFO:
    def __init__(self, devicetype, **kw):
        self.devicetype = devicetype
        self.__dict__.update(kw)

    def __str__(self):
        return "DEV_BROADCAST_INFO:" + str(self.__dict__)


# Support for unpacking the 'lparam'
def UnpackDEV_BROADCAST(lparam):
    if lparam == 0:
        return None
    hdr_format = "iii"
    hdr_size = struct.calcsize(hdr_format)
    hdr_buf = win32gui.PyGetMemory(lparam, hdr_size)
    size, devtype, reserved = struct.unpack("iii", hdr_buf)
    # Due to x64 alignment issues, we need to use the full format string over
    # the entire buffer.  ie, on x64:
    # calcsize('iiiP') != calcsize('iii')+calcsize('P')
    buf = win32gui.PyGetMemory(lparam, size)

    extra = x = {}
    if devtype == win32con.DBT_DEVTYP_HANDLE:
        # 2 handles, a GUID, a LONG and possibly an array following...
        fmt = hdr_format + "PP16sl"
        (
            _,
            _,
            _,
            x["handle"],
            x["hdevnotify"],
            guid_bytes,
            x["nameoffset"],
        ) = struct.unpack(fmt, buf[: struct.calcsize(fmt)])
        x["eventguid"] = pywintypes.IID(guid_bytes, True)
    elif devtype == win32con.DBT_DEVTYP_DEVICEINTERFACE:
        fmt = hdr_format + "16s"
        _, _, _, guid_bytes = struct.unpack(fmt, buf[: struct.calcsize(fmt)])
        x["classguid"] = pywintypes.IID(guid_bytes, True)
        x["name"] = win32gui.PyGetString(lparam + struct.calcsize(fmt))
    elif devtype == win32con.DBT_DEVTYP_VOLUME:
        # int mask and flags
        fmt = hdr_format + "II"
        _, _, _, x["unitmask"], x["flags"] = struct.unpack(
            fmt, buf[: struct.calcsize(fmt)]
        )
    else:
        raise NotImplementedError("unknown device type %d" % (devtype,))
    return DEV_BROADCAST_INFO(devtype, **extra)
"""--------------------------------------------------------------"""
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk, UnidentifiedImageError
from TkinterDnD2 import DND_FILES, TkinterDnD
import os, os.path
import sys
from threading import *
import cv2
from pygrabber.dshow_graph import FilterGraph
import time
import numpy as np
import traceback

import image

sys.path.append('scripts')
import ipbr
import config
import cam_modnet
from error_panel import error_handler, done_handler


if __name__ == "__main__":
    def background_panel_gui():
        # access variables
        global add_background_icon
        global backgrounds_array
        global yindex
        # refresh the yindex value every time the this function is called
        yindex = 0.1

        # create main panel
        background_panel = tk.Frame(mainwindow, height=720, width=360, relief="groove", bg="#525E75")
        background_panel.place(relx=0.72, rely=0)
        # create button widgets
        tk.Button(background_panel, height=20, width=20, bd=0, image=add_background_icon, cursor="hand2",
                  command=lambda: add_background(background_panel)).place(relx=0.05, rely=0.025)
        tk.Button(background_panel, height = 2, width = 15, text = "Add Background", font = ("Roboto", 10), fg = "#b3e3b3", bg = "#78938A", activebackground = "#858585", cursor = "hand2", borderwidth=0, highlightthickness=0,command=lambda: add_background(background_panel)).place(relx=0.15, rely=0.02)
        tk.Button(background_panel, height = 2, width = 8, text = "Close", font = ("Roboto", 10),borderwidth=0, highlightthickness=0, cursor = "hand2", fg="#b3e3b3", bg="#78938A", command = background_panel.destroy).place(relx = 0.7, rely = 0.02)
        # recreate the image gallery with current image and panel as paramenter
        for img in backgrounds_array:
            create_background_gallery(img, background_panel)

        return background_panel

    def add_background(panel):
        global backgrounds_array
        if len(backgrounds_array) <= 2:
            image_url = filedialog.askopenfilename(initialdir="/Desktop" if len(backgrounds_array) == 0 else  os.path.dirname(backgrounds_array[-1]),
                                                   filetypes=(("image files", ".jpg"), ("image files", ".png")))
            if image_url:
                create_background_gallery(image_url, panel)
                backgrounds_array.append(image_url)

                conf.set_array_backgrounds(backgrounds_array)
                conf.write()
        else:
            text = "Exceeded Number of Backgrounds"
            error_handler(text, True)

    def create_background_gallery(image_url, panel):
        # acces y index
        global yindex
        global im_index
        global isPortraitBackground
        global height_prev_bg

        try:
            if os.path.exists(image_url):
                img = Image.open(image_url)
                img.thumbnail((250, 250))
                img = ImageTk.PhotoImage(img)

                # create the image in image gallery, I used button for command attribute
                image_panel = tk.Frame(panel, height=180, width=img.width(), bg="#525E75")
                mainwindow.update()
                x = ((panel.winfo_width() - img.width()) / 2) / panel.winfo_width()
                image_panel.place(relx=x, rely=(yindex))
                image_view = tk.Button(image_panel, text="view", height=img.height(), width=img.width(), image=img,
                                       bg="#383d3a", cursor="hand2", borderwidth=0, highlightthickness=0,
                                       command=lambda: choosebackground(img, image_url, panel))
                image_view.place(relx=0, rely=0)

                # create a delete button
                mainwindow.update()
                bx = (image_view.winfo_width() - 25) / image_view.winfo_width()
                delete = tk.Button(image_panel, image=trash_image, height="20", width="20", bg="#161010",
                                   cursor="hand2",
                                   command=lambda: (deletebackground(image_url, image_view, panel)))
                delete.place(relx=bx, rely=0.00)

                im_index += 1
                # increase yindex for proper margin of succeeding images
                yindex += 0.285
            else:
                text = "Background Image Not Found\n" + image_url
                error_handler(text, True)
                backgrounds_array.remove(image_url)
                conf.set_array_backgrounds(backgrounds_array)
                conf.write()

            return panel

        except UnidentifiedImageError:
            text = "Some images are corrupted!"
            error_handler(text, True)

        except Exception as e:
            error_handler(f"Some Error Happened: \n {e}", True)

    def deletebackground(image_url, image_view, panel):
        global backgrounds_array
        global background_path
        global current_background

        print(image_url)
        print(current_background)

        if image_url != current_background:
            image_view.destroy()
            panel.destroy()

            backgrounds_array.remove(image_url)
            if len(backgrounds_array) == 0:
                background_preview.configure(height=160, width=310, image = "")
                background_path = ""
                conf.set_background("")
                conf.set_array_backgrounds(backgrounds_array)
                conf.write()

            conf.set_array_backgrounds(backgrounds_array)
            conf.write()
            background_panel_gui()
        else:
            del_error = "Current Background Should Net Be Deleted!"
            error_handler(del_error, True)

    def choosebackground(bgimg,image_url, panel):
        #access essential variable background image
        global background_path
        global background_image
        #assign it with the argument variable image
        background_image = bgimg
        background_path = image_url
        background_preview.configure(height=160, width=310, image=background_image)
        conf.set_background(background_path)
        conf.write()
        panel.destroy()

    def open_settings():
        # access temporary variables
        global output_loc
        global height_entry_var
        global width_entry_var
        global ifcheck_var
        global ch
        global temp
        global inputsize_checkbox
        global isSaveTransparent

        temp.set(ifcheck_var)

        height_entry_var.set(str(height_var))
        width_entry_var.set(str(width_var))

        setting_panel = tk.Frame(mainwindow, height=720, width=350, bg="#525E75")
        setting_panel.place(relx=0.730, rely=0)
        tk.Label(setting_panel, text="Settings", font=("Roboto", 20), fg="#4369D9", bg="#525E75").place(relx=0.05,
                                                                                                        rely=0.025)
        tk.Label(setting_panel, text="Output Location", font=("Roboto", 14), fg="#b3e3b3", bg="#525E75").place(relx=0.05,
                                                                                                               rely=0.1)
        output_loc_entry = tk.Label(setting_panel, width=25, font=("Roboto", 12), fg="#b3e3b3", bg="#525E75", bd=2,
                                    relief="groove")
        output_loc_entry.configure(text = output_loc)
        output_loc_entry.place(relx=0.05, rely=0.15, height=40)

        output_error_label = tk.Label(setting_panel, font=("Roboto", 10), fg="#b3e3b3", bg="#525E75")
        output_error_label.place(relx=0.25, rely=0.215)

        tk.Button(setting_panel, height=0, width=3, text="...", font=("Roboto", 15), fg="#b3e3b3", bg="#525E75", bd=2,
                  relief="groove", command=lambda: [get_output_loc(output_loc_entry)]).place(relx=0.7, rely=0.15)

        tk.Label(setting_panel, text = "Use Input Sizes", font = ("Roboto", 13), fg="#b3e3b3", bg="#525E75").place(relx = 0.05, rely = 0.3)
        tk.Label(setting_panel, text = "Save transparent background",font = ("Roboto", 13), fg="#b3e3b3", bg="#525E75").place(relx = 0.05, rely = 0.25)
        tk.Checkbutton(setting_panel,  bg="#525E75", variable=isSaveTransparent).place(relx = 0.7, rely = 0.25)

        tk.Checkbutton(setting_panel, variable = inputsize_checkbox, bg="#525E75", command = lambda: use_input_reso_handler(inputsize_checkbox.get(),customreso_cbeckbox, height_entry, width_entry)).place(relx = 0.7, rely = 0.298)

        tk.Label(setting_panel, text="Use Custom Sizes", font=("Roboto", 13), fg="#b3e3b3", bg="#525E75").place(relx=0.05, rely=0.35)

        customreso_cbeckbox = tk.Checkbutton(setting_panel, variable= temp, bg="#525E75",command=lambda: [checkbox(height_entry, width_entry)])
        customreso_cbeckbox.place(relx=0.7, rely=0.348)

        tk.Label(setting_panel, text="Height (Pixels): ", font=("Roboto", 12), fg="#b3e3b3", bg="#525E75").place(relx=0.05,rely=0.4)
        height_entry = tk.Entry(setting_panel, state='readonly',  textvariable=height_entry_var, width=5,
                                font=("Roboto", 12), fg="#331c09", bg="#FAFDD6", bd=3)

        height_entry.place(relx=0.4, rely=0.4)

        tk.Label(setting_panel, text="Width (Pixels): ", font=("Roboto", 12), fg="#b3e3b3", bg="#525E75").place(relx=0.05,rely=0.45)

        width_entry = tk.Entry(setting_panel, state='readonly', textvariable=width_entry_var, width=5, font=("Roboto", 12),
                               fg="#331c09", bg="#FAFDD6", bd=3)

        width_entry.place(relx=0.4, rely=0.45)
        height_error_label = tk.Label(setting_panel, font=("Roboto", 10), fg="#b3e3b3", bg="#525E75")
        height_error_label.place(relx=0.575, rely=0.405)
        width_error_label = tk.Label(setting_panel, font=("Roboto", 10), fg="#b3e3b3", bg="#525E75")
        width_error_label.place(relx=0.575, rely=0.455)

        tk.Button(setting_panel, height=2, width=30, text="Cancel", font=("Roboto", 13), fg="#b3e3b3", bg="#78938A", borderwidth=0, highlightthickness=0,
                  cursor="hand2", command=setting_panel.destroy).place(relx = 0.07, rely=0.85)
        tk.Button(setting_panel, height=2, width=30, text="Apply Changes", font=("Roboto", 13), fg="#b3e3b3", bg="#78938A", borderwidth=0, highlightthickness=0,
                  cursor="hand2", command=lambda: [
                save_settings(height_error_label, width_error_label, output_error_label, setting_panel,  ifcheck_var)]).place(
            relx=0.07, rely=0.75)

        tk.Label(setting_panel)

        checkbox(height_entry, width_entry)



        return height_error_label, width_error_label, output_error_label, output_loc_entry, setting_panel, customreso_cbeckbox, height_entry, width_entry


    # def template_size_handler(passedHeight, passwedWidth, height_entry, width_entry):
    #     global width_var
    #     global height_var
    #     global width_var
    #     global height_var
    #
    #     # set passed size to global variables
    #     height_var = passedHeight
    #     width_var = passwedWidth
    #     # set state of entry to normal
    #     height_entry.configure(state = "normal")
    #     height_entry.delete(0,"end")
    #     height_entry.insert(0, passedHeight)
    #     width_entry.configure(state="normal")
    #     width_entry.delete(0,"end")
    #     width_entry.insert(0, passwedWidth)
    #
    #     print(height_var)
    #     print(width_var)

    def use_input_reso_handler(inputsize_checkbox,customreso_cbeckbox,  height_entry, width_entry):
        global ifcheck_var

        if inputsize_checkbox == True:
            ifcheck_var = 0
            temp.set(False)
            height_entry.configure(state = "disabled")
            width_entry.configure(state = "disabled")
        else:
            ifcheck_var = 1
            temp.set(True)
            customreso_cbeckbox.configure(state="normal")
            height_entry.configure(state="normal")
            width_entry.configure(state="normal")

    def checkbox(height_entry, width_entry):
        # check if checkbox is checked or not
        global ifcheck_var
        global temp
        global inputsize_checkbox

        if temp.get() is True:
            inputsize_checkbox.set(False)
            height_entry.configure(state="normal")
            width_entry.configure(state="normal")

        else:
            height_entry.configure(state="readonly")
            width_entry.configure(state="readonly")

        ifcheck_var = temp.get()

    def get_output_loc(output_loc_entry):
        #access temporary location variables as holedr
        global temp_output_loc
        global output_loc
        #assign it with value <str> path from filedialog.askdirectory fpr folder path only
        temp_output_loc = filedialog.askdirectory(initialdir= "/Desktop" if temp_output_loc is None else temp_output_loc)

        if not temp_output_loc:
            temp_output_loc = output_loc

        output_loc_entry.configure(text = temp_output_loc)

    def save_settings(height_error_label, width_error_label, output_error_label , setting_panel ,ifcheck_var):
        # access permanent variables
        global height_entry_var
        global width_entry_var
        global temp_output_loc
        global output_loc
        global height_var
        global width_var

        height_error = False
        width_error = False

        if ifcheck_var is True:
            try:
                height_var = int(height_entry_var.get())
                height_error_label.configure(text="")
                height_error = False
                # check if height input is 512 or higher
                if height_var < 512:
                    height_error_label.configure(text="(512 MINIMUM)")
                    height_error = True
                elif height_var == "":
                    height_error_label.configure(text="Invalid Input")
                    height_error = True
                else:
                    height_error = False
            except ValueError:
                height_error_label.configure(text="Numbers Only (Integers)")
                height_error = True
            try:
                width_var = int(width_entry_var.get())
                width_error_label.configure(text="")
                width_error = False
                # check if height input is 512 or higher
                if width_var < 512:
                    width_error_label.configure(text="(512 MINIMUM)")
                    width_error = True
                elif width_var == "":
                    width_error_label.configure(text="(Invalid Input)")
                    width_error = True
                else:
                    width_error = False
            except ValueError:
                width_error_label.configure(text="Numbers Only (Integers)")
                width_error = True
        else:
            height_var = 900
            width_var = 600

        if height_error is False and width_error is False and os.path.exists(temp_output_loc) is True:
            # pass temporary output location to permanent output location
            output_loc = temp_output_loc
            conf.set_output_path(output_loc)
            conf.set_width(width_var)
            conf.set_height(height_var)
            conf.set_save_transparent(1 if isSaveTransparent.get() is True else 0)
            conf.set_checkbox_state(1 if ifcheck_var is True else 0)
            conf.write()
            setting_panel.destroy()
        else:
            output_error_label.configure(text="Path Not Found!")

    def update_preview(img):
        global imm
        global preview

        img = image.downscale(img, 300)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        p = ImageTk.PhotoImage(img)
        imm.append(p)
        preview.configure(height=330, width=315, image=p)

    def start_thread():
        global isLoaded

        if isLoaded:
            t1 = Thread(target=start_process)
            t1.daemon = True
            t1.start()

    def stop_process():
        global stopped
        global stop_btn

        stopped = True
        stop_btn.destroy()

    def start_process():
        #access permanent variables
        global background_path
        global input_folder_path
        global width_var
        global height_var
        global output_loc
        global preview
        global imm
        global column_size
        global inputsize_checkbox
        global stop_btn
        global stopped
        global isModelPresent
        global current_background

        #check if all needed variables are populated
        try:
            if not isModelPresent:
                text = "Pretrained Model not found!"
                error_handler(text, True)
            elif not os.path.exists(output_loc):
                text = "Output path does not exists or is not set!"
                error_handler(text, True)
            else:
                if os.path.isfile(background_path):
                    current_background = background_path
                    background = Image.open(background_path)
                    i = 0
                    stopped = False
                    stop_btn = tk.Button(menu_frame, height=3, width=25, text="STOP", font=("Roboto", 16), fg="#e0efff",
                                         bg="#DC4343", activebackground="#4a9eff", cursor="hand2", borderwidth=0,
                                         highlightthickness=0, command=stop_process)
                    stop_btn.place(relx=0.025, rely=0.87)

                    for im in input_array:

                        im_label_array[i].configure(text="Processing")
                        im_label_array[i].place(relx=0.05, rely=0.1)

                        if i > 0:
                            im_label_array[i - 1].configure(text="Done")

                        if stopped:
                            im_label_array[i].configure(text="Stopped")
                            break
                        start_time = time.time()
                        img = Image.open(im)
                        name = os.path.basename(im)
                        name = name.split('.')[0] + '.png'

                        if inputsize_checkbox.get():
                            img, transparent = main.process_v2(img, background, isSaveTransparent.get())
                        else:
                            img, transparent = main.process(img, background, (width_var, height_var),
                                                            isSaveTransparent.get())

                        cv2.imwrite(os.path.join(output_loc, name), img)

                        if transparent is not None:
                            transparent_name = name.split('.')[0] + "_transparent" + ".png"
                            try:
                                path = os.path.join(output_loc, "Transparent Images")
                                os.mkdir(path)
                            except:
                                pass

                            transparent = np.array(transparent)
                            cv2.imwrite(os.path.join(path, transparent_name), transparent)

                        update_preview(img)

                        print("Execution Time (seconds): ", (time.time() - start_time))

                        if i == len(input_array) - 1:
                            im_label_array[i].configure(text="Done")

                        i += 1

                    current_background = ""
                    text = "Processing done!"
                    done_handler(text)
                    stop_btn.destroy()
                else:
                    text = "Background Image Not Found "
                    error_handler(text, True)

        except AttributeError:
            text = "No selected Background!"
            error_handler(text, True)
            print(traceback.format_exc())
        except IOError:
            text = "Cannot Open Images! \nImages does not exist or deleted!"
            error_handler(text, True)
        except Exception as e:
            error_handler(f"Some Error happened! \n {e}", True)
            print(traceback.format_exc())

        column_size = 4

    def drop_inside_list_box(event):
        global isHomeBool
        #access essential variable
        global input_folder_path
        global input_array
        #assign it with data from event
        input_folder_path = event.data

        print(input_folder_path)


        if input_folder_path:
            isNotBigger = False
            isCorrupted = False
            index = 0

            if os.path.isdir(input_folder_path):
                temp_var = os.listdir(input_folder_path)

            if input_folder_path.endswith(".png") or input_folder_path.endswith(".jpeg") or input_folder_path.endswith(".jpg"):
                temp_var = input_folder_path.split()

            for file in temp_var:
                if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                    try:
                        temp = Image.open(os.path.join(input_folder_path, file))

                        if temp.width > 512 and temp.height > 512:
                            index += 1
                            input_array.append(os.path.join(input_folder_path, file))
                        else:
                            isNotBigger = True

                    except UnidentifiedImageError:
                        isCorrupted = True

            if isNotBigger and isCorrupted:
                text = "Some images are corrupted and smaller than 512x512!"
                error_handler(text, True)

            elif isNotBigger:
                text = "Images must be bigger than 512x512!"
                error_handler(text, False)

            elif isCorrupted:
                text = "Some images are corrupted!"
                error_handler(text, True)

            if index > 0:
                input_gallery_gui()
                isHomeBool = False
                checkI_home_handler()

    def get_input_handler():
        global isHomeBool
        #access essential variable
        global input_folder_path
        #assign it with data from event
        input_folder_path = filedialog.askdirectory(initialdir = "/Desktop" if input_folder_path is None else input_folder_path,title = "Select Input Path")

        if input_folder_path:
            isNotBigger = False
            isCorrupted = False
            index = 0
            for file in os.listdir(input_folder_path):
                if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                    try:
                        temp = Image.open(os.path.join(input_folder_path, file))

                        if temp.width > 512 and temp.height > 512:
                            index += 1
                            input_array.append(os.path.join(input_folder_path, file))
                        else:
                            isNotBigger = True

                    except UnidentifiedImageError:
                        isCorrupted = True

            if isNotBigger and isCorrupted:
                text = "Some images are corrupted and smaller than 512x512!"
                error_handler(text, True)

            elif isNotBigger:
                text = "Images must be bigger than 512x512!"
                error_handler(text, False)

            elif isCorrupted:
                text = "Some images are corrupted!"
                error_handler(text, True)

            if index > 0:
                input_gallery_gui()
                isHomeBool = False
                checkI_home_handler()
            else:
                text = "No images found!"
                error_handler(text, False)

    def clear():
        global foreground_input_list_box
        global input_array
        global isHomeBool
        global column_size
        global preview
        global imm
        global im_label_array
        global del_btn_disabled
        global clicked
        # global use_cam_btn_disabled

        input_array.clear()
        imm.clear()
        is_selected.clear()
        checkbox_array.clear()
        im_label_array.clear()
        #use_cam_btn_disabled.destroy()
        preview.configure(image = None)

        for widgets in foreground_input_list_box.winfo_children():
            widgets.destroy()

        tk.Label(foreground_input_list_box, text="Drop image folder here", font=("Roboto", 20), fg="#D6D2D2",
                 bg="#2C2B2B").place(relx=0.25, rely=0.35)
        tk.Label(foreground_input_list_box, text="or", font=("Roboto", 20), fg="#D6D2D2", bg="#2C2B2B").place(relx=0.35,
                                                                                                              rely=0.41)
        tk.Button(foreground_input_list_box, text="Browse", height=1, width=20, font=("Roboto", 17), fg = "#e0efff", bg = "#127DF4", activebackground="#4a9eff", cursor="hand2", borderwidth=0, highlightthickness=0, command=get_input_handler).place(
            relx=0.255, rely=0.50)

        if clicked:
            del_btn_disabled.destroy()
            clicked = False

        isHomeBool = True
        checkI_home_handler()

    def select_img():
        global checkbox_array
        global is_selected
        global clicked
        global isHomeBool
        global del_btn_disabled
        global select_lbl

        if not clicked:
            i=0
            select_lbl.configure(text = "Deselect")
            del_btn.configure(state="normal", cursor = "hand2")

            for btn in del_btn.winfo_children():
                btn.destroy()

            for frame in view_frame.winfo_children():
                is_selected.append(tk.BooleanVar())
                checkbox = tk.Checkbutton(frame, variable=is_selected[i])
                checkbox_array.append(checkbox)
                x = (int(frame.winfo_width()) - 25) / int(frame.winfo_width())
                checkbox.place(relx=x,rely=0.1)

                i += 1
            clicked = True
        else:
            select_lbl.configure(text="Select")
            del_btn.configure(state= "disabled", cursor = "arrow")

            for checkbox in checkbox_array:
                checkbox.destroy()

            is_selected.clear()
            checkbox_array.clear()
            clicked = False

    def delete_selected():
        global clicked
        global is_selected
        global checkbox_array

        i = 0
        j = -1

        for selected in is_selected:
            if selected.get():
                j += 1
                del input_array[i - j]
            i += 1

        if len(input_array) == 0:
            clear()
        else:
            del_btn_disabled = tk.Label(del_btn, image=delete_image_disable, bg="#323232")
            del_btn_disabled.place(relx=0, rely=0)
            input_gallery_gui()
            is_selected.clear()
            checkbox_array.clear()

        clicked = False
        select_lbl.configure(text="Select")
        del_btn.configure(state="disabled", cursor="arrow")

    def click_image(id):
        global clicked

        if not clicked:
            select_img()

        if len(is_selected) > 0:
            if not is_selected[id].get():
                is_selected[id].set(1)
            else:
                is_selected[id].set(0)

    def show_input_thread():
        global isLoaded

        if isLoaded:
            t3 = Thread(target = create_container)
            t3.daemon = True
            t3.start()

    def create_container():
        global view_frame
        global im_label_array
        global checkbox_array
        global container_array
        global isLoaded

        try:
            row_dimension = 0
            column_cimension = 0

            i = 0
            for file in input_array:
                isLoaded = False
                # if file is an image then create an image widget
                if os.path.isfile(file):
                    try:
                        img = cv2.imread(file)
                    except:
                        text = "Cannot open image!"
                        error_handler(text, True)

                    if column_size == 4:
                        img = image.downscale(img, 100)
                    if column_size == 3:
                        img = image.downscale(img, 150)
                    if column_size == 2:
                        img = image.downscale(img, 200)

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = ImageTk.PhotoImage(img)
                    images.append(img)

                    view_frame.update()
                    if view_frame is not None and view_frame.winfo_height() > 630:
                        scrollbar = ttk.Scrollbar(display_frame, command=display_canvas.yview)
                        scrollbar.place(relx=1, rely=0, relheight=0.89, anchor='ne')
                        display_canvas.configure(yscrollcommand=scrollbar.set)

                    if column_cimension < column_size:
                        image_frame = tk.Frame(view_frame, height=img.height(), width=img.width(), bg="#2C2B2B", bd=0,
                                               relief="groove")
                        image_frame.grid(row=row_dimension, column=column_cimension)
                        # change the h and w of tk.Button when trying display the image
                        container = tk.Button(image_frame, image=img, borderwidth=0, highlightthickness=0,
                                              command=lambda id=i: click_image(id))
                        container_array.append(container)
                        container.place(relx=0.05, rely=0.1)
                        im_label = tk.Label(image_frame)
                        im_label_array.append(im_label)
                        column_cimension += 1

                    if column_cimension == column_size:
                        row_dimension += 1
                        column_cimension = 0
                    i += 1
                else:
                    text = "Cannot Open Images! \nImages does not exist or deleted!"
                    error_handler(text, True)
        except:
            pass

        isLoaded = True

    def input_gallery_gui():
        global images
        global view_frame
        global foreground_input_list_box
        global input_array
        global view_frame
        global foreground_input_list_box
        global display_frame
        global display_canvas

        im_label_array.clear()

        def on_configure(event):
            display_canvas.configure(scrollregion=display_canvas.bbox('all'))

        def _on_mousewheel(event):
            display_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        ttkstyle = ttk.Style()
        ttkstyle.theme_use("classic")
        ttkstyle.configure("Vertical.TScrollbar", background="#FAFDD6",arrowsize = 1, borderwidth = 0, troughcolor = "#2b2a2a", relief = "groove")

        display_frame = tk.Frame(foreground_input_list_box, height=720, width=870, bg="#FAFDD6")
        display_frame.place(relx=0, rely=0)
        display_canvas = tk.Canvas(display_frame, bg="#FAFDD6", height=630, width=960, borderwidth=0, highlightthickness=0, )
        display_canvas.place(relx=0, rely=0)
        view_frame = tk.Frame(display_canvas, bg="#FAFDD6")
        view_frame.bind('<Configure>', on_configure)
        mainwindow.bind("<MouseWheel>", _on_mousewheel)
        display_canvas.create_window(0, 0, window=view_frame)

        show_input_thread()

    def checkI_home_handler():
        global isHomeBool
        global input_array
        global select_btn_disabled
        global del_btn_disabled
        global clean_btn_disabled
        # global use_cam_btn_disabled
        global column_size
        global column_handler_btn_disabled
        global column_label

        if len(input_array) > 0:
            if isHomeBool == True:
                select_btn.configure(state = "disabled",cursor="arrow", bg ="#e8e8e8")
                del_btn.configure(state="disabled",cursor="arrow", bg = "#e8e8e8")
                clean_btn.configure(state = "disabled",cursor="arrow", bg = "#e8e8e8")
                # use_cam_btn.configure(state="normal", cursor="hand2")
                column_handler_btn.configure(state = "disabled", cursor = "arrow")

                if column_label["text"] == "Large":
                    column_handler_btn_disabled = tk.Label(frame1, image= small_image_disabled, bg = "#323232")
                if column_label["text"] == "Medium":
                    column_handler_btn_disabled = tk.Label(frame1, image=medium_image_disabled, bg = "#323232")
                if column_label["text"] == "Small":
                    column_handler_btn_disabled = tk.Label(frame1, image=large_image_disabled, bg = "#323232")

                column_handler_btn_disabled.place(relx=0.235, rely=0.015)

                select_btn_disabled = tk.Label(frame1, image = select_image_disable, bg = "#323232")
                select_btn_disabled.place(relx=0.07,rely=0.015)

                del_btn_disabled = tk.Label(del_btn, image=delete_image_disable, bg = "#323232")
                del_btn_disabled.place(relx = 0, rely = 0)

                clean_btn_disabled = tk.Label(frame1, image=clear_image_disable, bg = "#323232")
                clean_btn_disabled.place(relx=0.18, rely=0.015)

                # try:
                #     use_cam_btn_disabled.destroy()
                # except:
                #     pass

            else:
                select_btn.configure(state = "normal", cursor = "hand2")
                clean_btn.configure(state = "normal", cursor = "hand2")
                column_handler_btn.configure(state="normal", cursor="hand2")

                select_btn_disabled.destroy()
                clean_btn_disabled.destroy()
                column_handler_btn_disabled.destroy()

                # try:
                #     use_cam_btn_disabled.destroy()
                # except:
                #     pass
                #
                # use_cam_btn_disabled = tk.Label(frame1, image=camera_image_disable, bg="#323232")
                # use_cam_btn_disabled.place(relx=0.655, rely=0.015)
        else:
            isHomeBool = True
            select_btn.configure(state="disabled",cursor="arrow")
            del_btn.configure(state="disabled",cursor="arrow")
            clean_btn.configure(state="disabled",cursor="arrow")
            # use_cam_btn.configure(state="normal", cursor="hand2")
            column_handler_btn.configure(state="disabled", cursor="arrow")

            select_btn_disabled = tk.Label(frame1, image=select_image_disable, bg = "#323232")
            select_btn_disabled.place(relx=0.07, rely=0.015)

            del_btn_disabled = tk.Label(del_btn, image=delete_image_disable, bg="#323232")
            del_btn_disabled.place(relx=0, rely=0)

            clean_btn_disabled = tk.Label(frame1, image=clear_image_disable, bg="#323232")
            clean_btn_disabled.place(relx=0.18, rely=0.015)

            if column_label["text"] == "Large":
                column_handler_btn_disabled = tk.Label(frame1, image=small_image_disabled, bg="#323232")
            if column_label["text"] == "Medium":
                column_handler_btn_disabled = tk.Label(frame1, image=medium_image_disabled, bg="#323232")
            if column_label["text"] == "Small":
                column_handler_btn_disabled = tk.Label(frame1, image=large_image_disabled, bg="#323232")

            column_handler_btn_disabled.place(relx=0.235, rely=0.015)

            # try:
            #     use_cam_btn_disabled.destroy()
            # except:
            #     pass

    def add_image_handler():
        global input_array
        global isHomeBool
        temp_len = len(input_array)
        index = 0
        num = 0
        isNotBigger = False
        isCorrupted = False

        for image in filedialog.askopenfilenames(initialdir = "/Desktop" if input_folder_path is None else input_folder_path, title = "Add Image/s", filetypes = (("image files",".jpg"),("image files",".png"), ("image files",".jpeg"))):
            num += 1
            try:
                temp = Image.open(image)

                if temp.width > 512 and temp.height > 512:
                    index += 1
                    input_array.append(image)
                else:
                    isNotBigger = True

            except UnidentifiedImageError:
                isCorrupted = True

        if isNotBigger and isCorrupted:
            text = "Some images are corrupted and smaller than 512x512!"
            error_handler(text, True)

        elif isNotBigger:
            text = "Images must be bigger than 512x512!"
            error_handler(text, False)

        elif isCorrupted:
            text = "Some images are corrupted!"
            error_handler(text, True)

        if index > 0 and len(input_array) > temp_len:
            #input_array += added_images
            select_btn.configure(state="normal")
            clean_btn.configure(state="normal")
            input_gallery_gui()
            isHomeBool = False
            checkI_home_handler()

    def update_column_handler():
        global column_size
        global view_frame
        global display_frame
        global display_canvas
        global isLoaded

        if isLoaded:
            if column_size == 4:
                column_size = 2
            else:
                column_size += 1

            if column_size == 4:
                column_handler_btn.configure(image = large_image)
                column_label.configure(text = "Small", bg = "#323232")
                column_label.place(relx=0.24)
            elif column_size == 3:
                column_handler_btn.configure(image = medium_image)
                column_label.configure(text="Medium", bg = "#323232")
                column_label.place(relx = 0.235)
            elif column_size == 2:
                column_handler_btn.configure(image = small_image)
                column_label.configure(text="Large", bg = "#323232")
                column_label.place(relx=0.24)

                try:
                    if view_frame is not None and view_frame.winfo_height() > 630:
                        scrollbar = ttk.Scrollbar(display_frame, command=display_canvas.yview)
                        scrollbar.place(relx=1, rely=0, relheight=0.89, anchor='ne')
                        display_canvas.configure(yscrollcommand=scrollbar.set)
                except:
                    pass

            for widget in foreground_input_list_box.winfo_children():
                widget.destroy()

            input_gallery_gui()

    def initialize_stream():
        global cmodnet

        try:
            pretrained_ckpt = "pretrained/modnet_webcam_portrait_matting.ckpt"
            cmodnet = cam_modnet.cam_modnet(pretrained_ckpt)
        except:
            text = "Pretrained Model not found!"
            error_handler(text, True)

    def capture():
        global frame_update
        global preview_frame
        global imm
        global isSaveTransparent
        global transparent

        if frame_update is not None:
            img = Image.fromarray(np.uint8(frame_update))

            name = time.strftime("%Y%m%d-%H%M%S") + '.png'
            img.save(os.path.join(output_loc, name))

            img.save(os.path.join(output_loc, name))

            if transparent is not None:
                transparent_name = name.split('.')[0] + "_transparent" + ".png"
                try:
                    path = os.path.join(output_loc, "Transparent Images")
                    os.mkdir(path)
                except:
                    pass

                transparent.save(os.path.join(path, transparent_name))

            img.thumbnail((400, 400))
            imgtk = ImageTk.PhotoImage(image=img)
            imm.append(imgtk)
            preview.configure(height=330, width=315, image=imgtk)

    def press(event):
        try:
            int(event.char)
        except:
            capture()

    def create_grid(frame):
        width, height = frame.size
        x1 = int(width/3)
        y1 = int(height/3)

        frame = np.array(frame)

        frame = cv2.line(frame, (x1,0), (x1, height), (0,0,0), thickness=1)
        frame = cv2.line(frame, (x1*2, 0), (x1*2, height), (0, 0, 0), thickness=1)
        frame = cv2.line(frame, (0, y1), (width, y1), (0, 0, 0), thickness=1)
        frame = cv2.line(frame, (0, y1*2), (width, y1*2), (0, 0, 0), thickness=1)

        return Image.fromarray(frame)

    def set_grid():
        global isGrid

        if isGrid:
            isGrid = False
        else:
            isGrid = True

    def thread_process_stream():
        global frame_update
        global current_background
        global streaming
        global frame_np
        global preview_stream
        global load_lbl
        global camera
        global width_var
        global height_var
        global t1
        global transparent
        global stop_camera_btn
        global grid_btn
        global isGrid

        current_background = background_path
        bg = Image.open(background_path)

        stop_camera_btn = tk.Button(use_camera_frame, height=2, width=9, text="Stop", font=("Roboto", 12), fg="#e0efff",
                                    bg="#ba6032",
                                    activebackground="#ba6032", borderwidth=0, highlightthickness=0, cursor="hand2",
                                    command=stop_camera_handler)

        grid_btn = tk.Button(use_camera_frame, height=2, width=9, text="Grid", font=("Roboto", 12), fg="#e0efff",
                                    bg="#4369D9",
                                    borderwidth=0, highlightthickness=0, cursor="hand2",
                                    command=set_grid)

        while True:
            try:
                if frame_np is not None and streaming:
                    if current_background != background_path:
                        bg = Image.open(background_path)
                        current_background = background_path

                    frame_update, transparent = cmodnet.update(frame_np, bg, inputsize_checkbox.get(), (width_var, height_var), isSaveTransparent.get())

                    load_lbl.destroy()
                    img = Image.fromarray(frame_update)
                    img.thumbnail((900,600))

                    if isGrid:
                        img = create_grid(img)

                    imgtk = ImageTk.PhotoImage(image=img)
                    preview_stream.config(image=imgtk)
                    preview_stream.image = imgtk

                    # center preview
                    x = ((930 - img.width) / 2) / 930
                    y = ((600 - img.height) / 2) / 600

                    preview_stream.place(relx=x, rely=y)
                    stop_camera_btn.place(relx=0.78, rely=0.05)
                    grid_btn.place(relx=0.66, rely = 0.05)


            except Exception as e:
                print(traceback.format_exc())
                streaming = False
                break

    def stream():
        global streaming
        global streamer
        global frame_np
        global fg_np
        global load_lbl
        global cap

        disconnect_img = cv2.imread("resources/images/disconnect.png")

        t2 = Thread(target = thread_process_stream)
        t2.daemon = True
        load_lbl.configure(text = "Connecting to camera")

        cap = cv2.VideoCapture(camera)

        while True:
            _, frame = cap.read()

            if np.sum(frame) == np.sum(disconnect_img):
                text = "Camera is disconnected!"
                error_handler(text, True)
                stop_camera_handler()
                break

            if frame is not None and streaming:
                frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if not t2.is_alive():
                    try:
                        t2.start()
                    except RuntimeError:
                        pass
            else:
                streaming = False
                break

    def stop_camera_handler():
        global isClick_camera
        global streaming
        global cap
        global preview_stream
        global start_cam_btn
        global stop_camera_btn
        global grid_btn

        streaming = False

        try:
            preview_stream.destroy()
            grid_btn.destroy()
            stop_camera_btn.destroy()
        except:
            pass

        start_cam_btn = tk.Button(frame_preview, height=3, width=25, text="Start Camera Capture", font=("Roboto", 14),
                                  fg="#ffffff", bg="#4369D9", activebackground="#314d9e", borderwidth=0,
                                  highlightthickness=0, cursor="hand2", command=start_stream)
        start_cam_btn.place(relx=0.35, rely=0.40)

        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            streaming = False

    def start_stream():
        global camera_chosen
        global camera
        global cmodnet
        global start_cam_btn
        global frame_preview
        global load_lbl
        global streaming
        global t1
        global preview_stream
        global use_camera_frame

        preview_stream = tk.Label(frame_preview, bg="#2C2B2B")

        if os.path.isfile(background_path):

            streaming = True

            start_cam_btn.destroy()
            load_lbl = tk.Label(frame_preview, text="Setting up camera", font=("Roboto", 20), fg="#D6D2D2",
                                bg="#2C2B2B")
            load_lbl.place(relx=0.35, rely=0.40)

            # setup camera
            camera = cam_lists.index(camera_chosen.get())

            t1 = Thread(target=stream)
            t1.daemon = True
            time.sleep(2)
            t1.start()

            mainwindow.bind('<KeyPress>', press)

        else:
            text = "No Background Found!"
            error_handler(text, True)

    def use_camera_handler():
        global isClick_camera
        global camera_chosen
        global cam_lists
        global init_thread
        global start_cam_btn
        global frame_preview
        global use_camera_frame
        global streaming
        global background_path

        if not isClick_camera:
            init_thread = Thread(target=initialize_stream)
            init_thread.daemon = True
            init_thread.start()

            camera_chosen = tk.StringVar()

            streaming = False

            # get camera lists
            graph = FilterGraph()
            cam_lists = graph.get_input_devices()

            use_camera_frame = tk.Frame(mainwindow, height= 720, width=850, bg = "#323232")
            use_camera_frame.place(relx = 0, rely = 0)

            frame_preview = tk.Frame(use_camera_frame, width = 930, height = 609,bg = "#2C2B2B")
            frame_preview.place(relx= 0.005, rely=0.141)

            camera_chosen.set(cam_lists[0])
            tk.Label(use_camera_frame, font = ("Roboto", 12), text = "Choose Camera Device: ", fg = "#D6D2D2", bg = "#323232").place(relx = 0.01, rely = 0.03)
            dropdown = ttk.OptionMenu(use_camera_frame, camera_chosen, cam_lists[0], *cam_lists)
            dropdown.place(relx = 0.02, rely = 0.08)

            start_cam_btn = tk.Button(frame_preview, height = 3, width = 25, text = "Start Camera Capture", font = ("Roboto", 14), fg = "#ffffff", bg = "#4369D9",activebackground="#314d9e", borderwidth= 0, highlightthickness= 0,cursor = "hand2", command = start_stream)
            start_cam_btn.place(relx=0.35,rely=0.40 )

            capturebtn = tk.Button(menu_frame, height=3, width=25, text="CAPTURE", font=("Roboto", 16), fg="#e0efff", bg="#4369D9",
                      activebackground="#4a9eff", cursor="hand2", borderwidth=0, highlightthickness=0,
                      command=capture)
            capturebtn.place(relx=0.025, rely=0.87)

            tk.Button(use_camera_frame,height = 2, width = 9, text = "Exit", font = ("Roboto", 12), fg = "#e0efff", bg = "#8f2615", activebackground="#ba6032", borderwidth= 0, highlightthickness= 0,cursor = "hand2",command = lambda: exit_handler()).place(relx=0.9,rely=0.05)
            isClick_camera = True

            def exit_handler():
                global isClick_camera
                global streaming
                global cap

                if streaming:
                    text = "Stop the camera first to exit!"
                    error_handler(text, True)
                else:
                    use_camera_frame.destroy()
                    isClick_camera = False
                    capturebtn.destroy()
                    try:
                        cap.release()
                        cv2.destroyAllWindows()
                    except:
                        streaming = False
                isClick_camera = False

    # start of main gui creationg with TkinterDnD wrapper
    mainwindow = TkinterDnD.Tk()

    # initialize ipbr
    def initialize_ipbr():
        global main
        global isModelPresent
        try:
            main = ipbr.main()
            isModelPresent = True
        except FileNotFoundError:
            text = "Pretrained Model not found!"
            error_handler(text, True)
            isModelPresent = False

    init_ipbr = Thread(target=initialize_ipbr)
    init_ipbr.start()

    # load config
    conf = config.conf()
    output_loc, background_path, save_transparent, ifcheck_var, width_var, height_var, backgrounds_array = conf.get_conf()

    # global variables
    height_entry_var = tk.StringVar()
    width_entry_var = tk.StringVar()
    temp = tk.BooleanVar()
    inputsize_checkbox = tk.BooleanVar()
    isSaveTransparent = tk.BooleanVar()
    temp_output_loc = output_loc
    yindex = 0.1
    im_index = 0
    view_frame = None
    input_folder_path = ""
    input_array = []
    im_label_array = []
    container_array = []
    checkbox_array = []
    is_selected = []
    id_array = []
    imm = []
    images = []
    col_d = 0
    row_d = 0
    isHomeBool = True
    column_size = 4
    clicked = False
    isClick_camera = False
    stopped = False
    isGrid = False
    isLoaded = True
    current_background = ""
    mainwindow_width = 1200
    mainwindow_height = 720

    # convert str to int
    width_var = int(width_var)
    height_var = int(height_var)

    # set checkboxes for settings
    if ifcheck_var == '1':
        inputsize_checkbox.set(False)
    else:
        inputsize_checkbox.set(True)

    if save_transparent == '1':
        isSaveTransparent.set(True)
    else:
        isSaveTransparent.set(False)

    # set default background preview
    if len(backgrounds_array) > 0:
        if os.path.isfile(background_path):
            try:
                background_image = Image.open(background_path)
                background_image.thumbnail((250, 250))
                background_image = ImageTk.PhotoImage(background_image)
            except:
                text = "Cannot Open Background Image to Preview!"
                error_handler(text, True)
        else:
            background_image = None
    else:
        background_image = None

    #create and assign icons image
    add_image_icon = Image.open("resources/images/add_image.png")
    add_image_icon.thumbnail((50,50))
    add_image_icon = ImageTk.PhotoImage(add_image_icon)

    select_image = Image.open("resources/images/select_image.png")
    select_image.thumbnail((50,50))
    select_image = ImageTk.PhotoImage(select_image)

    select_image_disable = Image.open("resources/images/Select Image-disabled.png")
    select_image_disable.thumbnail((50,50))
    select_image_disable = ImageTk.PhotoImage(select_image_disable)

    delete_image = Image.open("resources/images/delete_image.png")
    delete_image.thumbnail((50,50))
    delete_image = ImageTk.PhotoImage(delete_image)

    delete_image_disable = Image.open("resources/images/Delete Image-disabled.png")
    delete_image_disable.thumbnail((50,50))
    delete_image_disable = ImageTk.PhotoImage(delete_image_disable)

    clear_image = Image.open("resources/images/Clear.png")
    clear_image.thumbnail((50,50))
    clear_image = ImageTk.PhotoImage(clear_image)

    clear_image_disable = Image.open("resources/images/Clear-disabled.png")
    clear_image_disable.thumbnail((50,50))
    clear_image_disable = ImageTk.PhotoImage(clear_image_disable)

    # camera_image = Image.open("resources/images/Camera.png")
    # camera_image.thumbnail((50,50))
    # camera_image = ImageTk.PhotoImage(camera_image)
    #
    # camera_image_disable = Image.open("resources/images/Camera-disabled.png")
    # camera_image_disable.thumbnail((50,50))
    # camera_image_disable = ImageTk.PhotoImage(camera_image_disable)

    trash_image = Image.open("resources/images/trash.png")
    trash_image.thumbnail((20,20))
    trash_image = ImageTk.PhotoImage(trash_image)

    small_image = Image.open("resources/images/2.png")
    small_image.thumbnail((50, 50))
    small_image = ImageTk.PhotoImage(small_image)

    small_image_disabled = Image.open("resources/images/2_disabled.png")
    small_image_disabled.thumbnail((50,50))
    small_image_disabled = ImageTk.PhotoImage(small_image_disabled)

    medium_image = Image.open("resources/images/3.png")
    medium_image.thumbnail((50, 50))
    medium_image = ImageTk.PhotoImage(medium_image)

    medium_image_disabled = Image.open("resources/images/3-disabled.png")
    medium_image_disabled.thumbnail((50,50))
    medium_image_disabled = ImageTk.PhotoImage(medium_image_disabled)

    large_image = Image.open("resources/images/4.png")
    large_image.thumbnail((50, 50))
    large_image = ImageTk.PhotoImage(large_image)

    large_image_disabled = Image.open("resources/images/4-disabled.png")
    large_image_disabled.thumbnail((50,50))
    large_image_disabled = ImageTk.PhotoImage(large_image_disabled)

    add_background_icon = tk.PhotoImage(file = "resources/images/add_background_icon.png")
    icon2 = ("resources/images/logo.ico")


    # set mainwindow to pop in center
    screen_width = mainwindow.winfo_screenwidth()
    screen_height = mainwindow.winfo_screenheight()
    #configure mainwindow / root
    mainwindow.iconbitmap(icon2)
    mainwindow.geometry(f"{mainwindow_width}x{mainwindow_height}+{int((screen_width / 2) - (mainwindow_width / 2))}+{int((screen_height / 2) - (mainwindow_height / 2))}")
    mainwindow.title("Intelligent Portrait Background Replacement")
    mainwindow.configure(bg = "#343434")
    mainwindow.resizable(False, False)

    #create main window widgets
    frame1 = tk.Frame(mainwindow, height= 55, width = 900, bg = "#343434").place(x=0,y=0)
    add_btn = tk.Button(frame1, image = add_image_icon,bg = "#343434", height = 50, width = 50,  bd = 2, command = add_image_handler, cursor = "hand2", borderwidth= 0 , highlightthickness= 0)
    add_btn.place(relx = 0.015, rely = 0.015)
    tk.Label(frame1, text = "Add", font = ("Roboto", 10), fg = "#D6D2D2", bg = "#343434").place(relx = 0.02, rely = 0.1)
    select_btn = tk.Button(frame1, image = select_image, command = lambda : [del_btn.configure(state="normal"), select_img()], bg = "#323232", height = 50, width = 50, borderwidth= 0 , highlightthickness= 0)
    select_btn.place(relx=0.07,rely=0.015)
    select_lbl = tk.Label(frame1, text="Select", font = ("Roboto", 10), fg = "#D6D2D2", bg = "#343434")
    select_lbl.place(relx = 0.0725, rely = 0.1)
    del_btn = tk.Button(frame1, image = delete_image, command = delete_selected, bg = "#343434", height = 50, width = 50,borderwidth= 0 , highlightthickness= 0)
    del_btn.place(relx = 0.125, rely = 0.015)
    tk.Label(frame1, text="Delete", font = ("Roboto", 10), fg = "#D6D2D2", bg = "#343434").place(relx = 0.1275, rely = 0.1)
    clean_btn = tk.Button(frame1, image = clear_image, bg = "#343434", height = 50, width = 50,command = clear, borderwidth= 0 , highlightthickness= 0)
    clean_btn.place(relx = 0.18, rely = 0.015)
    tk.Label(frame1, text="Clear", font = ("Roboto", 10), fg = "#D6D2D2", bg = "#343434").place(relx = 0.1865, rely = 0.1)
    column_handler_btn = tk.Button(frame1, image = large_image, bg = "#343434", height = 50, width = 50, command = update_column_handler, borderwidth= 0 , highlightthickness= 0)
    column_handler_btn.place(relx=0.235,rely=0.015)
    column_label = tk.Label(frame1, text = "Small", font = ("Roboto", 10), fg = "#D6D2D2", bg = "#343434")
    column_label.place(relx = 0.24, rely = 0.1)
    # use_cam_btn = tk.Button(frame1, image = camera_image, command =use_camera_handler,bg = "#323232", height = 50, width = 50,borderwidth= 0 , highlightthickness= 0)
    # use_cam_btn.place(relx=0.655,rely=0.015)
    # tk.Label(frame1, text="Use Camera", font = ("Roboto", 10), fg = "#D6D2D2", bg = "#323232").place(relx = 0.6425, rely = 0.1)

    foreground_input_list_box = tk.Listbox(mainwindow, selectmode= tk.SINGLE, width = 200, height = 38, bg = "#343434", bd = 1, relief = "groove", borderwidth= 0, highlightthickness=0 )
    foreground_input_list_box.drop_target_register(DND_FILES)
    foreground_input_list_box.dnd_bind("<<Drop>>", drop_inside_list_box)
    foreground_input_list_box.place(relx= 0.005, rely=0.141)
    tk.Label(foreground_input_list_box, text= "Drop image folder here", font = ("Roboto", 20), fg = "white", bg = "#343434").place(relx= 0.25, rely = 0.35)
    tk.Label(foreground_input_list_box, text= "or", font = ("Roboto", 20), fg = "white", bg = "#343434").place(relx= 0.35, rely = 0.41)
    tk.Button(foreground_input_list_box, text = "Browse", height = 2, width=20, font = ("Roboto", 17),  fg = "white", bg = "#78938A", activebackground="#4a9eff", cursor = "hand2", borderwidth= 0, highlightthickness= 0,command= get_input_handler).place(relx= 0.255, rely = 0.50)

    #create menu frame widget
    menu_frame = tk.Frame(mainwindow, height= 720, width=350, relief="groove", bg = "#525E75")
    menu_frame.place(relx= 0.73, rely= 0)
    # tk.Label(menu_frame, text = "Chosen Background Image Preview: ", font = ("Roboto", 12), fg = "#D6D2D2", bg = "#2C2B2B").place(relx= 0.05, rely = 0.02)
    background_preview = tk.Label(menu_frame, bg = "#525E75", bd =2, relief = "groove", borderwidth= 0, highlightthickness=0)
    background_preview.place(relx = 0.02, rely = 0)
    background_preview.configure(height=160, width=310, image=background_image)
    preview_frame = tk.Frame(menu_frame, height= 500, width= 310, bg = "#525E75")
    preview_frame.place(relx=0.02, rely=0.385)
    preview = tk.Label(preview_frame, height= 22, width= 315, bg = "#525E75" )
    preview.place(relx = 0, rely =0)
    tk.Button(menu_frame, height = 2, width = 20, text = "Change Background", font = ("Roboto", 14), fg = "#b3e3b3", bg = "#78938A", activebackground="#d1d971", cursor ="hand2",borderwidth= 0, highlightthickness= 0, command=background_panel_gui).place(relx= 0.135, rely= 0.26)
    tk.Button(menu_frame, height = 2, width = 20, text = "Settings", font = ("Roboto", 14), fg = "#b3e3b3", bg = "#78938A", activebackground="#d1d971", cursor ="hand2",borderwidth= 0, highlightthickness= 0,command = open_settings).place(relx= 0.135, rely= 0.37)
    start_btn = tk.Button(menu_frame, height = 3, width = 25, text = "START", font = ("Roboto", 16), fg = "#b3e3b3", bg = "#78938A", activebackground="#d1d971", cursor ="hand2",borderwidth= 0, highlightthickness= 0, command = start_thread)
    start_btn.place(relx=0.025, rely=0.87)
    checkI_home_handler()

    #make main window display in loop
    mainwindow.mainloop()
"""------------------------------------------------------------------------------------"""
# --*-- coding: utf-8 --*--
# author: zwei
# email: suifeng20@hotmail.com

'''
Parse html file
'''
import re
import six
import urlparse
from bs4.element import Tag
from bs4 import BeautifulSoup

from weibo import exception
from weibo.jiexi import userinfo
from weibo.common.gettextutils import _
from weibo.common import log as logging

LOG = logging.getLogger(__name__)


class HBeautifulSoup(BeautifulSoup):
    pass


class Soup(object):
    def __init__(self, *args, **kwargs):
        self.parser_type = "html.parser"
        self.soup = None

    def __call__(self, wb, **kwargs):
        if not wb:
            raise

        self.soup = HBeautifulSoup(wb, self.parser_type)

    def __getattr__(self, key):
        if self.soup:
            return getattr(self.soup, key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class ZDetail(object):
    def __init__(self, z_jx, jdetail):
        self.z_jx = z_jx
        self.jx = jdetail

    def get_wb_text(self, zf=True):
        self._get_wb_mid_date()
        self._get_wb_uid_name()
        return self.jx.get_wb_text(self.z_jx, zf)

    def get_wb_img(self, zf=True):
        return self.jx.get_wb_img(self.z_jx, zf)

    def get_wb_videos(self, zf=True):
        return self.jx.get_wb_videos(self.z_jx, zf)

    def _get_wb_mid_date(self, zf=True):
        self.jx._get_wb_mid_date(self.z_jx, zf)

    def _get_wb_uid_name(self, zf=True):
        self.jx._get_wb_uid_name(self.z_jx, zf)

    def get_wb_mid(self, zf=True):
        return self.jx.get_wb_mid(self.z_jx, zf)

    def get_wb_uid(self, zf=True):
        return self.jx.get_wb_uid(self.z_jx, zf)

    def get_wb_date(self, zf=True):
        return self.jx.get_wb_date(self.z_jx, zf)

    def get_wb_name(self, zf=True):
        return self.jx.get_wb_name(self.z_jx, zf)


class JDetail(object):
    '''
    pasre WB_detail info
    '''
    def __init__(self, *args, **kwargs):
        self.content = kwargs.get('content', None)
        self.jx = Soup()
        self.z_jx = None
        self.is_zf = False
        self.wb_2frp = None
        self.wb_mid_date = {}
        self.wb_uid_name = {}

    def __call__(self, WB, **kwargs):
        wb = self._get_wb_html(WB)
        self.jx(wb)

    def _get_wb_html(self, WB, content=None):
        if content is None:
            content = self.content

        # 去掉 "\\n, \\r, \\t, \\/, \\"
        WB = WB.replace('\\n', '')
        WB = WB.replace('\\t', '')
        WB = WB.replace('\\r', '')
        WB = WB.replace('\\/', '/')
        WB = WB.replace('\\', '')

        # 确保完整的html 信息
        WB = WB[WB.find('>') + 1:WB.rfind('<')]
        return WB.strip()

    # 是否是转发微博
    @property
    def is_zf_wb(self):
        div_attrs = {'class': 'WB_feed_expand'}
        z_jx = self.jx.findChild(name='div', attrs=div_attrs)
        if z_jx and not self.is_zf_wb_and_delete(z_jx):
            self.is_zf = True
            self.get_zf_wb(z_jx)
        else:
            self.is_zf = False

        return self.is_zf

    # 是转发微博 但是 转发被删除
    def is_zf_wb_and_delete(self, z_jx):
        div_attrs = {'class': 'WB_empty'}
        try:
            self._get_children_tag(z_jx, name='div', attrs=div_attrs)
        except exception.NotFoundChildrenTag:
            return False
        return True

    def get_zf_wb(self, z_jx=None):
        if not isinstance(z_jx, Tag):
            raise exception.NotFoudZfweibo()

        div_attrs = {'node-type': 'feed_list_forwardContent'}
        z_jx = z_jx.findChild(name='div', attrs=div_attrs)
        self.z_jx = ZDetail(z_jx, self)

    # 获取 正常的 text 文本信息
    def _resolve_text(self, text):
        if not isinstance(text, six.text_type):
            text = unicode(text, 'utf-8')
        return text.strip()

    # 获取下一个 第一个 tag
    def _get_children_tag(self, jx, name=None, attrs={}, **kwargs):
        children = jx.findChild(name, attrs, **kwargs)
        if not children:
            attrs.setdefault('name', name)
            attrs.setdefault('class', None)
            attrs.setdefault('node-type', None)
            LOG.debug(_("children %(name)s, %(class)s, "
                        "%(node-type)s"), attrs)
            raise exception.NotFoundChildrenTag(attrs)
        return children

    # 获取 微博的uid, name 等
    def _get_wb_uid_name(self, jx=None, zf=False):
        self.wb_uid_name = {}
        if not jx:
            jx = self.jx
        if not zf:
            div_attrs = {'class': 'WB_info'}
            uidname_div = self._get_children_tag(jx,
                                                 name='div',
                                                 attrs=div_attrs)
            a_first = self._get_children_tag(uidname_div,
                                             name='a')
            name = a_first.get_text()

            usercard = a_first.attrs.get('usercard', None)
            usercard = urlparse.parse_qsl(usercard)
            usercard = dict(usercard)
            uid = usercard.get('id', None)

            self.wb_uid_name['name'] = name
            self.wb_uid_name['uid'] = uid
        else:
            div_attrs = {'class': 'WB_info'}
            uidname_div = self._get_children_tag(jx,
                                                 name='div',
                                                 attrs=div_attrs)
            a_first = self._get_children_tag(uidname_div,
                                             name='a')
            usercard = a_first.attrs.get('usercard', None)
            usercard = urlparse.parse_qsl(usercard)
            usercard = dict(usercard)
            uid = usercard.get('id', None)
            self.wb_uid_name['uid'] = uid

            nickname = a_first.attrs.get('nick-name', None)
            self.wb_uid_name['name'] = nickname

            midstr = a_first.attrs.get('suda-uatrack', None)
            midstr = urlparse.parse_qsl(midstr)
            midstr = dict(midstr)
            midstr = midstr.get('value', None)
            midnum = midstr.find(":") + 1
            mid = midstr[midnum:]

            time_at = '0'
            self.wb_mid_date['mid'] = mid
            self.wb_mid_date['time_at'] = time_at

    # 获取 微博的mid, time_at 等
    def _get_wb_mid_date(self, jx=None, zf=False):
        self.wb_mid_date = {}
        if not jx:
            jx = self.jx
        if not zf:
            div_attrs = {'class': 'WB_from S_txt2'}
            middate_div = self._get_children_tag(jx,
                                                 name='div',
                                                 attrs=div_attrs)
            a_first = self._get_children_tag(middate_div,
                                             name='a')
            mid = a_first.attrs.get('name', None)
            time_at = a_first.attrs.get('date', '0')
            self.wb_mid_date['mid'] = mid
            self.wb_mid_date['time_at'] = time_at
        else:
            self._get_wb_uid_name(jx, zf)

    # 获取 mid
    def get_wb_mid(self, jx=None, zf=False):
        return self.wb_mid_date.get('mid', None)

    # 获取 time_at 时间
    def get_wb_date(self, jx=None, zf=False):
        return self.wb_mid_date.get('time_at', None)

    # 获取 uid
    def get_wb_uid(self, jx=None, zf=False):
        return self.wb_uid_name.get('uid', None)

    # 获取 name
    def get_wb_name(self, jx=None, zf=False):
        return self.wb_uid_name.get('name', None)

    # 获取 文本信息
    # 获取 文本信息中的 表情 图片
    def _get_wb_img_link(self, jx=None, *args, **kwargs):
        pass

    # 获取 文本信息中的 视频 链接地址
    def _get_wb_a_link(self, jx=None, *args, **kwargs):
        pass

    # 获取文本信息中的 子信息
    def _get_wb_children(self, div):
        text = div.text
        text = self._resolve_text(text)
        return text

    # 获取文本信息中的 没有 子信息
    def _get_wb_no_children(self, div):
        text = div.text
        text = self._resolve_text(text)
        return text

    def _get_wb_text(self, jx, zf=False):
        if not zf:
            div_attrs = {'node-type': 'feed_list_content'}
        else:
            div_attrs = {'node-type': 'feed_list_reason'}
        text_div = self._get_children_tag(jx=jx, name='div', attrs=div_attrs)
        return text_div

    # 获取 text 信息
    def get_wb_text(self, jx=None, zf=False):
        if not jx:
            jx = self.jx
        texts = {'is_zf': zf}
        text_div = self._get_wb_text(jx, zf)
        if len(list(text_div.children)) > 1:
            text = self._get_wb_children(text_div)
        else:
            text = self._get_wb_no_children(text_div)
        texts.setdefault('text', text)
        return texts

    # start 获取img 或者 videos 的 url 地址
    def _get_wb_img_ul(self, i_v_div):
        img_urls = []
        uls = i_v_div.findAll('ul')
        for ul in uls:
            urls = self._get_wb_img_li(ul)
            img_urls.extend(urls)
        return img_urls

    def _get_wb_img_li(self, i_v_div_ul):
        img_urls = []
        lis = i_v_div_ul.findAll('li')
        for li in lis:
            url = self._get_wb_li_img(li)
            if url:
                img_urls.append(url)
        return img_urls

    def _get_wb_li_img(self, li):
        img = li.img
        attrs = img.attrs
        return attrs.get('src', None)

    # 查询下一个 子 的 div
    def _get_wb_img_or_videos(self, jx=None, zf=False):
        if not jx:
            jx = self.jx
        if zf:
            div_attrs = {'class': 'WB_media_wrap clearfix'}
        else:
            div_attrs = {'class': 'WB_media_wrap clearfix',
                         'node-type': 'feed_list_media_prev'}
        # find first img div
        try:
            img_videos_div = self._get_children_tag(jx=jx,
                                                    name='div',
                                                    attrs=div_attrs)
        except exception.NotFoundChildrenTag:
            return None
        return img_videos_div

    # 获取 img 和videos 的url 地址
    def get_wb_img_or_videos(self, jx, zf=False):
        if not jx:
            jx = self.jx
        img_urls = {'is_zf': zf}
        i_v_div = self._get_wb_img_or_videos(jx, zf)
        if not i_v_div:
            urls = None
        else:
            urls = self._get_wb_img_ul(i_v_div)
        img_urls.setdefault('urls', urls)
        return img_urls

    # 获取所有图片的信息
    def get_wb_img(self, jx=None, zf=False):
        if self.is_zf != zf:
            return
        return self.get_wb_img_or_videos(jx, zf)

    # 获取所有视频的信息
    def get_wb_videos(self, jx=None, zf=False):
        if self.is_zf_wb != zf:
            return
    # end 获取 img or videos 的 url

    # 获取 收藏, 转发, 评论, 点赞 数据
    def _get_wb_2frp_em(self, a, zf=False):
        ems = a.findAll('em')
        if not len(ems):
            return "收藏"
        em = self._resolve_text(ems[-1].text)
        return em

    def _get_wb_2frp_a(self, li, zf=False):
        a = li.find('a')
        return self._get_wb_2frp_em(a, zf)

    def _get_wb_2frp_li(self, ul, zf=False):
        lis = ul.findAll('li')
        li_texts = []
        for li in lis:
            li_text = self._get_wb_2frp_a(li, zf)
            li_texts.append(li_text)
        return li_texts

    def _get_wb_2frp_ul(self, div_2frp, zf=False):
        wb_2frp = []
        uls = div_2frp.findAll('ul')
        for ul in uls:
            ems = self._get_wb_2frp_li(ul)
            wb_2frp.extend(ems)
        return wb_2frp

    def _get_wb_2frp(self, jx, zf=False):
        div_attrs = {'class': 'WB_feed_handle',
                     'node-type': 'feed_list_options'}
        if not jx:
            jx = self.jx

        zfrp_div = self._get_children_tag(jx=jx,
                                          name='div',
                                          attrs=div_attrs)

        return zfrp_div

    def get_wb_2frp(self, jx=None, zf=False):
        if not jx:
            jx = self.jx
        wb_2frp = self._get_wb_2frp(jx, zf)
        self.wb_2frp = self._get_wb_2frp_ul(wb_2frp, zf)

    def get_wb_favorite(self, jx=None, zf=False):
        # 收藏
        return self.wb_2frp[0]

    def get_wb_forward(self, jx=None, zf=False):
        # 转发
        try:
            fd = eval(self.wb_2frp[1])
        except:
            return 0
        else:
            return fd

    def get_wb_repeat(self, jx=None, zf=False):
        # 评论
        try:
            rt = eval(self.wb_2frp[2])
        except:
            return 0
        else:
            return rt

    def get_wb_praised(self, jx=None, zf=False):
        # 点赞
        try:
            pd = eval(self.wb_2frp[3])
        except:
            return 0
        else:
            return pd


class Jhtml(object):
    '''
    This class pasre some html file
    use re module
    '''

    def __init__(self, *args, **kwargs):
        self.userinfo = userinfo.Userinfo()
        self.weibodata = []
        self.weibodata_dict = {}
        self.fl_values = None

    def __call__(self, content):
        self.weibodata = []
        self.jdetail = JDetail()
        wbinfo = self.jiexi2(content)
        self.get_userdata_info(wbinfo, content)

    def tmp_file(self, content):
        tmp = re.findall(r'pl\.content\.homeFeed\.'
                         r'index.*html\":\"(.*)\"}\)', content)
        for tmp_r in tmp:
            content = content.replace(tmp_r, 's')

        max = 0
        for i in tmp:
            if max < len(i):
                max = len(i)
                content = i

        content = content.replace('WB_detail', 'WB_detailWB_detail')
        return content

    def get_userdata_info(self, wbinfo, content):
        uid = wbinfo[0].get('uid', None)
        if uid:
            self.fl_values = self.userinfo.jiexi2(uid, content)
            self.weibodata_dict['userdata'] = self.fl_values
        self.weibodata_dict['weibodata'] = self.weibodata
        return self.weibodata_dict

    def wb_detail(self, content):
        # get all things from WB_detail
        WB_detail = re.findall(r"WB\_detail(.+?)WB\_detail", content)
        return WB_detail

    def is_zf_wb(self, WB=None):
        if WB:
            self.jdetail(WB)
        return self.jdetail.is_zf_wb

    def wb(self, WB):
        self.jdetail(WB)
        self.jdetail.get_wb_2frp()
        self.get_wb_and_zf_info()

    def wb_text(self, WB=None, zf=False):
        # 正文
        if zf:
            text = self.jdetail.z_jx.get_wb_text()
        else:
            text = self.jdetail.get_wb_text()
        return text

    def wb_img(self, WB=None, zf=False):
        # 图片
        if zf:
            img = self.jdetail.z_jx.get_wb_img()
        else:
            img = self.jdetail.get_wb_img()
        return img

    def wb_videos(self, WB, zf=False):
        # 视频
        pass

    def wb_favorite(self, WB, zf=False):
        # 收藏
        if zf:
            return
        return self.jdetail.get_wb_favorite()

    def wb_forward(self, WB, zf=False):
        # 转发
        if zf:
            return
        return self.jdetail.get_wb_forward()

    def wb_repeat(self, WB, zf=False):
        # 评论
        if zf:
            return
        return self.jdetail.get_wb_repeat()

    def wb_praised(self, WB, zf=False):
        # 点赞
        if zf:
            return
        return self.jdetail.get_wb_praised()

    def get_wb_and_zf_info(self, WB=None, zf=False):
        # 获取weibo 主人的信息
        # 获取 weibo 本身的 信息
        self.jdetail._get_wb_uid_name(zf=zf)
        self.jdetail._get_wb_mid_date(zf=zf)

    def wb_time(self, WB, zf=False):
        # weibo 发布时间
        # WB_timestamp = re.findall(r'date=\\"([^"]*)\\"', WB)[-1]
        # checked
        if not zf:
            WB_timestamp = self.jdetail.get_wb_date()
        else:
            WB_timestamp = self.jdetail.z_jx.get_wb_date()
        return int(WB_timestamp) / 1000

    def wb_mid(self, WB, zf=False):
        # 获取微博id
        # WB_mid = re.findall(r'mid=.*?(\d*)', WB)[-1]
        # checked
        if not zf:
            WB_mid = self.jdetail.get_wb_mid()
        else:
            WB_mid = self.jdetail.z_jx.get_wb_mid()
        return WB_mid

    def wb_like(self, WB, zf=False):
        # weibo like
        WB_like = ''.join(re.findall(r'WB\_text[^>]*>.*'
                                     r'praised.*?\(([0-9]*)', WB))
        # checked
        return WB_like

    def wb_name(self, WB, zf=False):
        # 微博作者的用户信息字段
        if not zf:
            # WB_name = ''.join(re.findall(r'nick-name=\\"([^"]*)\\"', WB))
            WB_name = self.jdetail.get_wb_name()
        else:
            WB_name = self.jdetail.z_jx.get_wb_name()
        return WB_name

    def wb_uid(self, WB, zf=False):
        # 微博的作者id
        if not zf:
            # WB_uid = ''.join(re.findall(r'fuid=([^"]*)\\"', WB))
            WB_uid = self.jdetail.get_wb_uid()
            # checked
        else:
            WB_uid = self.jdetail.z_jx.get_wb_uid()
        return WB_uid

    def get_wb_info(self, wb=None, zf=False):
        wb_info = {}
        wb_info['text'] = self.wb_text(wb, zf)
        wb_info['img'] = self.wb_img(wb, zf)
        wb_info['uid'] = self.wb_uid(wb, zf)
        wb_info['mid'] = self.wb_mid(wb, zf)
        wb_info['name'] = self.wb_name(wb, zf)
        wb_info['time_at'] = self.wb_time(wb, zf)
        wb_info['favorite'] = self.wb_favorite(wb, zf)
        wb_info['forward'] = self.wb_forward(wb, zf)
        wb_info['repeat'] = self.wb_repeat(wb, zf)
        wb_info['praised'] = self.wb_praised(wb, zf)
        return wb_info

    def wb_all_jiexi2(self, wb_detail):
        for wb in wb_detail:
            # 初始化wb 信息
            self.wb(wb)
            is_zf = self.is_zf_wb()
            wb_info = self.get_wb_info(wb, False)
            if is_zf:
                wb_info.setdefault('is_zf', is_zf)
                zf_wb = self.get_wb_info(wb, True)
                zf_wb.setdefault('pa_mid', wb_info.get('mid'))
                wb_info.setdefault('zf_wb', zf_wb)
                wb_info.setdefault('zf_mid', zf_wb.get('mid', None))
            self.weibodata.append(wb_info)
        return self.weibodata

    def jiexi2(self, content=None):
        if content is None:
            raise exception.NotFoundContent()

        if six.PY3:
            content = content.decode('utf-8')

        content = self.tmp_file(content)
        wb_detail = self.wb_detail(content)
        if not len(wb_detail):
            raise exception.DetailNotFound()

        return self.wb_all_jiexi2(wb_detail)
"""------------------------------------------------------------------------------"""
"""PyPI and direct package downloading"""
import sys
import os
import re
import io
import shutil
import socket
import base64
import hashlib
import itertools
import warnings
import configparser
import html
import http.client
import urllib.parse
import urllib.request
import urllib.error
from functools import wraps

import setuptools
from pkg_resources import (
    CHECKOUT_DIST, Distribution, BINARY_DIST, normalize_path, SOURCE_DIST,
    Environment, find_distributions, safe_name, safe_version,
    to_filename, Requirement, DEVELOP_DIST, EGG_DIST,
)
from distutils import log
from distutils.errors import DistutilsError
from fnmatch import translate
from setuptools.wheel import Wheel
from setuptools.extern.more_itertools import unique_everseen


EGG_FRAGMENT = re.compile(r'^egg=([-A-Za-z0-9_.+!]+)$')
HREF = re.compile(r"""href\s*=\s*['"]?([^'"> ]+)""", re.I)
PYPI_MD5 = re.compile(
    r'<a href="([^"#]+)">([^<]+)</a>\n\s+\(<a (?:title="MD5 hash"\n\s+)'
    r'href="[^?]+\?:action=show_md5&amp;digest=([0-9a-f]{32})">md5</a>\)'
)
URL_SCHEME = re.compile('([-+.a-z0-9]{2,}):', re.I).match
EXTENSIONS = ".tar.gz .tar.bz2 .tar .zip .tgz".split()

__all__ = [
    'PackageIndex', 'distros_for_url', 'parse_bdist_wininst',
    'interpret_distro_name',
]

_SOCKET_TIMEOUT = 15

_tmpl = "setuptools/{setuptools.__version__} Python-urllib/{py_major}"
user_agent = _tmpl.format(
    py_major='{}.{}'.format(*sys.version_info), setuptools=setuptools)


def parse_requirement_arg(spec):
    try:
        return Requirement.parse(spec)
    except ValueError as e:
        raise DistutilsError(
            "Not a URL, existing file, or requirement spec: %r" % (spec,)
        ) from e


def parse_bdist_wininst(name):
    """Return (base,pyversion) or (None,None) for possible .exe name"""

    lower = name.lower()
    base, py_ver, plat = None, None, None

    if lower.endswith('.exe'):
        if lower.endswith('.win32.exe'):
            base = name[:-10]
            plat = 'win32'
        elif lower.startswith('.win32-py', -16):
            py_ver = name[-7:-4]
            base = name[:-16]
            plat = 'win32'
        elif lower.endswith('.win-amd64.exe'):
            base = name[:-14]
            plat = 'win-amd64'
        elif lower.startswith('.win-amd64-py', -20):
            py_ver = name[-7:-4]
            base = name[:-20]
            plat = 'win-amd64'
    return base, py_ver, plat


def egg_info_for_url(url):
    parts = urllib.parse.urlparse(url)
    scheme, server, path, parameters, query, fragment = parts
    base = urllib.parse.unquote(path.split('/')[-1])
    if server == 'sourceforge.net' and base == 'download':  # XXX Yuck
        base = urllib.parse.unquote(path.split('/')[-2])
    if '#' in base:
        base, fragment = base.split('#', 1)
    return base, fragment


def distros_for_url(url, metadata=None):
    """Yield egg or source distribution objects that might be found at a URL"""
    base, fragment = egg_info_for_url(url)
    for dist in distros_for_location(url, base, metadata):
        yield dist
    if fragment:
        match = EGG_FRAGMENT.match(fragment)
        if match:
            for dist in interpret_distro_name(
                url, match.group(1), metadata, precedence=CHECKOUT_DIST
            ):
                yield dist


def distros_for_location(location, basename, metadata=None):
    """Yield egg or source distribution objects based on basename"""
    if basename.endswith('.egg.zip'):
        basename = basename[:-4]  # strip the .zip
    if basename.endswith('.egg') and '-' in basename:
        # only one, unambiguous interpretation
        return [Distribution.from_location(location, basename, metadata)]
    if basename.endswith('.whl') and '-' in basename:
        wheel = Wheel(basename)
        if not wheel.is_compatible():
            return []
        return [Distribution(
            location=location,
            project_name=wheel.project_name,
            version=wheel.version,
            # Increase priority over eggs.
            precedence=EGG_DIST + 1,
        )]
    if basename.endswith('.exe'):
        win_base, py_ver, platform = parse_bdist_wininst(basename)
        if win_base is not None:
            return interpret_distro_name(
                location, win_base, metadata, py_ver, BINARY_DIST, platform
            )
    # Try source distro extensions (.zip, .tgz, etc.)
    #
    for ext in EXTENSIONS:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            return interpret_distro_name(location, basename, metadata)
    return []  # no extension matched


def distros_for_filename(filename, metadata=None):
    """Yield possible egg or source distribution objects based on a filename"""
    return distros_for_location(
        normalize_path(filename), os.path.basename(filename), metadata
    )


def interpret_distro_name(
        location, basename, metadata, py_version=None, precedence=SOURCE_DIST,
        platform=None
):
    """Generate alternative interpretations of a source distro name

    Note: if `location` is a filesystem filename, you should call
    ``pkg_resources.normalize_path()`` on it before passing it to this
    routine!
    """
    # Generate alternative interpretations of a source distro name
    # Because some packages are ambiguous as to name/versions split
    # e.g. "adns-python-1.1.0", "egenix-mx-commercial", etc.
    # So, we generate each possible interpretation (e.g. "adns, python-1.1.0"
    # "adns-python, 1.1.0", and "adns-python-1.1.0, no version").  In practice,
    # the spurious interpretations should be ignored, because in the event
    # there's also an "adns" package, the spurious "python-1.1.0" version will
    # compare lower than any numeric version number, and is therefore unlikely
    # to match a request for it.  It's still a potential problem, though, and
    # in the long run PyPI and the distutils should go for "safe" names and
    # versions in distribution archive names (sdist and bdist).

    parts = basename.split('-')
    if not py_version and any(re.match(r'py\d\.\d$', p) for p in parts[2:]):
        # it is a bdist_dumb, not an sdist -- bail out
        return

    for p in range(1, len(parts) + 1):
        yield Distribution(
            location, metadata, '-'.join(parts[:p]), '-'.join(parts[p:]),
            py_version=py_version, precedence=precedence,
            platform=platform
        )


def unique_values(func):
    """
    Wrap a function returning an iterable such that the resulting iterable
    only ever yields unique items.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return unique_everseen(func(*args, **kwargs))

    return wrapper


REL = re.compile(r"""<([^>]*\srel\s*=\s*['"]?([^'">]+)[^>]*)>""", re.I)
# this line is here to fix emacs' cruddy broken syntax highlighting


@unique_values
def find_external_links(url, page):
    """Find rel="homepage" and rel="download" links in `page`, yielding URLs"""

    for match in REL.finditer(page):
        tag, rel = match.groups()
        rels = set(map(str.strip, rel.lower().split(',')))
        if 'homepage' in rels or 'download' in rels:
            for match in HREF.finditer(tag):
                yield urllib.parse.urljoin(url, htmldecode(match.group(1)))

    for tag in ("<th>Home Page", "<th>Download URL"):
        pos = page.find(tag)
        if pos != -1:
            match = HREF.search(page, pos)
            if match:
                yield urllib.parse.urljoin(url, htmldecode(match.group(1)))


class ContentChecker:
    """
    A null content checker that defines the interface for checking content
    """

    def feed(self, block):
        """
        Feed a block of data to the hash.
        """
        return

    def is_valid(self):
        """
        Check the hash. Return False if validation fails.
        """
        return True

    def report(self, reporter, template):
        """
        Call reporter with information about the checker (hash name)
        substituted into the template.
        """
        return


class HashChecker(ContentChecker):
    pattern = re.compile(
        r'(?P<hash_name>sha1|sha224|sha384|sha256|sha512|md5)='
        r'(?P<expected>[a-f0-9]+)'
    )

    def __init__(self, hash_name, expected):
        self.hash_name = hash_name
        self.hash = hashlib.new(hash_name)
        self.expected = expected

    @classmethod
    def from_url(cls, url):
        "Construct a (possibly null) ContentChecker from a URL"
        fragment = urllib.parse.urlparse(url)[-1]
        if not fragment:
            return ContentChecker()
        match = cls.pattern.search(fragment)
        if not match:
            return ContentChecker()
        return cls(**match.groupdict())

    def feed(self, block):
        self.hash.update(block)

    def is_valid(self):
        return self.hash.hexdigest() == self.expected

    def report(self, reporter, template):
        msg = template % self.hash_name
        return reporter(msg)


class PackageIndex(Environment):
    """A distribution index that scans web pages for download URLs"""

    def __init__(
            self, index_url="https://pypi.org/simple/", hosts=('*',),
            ca_bundle=None, verify_ssl=True, *args, **kw
    ):
        Environment.__init__(self, *args, **kw)
        self.index_url = index_url + "/" [:not index_url.endswith('/')]
        self.scanned_urls = {}
        self.fetched_urls = {}
        self.package_pages = {}
        self.allows = re.compile('|'.join(map(translate, hosts))).match
        self.to_scan = []
        self.opener = urllib.request.urlopen

    # FIXME: 'PackageIndex.process_url' is too complex (14)
    def process_url(self, url, retrieve=False):  # noqa: C901
        """Evaluate a URL as a possible download, and maybe retrieve it"""
        if url in self.scanned_urls and not retrieve:
            return
        self.scanned_urls[url] = True
        if not URL_SCHEME(url):
            self.process_filename(url)
            return
        else:
            dists = list(distros_for_url(url))
            if dists:
                if not self.url_ok(url):
                    return
                self.debug("Found link: %s", url)

        if dists or not retrieve or url in self.fetched_urls:
            list(map(self.add, dists))
            return  # don't need the actual page

        if not self.url_ok(url):
            self.fetched_urls[url] = True
            return

        self.info("Reading %s", url)
        self.fetched_urls[url] = True  # prevent multiple fetch attempts
        tmpl = "Download error on %s: %%s -- Some packages may not be found!"
        f = self.open_url(url, tmpl % url)
        if f is None:
            return
        if isinstance(f, urllib.error.HTTPError) and f.code == 401:
            self.info("Authentication error: %s" % f.msg)
        self.fetched_urls[f.url] = True
        if 'html' not in f.headers.get('content-type', '').lower():
            f.close()  # not html, we can't process it
            return

        base = f.url  # handle redirects
        page = f.read()
        if not isinstance(page, str):
            # In Python 3 and got bytes but want str.
            if isinstance(f, urllib.error.HTTPError):
                # Errors have no charset, assume latin1:
                charset = 'latin-1'
            else:
                charset = f.headers.get_param('charset') or 'latin-1'
            page = page.decode(charset, "ignore")
        f.close()
        for match in HREF.finditer(page):
            link = urllib.parse.urljoin(base, htmldecode(match.group(1)))
            self.process_url(link)
        if url.startswith(self.index_url) and getattr(f, 'code', None) != 404:
            page = self.process_index(url, page)

    def process_filename(self, fn, nested=False):
        # process filenames or directories
        if not os.path.exists(fn):
            self.warn("Not found: %s", fn)
            return

        if os.path.isdir(fn) and not nested:
            path = os.path.realpath(fn)
            for item in os.listdir(path):
                self.process_filename(os.path.join(path, item), True)

        dists = distros_for_filename(fn)
        if dists:
            self.debug("Found: %s", fn)
            list(map(self.add, dists))

    def url_ok(self, url, fatal=False):
        s = URL_SCHEME(url)
        is_file = s and s.group(1).lower() == 'file'
        if is_file or self.allows(urllib.parse.urlparse(url)[1]):
            return True
        msg = (
            "\nNote: Bypassing %s (disallowed host; see "
            "http://bit.ly/2hrImnY for details).\n")
        if fatal:
            raise DistutilsError(msg % url)
        else:
            self.warn(msg, url)

    def scan_egg_links(self, search_path):
        dirs = filter(os.path.isdir, search_path)
        egg_links = (
            (path, entry)
            for path in dirs
            for entry in os.listdir(path)
            if entry.endswith('.egg-link')
        )
        list(itertools.starmap(self.scan_egg_link, egg_links))

    def scan_egg_link(self, path, entry):
        with open(os.path.join(path, entry)) as raw_lines:
            # filter non-empty lines
            lines = list(filter(None, map(str.strip, raw_lines)))

        if len(lines) != 2:
            # format is not recognized; punt
            return

        egg_path, setup_path = lines

        for dist in find_distributions(os.path.join(path, egg_path)):
            dist.location = os.path.join(path, *lines)
            dist.precedence = SOURCE_DIST
            self.add(dist)

    def _scan(self, link):
        # Process a URL to see if it's for a package page
        NO_MATCH_SENTINEL = None, None
        if not link.startswith(self.index_url):
            return NO_MATCH_SENTINEL

        parts = list(map(
            urllib.parse.unquote, link[len(self.index_url):].split('/')
        ))
        if len(parts) != 2 or '#' in parts[1]:
            return NO_MATCH_SENTINEL

        # it's a package page, sanitize and index it
        pkg = safe_name(parts[0])
        ver = safe_version(parts[1])
        self.package_pages.setdefault(pkg.lower(), {})[link] = True
        return to_filename(pkg), to_filename(ver)

    def process_index(self, url, page):
        """Process the contents of a PyPI page"""

        # process an index page into the package-page index
        for match in HREF.finditer(page):
            try:
                self._scan(urllib.parse.urljoin(url, htmldecode(match.group(1))))
            except ValueError:
                pass

        pkg, ver = self._scan(url)  # ensure this page is in the page index
        if not pkg:
            return ""  # no sense double-scanning non-package pages

        # process individual package page
        for new_url in find_external_links(url, page):
            # Process the found URL
            base, frag = egg_info_for_url(new_url)
            if base.endswith('.py') and not frag:
                if ver:
                    new_url += '#egg=%s-%s' % (pkg, ver)
                else:
                    self.need_version_info(url)
            self.scan_url(new_url)

        return PYPI_MD5.sub(
            lambda m: '<a href="%s#md5=%s">%s</a>' % m.group(1, 3, 2), page
        )

    def need_version_info(self, url):
        self.scan_all(
            "Page at %s links to .py file(s) without version info; an index "
            "scan is required.", url
        )

    def scan_all(self, msg=None, *args):
        if self.index_url not in self.fetched_urls:
            if msg:
                self.warn(msg, *args)
            self.info(
                "Scanning index of all packages (this may take a while)"
            )
        self.scan_url(self.index_url)

    def find_packages(self, requirement):
        self.scan_url(self.index_url + requirement.unsafe_name + '/')

        if not self.package_pages.get(requirement.key):
            # Fall back to safe version of the name
            self.scan_url(self.index_url + requirement.project_name + '/')

        if not self.package_pages.get(requirement.key):
            # We couldn't find the target package, so search the index page too
            self.not_found_in_index(requirement)

        for url in list(self.package_pages.get(requirement.key, ())):
            # scan each page that might be related to the desired package
            self.scan_url(url)

    def obtain(self, requirement, installer=None):
        self.prescan()
        self.find_packages(requirement)
        for dist in self[requirement.key]:
            if dist in requirement:
                return dist
            self.debug("%s does not match %s", requirement, dist)
        return super(PackageIndex, self).obtain(requirement, installer)

    def check_hash(self, checker, filename, tfp):
        """
        checker is a ContentChecker
        """
        checker.report(
            self.debug,
            "Validating %%s checksum for %s" % filename)
        if not checker.is_valid():
            tfp.close()
            os.unlink(filename)
            raise DistutilsError(
                "%s validation failed for %s; "
                "possible download problem?"
                % (checker.hash.name, os.path.basename(filename))
            )

    def add_find_links(self, urls):
        """Add `urls` to the list that will be prescanned for searches"""
        for url in urls:
            if (
                self.to_scan is None  # if we have already "gone online"
                or not URL_SCHEME(url)  # or it's a local file/directory
                or url.startswith('file:')
                or list(distros_for_url(url))  # or a direct package link
            ):
                # then go ahead and process it now
                self.scan_url(url)
            else:
                # otherwise, defer retrieval till later
                self.to_scan.append(url)

    def prescan(self):
        """Scan urls scheduled for prescanning (e.g. --find-links)"""
        if self.to_scan:
            list(map(self.scan_url, self.to_scan))
        self.to_scan = None  # from now on, go ahead and process immediately

    def not_found_in_index(self, requirement):
        if self[requirement.key]:  # we've seen at least one distro
            meth, msg = self.info, "Couldn't retrieve index page for %r"
        else:  # no distros seen for this name, might be misspelled
            meth, msg = (
                self.warn,
                "Couldn't find index page for %r (maybe misspelled?)")
        meth(msg, requirement.unsafe_name)
        self.scan_all()

    def download(self, spec, tmpdir):
        """Locate and/or download `spec` to `tmpdir`, returning a local path

        `spec` may be a ``Requirement`` object, or a string containing a URL,
        an existing local filename, or a project/version requirement spec
        (i.e. the string form of a ``Requirement`` object).  If it is the URL
        of a .py file with an unambiguous ``#egg=name-version`` tag (i.e., one
        that escapes ``-`` as ``_`` throughout), a trivial ``setup.py`` is
        automatically created alongside the downloaded file.

        If `spec` is a ``Requirement`` object or a string containing a
        project/version requirement spec, this method returns the location of
        a matching distribution (possibly after downloading it to `tmpdir`).
        If `spec` is a locally existing file or directory name, it is simply
        returned unchanged.  If `spec` is a URL, it is downloaded to a subpath
        of `tmpdir`, and the local filename is returned.  Various errors may be
        raised if a problem occurs during downloading.
        """
        if not isinstance(spec, Requirement):
            scheme = URL_SCHEME(spec)
            if scheme:
                # It's a url, download it to tmpdir
                found = self._download_url(scheme.group(1), spec, tmpdir)
                base, fragment = egg_info_for_url(spec)
                if base.endswith('.py'):
                    found = self.gen_setup(found, fragment, tmpdir)
                return found
            elif os.path.exists(spec):
                # Existing file or directory, just return it
                return spec
            else:
                spec = parse_requirement_arg(spec)
        return getattr(self.fetch_distribution(spec, tmpdir), 'location', None)

    def fetch_distribution(  # noqa: C901  # is too complex (14)  # FIXME
            self, requirement, tmpdir, force_scan=False, source=False,
            develop_ok=False, local_index=None):
        """Obtain a distribution suitable for fulfilling `requirement`

        `requirement` must be a ``pkg_resources.Requirement`` instance.
        If necessary, or if the `force_scan` flag is set, the requirement is
        searched for in the (online) package index as well as the locally
        installed packages.  If a distribution matching `requirement` is found,
        the returned distribution's ``location`` is the value you would have
        gotten from calling the ``download()`` method with the matching
        distribution's URL or filename.  If no matching distribution is found,
        ``None`` is returned.

        If the `source` flag is set, only source distributions and source
        checkout links will be considered.  Unless the `develop_ok` flag is
        set, development and system eggs (i.e., those using the ``.egg-info``
        format) will be ignored.
        """
        # process a Requirement
        self.info("Searching for %s", requirement)
        skipped = {}
        dist = None

        def find(req, env=None):
            if env is None:
                env = self
            # Find a matching distribution; may be called more than once

            for dist in env[req.key]:

                if dist.precedence == DEVELOP_DIST and not develop_ok:
                    if dist not in skipped:
                        self.warn(
                            "Skipping development or system egg: %s", dist,
                        )
                        skipped[dist] = 1
                    continue

                test = (
                    dist in req
                    and (dist.precedence <= SOURCE_DIST or not source)
                )
                if test:
                    loc = self.download(dist.location, tmpdir)
                    dist.download_location = loc
                    if os.path.exists(dist.download_location):
                        return dist

        if force_scan:
            self.prescan()
            self.find_packages(requirement)
            dist = find(requirement)

        if not dist and local_index is not None:
            dist = find(requirement, local_index)

        if dist is None:
            if self.to_scan is not None:
                self.prescan()
            dist = find(requirement)

        if dist is None and not force_scan:
            self.find_packages(requirement)
            dist = find(requirement)

        if dist is None:
            self.warn(
                "No local packages or working download links found for %s%s",
                (source and "a source distribution of " or ""),
                requirement,
            )
        else:
            self.info("Best match: %s", dist)
            return dist.clone(location=dist.download_location)

    def fetch(self, requirement, tmpdir, force_scan=False, source=False):
        """Obtain a file suitable for fulfilling `requirement`

        DEPRECATED; use the ``fetch_distribution()`` method now instead.  For
        backward compatibility, this routine is identical but returns the
        ``location`` of the downloaded distribution instead of a distribution
        object.
        """
        dist = self.fetch_distribution(requirement, tmpdir, force_scan, source)
        if dist is not None:
            return dist.location
        return None

    def gen_setup(self, filename, fragment, tmpdir):
        match = EGG_FRAGMENT.match(fragment)
        dists = match and [
            d for d in
            interpret_distro_name(filename, match.group(1), None) if d.version
        ] or []

        if len(dists) == 1:  # unambiguous ``#egg`` fragment
            basename = os.path.basename(filename)

            # Make sure the file has been downloaded to the temp dir.
            if os.path.dirname(filename) != tmpdir:
                dst = os.path.join(tmpdir, basename)
                from setuptools.command.easy_install import samefile
                if not samefile(filename, dst):
                    shutil.copy2(filename, dst)
                    filename = dst

            with open(os.path.join(tmpdir, 'setup.py'), 'w') as file:
                file.write(
                    "from setuptools import setup\n"
                    "setup(name=%r, version=%r, py_modules=[%r])\n"
                    % (
                        dists[0].project_name, dists[0].version,
                        os.path.splitext(basename)[0]
                    )
                )
            return filename

        elif match:
            raise DistutilsError(
                "Can't unambiguously interpret project/version identifier %r; "
                "any dashes in the name or version should be escaped using "
                "underscores. %r" % (fragment, dists)
            )
        else:
            raise DistutilsError(
                "Can't process plain .py files without an '#egg=name-version'"
                " suffix to enable automatic setup script generation."
            )

    dl_blocksize = 8192

    def _download_to(self, url, filename):
        self.info("Downloading %s", url)
        # Download the file
        fp = None
        try:
            checker = HashChecker.from_url(url)
            fp = self.open_url(url)
            if isinstance(fp, urllib.error.HTTPError):
                raise DistutilsError(
                    "Can't download %s: %s %s" % (url, fp.code, fp.msg)
                )
            headers = fp.info()
            blocknum = 0
            bs = self.dl_blocksize
            size = -1
            if "content-length" in headers:
                # Some servers return multiple Content-Length headers :(
                sizes = headers.get_all('Content-Length')
                size = max(map(int, sizes))
                self.reporthook(url, filename, blocknum, bs, size)
            with open(filename, 'wb') as tfp:
                while True:
                    block = fp.read(bs)
                    if block:
                        checker.feed(block)
                        tfp.write(block)
                        blocknum += 1
                        self.reporthook(url, filename, blocknum, bs, size)
                    else:
                        break
                self.check_hash(checker, filename, tfp)
            return headers
        finally:
            if fp:
                fp.close()

    def reporthook(self, url, filename, blocknum, blksize, size):
        pass  # no-op

    # FIXME:
    def open_url(self, url, warning=None):  # noqa: C901  # is too complex (12)
        if url.startswith('file:'):
            return local_open(url)
        try:
            return open_with_auth(url, self.opener)
        except (ValueError, http.client.InvalidURL) as v:
            msg = ' '.join([str(arg) for arg in v.args])
            if warning:
                self.warn(warning, msg)
            else:
                raise DistutilsError('%s %s' % (url, msg)) from v
        except urllib.error.HTTPError as v:
            return v
        except urllib.error.URLError as v:
            if warning:
                self.warn(warning, v.reason)
            else:
                raise DistutilsError("Download error for %s: %s"
                                     % (url, v.reason)) from v
        except http.client.BadStatusLine as v:
            if warning:
                self.warn(warning, v.line)
            else:
                raise DistutilsError(
                    '%s returned a bad status line. The server might be '
                    'down, %s' %
                    (url, v.line)
                ) from v
        except (http.client.HTTPException, socket.error) as v:
            if warning:
                self.warn(warning, v)
            else:
                raise DistutilsError("Download error for %s: %s"
                                     % (url, v)) from v

    def _download_url(self, scheme, url, tmpdir):
        # Determine download filename
        #
        name, fragment = egg_info_for_url(url)
        if name:
            while '..' in name:
                name = name.replace('..', '.').replace('\\', '_')
        else:
            name = "__downloaded__"  # default if URL has no path contents

        if name.endswith('.egg.zip'):
            name = name[:-4]  # strip the extra .zip before download

        filename = os.path.join(tmpdir, name)

        # Download the file
        #
        if scheme == 'svn' or scheme.startswith('svn+'):
            return self._download_svn(url, filename)
        elif scheme == 'git' or scheme.startswith('git+'):
            return self._download_git(url, filename)
        elif scheme.startswith('hg+'):
            return self._download_hg(url, filename)
        elif scheme == 'file':
            return urllib.request.url2pathname(urllib.parse.urlparse(url)[2])
        else:
            self.url_ok(url, True)  # raises error if not allowed
            return self._attempt_download(url, filename)

    def scan_url(self, url):
        self.process_url(url, True)

    def _attempt_download(self, url, filename):
        headers = self._download_to(url, filename)
        if 'html' in headers.get('content-type', '').lower():
            return self._download_html(url, headers, filename)
        else:
            return filename

    def _download_html(self, url, headers, filename):
        file = open(filename)
        for line in file:
            if line.strip():
                # Check for a subversion index page
                if re.search(r'<title>([^- ]+ - )?Revision \d+:', line):
                    # it's a subversion index page:
                    file.close()
                    os.unlink(filename)
                    return self._download_svn(url, filename)
                break  # not an index page
        file.close()
        os.unlink(filename)
        raise DistutilsError("Unexpected HTML page found at " + url)

    def _download_svn(self, url, filename):
        warnings.warn("SVN download support is deprecated", UserWarning)
        url = url.split('#', 1)[0]  # remove any fragment for svn's sake
        creds = ''
        if url.lower().startswith('svn:') and '@' in url:
            scheme, netloc, path, p, q, f = urllib.parse.urlparse(url)
            if not netloc and path.startswith('//') and '/' in path[2:]:
                netloc, path = path[2:].split('/', 1)
                auth, host = _splituser(netloc)
                if auth:
                    if ':' in auth:
                        user, pw = auth.split(':', 1)
                        creds = " --username=%s --password=%s" % (user, pw)
                    else:
                        creds = " --username=" + auth
                    netloc = host
                    parts = scheme, netloc, url, p, q, f
                    url = urllib.parse.urlunparse(parts)
        self.info("Doing subversion checkout from %s to %s", url, filename)
        os.system("svn checkout%s -q %s %s" % (creds, url, filename))
        return filename

    @staticmethod
    def _vcs_split_rev_from_url(url, pop_prefix=False):
        scheme, netloc, path, query, frag = urllib.parse.urlsplit(url)

        scheme = scheme.split('+', 1)[-1]

        # Some fragment identification fails
        path = path.split('#', 1)[0]

        rev = None
        if '@' in path:
            path, rev = path.rsplit('@', 1)

        # Also, discard fragment
        url = urllib.parse.urlunsplit((scheme, netloc, path, query, ''))

        return url, rev

    def _download_git(self, url, filename):
        filename = filename.split('#', 1)[0]
        url, rev = self._vcs_split_rev_from_url(url, pop_prefix=True)

        self.info("Doing git clone from %s to %s", url, filename)
        os.system("git clone --quiet %s %s" % (url, filename))

        if rev is not None:
            self.info("Checking out %s", rev)
            os.system("git -C %s checkout --quiet %s" % (
                filename,
                rev,
            ))

        return filename

    def _download_hg(self, url, filename):
        filename = filename.split('#', 1)[0]
        url, rev = self._vcs_split_rev_from_url(url, pop_prefix=True)

        self.info("Doing hg clone from %s to %s", url, filename)
        os.system("hg clone --quiet %s %s" % (url, filename))

        if rev is not None:
            self.info("Updating to %s", rev)
            os.system("hg --cwd %s up -C -r %s -q" % (
                filename,
                rev,
            ))

        return filename

    def debug(self, msg, *args):
        log.debug(msg, *args)

    def info(self, msg, *args):
        log.info(msg, *args)

    def warn(self, msg, *args):
        log.warn(msg, *args)


# This pattern matches a character entity reference (a decimal numeric
# references, a hexadecimal numeric reference, or a named reference).
entity_sub = re.compile(r'&(#(\d+|x[\da-fA-F]+)|[\w.:-]+);?').sub


def decode_entity(match):
    what = match.group(0)
    return html.unescape(what)


def htmldecode(text):
    """
    Decode HTML entities in the given text.

    >>> htmldecode(
    ...     'https://../package_name-0.1.2.tar.gz'
    ...     '?tokena=A&amp;tokenb=B">package_name-0.1.2.tar.gz')
    'https://../package_name-0.1.2.tar.gz?tokena=A&tokenb=B">package_name-0.1.2.tar.gz'
    """
    return entity_sub(decode_entity, text)


def socket_timeout(timeout=15):
    def _socket_timeout(func):
        def _socket_timeout(*args, **kwargs):
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            try:
                return func(*args, **kwargs)
            finally:
                socket.setdefaulttimeout(old_timeout)

        return _socket_timeout

    return _socket_timeout


def _encode_auth(auth):
    """
    Encode auth from a URL suitable for an HTTP header.
    >>> str(_encode_auth('username%3Apassword'))
    'dXNlcm5hbWU6cGFzc3dvcmQ='

    Long auth strings should not cause a newline to be inserted.
    >>> long_auth = 'username:' + 'password'*10
    >>> chr(10) in str(_encode_auth(long_auth))
    False
    """
    auth_s = urllib.parse.unquote(auth)
    # convert to bytes
    auth_bytes = auth_s.encode()
    encoded_bytes = base64.b64encode(auth_bytes)
    # convert back to a string
    encoded = encoded_bytes.decode()
    # strip the trailing carriage return
    return encoded.replace('\n', '')


class Credential:
    """
    A username/password pair. Use like a namedtuple.
    """

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __iter__(self):
        yield self.username
        yield self.password

    def __str__(self):
        return '%(username)s:%(password)s' % vars(self)


class PyPIConfig(configparser.RawConfigParser):
    def __init__(self):
        """
        Load from ~/.pypirc
        """
        defaults = dict.fromkeys(['username', 'password', 'repository'], '')
        configparser.RawConfigParser.__init__(self, defaults)

        rc = os.path.join(os.path.expanduser('~'), '.pypirc')
        if os.path.exists(rc):
            self.read(rc)

    @property
    def creds_by_repository(self):
        sections_with_repositories = [
            section for section in self.sections()
            if self.get(section, 'repository').strip()
        ]

        return dict(map(self._get_repo_cred, sections_with_repositories))

    def _get_repo_cred(self, section):
        repo = self.get(section, 'repository').strip()
        return repo, Credential(
            self.get(section, 'username').strip(),
            self.get(section, 'password').strip(),
        )

    def find_credential(self, url):
        """
        If the URL indicated appears to be a repository defined in this
        config, return the credential for that repository.
        """
        for repository, cred in self.creds_by_repository.items():
            if url.startswith(repository):
                return cred


def open_with_auth(url, opener=urllib.request.urlopen):
    """Open a urllib2 request, handling HTTP authentication"""

    parsed = urllib.parse.urlparse(url)
    scheme, netloc, path, params, query, frag = parsed

    # Double scheme does not raise on macOS as revealed by a
    # failing test. We would expect "nonnumeric port". Refs #20.
    if netloc.endswith(':'):
        raise http.client.InvalidURL("nonnumeric port: ''")

    if scheme in ('http', 'https'):
        auth, address = _splituser(netloc)
    else:
        auth = None

    if not auth:
        cred = PyPIConfig().find_credential(url)
        if cred:
            auth = str(cred)
            info = cred.username, url
            log.info('Authenticating as %s for %s (from .pypirc)', *info)

    if auth:
        auth = "Basic " + _encode_auth(auth)
        parts = scheme, address, path, params, query, frag
        new_url = urllib.parse.urlunparse(parts)
        request = urllib.request.Request(new_url)
        request.add_header("Authorization", auth)
    else:
        request = urllib.request.Request(url)

    request.add_header('User-Agent', user_agent)
    fp = opener(request)

    if auth:
        # Put authentication info back into request URL if same host,
        # so that links found on the page will work
        s2, h2, path2, param2, query2, frag2 = urllib.parse.urlparse(fp.url)
        if s2 == scheme and h2 == address:
            parts = s2, netloc, path2, param2, query2, frag2
            fp.url = urllib.parse.urlunparse(parts)

    return fp


# copy of urllib.parse._splituser from Python 3.8
def _splituser(host):
    """splituser('user[:passwd]@host[:port]')
    --> 'user[:passwd]', 'host[:port]'."""
    user, delim, host = host.rpartition('@')
    return (user if delim else None), host


# adding a timeout to avoid freezing package_index
open_with_auth = socket_timeout(_SOCKET_TIMEOUT)(open_with_auth)


def fix_sf_url(url):
    return url  # backward compatibility


def local_open(url):
    """Read a local path, with special support for directories"""
    scheme, server, path, param, query, frag = urllib.parse.urlparse(url)
    filename = urllib.request.url2pathname(path)
    if os.path.isfile(filename):
        return urllib.request.urlopen(url)
    elif path.endswith('/') and os.path.isdir(filename):
        files = []
        for f in os.listdir(filename):
            filepath = os.path.join(filename, f)
            if f == 'index.html':
                with open(filepath, 'r') as fp:
                    body = fp.read()
                break
            elif os.path.isdir(filepath):
                f += '/'
            files.append('<a href="{name}">{name}</a>'.format(name=f))
        else:
            tmpl = (
                "<html><head><title>{url}</title>"
                "</head><body>{files}</body></html>")
            body = tmpl.format(url=url, files='\n'.join(files))
        status, message = 200, "OK"
    else:
        status, message, body = 404, "Path not found", "Not found"

    headers = {'content-type': 'text/html'}
    body_stream = io.StringIO(body)
    return urllib.error.HTTPError(url, status, message, headers, body_stream)
"""-------------------------------------------------------------------------------------"""
"""distutils.msvc9compiler

Contains MSVCCompiler, an implementation of the abstract CCompiler class
for the Microsoft Visual Studio 2008.

The module is compatible with VS 2005 and VS 2008. You can find legacy support
for older versions of VS in distutils.msvccompiler.
"""

# Written by Perry Stoll
# hacked by Robin Becker and Thomas Heller to do a better job of
#   finding DevStudio (through the registry)
# ported to VS2005 and VS 2008 by Christian Heimes

import os
import subprocess
import sys
import re

from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
                             CompileError, LibError, LinkError
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform

import winreg

RegOpenKeyEx = winreg.OpenKeyEx
RegEnumKey = winreg.EnumKey
RegEnumValue = winreg.EnumValue
RegError = winreg.error

HKEYS = (winreg.HKEY_USERS,
         winreg.HKEY_CURRENT_USER,
         winreg.HKEY_LOCAL_MACHINE,
         winreg.HKEY_CLASSES_ROOT)

NATIVE_WIN64 = (sys.platform == 'win32' and sys.maxsize > 2**32)
if NATIVE_WIN64:
    # Visual C++ is a 32-bit application, so we need to look in
    # the corresponding registry branch, if we're running a
    # 64-bit Python on Win64
    VS_BASE = r"Software\Wow6432Node\Microsoft\VisualStudio\%0.1f"
    WINSDK_BASE = r"Software\Wow6432Node\Microsoft\Microsoft SDKs\Windows"
    NET_BASE = r"Software\Wow6432Node\Microsoft\.NETFramework"
else:
    VS_BASE = r"Software\Microsoft\VisualStudio\%0.1f"
    WINSDK_BASE = r"Software\Microsoft\Microsoft SDKs\Windows"
    NET_BASE = r"Software\Microsoft\.NETFramework"

# A map keyed by get_platform() return values to values accepted by
# 'vcvarsall.bat'.  Note a cross-compile may combine these (eg, 'x86_amd64' is
# the param to cross-compile on x86 targeting amd64.)
PLAT_TO_VCVARS = {
    'win32' : 'x86',
    'win-amd64' : 'amd64',
}

class Reg:
    """Helper class to read values from the registry
    """

    def get_value(cls, path, key):
        for base in HKEYS:
            d = cls.read_values(base, path)
            if d and key in d:
                return d[key]
        raise KeyError(key)
    get_value = classmethod(get_value)

    def read_keys(cls, base, key):
        """Return list of registry keys."""
        try:
            handle = RegOpenKeyEx(base, key)
        except RegError:
            return None
        L = []
        i = 0
        while True:
            try:
                k = RegEnumKey(handle, i)
            except RegError:
                break
            L.append(k)
            i += 1
        return L
    read_keys = classmethod(read_keys)

    def read_values(cls, base, key):
        """Return dict of registry keys and values.

        All names are converted to lowercase.
        """
        try:
            handle = RegOpenKeyEx(base, key)
        except RegError:
            return None
        d = {}
        i = 0
        while True:
            try:
                name, value, type = RegEnumValue(handle, i)
            except RegError:
                break
            name = name.lower()
            d[cls.convert_mbcs(name)] = cls.convert_mbcs(value)
            i += 1
        return d
    read_values = classmethod(read_values)

    def convert_mbcs(s):
        dec = getattr(s, "decode", None)
        if dec is not None:
            try:
                s = dec("mbcs")
            except UnicodeError:
                pass
        return s
    convert_mbcs = staticmethod(convert_mbcs)

class MacroExpander:

    def __init__(self, version):
        self.macros = {}
        self.vsbase = VS_BASE % version
        self.load_macros(version)

    def set_macro(self, macro, path, key):
        self.macros["$(%s)" % macro] = Reg.get_value(path, key)

    def load_macros(self, version):
        self.set_macro("VCInstallDir", self.vsbase + r"\Setup\VC", "productdir")
        self.set_macro("VSInstallDir", self.vsbase + r"\Setup\VS", "productdir")
        self.set_macro("FrameworkDir", NET_BASE, "installroot")
        try:
            if version >= 8.0:
                self.set_macro("FrameworkSDKDir", NET_BASE,
                               "sdkinstallrootv2.0")
            else:
                raise KeyError("sdkinstallrootv2.0")
        except KeyError:
            raise DistutilsPlatformError(
            """Python was built with Visual Studio 2008;
extensions must be built with a compiler than can generate compatible binaries.
Visual Studio 2008 was not found on this system. If you have Cygwin installed,
you can try compiling with MingW32, by passing "-c mingw32" to setup.py.""")

        if version >= 9.0:
            self.set_macro("FrameworkVersion", self.vsbase, "clr version")
            self.set_macro("WindowsSdkDir", WINSDK_BASE, "currentinstallfolder")
        else:
            p = r"Software\Microsoft\NET Framework Setup\Product"
            for base in HKEYS:
                try:
                    h = RegOpenKeyEx(base, p)
                except RegError:
                    continue
                key = RegEnumKey(h, 0)
                d = Reg.get_value(base, r"%s\%s" % (p, key))
                self.macros["$(FrameworkVersion)"] = d["version"]

    def sub(self, s):
        for k, v in self.macros.items():
            s = s.replace(k, v)
        return s

def get_build_version():
    """Return the version of MSVC that was used to build Python.

    For Python 2.3 and up, the version number is included in
    sys.version.  For earlier versions, assume the compiler is MSVC 6.
    """
    prefix = "MSC v."
    i = sys.version.find(prefix)
    if i == -1:
        return 6
    i = i + len(prefix)
    s, rest = sys.version[i:].split(" ", 1)
    majorVersion = int(s[:-2]) - 6
    if majorVersion >= 13:
        # v13 was skipped and should be v14
        majorVersion += 1
    minorVersion = int(s[2:3]) / 10.0
    # I don't think paths are affected by minor version in version 6
    if majorVersion == 6:
        minorVersion = 0
    if majorVersion >= 6:
        return majorVersion + minorVersion
    # else we don't know what version of the compiler this is
    return None

def normalize_and_reduce_paths(paths):
    """Return a list of normalized paths with duplicates removed.

    The current order of paths is maintained.
    """
    # Paths are normalized so things like:  /a and /a/ aren't both preserved.
    reduced_paths = []
    for p in paths:
        np = os.path.normpath(p)
        # XXX(nnorwitz): O(n**2), if reduced_paths gets long perhaps use a set.
        if np not in reduced_paths:
            reduced_paths.append(np)
    return reduced_paths

def removeDuplicates(variable):
    """Remove duplicate values of an environment variable.
    """
    oldList = variable.split(os.pathsep)
    newList = []
    for i in oldList:
        if i not in newList:
            newList.append(i)
    newVariable = os.pathsep.join(newList)
    return newVariable

def find_vcvarsall(version):
    """Find the vcvarsall.bat file

    At first it tries to find the productdir of VS 2008 in the registry. If
    that fails it falls back to the VS90COMNTOOLS env var.
    """
    vsbase = VS_BASE % version
    try:
        productdir = Reg.get_value(r"%s\Setup\VC" % vsbase,
                                   "productdir")
    except KeyError:
        log.debug("Unable to find productdir in registry")
        productdir = None

    if not productdir or not os.path.isdir(productdir):
        toolskey = "VS%0.f0COMNTOOLS" % version
        toolsdir = os.environ.get(toolskey, None)

        if toolsdir and os.path.isdir(toolsdir):
            productdir = os.path.join(toolsdir, os.pardir, os.pardir, "VC")
            productdir = os.path.abspath(productdir)
            if not os.path.isdir(productdir):
                log.debug("%s is not a valid directory" % productdir)
                return None
        else:
            log.debug("Env var %s is not set or invalid" % toolskey)
    if not productdir:
        log.debug("No productdir found")
        return None
    vcvarsall = os.path.join(productdir, "vcvarsall.bat")
    if os.path.isfile(vcvarsall):
        return vcvarsall
    log.debug("Unable to find vcvarsall.bat")
    return None

def query_vcvarsall(version, arch="x86"):
    """Launch vcvarsall.bat and read the settings from its environment
    """
    vcvarsall = find_vcvarsall(version)
    interesting = {"include", "lib", "libpath", "path"}
    result = {}

    if vcvarsall is None:
        raise DistutilsPlatformError("Unable to find vcvarsall.bat")
    log.debug("Calling 'vcvarsall.bat %s' (version=%s)", arch, version)
    popen = subprocess.Popen('"%s" %s & set' % (vcvarsall, arch),
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    try:
        stdout, stderr = popen.communicate()
        if popen.wait() != 0:
            raise DistutilsPlatformError(stderr.decode("mbcs"))

        stdout = stdout.decode("mbcs")
        for line in stdout.split("\n"):
            line = Reg.convert_mbcs(line)
            if '=' not in line:
                continue
            line = line.strip()
            key, value = line.split('=', 1)
            key = key.lower()
            if key in interesting:
                if value.endswith(os.pathsep):
                    value = value[:-1]
                result[key] = removeDuplicates(value)

    finally:
        popen.stdout.close()
        popen.stderr.close()

    if len(result) != len(interesting):
        raise ValueError(str(list(result.keys())))

    return result

# More globals
VERSION = get_build_version()
if VERSION < 8.0:
    raise DistutilsPlatformError("VC %0.1f is not supported by this module" % VERSION)
# MACROS = MacroExpander(VERSION)

class MSVCCompiler(CCompiler) :
    """Concrete class that implements an interface to Microsoft Visual C++,
       as defined by the CCompiler abstract class."""

    compiler_type = 'msvc'

    # Just set this so CCompiler's constructor doesn't barf.  We currently
    # don't use the 'set_executables()' bureaucracy provided by CCompiler,
    # as it really isn't necessary for this sort of single-compiler class.
    # Would be nice to have a consistent interface with UnixCCompiler,
    # though, so it's worth thinking about.
    executables = {}

    # Private class data (need to distinguish C from C++ source for compiler)
    _c_extensions = ['.c']
    _cpp_extensions = ['.cc', '.cpp', '.cxx']
    _rc_extensions = ['.rc']
    _mc_extensions = ['.mc']

    # Needed for the filename generation methods provided by the
    # base class, CCompiler.
    src_extensions = (_c_extensions + _cpp_extensions +
                      _rc_extensions + _mc_extensions)
    res_extension = '.res'
    obj_extension = '.obj'
    static_lib_extension = '.lib'
    shared_lib_extension = '.dll'
    static_lib_format = shared_lib_format = '%s%s'
    exe_extension = '.exe'

    def __init__(self, verbose=0, dry_run=0, force=0):
        CCompiler.__init__ (self, verbose, dry_run, force)
        self.__version = VERSION
        self.__root = r"Software\Microsoft\VisualStudio"
        # self.__macros = MACROS
        self.__paths = []
        # target platform (.plat_name is consistent with 'bdist')
        self.plat_name = None
        self.__arch = None # deprecated name
        self.initialized = False

    def initialize(self, plat_name=None):
        # multi-init means we would need to check platform same each time...
        assert not self.initialized, "don't init multiple times"
        if plat_name is None:
            plat_name = get_platform()
        # sanity check for platforms to prevent obscure errors later.
        ok_plats = 'win32', 'win-amd64'
        if plat_name not in ok_plats:
            raise DistutilsPlatformError("--plat-name must be one of %s" %
                                         (ok_plats,))

        if "DISTUTILS_USE_SDK" in os.environ and "MSSdk" in os.environ and self.find_exe("cl.exe"):
            # Assume that the SDK set up everything alright; don't try to be
            # smarter
            self.cc = "cl.exe"
            self.linker = "link.exe"
            self.lib = "lib.exe"
            self.rc = "rc.exe"
            self.mc = "mc.exe"
        else:
            # On x86, 'vcvars32.bat amd64' creates an env that doesn't work;
            # to cross compile, you use 'x86_amd64'.
            # On AMD64, 'vcvars32.bat amd64' is a native build env; to cross
            # compile use 'x86' (ie, it runs the x86 compiler directly)
            if plat_name == get_platform() or plat_name == 'win32':
                # native build or cross-compile to win32
                plat_spec = PLAT_TO_VCVARS[plat_name]
            else:
                # cross compile from win32 -> some 64bit
                plat_spec = PLAT_TO_VCVARS[get_platform()] + '_' + \
                            PLAT_TO_VCVARS[plat_name]

            vc_env = query_vcvarsall(VERSION, plat_spec)

            self.__paths = vc_env['path'].split(os.pathsep)
            os.environ['lib'] = vc_env['lib']
            os.environ['include'] = vc_env['include']

            if len(self.__paths) == 0:
                raise DistutilsPlatformError("Python was built with %s, "
                       "and extensions need to be built with the same "
                       "version of the compiler, but it isn't installed."
                       % self.__product)

            self.cc = self.find_exe("cl.exe")
            self.linker = self.find_exe("link.exe")
            self.lib = self.find_exe("lib.exe")
            self.rc = self.find_exe("rc.exe")   # resource compiler
            self.mc = self.find_exe("mc.exe")   # message compiler
            #self.set_path_env_var('lib')
            #self.set_path_env_var('include')

        # extend the MSVC path with the current path
        try:
            for p in os.environ['path'].split(';'):
                self.__paths.append(p)
        except KeyError:
            pass
        self.__paths = normalize_and_reduce_paths(self.__paths)
        os.environ['path'] = ";".join(self.__paths)

        self.preprocess_options = None
        if self.__arch == "x86":
            self.compile_options = [ '/nologo', '/O2', '/MD', '/W3',
                                     '/DNDEBUG']
            self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3',
                                          '/Z7', '/D_DEBUG']
        else:
            # Win64
            self.compile_options = [ '/nologo', '/O2', '/MD', '/W3', '/GS-' ,
                                     '/DNDEBUG']
            self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3', '/GS-',
                                          '/Z7', '/D_DEBUG']

        self.ldflags_shared = ['/DLL', '/nologo', '/INCREMENTAL:NO']
        if self.__version >= 7:
            self.ldflags_shared_debug = [
                '/DLL', '/nologo', '/INCREMENTAL:no', '/DEBUG'
                ]
        self.ldflags_static = [ '/nologo']

        self.initialized = True

    # -- Worker methods ------------------------------------------------

    def object_filenames(self,
                         source_filenames,
                         strip_dir=0,
                         output_dir=''):
        # Copied from ccompiler.py, extended to return .res as 'object'-file
        # for .rc input file
        if output_dir is None: output_dir = ''
        obj_names = []
        for src_name in source_filenames:
            (base, ext) = os.path.splitext (src_name)
            base = os.path.splitdrive(base)[1] # Chop off the drive
            base = base[os.path.isabs(base):]  # If abs, chop off leading /
            if ext not in self.src_extensions:
                # Better to raise an exception instead of silently continuing
                # and later complain about sources and targets having
                # different lengths
                raise CompileError ("Don't know how to compile %s" % src_name)
            if strip_dir:
                base = os.path.basename (base)
            if ext in self._rc_extensions:
                obj_names.append (os.path.join (output_dir,
                                                base + self.res_extension))
            elif ext in self._mc_extensions:
                obj_names.append (os.path.join (output_dir,
                                                base + self.res_extension))
            else:
                obj_names.append (os.path.join (output_dir,
                                                base + self.obj_extension))
        return obj_names


    def compile(self, sources,
                output_dir=None, macros=None, include_dirs=None, debug=0,
                extra_preargs=None, extra_postargs=None, depends=None):

        if not self.initialized:
            self.initialize()
        compile_info = self._setup_compile(output_dir, macros, include_dirs,
                                           sources, depends, extra_postargs)
        macros, objects, extra_postargs, pp_opts, build = compile_info

        compile_opts = extra_preargs or []
        compile_opts.append ('/c')
        if debug:
            compile_opts.extend(self.compile_options_debug)
        else:
            compile_opts.extend(self.compile_options)

        for obj in objects:
            try:
                src, ext = build[obj]
            except KeyError:
                continue
            if debug:
                # pass the full pathname to MSVC in debug mode,
                # this allows the debugger to find the source file
                # without asking the user to browse for it
                src = os.path.abspath(src)

            if ext in self._c_extensions:
                input_opt = "/Tc" + src
            elif ext in self._cpp_extensions:
                input_opt = "/Tp" + src
            elif ext in self._rc_extensions:
                # compile .RC to .RES file
                input_opt = src
                output_opt = "/fo" + obj
                try:
                    self.spawn([self.rc] + pp_opts +
                               [output_opt] + [input_opt])
                except DistutilsExecError as msg:
                    raise CompileError(msg)
                continue
            elif ext in self._mc_extensions:
                # Compile .MC to .RC file to .RES file.
                #   * '-h dir' specifies the directory for the
                #     generated include file
                #   * '-r dir' specifies the target directory of the
                #     generated RC file and the binary message resource
                #     it includes
                #
                # For now (since there are no options to change this),
                # we use the source-directory for the include file and
                # the build directory for the RC file and message
                # resources. This works at least for win32all.
                h_dir = os.path.dirname(src)
                rc_dir = os.path.dirname(obj)
                try:
                    # first compile .MC to .RC and .H file
                    self.spawn([self.mc] +
                               ['-h', h_dir, '-r', rc_dir] + [src])
                    base, _ = os.path.splitext (os.path.basename (src))
                    rc_file = os.path.join (rc_dir, base + '.rc')
                    # then compile .RC to .RES file
                    self.spawn([self.rc] +
                               ["/fo" + obj] + [rc_file])

                except DistutilsExecError as msg:
                    raise CompileError(msg)
                continue
            else:
                # how to handle this file?
                raise CompileError("Don't know how to compile %s to %s"
                                   % (src, obj))

            output_opt = "/Fo" + obj
            try:
                self.spawn([self.cc] + compile_opts + pp_opts +
                           [input_opt, output_opt] +
                           extra_postargs)
            except DistutilsExecError as msg:
                raise CompileError(msg)

        return objects


    def create_static_lib(self,
                          objects,
                          output_libname,
                          output_dir=None,
                          debug=0,
                          target_lang=None):

        if not self.initialized:
            self.initialize()
        (objects, output_dir) = self._fix_object_args(objects, output_dir)
        output_filename = self.library_filename(output_libname,
                                                output_dir=output_dir)

        if self._need_link(objects, output_filename):
            lib_args = objects + ['/OUT:' + output_filename]
            if debug:
                pass # XXX what goes here?
            try:
                self.spawn([self.lib] + lib_args)
            except DistutilsExecError as msg:
                raise LibError(msg)
        else:
            log.debug("skipping %s (up-to-date)", output_filename)


    def link(self,
             target_desc,
             objects,
             output_filename,
             output_dir=None,
             libraries=None,
             library_dirs=None,
             runtime_library_dirs=None,
             export_symbols=None,
             debug=0,
             extra_preargs=None,
             extra_postargs=None,
             build_temp=None,
             target_lang=None):

        if not self.initialized:
            self.initialize()
        (objects, output_dir) = self._fix_object_args(objects, output_dir)
        fixed_args = self._fix_lib_args(libraries, library_dirs,
                                        runtime_library_dirs)
        (libraries, library_dirs, runtime_library_dirs) = fixed_args

        if runtime_library_dirs:
            self.warn ("I don't know what to do with 'runtime_library_dirs': "
                       + str (runtime_library_dirs))

        lib_opts = gen_lib_options(self,
                                   library_dirs, runtime_library_dirs,
                                   libraries)
        if output_dir is not None:
            output_filename = os.path.join(output_dir, output_filename)

        if self._need_link(objects, output_filename):
            if target_desc == CCompiler.EXECUTABLE:
                if debug:
                    ldflags = self.ldflags_shared_debug[1:]
                else:
                    ldflags = self.ldflags_shared[1:]
            else:
                if debug:
                    ldflags = self.ldflags_shared_debug
                else:
                    ldflags = self.ldflags_shared

            export_opts = []
            for sym in (export_symbols or []):
                export_opts.append("/EXPORT:" + sym)

            ld_args = (ldflags + lib_opts + export_opts +
                       objects + ['/OUT:' + output_filename])

            # The MSVC linker generates .lib and .exp files, which cannot be
            # suppressed by any linker switches. The .lib files may even be
            # needed! Make sure they are generated in the temporary build
            # directory. Since they have different names for debug and release
            # builds, they can go into the same directory.
            build_temp = os.path.dirname(objects[0])
            if export_symbols is not None:
                (dll_name, dll_ext) = os.path.splitext(
                    os.path.basename(output_filename))
                implib_file = os.path.join(
                    build_temp,
                    self.library_filename(dll_name))
                ld_args.append ('/IMPLIB:' + implib_file)

            self.manifest_setup_ldargs(output_filename, build_temp, ld_args)

            if extra_preargs:
                ld_args[:0] = extra_preargs
            if extra_postargs:
                ld_args.extend(extra_postargs)

            self.mkpath(os.path.dirname(output_filename))
            try:
                self.spawn([self.linker] + ld_args)
            except DistutilsExecError as msg:
                raise LinkError(msg)

            # embed the manifest
            # XXX - this is somewhat fragile - if mt.exe fails, distutils
            # will still consider the DLL up-to-date, but it will not have a
            # manifest.  Maybe we should link to a temp file?  OTOH, that
            # implies a build environment error that shouldn't go undetected.
            mfinfo = self.manifest_get_embed_info(target_desc, ld_args)
            if mfinfo is not None:
                mffilename, mfid = mfinfo
                out_arg = '-outputresource:%s;%s' % (output_filename, mfid)
                try:
                    self.spawn(['mt.exe', '-nologo', '-manifest',
                                mffilename, out_arg])
                except DistutilsExecError as msg:
                    raise LinkError(msg)
        else:
            log.debug("skipping %s (up-to-date)", output_filename)

    def manifest_setup_ldargs(self, output_filename, build_temp, ld_args):
        # If we need a manifest at all, an embedded manifest is recommended.
        # See MSDN article titled
        # "How to: Embed a Manifest Inside a C/C++ Application"
        # (currently at http://msdn2.microsoft.com/en-us/library/ms235591(VS.80).aspx)
        # Ask the linker to generate the manifest in the temp dir, so
        # we can check it, and possibly embed it, later.
        temp_manifest = os.path.join(
                build_temp,
                os.path.basename(output_filename) + ".manifest")
        ld_args.append('/MANIFESTFILE:' + temp_manifest)

    def manifest_get_embed_info(self, target_desc, ld_args):
        # If a manifest should be embedded, return a tuple of
        # (manifest_filename, resource_id).  Returns None if no manifest
        # should be embedded.  See http://bugs.python.org/issue7833 for why
        # we want to avoid any manifest for extension modules if we can)
        for arg in ld_args:
            if arg.startswith("/MANIFESTFILE:"):
                temp_manifest = arg.split(":", 1)[1]
                break
        else:
            # no /MANIFESTFILE so nothing to do.
            return None
        if target_desc == CCompiler.EXECUTABLE:
            # by default, executables always get the manifest with the
            # CRT referenced.
            mfid = 1
        else:
            # Extension modules try and avoid any manifest if possible.
            mfid = 2
            temp_manifest = self._remove_visual_c_ref(temp_manifest)
        if temp_manifest is None:
            return None
        return temp_manifest, mfid

    def _remove_visual_c_ref(self, manifest_file):
        try:
            # Remove references to the Visual C runtime, so they will
            # fall through to the Visual C dependency of Python.exe.
            # This way, when installed for a restricted user (e.g.
            # runtimes are not in WinSxS folder, but in Python's own
            # folder), the runtimes do not need to be in every folder
            # with .pyd's.
            # Returns either the filename of the modified manifest or
            # None if no manifest should be embedded.
            manifest_f = open(manifest_file)
            try:
                manifest_buf = manifest_f.read()
            finally:
                manifest_f.close()
            pattern = re.compile(
                r"""<assemblyIdentity.*?name=("|')Microsoft\."""\
                r"""VC\d{2}\.CRT("|').*?(/>|</assemblyIdentity>)""",
                re.DOTALL)
            manifest_buf = re.sub(pattern, "", manifest_buf)
            pattern = r"<dependentAssembly>\s*</dependentAssembly>"
            manifest_buf = re.sub(pattern, "", manifest_buf)
            # Now see if any other assemblies are referenced - if not, we
            # don't want a manifest embedded.
            pattern = re.compile(
                r"""<assemblyIdentity.*?name=(?:"|')(.+?)(?:"|')"""
                r""".*?(?:/>|</assemblyIdentity>)""", re.DOTALL)
            if re.search(pattern, manifest_buf) is None:
                return None

            manifest_f = open(manifest_file, 'w')
            try:
                manifest_f.write(manifest_buf)
                return manifest_file
            finally:
                manifest_f.close()
        except OSError:
            pass

    # -- Miscellaneous methods -----------------------------------------
    # These are all used by the 'gen_lib_options() function, in
    # ccompiler.py.

    def library_dir_option(self, dir):
        return "/LIBPATH:" + dir

    def runtime_library_dir_option(self, dir):
        raise DistutilsPlatformError(
              "don't know how to set runtime library search path for MSVC++")

    def library_option(self, lib):
        return self.library_filename(lib)


    def find_library_file(self, dirs, lib, debug=0):
        # Prefer a debugging library if found (and requested), but deal
        # with it if we don't have one.
        if debug:
            try_names = [lib + "_d", lib]
        else:
            try_names = [lib]
        for dir in dirs:
            for name in try_names:
                libfile = os.path.join(dir, self.library_filename (name))
                if os.path.exists(libfile):
                    return libfile
        else:
            # Oops, didn't find it in *any* of 'dirs'
            return None

    # Helper methods for using the MSVC registry settings

    def find_exe(self, exe):
        """Return path to an MSVC executable program.

        Tries to find the program in several places: first, one of the
        MSVC program search paths from the registry; next, the directories
        in the PATH environment variable.  If any of those work, return an
        absolute path that is known to exist.  If none of them work, just
        return the original program name, 'exe'.
        """
        for p in self.__paths:
            fn = os.path.join(os.path.abspath(p), exe)
            if os.path.isfile(fn):
                return fn

        # didn't find it; try existing path
        for p in os.environ['Path'].split(';'):
            fn = os.path.join(os.path.abspath(p),exe)
            if os.path.isfile(fn):
                return fn

        return exe
"""---------------------------------------------------------------------------------------------"""
"""Provide access to Python's configuration information.  The specific
configuration variables available depend heavily on the platform and
configuration.  The values may be retrieved using
get_config_var(name), and the list of variables is available via
get_config_vars().keys().  Additional convenience functions are also
available.

Written by:   Fred L. Drake, Jr.
Email:        <fdrake@acm.org>
"""

import _imp
import os
import re
import sys

from .errors import DistutilsPlatformError

IS_PYPY = '__pypy__' in sys.builtin_module_names

# These are needed in a couple of spots, so just compute them once.
PREFIX = os.path.normpath(sys.prefix)
EXEC_PREFIX = os.path.normpath(sys.exec_prefix)
BASE_PREFIX = os.path.normpath(sys.base_prefix)
BASE_EXEC_PREFIX = os.path.normpath(sys.base_exec_prefix)

# Path to the base directory of the project. On Windows the binary may
# live in project/PCbuild/win32 or project/PCbuild/amd64.
# set for cross builds
if "_PYTHON_PROJECT_BASE" in os.environ:
    project_base = os.path.abspath(os.environ["_PYTHON_PROJECT_BASE"])
else:
    if sys.executable:
        project_base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # sys.executable can be empty if argv[0] has been changed and Python is
        # unable to retrieve the real program name
        project_base = os.getcwd()


# python_build: (Boolean) if true, we're either building Python or
# building an extension with an un-installed Python, so we use
# different (hard-wired) directories.
def _is_python_source_dir(d):
    for fn in ("Setup", "Setup.local"):
        if os.path.isfile(os.path.join(d, "Modules", fn)):
            return True
    return False

_sys_home = getattr(sys, '_home', None)

if os.name == 'nt':
    def _fix_pcbuild(d):
        if d and os.path.normcase(d).startswith(
                os.path.normcase(os.path.join(PREFIX, "PCbuild"))):
            return PREFIX
        return d
    project_base = _fix_pcbuild(project_base)
    _sys_home = _fix_pcbuild(_sys_home)

def _python_build():
    if _sys_home:
        return _is_python_source_dir(_sys_home)
    return _is_python_source_dir(project_base)

python_build = _python_build()


# Calculate the build qualifier flags if they are defined.  Adding the flags
# to the include and lib directories only makes sense for an installation, not
# an in-source build.
build_flags = ''
try:
    if not python_build:
        build_flags = sys.abiflags
except AttributeError:
    # It's not a configure-based build, so the sys module doesn't have
    # this attribute, which is fine.
    pass

def get_python_version():
    """Return a string containing the major and minor Python version,
    leaving off the patchlevel.  Sample return values could be '1.5'
    or '2.2'.
    """
    return '%d.%d' % sys.version_info[:2]


def get_python_inc(plat_specific=0, prefix=None):
    """Return the directory containing installed Python header files.

    If 'plat_specific' is false (the default), this is the path to the
    non-platform-specific header files, i.e. Python.h and so on;
    otherwise, this is the path to platform-specific header files
    (namely pyconfig.h).

    If 'prefix' is supplied, use it instead of sys.base_prefix or
    sys.base_exec_prefix -- i.e., ignore 'plat_specific'.
    """
    if prefix is None:
        prefix = plat_specific and BASE_EXEC_PREFIX or BASE_PREFIX
    if os.name == "posix":
        if IS_PYPY and sys.version_info < (3, 8):
            return os.path.join(prefix, 'include')
        if python_build:
            # Assume the executable is in the build directory.  The
            # pyconfig.h file should be in the same directory.  Since
            # the build directory may not be the source directory, we
            # must use "srcdir" from the makefile to find the "Include"
            # directory.
            if plat_specific:
                return _sys_home or project_base
            else:
                incdir = os.path.join(get_config_var('srcdir'), 'Include')
                return os.path.normpath(incdir)
        implementation = 'pypy' if IS_PYPY else 'python'
        python_dir = implementation + get_python_version() + build_flags
        return os.path.join(prefix, "include", python_dir)
    elif os.name == "nt":
        if python_build:
            # Include both the include and PC dir to ensure we can find
            # pyconfig.h
            return (os.path.join(prefix, "include") + os.path.pathsep +
                    os.path.join(prefix, "PC"))
        return os.path.join(prefix, "include")
    else:
        raise DistutilsPlatformError(
            "I don't know where Python installs its C header files "
            "on platform '%s'" % os.name)


def get_python_lib(plat_specific=0, standard_lib=0, prefix=None):
    """Return the directory containing the Python library (standard or
    site additions).

    If 'plat_specific' is true, return the directory containing
    platform-specific modules, i.e. any module from a non-pure-Python
    module distribution; otherwise, return the platform-shared library
    directory.  If 'standard_lib' is true, return the directory
    containing standard Python library modules; otherwise, return the
    directory for site-specific modules.

    If 'prefix' is supplied, use it instead of sys.base_prefix or
    sys.base_exec_prefix -- i.e., ignore 'plat_specific'.
    """

    if IS_PYPY and sys.version_info < (3, 8):
        # PyPy-specific schema
        if prefix is None:
            prefix = PREFIX
        if standard_lib:
            return os.path.join(prefix, "lib-python", sys.version[0])
        return os.path.join(prefix, 'site-packages')

    if prefix is None:
        if standard_lib:
            prefix = plat_specific and BASE_EXEC_PREFIX or BASE_PREFIX
        else:
            prefix = plat_specific and EXEC_PREFIX or PREFIX

    if os.name == "posix":
        if plat_specific or standard_lib:
            # Platform-specific modules (any module from a non-pure-Python
            # module distribution) or standard Python library modules.
            libdir = getattr(sys, "platlibdir", "lib")
        else:
            # Pure Python
            libdir = "lib"
        implementation = 'pypy' if IS_PYPY else 'python'
        libpython = os.path.join(prefix, libdir,
                                 implementation + get_python_version())
        if standard_lib:
            return libpython
        else:
            return os.path.join(libpython, "site-packages")
    elif os.name == "nt":
        if standard_lib:
            return os.path.join(prefix, "Lib")
        else:
            return os.path.join(prefix, "Lib", "site-packages")
    else:
        raise DistutilsPlatformError(
            "I don't know where Python installs its library "
            "on platform '%s'" % os.name)



def customize_compiler(compiler):
    """Do any platform-specific customization of a CCompiler instance.

    Mainly needed on Unix, so we can plug in the information that
    varies across Unices and is stored in Python's Makefile.
    """
    if compiler.compiler_type == "unix":
        if sys.platform == "darwin":
            # Perform first-time customization of compiler-related
            # config vars on OS X now that we know we need a compiler.
            # This is primarily to support Pythons from binary
            # installers.  The kind and paths to build tools on
            # the user system may vary significantly from the system
            # that Python itself was built on.  Also the user OS
            # version and build tools may not support the same set
            # of CPU architectures for universal builds.
            global _config_vars
            # Use get_config_var() to ensure _config_vars is initialized.
            if not get_config_var('CUSTOMIZED_OSX_COMPILER'):
                import _osx_support
                _osx_support.customize_compiler(_config_vars)
                _config_vars['CUSTOMIZED_OSX_COMPILER'] = 'True'

        (cc, cxx, cflags, ccshared, ldshared, shlib_suffix, ar, ar_flags) = \
            get_config_vars('CC', 'CXX', 'CFLAGS',
                            'CCSHARED', 'LDSHARED', 'SHLIB_SUFFIX', 'AR', 'ARFLAGS')

        if 'CC' in os.environ:
            newcc = os.environ['CC']
            if('LDSHARED' not in os.environ
                    and ldshared.startswith(cc)):
                # If CC is overridden, use that as the default
                #       command for LDSHARED as well
                ldshared = newcc + ldshared[len(cc):]
            cc = newcc
        if 'CXX' in os.environ:
            cxx = os.environ['CXX']
        if 'LDSHARED' in os.environ:
            ldshared = os.environ['LDSHARED']
        if 'CPP' in os.environ:
            cpp = os.environ['CPP']
        else:
            cpp = cc + " -E"           # not always
        if 'LDFLAGS' in os.environ:
            ldshared = ldshared + ' ' + os.environ['LDFLAGS']
        if 'CFLAGS' in os.environ:
            cflags = cflags + ' ' + os.environ['CFLAGS']
            ldshared = ldshared + ' ' + os.environ['CFLAGS']
        if 'CPPFLAGS' in os.environ:
            cpp = cpp + ' ' + os.environ['CPPFLAGS']
            cflags = cflags + ' ' + os.environ['CPPFLAGS']
            ldshared = ldshared + ' ' + os.environ['CPPFLAGS']
        if 'AR' in os.environ:
            ar = os.environ['AR']
        if 'ARFLAGS' in os.environ:
            archiver = ar + ' ' + os.environ['ARFLAGS']
        else:
            archiver = ar + ' ' + ar_flags

        cc_cmd = cc + ' ' + cflags
        compiler.set_executables(
            preprocessor=cpp,
            compiler=cc_cmd,
            compiler_so=cc_cmd + ' ' + ccshared,
            compiler_cxx=cxx,
            linker_so=ldshared,
            linker_exe=cc,
            archiver=archiver)

        if 'RANLIB' in os.environ and compiler.executables.get('ranlib', None):
            compiler.set_executables(ranlib=os.environ['RANLIB'])

        compiler.shared_lib_extension = shlib_suffix


def get_config_h_filename():
    """Return full pathname of installed pyconfig.h file."""
    if python_build:
        if os.name == "nt":
            inc_dir = os.path.join(_sys_home or project_base, "PC")
        else:
            inc_dir = _sys_home or project_base
    else:
        inc_dir = get_python_inc(plat_specific=1)

    return os.path.join(inc_dir, 'pyconfig.h')


def get_makefile_filename():
    """Return full pathname of installed Makefile from the Python build."""
    if python_build:
        return os.path.join(_sys_home or project_base, "Makefile")
    lib_dir = get_python_lib(plat_specific=0, standard_lib=1)
    config_file = 'config-{}{}'.format(get_python_version(), build_flags)
    if hasattr(sys.implementation, '_multiarch'):
        config_file += '-%s' % sys.implementation._multiarch
    return os.path.join(lib_dir, config_file, 'Makefile')


def parse_config_h(fp, g=None):
    """Parse a config.h-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    if g is None:
        g = {}
    define_rx = re.compile("#define ([A-Z][A-Za-z0-9_]+) (.*)\n")
    undef_rx = re.compile("/[*] #undef ([A-Z][A-Za-z0-9_]+) [*]/\n")
    #
    while True:
        line = fp.readline()
        if not line:
            break
        m = define_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            try: v = int(v)
            except ValueError: pass
            g[n] = v
        else:
            m = undef_rx.match(line)
            if m:
                g[m.group(1)] = 0
    return g


# Regexes needed for parsing Makefile (and similar syntaxes,
# like old-style Setup files).
_variable_rx = re.compile(r"([a-zA-Z][a-zA-Z0-9_]+)\s*=\s*(.*)")
_findvar1_rx = re.compile(r"\$\(([A-Za-z][A-Za-z0-9_]*)\)")
_findvar2_rx = re.compile(r"\${([A-Za-z][A-Za-z0-9_]*)}")

def parse_makefile(fn, g=None):
    """Parse a Makefile-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    from distutils.text_file import TextFile
    fp = TextFile(fn, strip_comments=1, skip_blanks=1, join_lines=1, errors="surrogateescape")

    if g is None:
        g = {}
    done = {}
    notdone = {}

    while True:
        line = fp.readline()
        if line is None: # eof
            break
        m = _variable_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            v = v.strip()
            # `$$' is a literal `$' in make
            tmpv = v.replace('$$', '')

            if "$" in tmpv:
                notdone[n] = v
            else:
                try:
                    v = int(v)
                except ValueError:
                    # insert literal `$'
                    done[n] = v.replace('$$', '$')
                else:
                    done[n] = v

    # Variables with a 'PY_' prefix in the makefile. These need to
    # be made available without that prefix through sysconfig.
    # Special care is needed to ensure that variable expansion works, even
    # if the expansion uses the name without a prefix.
    renamed_variables = ('CFLAGS', 'LDFLAGS', 'CPPFLAGS')

    # do variable interpolation here
    while notdone:
        for name in list(notdone):
            value = notdone[name]
            m = _findvar1_rx.search(value) or _findvar2_rx.search(value)
            if m:
                n = m.group(1)
                found = True
                if n in done:
                    item = str(done[n])
                elif n in notdone:
                    # get it on a subsequent round
                    found = False
                elif n in os.environ:
                    # do it like make: fall back to environment
                    item = os.environ[n]

                elif n in renamed_variables:
                    if name.startswith('PY_') and name[3:] in renamed_variables:
                        item = ""

                    elif 'PY_' + n in notdone:
                        found = False

                    else:
                        item = str(done['PY_' + n])
                else:
                    done[n] = item = ""
                if found:
                    after = value[m.end():]
                    value = value[:m.start()] + item + after
                    if "$" in after:
                        notdone[name] = value
                    else:
                        try: value = int(value)
                        except ValueError:
                            done[name] = value.strip()
                        else:
                            done[name] = value
                        del notdone[name]

                        if name.startswith('PY_') \
                            and name[3:] in renamed_variables:

                            name = name[3:]
                            if name not in done:
                                done[name] = value
            else:
                # bogus variable reference; just drop it since we can't deal
                del notdone[name]

    fp.close()

    # strip spurious spaces
    for k, v in done.items():
        if isinstance(v, str):
            done[k] = v.strip()

    # save the results in the global dictionary
    g.update(done)
    return g


def expand_makefile_vars(s, vars):
    """Expand Makefile-style variables -- "${foo}" or "$(foo)" -- in
    'string' according to 'vars' (a dictionary mapping variable names to
    values).  Variables not present in 'vars' are silently expanded to the
    empty string.  The variable values in 'vars' should not contain further
    variable expansions; if 'vars' is the output of 'parse_makefile()',
    you're fine.  Returns a variable-expanded version of 's'.
    """

    # This algorithm does multiple expansion, so if vars['foo'] contains
    # "${bar}", it will expand ${foo} to ${bar}, and then expand
    # ${bar}... and so forth.  This is fine as long as 'vars' comes from
    # 'parse_makefile()', which takes care of such expansions eagerly,
    # according to make's variable expansion semantics.

    while True:
        m = _findvar1_rx.search(s) or _findvar2_rx.search(s)
        if m:
            (beg, end) = m.span()
            s = s[0:beg] + vars.get(m.group(1)) + s[end:]
        else:
            break
    return s


_config_vars = None

def _init_posix():
    """Initialize the module as appropriate for POSIX systems."""
    # _sysconfigdata is generated at build time, see the sysconfig module
    name = os.environ.get('_PYTHON_SYSCONFIGDATA_NAME',
        '_sysconfigdata_{abi}_{platform}_{multiarch}'.format(
        abi=sys.abiflags,
        platform=sys.platform,
        multiarch=getattr(sys.implementation, '_multiarch', ''),
    ))
    try:
        _temp = __import__(name, globals(), locals(), ['build_time_vars'], 0)
    except ImportError:
        # Python 3.5 and pypy 7.3.1
        _temp = __import__(
            '_sysconfigdata', globals(), locals(), ['build_time_vars'], 0)
    build_time_vars = _temp.build_time_vars
    global _config_vars
    _config_vars = {}
    _config_vars.update(build_time_vars)


def _init_nt():
    """Initialize the module as appropriate for NT"""
    g = {}
    # set basic install directories
    g['LIBDEST'] = get_python_lib(plat_specific=0, standard_lib=1)
    g['BINLIBDEST'] = get_python_lib(plat_specific=1, standard_lib=1)

    # XXX hmmm.. a normal install puts include files here
    g['INCLUDEPY'] = get_python_inc(plat_specific=0)

    g['EXT_SUFFIX'] = _imp.extension_suffixes()[0]
    g['EXE'] = ".exe"
    g['VERSION'] = get_python_version().replace(".", "")
    g['BINDIR'] = os.path.dirname(os.path.abspath(sys.executable))

    global _config_vars
    _config_vars = g


def get_config_vars(*args):
    """With no arguments, return a dictionary of all configuration
    variables relevant for the current platform.  Generally this includes
    everything needed to build extensions and install both pure modules and
    extensions.  On Unix, this means every variable defined in Python's
    installed Makefile; on Windows it's a much smaller set.

    With arguments, return a list of values that result from looking up
    each argument in the configuration variable dictionary.
    """
    global _config_vars
    if _config_vars is None:
        func = globals().get("_init_" + os.name)
        if func:
            func()
        else:
            _config_vars = {}

        # Normalized versions of prefix and exec_prefix are handy to have;
        # in fact, these are the standard versions used most places in the
        # Distutils.
        _config_vars['prefix'] = PREFIX
        _config_vars['exec_prefix'] = EXEC_PREFIX

        if not IS_PYPY:
            # For backward compatibility, see issue19555
            SO = _config_vars.get('EXT_SUFFIX')
            if SO is not None:
                _config_vars['SO'] = SO

            # Always convert srcdir to an absolute path
            srcdir = _config_vars.get('srcdir', project_base)
            if os.name == 'posix':
                if python_build:
                    # If srcdir is a relative path (typically '.' or '..')
                    # then it should be interpreted relative to the directory
                    # containing Makefile.
                    base = os.path.dirname(get_makefile_filename())
                    srcdir = os.path.join(base, srcdir)
                else:
                    # srcdir is not meaningful since the installation is
                    # spread about the filesystem.  We choose the
                    # directory containing the Makefile since we know it
                    # exists.
                    srcdir = os.path.dirname(get_makefile_filename())
            _config_vars['srcdir'] = os.path.abspath(os.path.normpath(srcdir))

            # Convert srcdir into an absolute path if it appears necessary.
            # Normally it is relative to the build directory.  However, during
            # testing, for example, we might be running a non-installed python
            # from a different directory.
            if python_build and os.name == "posix":
                base = project_base
                if (not os.path.isabs(_config_vars['srcdir']) and
                    base != os.getcwd()):
                    # srcdir is relative and we are not in the same directory
                    # as the executable. Assume executable is in the build
                    # directory and make srcdir absolute.
                    srcdir = os.path.join(base, _config_vars['srcdir'])
                    _config_vars['srcdir'] = os.path.normpath(srcdir)

        # OS X platforms require special customization to handle
        # multi-architecture, multi-os-version installers
        if sys.platform == 'darwin':
            import _osx_support
            _osx_support.customize_config_vars(_config_vars)

    if args:
        vals = []
        for name in args:
            vals.append(_config_vars.get(name))
        return vals
    else:
        return _config_vars

def get_config_var(name):
    """Return the value of a single variable using the dictionary
    returned by 'get_config_vars()'.  Equivalent to
    get_config_vars().get(name)
    """
    if name == 'SO':
        import warnings
        warnings.warn('SO is deprecated, use EXT_SUFFIX', DeprecationWarning, 2)
    return get_config_vars().get(name)
"""------------------------------------------------------------------------------------------------------"""
# -*- coding: utf-8 -*-
__all__ = ['Distribution']

import io
import sys
import re
import os
import warnings
import numbers
import distutils.log
import distutils.core
import distutils.cmd
import distutils.dist
import distutils.command
from distutils.util import strtobool
from distutils.debug import DEBUG
from distutils.fancy_getopt import translate_longopt
from glob import iglob
import itertools
import textwrap
from typing import List, Optional, TYPE_CHECKING

from collections import defaultdict
from email import message_from_file

from distutils.errors import DistutilsOptionError, DistutilsSetupError
from distutils.util import rfc822_escape
from distutils.version import StrictVersion

from setuptools.extern import packaging
from setuptools.extern import ordered_set
from setuptools.extern.more_itertools import unique_everseen

from . import SetuptoolsDeprecationWarning

import setuptools
import setuptools.command
from setuptools import windows_support
from setuptools.monkey import get_unpatched
from setuptools.config import parse_configuration
import pkg_resources

if TYPE_CHECKING:
    from email.message import Message

__import__('setuptools.extern.packaging.specifiers')
__import__('setuptools.extern.packaging.version')


def _get_unpatched(cls):
    warnings.warn("Do not call this function", DistDeprecationWarning)
    return get_unpatched(cls)


def get_metadata_version(self):
    mv = getattr(self, 'metadata_version', None)
    if mv is None:
        mv = StrictVersion('2.1')
        self.metadata_version = mv
    return mv


def rfc822_unescape(content: str) -> str:
    """Reverse RFC-822 escaping by removing leading whitespaces from content."""
    lines = content.splitlines()
    if len(lines) == 1:
        return lines[0].lstrip()
    return '\n'.join((lines[0].lstrip(), textwrap.dedent('\n'.join(lines[1:]))))


def _read_field_from_msg(msg: "Message", field: str) -> Optional[str]:
    """Read Message header field."""
    value = msg[field]
    if value == 'UNKNOWN':
        return None
    return value


def _read_field_unescaped_from_msg(msg: "Message", field: str) -> Optional[str]:
    """Read Message header field and apply rfc822_unescape."""
    value = _read_field_from_msg(msg, field)
    if value is None:
        return value
    return rfc822_unescape(value)


def _read_list_from_msg(msg: "Message", field: str) -> Optional[List[str]]:
    """Read Message header field and return all results as list."""
    values = msg.get_all(field, None)
    if values == []:
        return None
    return values


def _read_payload_from_msg(msg: "Message") -> Optional[str]:
    value = msg.get_payload().strip()
    if value == 'UNKNOWN':
        return None
    return value


def read_pkg_file(self, file):
    """Reads the metadata values from a file object."""
    msg = message_from_file(file)

    self.metadata_version = StrictVersion(msg['metadata-version'])
    self.name = _read_field_from_msg(msg, 'name')
    self.version = _read_field_from_msg(msg, 'version')
    self.description = _read_field_from_msg(msg, 'summary')
    # we are filling author only.
    self.author = _read_field_from_msg(msg, 'author')
    self.maintainer = None
    self.author_email = _read_field_from_msg(msg, 'author-email')
    self.maintainer_email = None
    self.url = _read_field_from_msg(msg, 'home-page')
    self.license = _read_field_unescaped_from_msg(msg, 'license')

    if 'download-url' in msg:
        self.download_url = _read_field_from_msg(msg, 'download-url')
    else:
        self.download_url = None

    self.long_description = _read_field_unescaped_from_msg(msg, 'description')
    if self.long_description is None and self.metadata_version >= StrictVersion('2.1'):
        self.long_description = _read_payload_from_msg(msg)
    self.description = _read_field_from_msg(msg, 'summary')

    if 'keywords' in msg:
        self.keywords = _read_field_from_msg(msg, 'keywords').split(',')

    self.platforms = _read_list_from_msg(msg, 'platform')
    self.classifiers = _read_list_from_msg(msg, 'classifier')

    # PEP 314 - these fields only exist in 1.1
    if self.metadata_version == StrictVersion('1.1'):
        self.requires = _read_list_from_msg(msg, 'requires')
        self.provides = _read_list_from_msg(msg, 'provides')
        self.obsoletes = _read_list_from_msg(msg, 'obsoletes')
    else:
        self.requires = None
        self.provides = None
        self.obsoletes = None

    self.license_files = _read_list_from_msg(msg, 'license-file')


def single_line(val):
    # quick and dirty validation for description pypa/setuptools#1390
    if '\n' in val:
        # TODO after 2021-07-31: Replace with `raise ValueError("newlines not allowed")`
        warnings.warn("newlines not allowed and will break in the future")
        val = val.replace('\n', ' ')
    return val


# Based on Python 3.5 version
def write_pkg_file(self, file):  # noqa: C901  # is too complex (14)  # FIXME
    """Write the PKG-INFO format data to a file object."""
    version = self.get_metadata_version()

    def write_field(key, value):
        file.write("%s: %s\n" % (key, value))

    write_field('Metadata-Version', str(version))
    write_field('Name', self.get_name())
    write_field('Version', self.get_version())
    write_field('Summary', single_line(self.get_description()))
    write_field('Home-page', self.get_url())

    optional_fields = (
        ('Author', 'author'),
        ('Author-email', 'author_email'),
        ('Maintainer', 'maintainer'),
        ('Maintainer-email', 'maintainer_email'),
    )

    for field, attr in optional_fields:
        attr_val = getattr(self, attr, None)
        if attr_val is not None:
            write_field(field, attr_val)

    license = rfc822_escape(self.get_license())
    write_field('License', license)
    if self.download_url:
        write_field('Download-URL', self.download_url)
    for project_url in self.project_urls.items():
        write_field('Project-URL', '%s, %s' % project_url)

    keywords = ','.join(self.get_keywords())
    if keywords:
        write_field('Keywords', keywords)

    for platform in self.get_platforms():
        write_field('Platform', platform)

    self._write_list(file, 'Classifier', self.get_classifiers())

    # PEP 314
    self._write_list(file, 'Requires', self.get_requires())
    self._write_list(file, 'Provides', self.get_provides())
    self._write_list(file, 'Obsoletes', self.get_obsoletes())

    # Setuptools specific for PEP 345
    if hasattr(self, 'python_requires'):
        write_field('Requires-Python', self.python_requires)

    # PEP 566
    if self.long_description_content_type:
        write_field('Description-Content-Type', self.long_description_content_type)
    if self.provides_extras:
        for extra in self.provides_extras:
            write_field('Provides-Extra', extra)

    self._write_list(file, 'License-File', self.license_files or [])

    file.write("\n%s\n\n" % self.get_long_description())


sequence = tuple, list


def check_importable(dist, attr, value):
    try:
        ep = pkg_resources.EntryPoint.parse('x=' + value)
        assert not ep.extras
    except (TypeError, ValueError, AttributeError, AssertionError) as e:
        raise DistutilsSetupError(
            "%r must be importable 'module:attrs' string (got %r)" % (attr, value)
        ) from e


def assert_string_list(dist, attr, value):
    """Verify that value is a string list"""
    try:
        # verify that value is a list or tuple to exclude unordered
        # or single-use iterables
        assert isinstance(value, (list, tuple))
        # verify that elements of value are strings
        assert ''.join(value) != value
    except (TypeError, ValueError, AttributeError, AssertionError) as e:
        raise DistutilsSetupError(
            "%r must be a list of strings (got %r)" % (attr, value)
        ) from e


def check_nsp(dist, attr, value):
    """Verify that namespace packages are valid"""
    ns_packages = value
    assert_string_list(dist, attr, ns_packages)
    for nsp in ns_packages:
        if not dist.has_contents_for(nsp):
            raise DistutilsSetupError(
                "Distribution contains no modules or packages for "
                + "namespace package %r" % nsp
            )
        parent, sep, child = nsp.rpartition('.')
        if parent and parent not in ns_packages:
            distutils.log.warn(
                "WARNING: %r is declared as a package namespace, but %r"
                " is not: please correct this in setup.py",
                nsp,
                parent,
            )


def check_extras(dist, attr, value):
    """Verify that extras_require mapping is valid"""
    try:
        list(itertools.starmap(_check_extra, value.items()))
    except (TypeError, ValueError, AttributeError) as e:
        raise DistutilsSetupError(
            "'extras_require' must be a dictionary whose values are "
            "strings or lists of strings containing valid project/version "
            "requirement specifiers."
        ) from e


def _check_extra(extra, reqs):
    name, sep, marker = extra.partition(':')
    if marker and pkg_resources.invalid_marker(marker):
        raise DistutilsSetupError("Invalid environment marker: " + marker)
    list(pkg_resources.parse_requirements(reqs))


def assert_bool(dist, attr, value):
    """Verify that value is True, False, 0, or 1"""
    if bool(value) != value:
        tmpl = "{attr!r} must be a boolean value (got {value!r})"
        raise DistutilsSetupError(tmpl.format(attr=attr, value=value))


def invalid_unless_false(dist, attr, value):
    if not value:
        warnings.warn(f"{attr} is ignored.", DistDeprecationWarning)
        return
    raise DistutilsSetupError(f"{attr} is invalid.")


def check_requirements(dist, attr, value):
    """Verify that install_requires is a valid requirements list"""
    try:
        list(pkg_resources.parse_requirements(value))
        if isinstance(value, (dict, set)):
            raise TypeError("Unordered types are not allowed")
    except (TypeError, ValueError) as error:
        tmpl = (
            "{attr!r} must be a string or list of strings "
            "containing valid project/version requirement specifiers; {error}"
        )
        raise DistutilsSetupError(tmpl.format(attr=attr, error=error)) from error


def check_specifier(dist, attr, value):
    """Verify that value is a valid version specifier"""
    try:
        packaging.specifiers.SpecifierSet(value)
    except (packaging.specifiers.InvalidSpecifier, AttributeError) as error:
        tmpl = (
            "{attr!r} must be a string " "containing valid version specifiers; {error}"
        )
        raise DistutilsSetupError(tmpl.format(attr=attr, error=error)) from error


def check_entry_points(dist, attr, value):
    """Verify that entry_points map is parseable"""
    try:
        pkg_resources.EntryPoint.parse_map(value)
    except ValueError as e:
        raise DistutilsSetupError(e) from e


def check_test_suite(dist, attr, value):
    if not isinstance(value, str):
        raise DistutilsSetupError("test_suite must be a string")


def check_package_data(dist, attr, value):
    """Verify that value is a dictionary of package names to glob lists"""
    if not isinstance(value, dict):
        raise DistutilsSetupError(
            "{!r} must be a dictionary mapping package names to lists of "
            "string wildcard patterns".format(attr)
        )
    for k, v in value.items():
        if not isinstance(k, str):
            raise DistutilsSetupError(
                "keys of {!r} dict must be strings (got {!r})".format(attr, k)
            )
        assert_string_list(dist, 'values of {!r} dict'.format(attr), v)


def check_packages(dist, attr, value):
    for pkgname in value:
        if not re.match(r'\w+(\.\w+)*', pkgname):
            distutils.log.warn(
                "WARNING: %r not a valid package name; please use only "
                ".-separated package names in setup.py",
                pkgname,
            )


_Distribution = get_unpatched(distutils.core.Distribution)


class Distribution(_Distribution):
    """Distribution with support for tests and package data

    This is an enhanced version of 'distutils.dist.Distribution' that
    effectively adds the following new optional keyword arguments to 'setup()':

     'install_requires' -- a string or sequence of strings specifying project
        versions that the distribution requires when installed, in the format
        used by 'pkg_resources.require()'.  They will be installed
        automatically when the package is installed.  If you wish to use
        packages that are not available in PyPI, or want to give your users an
        alternate download location, you can add a 'find_links' option to the
        '[easy_install]' section of your project's 'setup.cfg' file, and then
        setuptools will scan the listed web pages for links that satisfy the
        requirements.

     'extras_require' -- a dictionary mapping names of optional "extras" to the
        additional requirement(s) that using those extras incurs. For example,
        this::

            extras_require = dict(reST = ["docutils>=0.3", "reSTedit"])

        indicates that the distribution can optionally provide an extra
        capability called "reST", but it can only be used if docutils and
        reSTedit are installed.  If the user installs your package using
        EasyInstall and requests one of your extras, the corresponding
        additional requirements will be installed if needed.

     'test_suite' -- the name of a test suite to run for the 'test' command.
        If the user runs 'python setup.py test', the package will be installed,
        and the named test suite will be run.  The format is the same as
        would be used on a 'unittest.py' command line.  That is, it is the
        dotted name of an object to import and call to generate a test suite.

     'package_data' -- a dictionary mapping package names to lists of filenames
        or globs to use to find data files contained in the named packages.
        If the dictionary has filenames or globs listed under '""' (the empty
        string), those names will be searched for in every package, in addition
        to any names for the specific package.  Data files found using these
        names/globs will be installed along with the package, in the same
        location as the package.  Note that globs are allowed to reference
        the contents of non-package subdirectories, as long as you use '/' as
        a path separator.  (Globs are automatically converted to
        platform-specific paths at runtime.)

    In addition to these new keywords, this class also has several new methods
    for manipulating the distribution's contents.  For example, the 'include()'
    and 'exclude()' methods can be thought of as in-place add and subtract
    commands that add or remove packages, modules, extensions, and so on from
    the distribution.
    """

    _DISTUTILS_UNSUPPORTED_METADATA = {
        'long_description_content_type': lambda: None,
        'project_urls': dict,
        'provides_extras': ordered_set.OrderedSet,
        'license_file': lambda: None,
        'license_files': lambda: None,
    }

    _patched_dist = None

    def patch_missing_pkg_info(self, attrs):
        # Fake up a replacement for the data that would normally come from
        # PKG-INFO, but which might not yet be built if this is a fresh
        # checkout.
        #
        if not attrs or 'name' not in attrs or 'version' not in attrs:
            return
        key = pkg_resources.safe_name(str(attrs['name'])).lower()
        dist = pkg_resources.working_set.by_key.get(key)
        if dist is not None and not dist.has_metadata('PKG-INFO'):
            dist._version = pkg_resources.safe_version(str(attrs['version']))
            self._patched_dist = dist

    def __init__(self, attrs=None):
        have_package_data = hasattr(self, "package_data")
        if not have_package_data:
            self.package_data = {}
        attrs = attrs or {}
        self.dist_files = []
        # Filter-out setuptools' specific options.
        self.src_root = attrs.pop("src_root", None)
        self.patch_missing_pkg_info(attrs)
        self.dependency_links = attrs.pop('dependency_links', [])
        self.setup_requires = attrs.pop('setup_requires', [])
        for ep in pkg_resources.iter_entry_points('distutils.setup_keywords'):
            vars(self).setdefault(ep.name, None)
        _Distribution.__init__(
            self,
            {
                k: v
                for k, v in attrs.items()
                if k not in self._DISTUTILS_UNSUPPORTED_METADATA
            },
        )

        self._set_metadata_defaults(attrs)

        self.metadata.version = self._normalize_version(
            self._validate_version(self.metadata.version)
        )
        self._finalize_requires()

    def _set_metadata_defaults(self, attrs):
        """
        Fill-in missing metadata fields not supported by distutils.
        Some fields may have been set by other tools (e.g. pbr).
        Those fields (vars(self.metadata)) take precedence to
        supplied attrs.
        """
        for option, default in self._DISTUTILS_UNSUPPORTED_METADATA.items():
            vars(self.metadata).setdefault(option, attrs.get(option, default()))

    @staticmethod
    def _normalize_version(version):
        if isinstance(version, setuptools.sic) or version is None:
            return version

        normalized = str(packaging.version.Version(version))
        if version != normalized:
            tmpl = "Normalizing '{version}' to '{normalized}'"
            warnings.warn(tmpl.format(**locals()))
            return normalized
        return version

    @staticmethod
    def _validate_version(version):
        if isinstance(version, numbers.Number):
            # Some people apparently take "version number" too literally :)
            version = str(version)

        if version is not None:
            try:
                packaging.version.Version(version)
            except (packaging.version.InvalidVersion, TypeError):
                warnings.warn(
                    "The version specified (%r) is an invalid version, this "
                    "may not work as expected with newer versions of "
                    "setuptools, pip, and PyPI. Please see PEP 440 for more "
                    "details." % version
                )
                return setuptools.sic(version)
        return version

    def _finalize_requires(self):
        """
        Set `metadata.python_requires` and fix environment markers
        in `install_requires` and `extras_require`.
        """
        if getattr(self, 'python_requires', None):
            self.metadata.python_requires = self.python_requires

        if getattr(self, 'extras_require', None):
            for extra in self.extras_require.keys():
                # Since this gets called multiple times at points where the
                # keys have become 'converted' extras, ensure that we are only
                # truly adding extras we haven't seen before here.
                extra = extra.split(':')[0]
                if extra:
                    self.metadata.provides_extras.add(extra)

        self._convert_extras_requirements()
        self._move_install_requirements_markers()

    def _convert_extras_requirements(self):
        """
        Convert requirements in `extras_require` of the form
        `"extra": ["barbazquux; {marker}"]` to
        `"extra:{marker}": ["barbazquux"]`.
        """
        spec_ext_reqs = getattr(self, 'extras_require', None) or {}
        self._tmp_extras_require = defaultdict(list)
        for section, v in spec_ext_reqs.items():
            # Do not strip empty sections.
            self._tmp_extras_require[section]
            for r in pkg_resources.parse_requirements(v):
                suffix = self._suffix_for(r)
                self._tmp_extras_require[section + suffix].append(r)

    @staticmethod
    def _suffix_for(req):
        """
        For a requirement, return the 'extras_require' suffix for
        that requirement.
        """
        return ':' + str(req.marker) if req.marker else ''

    def _move_install_requirements_markers(self):
        """
        Move requirements in `install_requires` that are using environment
        markers `extras_require`.
        """

        # divide the install_requires into two sets, simple ones still
        # handled by install_requires and more complex ones handled
        # by extras_require.

        def is_simple_req(req):
            return not req.marker

        spec_inst_reqs = getattr(self, 'install_requires', None) or ()
        inst_reqs = list(pkg_resources.parse_requirements(spec_inst_reqs))
        simple_reqs = filter(is_simple_req, inst_reqs)
        complex_reqs = itertools.filterfalse(is_simple_req, inst_reqs)
        self.install_requires = list(map(str, simple_reqs))

        for r in complex_reqs:
            self._tmp_extras_require[':' + str(r.marker)].append(r)
        self.extras_require = dict(
            (k, [str(r) for r in map(self._clean_req, v)])
            for k, v in self._tmp_extras_require.items()
        )

    def _clean_req(self, req):
        """
        Given a Requirement, remove environment markers and return it.
        """
        req.marker = None
        return req

    def _finalize_license_files(self):
        """Compute names of all license files which should be included."""
        license_files: Optional[List[str]] = self.metadata.license_files
        patterns: List[str] = license_files if license_files else []

        license_file: Optional[str] = self.metadata.license_file
        if license_file and license_file not in patterns:
            patterns.append(license_file)

        if license_files is None and license_file is None:
            # Default patterns match the ones wheel uses
            # See https://wheel.readthedocs.io/en/stable/user_guide.html
            # -> 'Including license files in the generated wheel file'
            patterns = ('LICEN[CS]E*', 'COPYING*', 'NOTICE*', 'AUTHORS*')

        self.metadata.license_files = list(
            unique_everseen(self._expand_patterns(patterns))
        )

    @staticmethod
    def _expand_patterns(patterns):
        """
        >>> list(Distribution._expand_patterns(['LICENSE']))
        ['LICENSE']
        >>> list(Distribution._expand_patterns(['setup.cfg', 'LIC*']))
        ['setup.cfg', 'LICENSE']
        """
        return (
            path
            for pattern in patterns
            for path in sorted(iglob(pattern))
            if not path.endswith('~') and os.path.isfile(path)
        )

    # FIXME: 'Distribution._parse_config_files' is too complex (14)
    def _parse_config_files(self, filenames=None):  # noqa: C901
        """
        Adapted from distutils.dist.Distribution.parse_config_files,
        this method provides the same functionality in subtly-improved
        ways.
        """
        from configparser import ConfigParser

        # Ignore install directory options if we have a venv
        ignore_options = (
            []
            if sys.prefix == sys.base_prefix
            else [
                'install-base',
                'install-platbase',
                'install-lib',
                'install-platlib',
                'install-purelib',
                'install-headers',
                'install-scripts',
                'install-data',
                'prefix',
                'exec-prefix',
                'home',
                'user',
                'root',
            ]
        )

        ignore_options = frozenset(ignore_options)

        if filenames is None:
            filenames = self.find_config_files()

        if DEBUG:
            self.announce("Distribution.parse_config_files():")

        parser = ConfigParser()
        parser.optionxform = str
        for filename in filenames:
            with io.open(filename, encoding='utf-8') as reader:
                if DEBUG:
                    self.announce("  reading {filename}".format(**locals()))
                parser.read_file(reader)
            for section in parser.sections():
                options = parser.options(section)
                opt_dict = self.get_option_dict(section)

                for opt in options:
                    if opt == '__name__' or opt in ignore_options:
                        continue

                    val = parser.get(section, opt)
                    opt = self.warn_dash_deprecation(opt, section)
                    opt = self.make_option_lowercase(opt, section)
                    opt_dict[opt] = (filename, val)

            # Make the ConfigParser forget everything (so we retain
            # the original filenames that options come from)
            parser.__init__()

        if 'global' not in self.command_options:
            return

        # If there was a "global" section in the config file, use it
        # to set Distribution options.

        for (opt, (src, val)) in self.command_options['global'].items():
            alias = self.negative_opt.get(opt)
            if alias:
                val = not strtobool(val)
            elif opt in ('verbose', 'dry_run'):  # ugh!
                val = strtobool(val)

            try:
                setattr(self, alias or opt, val)
            except ValueError as e:
                raise DistutilsOptionError(e) from e

    def warn_dash_deprecation(self, opt, section):
        if section in (
            'options.extras_require',
            'options.data_files',
        ):
            return opt

        underscore_opt = opt.replace('-', '_')
        commands = distutils.command.__all__ + self._setuptools_commands()
        if (
            not section.startswith('options')
            and section != 'metadata'
            and section not in commands
        ):
            return underscore_opt

        if '-' in opt:
            warnings.warn(
                "Usage of dash-separated '%s' will not be supported in future "
                "versions. Please use the underscore name '%s' instead"
                % (opt, underscore_opt)
            )
        return underscore_opt

    def _setuptools_commands(self):
        try:
            dist = pkg_resources.get_distribution('setuptools')
            return list(dist.get_entry_map('distutils.commands'))
        except pkg_resources.DistributionNotFound:
            # during bootstrapping, distribution doesn't exist
            return []

    def make_option_lowercase(self, opt, section):
        if section != 'metadata' or opt.islower():
            return opt

        lowercase_opt = opt.lower()
        warnings.warn(
            "Usage of uppercase key '%s' in '%s' will be deprecated in future "
            "versions. Please use lowercase '%s' instead"
            % (opt, section, lowercase_opt)
        )
        return lowercase_opt

    # FIXME: 'Distribution._set_command_options' is too complex (14)
    def _set_command_options(self, command_obj, option_dict=None):  # noqa: C901
        """
        Set the options for 'command_obj' from 'option_dict'.  Basically
        this means copying elements of a dictionary ('option_dict') to
        attributes of an instance ('command').

        'command_obj' must be a Command instance.  If 'option_dict' is not
        supplied, uses the standard option dictionary for this command
        (from 'self.command_options').

        (Adopted from distutils.dist.Distribution._set_command_options)
        """
        command_name = command_obj.get_command_name()
        if option_dict is None:
            option_dict = self.get_option_dict(command_name)

        if DEBUG:
            self.announce("  setting options for '%s' command:" % command_name)
        for (option, (source, value)) in option_dict.items():
            if DEBUG:
                self.announce("    %s = %s (from %s)" % (option, value, source))
            try:
                bool_opts = [translate_longopt(o) for o in command_obj.boolean_options]
            except AttributeError:
                bool_opts = []
            try:
                neg_opt = command_obj.negative_opt
            except AttributeError:
                neg_opt = {}

            try:
                is_string = isinstance(value, str)
                if option in neg_opt and is_string:
                    setattr(command_obj, neg_opt[option], not strtobool(value))
                elif option in bool_opts and is_string:
                    setattr(command_obj, option, strtobool(value))
                elif hasattr(command_obj, option):
                    setattr(command_obj, option, value)
                else:
                    raise DistutilsOptionError(
                        "error in %s: command '%s' has no such option '%s'"
                        % (source, command_name, option)
                    )
            except ValueError as e:
                raise DistutilsOptionError(e) from e

    def parse_config_files(self, filenames=None, ignore_option_errors=False):
        """Parses configuration files from various levels
        and loads configuration.

        """
        self._parse_config_files(filenames=filenames)

        parse_configuration(
            self, self.command_options, ignore_option_errors=ignore_option_errors
        )
        self._finalize_requires()
        self._finalize_license_files()

    def fetch_build_eggs(self, requires):
        """Resolve pre-setup requirements"""
        resolved_dists = pkg_resources.working_set.resolve(
            pkg_resources.parse_requirements(requires),
            installer=self.fetch_build_egg,
            replace_conflicting=True,
        )
        for dist in resolved_dists:
            pkg_resources.working_set.add(dist, replace=True)
        return resolved_dists

    def finalize_options(self):
        """
        Allow plugins to apply arbitrary operations to the
        distribution. Each hook may optionally define a 'order'
        to influence the order of execution. Smaller numbers
        go first and the default is 0.
        """
        group = 'setuptools.finalize_distribution_options'

        def by_order(hook):
            return getattr(hook, 'order', 0)

        defined = pkg_resources.iter_entry_points(group)
        filtered = itertools.filterfalse(self._removed, defined)
        loaded = map(lambda e: e.load(), filtered)
        for ep in sorted(loaded, key=by_order):
            ep(self)

    @staticmethod
    def _removed(ep):
        """
        When removing an entry point, if metadata is loaded
        from an older version of Setuptools, that removed
        entry point will attempt to be loaded and will fail.
        See #2765 for more details.
        """
        removed = {
            # removed 2021-09-05
            '2to3_doctests',
        }
        return ep.name in removed

    def _finalize_setup_keywords(self):
        for ep in pkg_resources.iter_entry_points('distutils.setup_keywords'):
            value = getattr(self, ep.name, None)
            if value is not None:
                ep.require(installer=self.fetch_build_egg)
                ep.load()(self, ep.name, value)

    def get_egg_cache_dir(self):
        egg_cache_dir = os.path.join(os.curdir, '.eggs')
        if not os.path.exists(egg_cache_dir):
            os.mkdir(egg_cache_dir)
            windows_support.hide_file(egg_cache_dir)
            readme_txt_filename = os.path.join(egg_cache_dir, 'README.txt')
            with open(readme_txt_filename, 'w') as f:
                f.write(
                    'This directory contains eggs that were downloaded '
                    'by setuptools to build, test, and run plug-ins.\n\n'
                )
                f.write(
                    'This directory caches those eggs to prevent '
                    'repeated downloads.\n\n'
                )
                f.write('However, it is safe to delete this directory.\n\n')

        return egg_cache_dir

    def fetch_build_egg(self, req):
        """Fetch an egg needed for building"""
        from setuptools.installer import fetch_build_egg

        return fetch_build_egg(self, req)

    def get_command_class(self, command):
        """Pluggable version of get_command_class()"""
        if command in self.cmdclass:
            return self.cmdclass[command]

        eps = pkg_resources.iter_entry_points('distutils.commands', command)
        for ep in eps:
            ep.require(installer=self.fetch_build_egg)
            self.cmdclass[command] = cmdclass = ep.load()
            return cmdclass
        else:
            return _Distribution.get_command_class(self, command)

    def print_commands(self):
        for ep in pkg_resources.iter_entry_points('distutils.commands'):
            if ep.name not in self.cmdclass:
                # don't require extras as the commands won't be invoked
                cmdclass = ep.resolve()
                self.cmdclass[ep.name] = cmdclass
        return _Distribution.print_commands(self)

    def get_command_list(self):
        for ep in pkg_resources.iter_entry_points('distutils.commands'):
            if ep.name not in self.cmdclass:
                # don't require extras as the commands won't be invoked
                cmdclass = ep.resolve()
                self.cmdclass[ep.name] = cmdclass
        return _Distribution.get_command_list(self)

    def include(self, **attrs):
        """Add items to distribution that are named in keyword arguments

        For example, 'dist.include(py_modules=["x"])' would add 'x' to
        the distribution's 'py_modules' attribute, if it was not already
        there.

        Currently, this method only supports inclusion for attributes that are
        lists or tuples.  If you need to add support for adding to other
        attributes in this or a subclass, you can add an '_include_X' method,
        where 'X' is the name of the attribute.  The method will be called with
        the value passed to 'include()'.  So, 'dist.include(foo={"bar":"baz"})'
        will try to call 'dist._include_foo({"bar":"baz"})', which can then
        handle whatever special inclusion logic is needed.
        """
        for k, v in attrs.items():
            include = getattr(self, '_include_' + k, None)
            if include:
                include(v)
            else:
                self._include_misc(k, v)

    def exclude_package(self, package):
        """Remove packages, modules, and extensions in named package"""

        pfx = package + '.'
        if self.packages:
            self.packages = [
                p for p in self.packages if p != package and not p.startswith(pfx)
            ]

        if self.py_modules:
            self.py_modules = [
                p for p in self.py_modules if p != package and not p.startswith(pfx)
            ]

        if self.ext_modules:
            self.ext_modules = [
                p
                for p in self.ext_modules
                if p.name != package and not p.name.startswith(pfx)
            ]

    def has_contents_for(self, package):
        """Return true if 'exclude_package(package)' would do something"""

        pfx = package + '.'

        for p in self.iter_distribution_names():
            if p == package or p.startswith(pfx):
                return True

    def _exclude_misc(self, name, value):
        """Handle 'exclude()' for list/tuple attrs without a special handler"""
        if not isinstance(value, sequence):
            raise DistutilsSetupError(
                "%s: setting must be a list or tuple (%r)" % (name, value)
            )
        try:
            old = getattr(self, name)
        except AttributeError as e:
            raise DistutilsSetupError("%s: No such distribution setting" % name) from e
        if old is not None and not isinstance(old, sequence):
            raise DistutilsSetupError(
                name + ": this setting cannot be changed via include/exclude"
            )
        elif old:
            setattr(self, name, [item for item in old if item not in value])

    def _include_misc(self, name, value):
        """Handle 'include()' for list/tuple attrs without a special handler"""

        if not isinstance(value, sequence):
            raise DistutilsSetupError("%s: setting must be a list (%r)" % (name, value))
        try:
            old = getattr(self, name)
        except AttributeError as e:
            raise DistutilsSetupError("%s: No such distribution setting" % name) from e
        if old is None:
            setattr(self, name, value)
        elif not isinstance(old, sequence):
            raise DistutilsSetupError(
                name + ": this setting cannot be changed via include/exclude"
            )
        else:
            new = [item for item in value if item not in old]
            setattr(self, name, old + new)

    def exclude(self, **attrs):
        """Remove items from distribution that are named in keyword arguments

        For example, 'dist.exclude(py_modules=["x"])' would remove 'x' from
        the distribution's 'py_modules' attribute.  Excluding packages uses
        the 'exclude_package()' method, so all of the package's contained
        packages, modules, and extensions are also excluded.

        Currently, this method only supports exclusion from attributes that are
        lists or tuples.  If you need to add support for excluding from other
        attributes in this or a subclass, you can add an '_exclude_X' method,
        where 'X' is the name of the attribute.  The method will be called with
        the value passed to 'exclude()'.  So, 'dist.exclude(foo={"bar":"baz"})'
        will try to call 'dist._exclude_foo({"bar":"baz"})', which can then
        handle whatever special exclusion logic is needed.
        """
        for k, v in attrs.items():
            exclude = getattr(self, '_exclude_' + k, None)
            if exclude:
                exclude(v)
            else:
                self._exclude_misc(k, v)

    def _exclude_packages(self, packages):
        if not isinstance(packages, sequence):
            raise DistutilsSetupError(
                "packages: setting must be a list or tuple (%r)" % (packages,)
            )
        list(map(self.exclude_package, packages))

    def _parse_command_opts(self, parser, args):
        # Remove --with-X/--without-X options when processing command args
        self.global_options = self.__class__.global_options
        self.negative_opt = self.__class__.negative_opt

        # First, expand any aliases
        command = args[0]
        aliases = self.get_option_dict('aliases')
        while command in aliases:
            src, alias = aliases[command]
            del aliases[command]  # ensure each alias can expand only once!
            import shlex

            args[:1] = shlex.split(alias, True)
            command = args[0]

        nargs = _Distribution._parse_command_opts(self, parser, args)

        # Handle commands that want to consume all remaining arguments
        cmd_class = self.get_command_class(command)
        if getattr(cmd_class, 'command_consumes_arguments', None):
            self.get_option_dict(command)['args'] = ("command line", nargs)
            if nargs is not None:
                return []

        return nargs

    def get_cmdline_options(self):
        """Return a '{cmd: {opt:val}}' map of all command-line options

        Option names are all long, but do not include the leading '--', and
        contain dashes rather than underscores.  If the option doesn't take
        an argument (e.g. '--quiet'), the 'val' is 'None'.

        Note that options provided by config files are intentionally excluded.
        """

        d = {}

        for cmd, opts in self.command_options.items():

            for opt, (src, val) in opts.items():

                if src != "command line":
                    continue

                opt = opt.replace('_', '-')

                if val == 0:
                    cmdobj = self.get_command_obj(cmd)
                    neg_opt = self.negative_opt.copy()
                    neg_opt.update(getattr(cmdobj, 'negative_opt', {}))
                    for neg, pos in neg_opt.items():
                        if pos == opt:
                            opt = neg
                            val = None
                            break
                    else:
                        raise AssertionError("Shouldn't be able to get here")

                elif val == 1:
                    val = None

                d.setdefault(cmd, {})[opt] = val

        return d

    def iter_distribution_names(self):
        """Yield all packages, modules, and extension names in distribution"""

        for pkg in self.packages or ():
            yield pkg

        for module in self.py_modules or ():
            yield module

        for ext in self.ext_modules or ():
            if isinstance(ext, tuple):
                name, buildinfo = ext
            else:
                name = ext.name
            if name.endswith('module'):
                name = name[:-6]
            yield name

    def handle_display_options(self, option_order):
        """If there were any non-global "display-only" options
        (--help-commands or the metadata display options) on the command
        line, display the requested info and return true; else return
        false.
        """
        import sys

        if self.help_commands:
            return _Distribution.handle_display_options(self, option_order)

        # Stdout may be StringIO (e.g. in tests)
        if not isinstance(sys.stdout, io.TextIOWrapper):
            return _Distribution.handle_display_options(self, option_order)

        # Don't wrap stdout if utf-8 is already the encoding. Provides
        #  workaround for #334.
        if sys.stdout.encoding.lower() in ('utf-8', 'utf8'):
            return _Distribution.handle_display_options(self, option_order)

        # Print metadata in UTF-8 no matter the platform
        encoding = sys.stdout.encoding
        errors = sys.stdout.errors
        newline = sys.platform != 'win32' and '\n' or None
        line_buffering = sys.stdout.line_buffering

        sys.stdout = io.TextIOWrapper(
            sys.stdout.detach(), 'utf-8', errors, newline, line_buffering
        )
        try:
            return _Distribution.handle_display_options(self, option_order)
        finally:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.detach(), encoding, errors, newline, line_buffering
            )


class DistDeprecationWarning(SetuptoolsDeprecationWarning):
    """Class for warning about deprecations in dist in
    setuptools. Not ignored by default, unlike DeprecationWarning."""
"""------------------------------------------------------------------------------------------"""
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

import abc
import functools
import itertools
import re

from ._compat import string_types, with_metaclass
from ._typing import TYPE_CHECKING
from .utils import canonicalize_version
from .version import Version, LegacyVersion, parse

if TYPE_CHECKING:  # pragma: no cover
    from typing import (
        List,
        Dict,
        Union,
        Iterable,
        Iterator,
        Optional,
        Callable,
        Tuple,
        FrozenSet,
    )

    ParsedVersion = Union[Version, LegacyVersion]
    UnparsedVersion = Union[Version, LegacyVersion, str]
    CallableOperator = Callable[[ParsedVersion, str], bool]


class InvalidSpecifier(ValueError):
    """
    An invalid specifier was found, users should refer to PEP 440.
    """


class BaseSpecifier(with_metaclass(abc.ABCMeta, object)):  # type: ignore
    @abc.abstractmethod
    def __str__(self):
        # type: () -> str
        """
        Returns the str representation of this Specifier like object. This
        should be representative of the Specifier itself.
        """

    @abc.abstractmethod
    def __hash__(self):
        # type: () -> int
        """
        Returns a hash value for this Specifier like object.
        """

    @abc.abstractmethod
    def __eq__(self, other):
        # type: (object) -> bool
        """
        Returns a boolean representing whether or not the two Specifier like
        objects are equal.
        """

    @abc.abstractmethod
    def __ne__(self, other):
        # type: (object) -> bool
        """
        Returns a boolean representing whether or not the two Specifier like
        objects are not equal.
        """

    @abc.abstractproperty
    def prereleases(self):
        # type: () -> Optional[bool]
        """
        Returns whether or not pre-releases as a whole are allowed by this
        specifier.
        """

    @prereleases.setter
    def prereleases(self, value):
        # type: (bool) -> None
        """
        Sets whether or not pre-releases as a whole are allowed by this
        specifier.
        """

    @abc.abstractmethod
    def contains(self, item, prereleases=None):
        # type: (str, Optional[bool]) -> bool
        """
        Determines if the given item is contained within this specifier.
        """

    @abc.abstractmethod
    def filter(self, iterable, prereleases=None):
        # type: (Iterable[UnparsedVersion], Optional[bool]) -> Iterable[UnparsedVersion]
        """
        Takes an iterable of items and filters them so that only items which
        are contained within this specifier are allowed in it.
        """


class _IndividualSpecifier(BaseSpecifier):

    _operators = {}  # type: Dict[str, str]

    def __init__(self, spec="", prereleases=None):
        # type: (str, Optional[bool]) -> None
        match = self._regex.search(spec)
        if not match:
            raise InvalidSpecifier("Invalid specifier: '{0}'".format(spec))

        self._spec = (
            match.group("operator").strip(),
            match.group("version").strip(),
        )  # type: Tuple[str, str]

        # Store whether or not this Specifier should accept prereleases
        self._prereleases = prereleases

    def __repr__(self):
        # type: () -> str
        pre = (
            ", prereleases={0!r}".format(self.prereleases)
            if self._prereleases is not None
            else ""
        )

        return "<{0}({1!r}{2})>".format(self.__class__.__name__, str(self), pre)

    def __str__(self):
        # type: () -> str
        return "{0}{1}".format(*self._spec)

    @property
    def _canonical_spec(self):
        # type: () -> Tuple[str, Union[Version, str]]
        return self._spec[0], canonicalize_version(self._spec[1])

    def __hash__(self):
        # type: () -> int
        return hash(self._canonical_spec)

    def __eq__(self, other):
        # type: (object) -> bool
        if isinstance(other, string_types):
            try:
                other = self.__class__(str(other))
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._canonical_spec == other._canonical_spec

    def __ne__(self, other):
        # type: (object) -> bool
        if isinstance(other, string_types):
            try:
                other = self.__class__(str(other))
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._spec != other._spec

    def _get_operator(self, op):
        # type: (str) -> CallableOperator
        operator_callable = getattr(
            self, "_compare_{0}".format(self._operators[op])
        )  # type: CallableOperator
        return operator_callable

    def _coerce_version(self, version):
        # type: (UnparsedVersion) -> ParsedVersion
        if not isinstance(version, (LegacyVersion, Version)):
            version = parse(version)
        return version

    @property
    def operator(self):
        # type: () -> str
        return self._spec[0]

    @property
    def version(self):
        # type: () -> str
        return self._spec[1]

    @property
    def prereleases(self):
        # type: () -> Optional[bool]
        return self._prereleases

    @prereleases.setter
    def prereleases(self, value):
        # type: (bool) -> None
        self._prereleases = value

    def __contains__(self, item):
        # type: (str) -> bool
        return self.contains(item)

    def contains(self, item, prereleases=None):
        # type: (UnparsedVersion, Optional[bool]) -> bool

        # Determine if prereleases are to be allowed or not.
        if prereleases is None:
            prereleases = self.prereleases

        # Normalize item to a Version or LegacyVersion, this allows us to have
        # a shortcut for ``"2.0" in Specifier(">=2")
        normalized_item = self._coerce_version(item)

        # Determine if we should be supporting prereleases in this specifier
        # or not, if we do not support prereleases than we can short circuit
        # logic if this version is a prereleases.
        if normalized_item.is_prerelease and not prereleases:
            return False

        # Actually do the comparison to determine if this item is contained
        # within this Specifier or not.
        operator_callable = self._get_operator(self.operator)  # type: CallableOperator
        return operator_callable(normalized_item, self.version)

    def filter(self, iterable, prereleases=None):
        # type: (Iterable[UnparsedVersion], Optional[bool]) -> Iterable[UnparsedVersion]

        yielded = False
        found_prereleases = []

        kw = {"prereleases": prereleases if prereleases is not None else True}

        # Attempt to iterate over all the values in the iterable and if any of
        # them match, yield them.
        for version in iterable:
            parsed_version = self._coerce_version(version)

            if self.contains(parsed_version, **kw):
                # If our version is a prerelease, and we were not set to allow
                # prereleases, then we'll store it for later incase nothing
                # else matches this specifier.
                if parsed_version.is_prerelease and not (
                    prereleases or self.prereleases
                ):
                    found_prereleases.append(version)
                # Either this is not a prerelease, or we should have been
                # accepting prereleases from the beginning.
                else:
                    yielded = True
                    yield version

        # Now that we've iterated over everything, determine if we've yielded
        # any values, and if we have not and we have any prereleases stored up
        # then we will go ahead and yield the prereleases.
        if not yielded and found_prereleases:
            for version in found_prereleases:
                yield version


class LegacySpecifier(_IndividualSpecifier):

    _regex_str = r"""
        (?P<operator>(==|!=|<=|>=|<|>))
        \s*
        (?P<version>
            [^,;\s)]* # Since this is a "legacy" specifier, and the version
                      # string can be just about anything, we match everything
                      # except for whitespace, a semi-colon for marker support,
                      # a closing paren since versions can be enclosed in
                      # them, and a comma since it's a version separator.
        )
        """

    _regex = re.compile(r"^\s*" + _regex_str + r"\s*$", re.VERBOSE | re.IGNORECASE)

    _operators = {
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
    }

    def _coerce_version(self, version):
        # type: (Union[ParsedVersion, str]) -> LegacyVersion
        if not isinstance(version, LegacyVersion):
            version = LegacyVersion(str(version))
        return version

    def _compare_equal(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective == self._coerce_version(spec)

    def _compare_not_equal(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective != self._coerce_version(spec)

    def _compare_less_than_equal(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective <= self._coerce_version(spec)

    def _compare_greater_than_equal(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective >= self._coerce_version(spec)

    def _compare_less_than(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective < self._coerce_version(spec)

    def _compare_greater_than(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective > self._coerce_version(spec)


def _require_version_compare(
    fn  # type: (Callable[[Specifier, ParsedVersion, str], bool])
):
    # type: (...) -> Callable[[Specifier, ParsedVersion, str], bool]
    @functools.wraps(fn)
    def wrapped(self, prospective, spec):
        # type: (Specifier, ParsedVersion, str) -> bool
        if not isinstance(prospective, Version):
            return False
        return fn(self, prospective, spec)

    return wrapped


class Specifier(_IndividualSpecifier):

    _regex_str = r"""
        (?P<operator>(~=|==|!=|<=|>=|<|>|===))
        (?P<version>
            (?:
                # The identity operators allow for an escape hatch that will
                # do an exact string match of the version you wish to install.
                # This will not be parsed by PEP 440 and we cannot determine
                # any semantic meaning from it. This operator is discouraged
                # but included entirely as an escape hatch.
                (?<====)  # Only match for the identity operator
                \s*
                [^\s]*    # We just match everything, except for whitespace
                          # since we are only testing for strict identity.
            )
            |
            (?:
                # The (non)equality operators allow for wild card and local
                # versions to be specified so we have to define these two
                # operators separately to enable that.
                (?<===|!=)            # Only match for equals and not equals

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?:                   # pre release
                    [-_\.]?
                    (a|b|c|rc|alpha|beta|pre|preview)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?

                # You cannot use a wild card and a dev or local version
                # together so group them with a | and make them optional.
                (?:
                    (?:[-_\.]?dev[-_\.]?[0-9]*)?         # dev release
                    (?:\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*)? # local
                    |
                    \.\*  # Wild card syntax of .*
                )?
            )
            |
            (?:
                # The compatible operator requires at least two digits in the
                # release segment.
                (?<=~=)               # Only match for the compatible operator

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)+   # release  (We have a + instead of a *)
                (?:                   # pre release
                    [-_\.]?
                    (a|b|c|rc|alpha|beta|pre|preview)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
            |
            (?:
                # All other operators only allow a sub set of what the
                # (non)equality operators do. Specifically they do not allow
                # local versions to be specified nor do they allow the prefix
                # matching wild cards.
                (?<!==|!=|~=)         # We have special cases for these
                                      # operators so we want to make sure they
                                      # don't match here.

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?:                   # pre release
                    [-_\.]?
                    (a|b|c|rc|alpha|beta|pre|preview)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
        )
        """

    _regex = re.compile(r"^\s*" + _regex_str + r"\s*$", re.VERBOSE | re.IGNORECASE)

    _operators = {
        "~=": "compatible",
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
        "===": "arbitrary",
    }

    @_require_version_compare
    def _compare_compatible(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool

        # Compatible releases have an equivalent combination of >= and ==. That
        # is that ~=2.2 is equivalent to >=2.2,==2.*. This allows us to
        # implement this in terms of the other specifiers instead of
        # implementing it ourselves. The only thing we need to do is construct
        # the other specifiers.

        # We want everything but the last item in the version, but we want to
        # ignore post and dev releases and we want to treat the pre-release as
        # it's own separate segment.
        prefix = ".".join(
            list(
                itertools.takewhile(
                    lambda x: (not x.startswith("post") and not x.startswith("dev")),
                    _version_split(spec),
                )
            )[:-1]
        )

        # Add the prefix notation to the end of our string
        prefix += ".*"

        return self._get_operator(">=")(prospective, spec) and self._get_operator("==")(
            prospective, prefix
        )

    @_require_version_compare
    def _compare_equal(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool

        # We need special logic to handle prefix matching
        if spec.endswith(".*"):
            # In the case of prefix matching we want to ignore local segment.
            prospective = Version(prospective.public)
            # Split the spec out by dots, and pretend that there is an implicit
            # dot in between a release segment and a pre-release segment.
            split_spec = _version_split(spec[:-2])  # Remove the trailing .*

            # Split the prospective version out by dots, and pretend that there
            # is an implicit dot in between a release segment and a pre-release
            # segment.
            split_prospective = _version_split(str(prospective))

            # Shorten the prospective version to be the same length as the spec
            # so that we can determine if the specifier is a prefix of the
            # prospective version or not.
            shortened_prospective = split_prospective[: len(split_spec)]

            # Pad out our two sides with zeros so that they both equal the same
            # length.
            padded_spec, padded_prospective = _pad_version(
                split_spec, shortened_prospective
            )

            return padded_prospective == padded_spec
        else:
            # Convert our spec string into a Version
            spec_version = Version(spec)

            # If the specifier does not have a local segment, then we want to
            # act as if the prospective version also does not have a local
            # segment.
            if not spec_version.local:
                prospective = Version(prospective.public)

            return prospective == spec_version

    @_require_version_compare
    def _compare_not_equal(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool
        return not self._compare_equal(prospective, spec)

    @_require_version_compare
    def _compare_less_than_equal(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool

        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return Version(prospective.public) <= Version(spec)

    @_require_version_compare
    def _compare_greater_than_equal(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool

        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return Version(prospective.public) >= Version(spec)

    @_require_version_compare
    def _compare_less_than(self, prospective, spec_str):
        # type: (ParsedVersion, str) -> bool

        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec_str)

        # Check to see if the prospective version is less than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective < spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a pre-release version, that we do not accept pre-release
        # versions for the version mentioned in the specifier (e.g. <3.1 should
        # not match 3.1.dev0, but should match 3.0.dev0).
        if not spec.is_prerelease and prospective.is_prerelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # less than the spec version *and* it's not a pre-release of the same
        # version in the spec.
        return True

    @_require_version_compare
    def _compare_greater_than(self, prospective, spec_str):
        # type: (ParsedVersion, str) -> bool

        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec_str)

        # Check to see if the prospective version is greater than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective > spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a post-release version, that we do not accept
        # post-release versions for the version mentioned in the specifier
        # (e.g. >3.1 should not match 3.0.post0, but should match 3.2.post0).
        if not spec.is_postrelease and prospective.is_postrelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # Ensure that we do not allow a local version of the version mentioned
        # in the specifier, which is technically greater than, to match.
        if prospective.local is not None:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # greater than the spec version *and* it's not a pre-release of the
        # same version in the spec.
        return True

    def _compare_arbitrary(self, prospective, spec):
        # type: (Version, str) -> bool
        return str(prospective).lower() == str(spec).lower()

    @property
    def prereleases(self):
        # type: () -> bool

        # If there is an explicit prereleases set for this, then we'll just
        # blindly use that.
        if self._prereleases is not None:
            return self._prereleases

        # Look at all of our specifiers and determine if they are inclusive
        # operators, and if they are if they are including an explicit
        # prerelease.
        operator, version = self._spec
        if operator in ["==", ">=", "<=", "~=", "==="]:
            # The == specifier can include a trailing .*, if it does we
            # want to remove before parsing.
            if operator == "==" and version.endswith(".*"):
                version = version[:-2]

            # Parse the version, and if it is a pre-release than this
            # specifier allows pre-releases.
            if parse(version).is_prerelease:
                return True

        return False

    @prereleases.setter
    def prereleases(self, value):
        # type: (bool) -> None
        self._prereleases = value


_prefix_regex = re.compile(r"^([0-9]+)((?:a|b|c|rc)[0-9]+)$")


def _version_split(version):
    # type: (str) -> List[str]
    result = []  # type: List[str]
    for item in version.split("."):
        match = _prefix_regex.search(item)
        if match:
            result.extend(match.groups())
        else:
            result.append(item)
    return result


def _pad_version(left, right):
    # type: (List[str], List[str]) -> Tuple[List[str], List[str]]
    left_split, right_split = [], []

    # Get the release segment of our versions
    left_split.append(list(itertools.takewhile(lambda x: x.isdigit(), left)))
    right_split.append(list(itertools.takewhile(lambda x: x.isdigit(), right)))

    # Get the rest of our versions
    left_split.append(left[len(left_split[0]) :])
    right_split.append(right[len(right_split[0]) :])

    # Insert our padding
    left_split.insert(1, ["0"] * max(0, len(right_split[0]) - len(left_split[0])))
    right_split.insert(1, ["0"] * max(0, len(left_split[0]) - len(right_split[0])))

    return (list(itertools.chain(*left_split)), list(itertools.chain(*right_split)))


class SpecifierSet(BaseSpecifier):
    def __init__(self, specifiers="", prereleases=None):
        # type: (str, Optional[bool]) -> None

        # Split on , to break each individual specifier into it's own item, and
        # strip each item to remove leading/trailing whitespace.
        split_specifiers = [s.strip() for s in specifiers.split(",") if s.strip()]

        # Parsed each individual specifier, attempting first to make it a
        # Specifier and falling back to a LegacySpecifier.
        parsed = set()
        for specifier in split_specifiers:
            try:
                parsed.add(Specifier(specifier))
            except InvalidSpecifier:
                parsed.add(LegacySpecifier(specifier))

        # Turn our parsed specifiers into a frozen set and save them for later.
        self._specs = frozenset(parsed)

        # Store our prereleases value so we can use it later to determine if
        # we accept prereleases or not.
        self._prereleases = prereleases

    def __repr__(self):
        # type: () -> str
        pre = (
            ", prereleases={0!r}".format(self.prereleases)
            if self._prereleases is not None
            else ""
        )

        return "<SpecifierSet({0!r}{1})>".format(str(self), pre)

    def __str__(self):
        # type: () -> str
        return ",".join(sorted(str(s) for s in self._specs))

    def __hash__(self):
        # type: () -> int
        return hash(self._specs)

    def __and__(self, other):
        # type: (Union[SpecifierSet, str]) -> SpecifierSet
        if isinstance(other, string_types):
            other = SpecifierSet(other)
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        specifier = SpecifierSet()
        specifier._specs = frozenset(self._specs | other._specs)

        if self._prereleases is None and other._prereleases is not None:
            specifier._prereleases = other._prereleases
        elif self._prereleases is not None and other._prereleases is None:
            specifier._prereleases = self._prereleases
        elif self._prereleases == other._prereleases:
            specifier._prereleases = self._prereleases
        else:
            raise ValueError(
                "Cannot combine SpecifierSets with True and False prerelease "
                "overrides."
            )

        return specifier

    def __eq__(self, other):
        # type: (object) -> bool
        if isinstance(other, (string_types, _IndividualSpecifier)):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        return self._specs == other._specs

    def __ne__(self, other):
        # type: (object) -> bool
        if isinstance(other, (string_types, _IndividualSpecifier)):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        return self._specs != other._specs

    def __len__(self):
        # type: () -> int
        return len(self._specs)

    def __iter__(self):
        # type: () -> Iterator[FrozenSet[_IndividualSpecifier]]
        return iter(self._specs)

    @property
    def prereleases(self):
        # type: () -> Optional[bool]

        # If we have been given an explicit prerelease modifier, then we'll
        # pass that through here.
        if self._prereleases is not None:
            return self._prereleases

        # If we don't have any specifiers, and we don't have a forced value,
        # then we'll just return None since we don't know if this should have
        # pre-releases or not.
        if not self._specs:
            return None

        # Otherwise we'll see if any of the given specifiers accept
        # prereleases, if any of them do we'll return True, otherwise False.
        return any(s.prereleases for s in self._specs)

    @prereleases.setter
    def prereleases(self, value):
        # type: (bool) -> None
        self._prereleases = value

    def __contains__(self, item):
        # type: (Union[ParsedVersion, str]) -> bool
        return self.contains(item)

    def contains(self, item, prereleases=None):
        # type: (Union[ParsedVersion, str], Optional[bool]) -> bool

        # Ensure that our item is a Version or LegacyVersion instance.
        if not isinstance(item, (LegacyVersion, Version)):
            item = parse(item)

        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # We can determine if we're going to allow pre-releases by looking to
        # see if any of the underlying items supports them. If none of them do
        # and this item is a pre-release then we do not allow it and we can
        # short circuit that here.
        # Note: This means that 1.0.dev1 would not be contained in something
        #       like >=1.0.devabc however it would be in >=1.0.debabc,>0.0.dev0
        if not prereleases and item.is_prerelease:
            return False

        # We simply dispatch to the underlying specs here to make sure that the
        # given version is contained within all of them.
        # Note: This use of all() here means that an empty set of specifiers
        #       will always return True, this is an explicit design decision.
        return all(s.contains(item, prereleases=prereleases) for s in self._specs)

    def filter(
        self,
        iterable,  # type: Iterable[Union[ParsedVersion, str]]
        prereleases=None,  # type: Optional[bool]
    ):
        # type: (...) -> Iterable[Union[ParsedVersion, str]]

        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # If we have any specifiers, then we want to wrap our iterable in the
        # filter method for each one, this will act as a logical AND amongst
        # each specifier.
        if self._specs:
            for spec in self._specs:
                iterable = spec.filter(iterable, prereleases=bool(prereleases))
            return iterable
        # If we do not have any specifiers, then we need to have a rough filter
        # which will filter out any pre-releases, unless there are no final
        # releases, and which will filter out LegacyVersion in general.
        else:
            filtered = []  # type: List[Union[ParsedVersion, str]]
            found_prereleases = []  # type: List[Union[ParsedVersion, str]]

            for item in iterable:
                # Ensure that we some kind of Version class for this item.
                if not isinstance(item, (LegacyVersion, Version)):
                    parsed_version = parse(item)
                else:
                    parsed_version = item

                # Filter out any item which is parsed as a LegacyVersion
                if isinstance(parsed_version, LegacyVersion):
                    continue

                # Store any item which is a pre-release for later unless we've
                # already found a final version or we are accepting prereleases
                if parsed_version.is_prerelease and not prereleases:
                    if not filtered:
                        found_prereleases.append(item)
                else:
                    filtered.append(item)

            # If we've found no items except for pre-releases, then we'll go
            # ahead and use the pre-releases
            if not filtered and found_prereleases and prereleases is None:
                return found_prereleases

            return filtered
"""----------------------------------------------------------------------------------"""
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

import abc
import functools
import itertools
import re

from ._compat import string_types, with_metaclass
from ._typing import TYPE_CHECKING
from .utils import canonicalize_version
from .version import Version, LegacyVersion, parse

if TYPE_CHECKING:  # pragma: no cover
    from typing import (
        List,
        Dict,
        Union,
        Iterable,
        Iterator,
        Optional,
        Callable,
        Tuple,
        FrozenSet,
    )

    ParsedVersion = Union[Version, LegacyVersion]
    UnparsedVersion = Union[Version, LegacyVersion, str]
    CallableOperator = Callable[[ParsedVersion, str], bool]


class InvalidSpecifier(ValueError):
    """
    An invalid specifier was found, users should refer to PEP 440.
    """


class BaseSpecifier(with_metaclass(abc.ABCMeta, object)):  # type: ignore
    @abc.abstractmethod
    def __str__(self):
        # type: () -> str
        """
        Returns the str representation of this Specifier like object. This
        should be representative of the Specifier itself.
        """

    @abc.abstractmethod
    def __hash__(self):
        # type: () -> int
        """
        Returns a hash value for this Specifier like object.
        """

    @abc.abstractmethod
    def __eq__(self, other):
        # type: (object) -> bool
        """
        Returns a boolean representing whether or not the two Specifier like
        objects are equal.
        """

    @abc.abstractmethod
    def __ne__(self, other):
        # type: (object) -> bool
        """
        Returns a boolean representing whether or not the two Specifier like
        objects are not equal.
        """

    @abc.abstractproperty
    def prereleases(self):
        # type: () -> Optional[bool]
        """
        Returns whether or not pre-releases as a whole are allowed by this
        specifier.
        """

    @prereleases.setter
    def prereleases(self, value):
        # type: (bool) -> None
        """
        Sets whether or not pre-releases as a whole are allowed by this
        specifier.
        """

    @abc.abstractmethod
    def contains(self, item, prereleases=None):
        # type: (str, Optional[bool]) -> bool
        """
        Determines if the given item is contained within this specifier.
        """

    @abc.abstractmethod
    def filter(self, iterable, prereleases=None):
        # type: (Iterable[UnparsedVersion], Optional[bool]) -> Iterable[UnparsedVersion]
        """
        Takes an iterable of items and filters them so that only items which
        are contained within this specifier are allowed in it.
        """


class _IndividualSpecifier(BaseSpecifier):

    _operators = {}  # type: Dict[str, str]

    def __init__(self, spec="", prereleases=None):
        # type: (str, Optional[bool]) -> None
        match = self._regex.search(spec)
        if not match:
            raise InvalidSpecifier("Invalid specifier: '{0}'".format(spec))

        self._spec = (
            match.group("operator").strip(),
            match.group("version").strip(),
        )  # type: Tuple[str, str]

        # Store whether or not this Specifier should accept prereleases
        self._prereleases = prereleases

    def __repr__(self):
        # type: () -> str
        pre = (
            ", prereleases={0!r}".format(self.prereleases)
            if self._prereleases is not None
            else ""
        )

        return "<{0}({1!r}{2})>".format(self.__class__.__name__, str(self), pre)

    def __str__(self):
        # type: () -> str
        return "{0}{1}".format(*self._spec)

    @property
    def _canonical_spec(self):
        # type: () -> Tuple[str, Union[Version, str]]
        return self._spec[0], canonicalize_version(self._spec[1])

    def __hash__(self):
        # type: () -> int
        return hash(self._canonical_spec)

    def __eq__(self, other):
        # type: (object) -> bool
        if isinstance(other, string_types):
            try:
                other = self.__class__(str(other))
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._canonical_spec == other._canonical_spec

    def __ne__(self, other):
        # type: (object) -> bool
        if isinstance(other, string_types):
            try:
                other = self.__class__(str(other))
            except InvalidSpecifier:
                return NotImplemented
        elif not isinstance(other, self.__class__):
            return NotImplemented

        return self._spec != other._spec

    def _get_operator(self, op):
        # type: (str) -> CallableOperator
        operator_callable = getattr(
            self, "_compare_{0}".format(self._operators[op])
        )  # type: CallableOperator
        return operator_callable

    def _coerce_version(self, version):
        # type: (UnparsedVersion) -> ParsedVersion
        if not isinstance(version, (LegacyVersion, Version)):
            version = parse(version)
        return version

    @property
    def operator(self):
        # type: () -> str
        return self._spec[0]

    @property
    def version(self):
        # type: () -> str
        return self._spec[1]

    @property
    def prereleases(self):
        # type: () -> Optional[bool]
        return self._prereleases

    @prereleases.setter
    def prereleases(self, value):
        # type: (bool) -> None
        self._prereleases = value

    def __contains__(self, item):
        # type: (str) -> bool
        return self.contains(item)

    def contains(self, item, prereleases=None):
        # type: (UnparsedVersion, Optional[bool]) -> bool

        # Determine if prereleases are to be allowed or not.
        if prereleases is None:
            prereleases = self.prereleases

        # Normalize item to a Version or LegacyVersion, this allows us to have
        # a shortcut for ``"2.0" in Specifier(">=2")
        normalized_item = self._coerce_version(item)

        # Determine if we should be supporting prereleases in this specifier
        # or not, if we do not support prereleases than we can short circuit
        # logic if this version is a prereleases.
        if normalized_item.is_prerelease and not prereleases:
            return False

        # Actually do the comparison to determine if this item is contained
        # within this Specifier or not.
        operator_callable = self._get_operator(self.operator)  # type: CallableOperator
        return operator_callable(normalized_item, self.version)

    def filter(self, iterable, prereleases=None):
        # type: (Iterable[UnparsedVersion], Optional[bool]) -> Iterable[UnparsedVersion]

        yielded = False
        found_prereleases = []

        kw = {"prereleases": prereleases if prereleases is not None else True}

        # Attempt to iterate over all the values in the iterable and if any of
        # them match, yield them.
        for version in iterable:
            parsed_version = self._coerce_version(version)

            if self.contains(parsed_version, **kw):
                # If our version is a prerelease, and we were not set to allow
                # prereleases, then we'll store it for later incase nothing
                # else matches this specifier.
                if parsed_version.is_prerelease and not (
                    prereleases or self.prereleases
                ):
                    found_prereleases.append(version)
                # Either this is not a prerelease, or we should have been
                # accepting prereleases from the beginning.
                else:
                    yielded = True
                    yield version

        # Now that we've iterated over everything, determine if we've yielded
        # any values, and if we have not and we have any prereleases stored up
        # then we will go ahead and yield the prereleases.
        if not yielded and found_prereleases:
            for version in found_prereleases:
                yield version


class LegacySpecifier(_IndividualSpecifier):

    _regex_str = r"""
        (?P<operator>(==|!=|<=|>=|<|>))
        \s*
        (?P<version>
            [^,;\s)]* # Since this is a "legacy" specifier, and the version
                      # string can be just about anything, we match everything
                      # except for whitespace, a semi-colon for marker support,
                      # a closing paren since versions can be enclosed in
                      # them, and a comma since it's a version separator.
        )
        """

    _regex = re.compile(r"^\s*" + _regex_str + r"\s*$", re.VERBOSE | re.IGNORECASE)

    _operators = {
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
    }

    def _coerce_version(self, version):
        # type: (Union[ParsedVersion, str]) -> LegacyVersion
        if not isinstance(version, LegacyVersion):
            version = LegacyVersion(str(version))
        return version

    def _compare_equal(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective == self._coerce_version(spec)

    def _compare_not_equal(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective != self._coerce_version(spec)

    def _compare_less_than_equal(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective <= self._coerce_version(spec)

    def _compare_greater_than_equal(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective >= self._coerce_version(spec)

    def _compare_less_than(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective < self._coerce_version(spec)

    def _compare_greater_than(self, prospective, spec):
        # type: (LegacyVersion, str) -> bool
        return prospective > self._coerce_version(spec)


def _require_version_compare(
    fn  # type: (Callable[[Specifier, ParsedVersion, str], bool])
):
    # type: (...) -> Callable[[Specifier, ParsedVersion, str], bool]
    @functools.wraps(fn)
    def wrapped(self, prospective, spec):
        # type: (Specifier, ParsedVersion, str) -> bool
        if not isinstance(prospective, Version):
            return False
        return fn(self, prospective, spec)

    return wrapped


class Specifier(_IndividualSpecifier):

    _regex_str = r"""
        (?P<operator>(~=|==|!=|<=|>=|<|>|===))
        (?P<version>
            (?:
                # The identity operators allow for an escape hatch that will
                # do an exact string match of the version you wish to install.
                # This will not be parsed by PEP 440 and we cannot determine
                # any semantic meaning from it. This operator is discouraged
                # but included entirely as an escape hatch.
                (?<====)  # Only match for the identity operator
                \s*
                [^\s]*    # We just match everything, except for whitespace
                          # since we are only testing for strict identity.
            )
            |
            (?:
                # The (non)equality operators allow for wild card and local
                # versions to be specified so we have to define these two
                # operators separately to enable that.
                (?<===|!=)            # Only match for equals and not equals

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?:                   # pre release
                    [-_\.]?
                    (a|b|c|rc|alpha|beta|pre|preview)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?

                # You cannot use a wild card and a dev or local version
                # together so group them with a | and make them optional.
                (?:
                    (?:[-_\.]?dev[-_\.]?[0-9]*)?         # dev release
                    (?:\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*)? # local
                    |
                    \.\*  # Wild card syntax of .*
                )?
            )
            |
            (?:
                # The compatible operator requires at least two digits in the
                # release segment.
                (?<=~=)               # Only match for the compatible operator

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)+   # release  (We have a + instead of a *)
                (?:                   # pre release
                    [-_\.]?
                    (a|b|c|rc|alpha|beta|pre|preview)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
            |
            (?:
                # All other operators only allow a sub set of what the
                # (non)equality operators do. Specifically they do not allow
                # local versions to be specified nor do they allow the prefix
                # matching wild cards.
                (?<!==|!=|~=)         # We have special cases for these
                                      # operators so we want to make sure they
                                      # don't match here.

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?:                   # pre release
                    [-_\.]?
                    (a|b|c|rc|alpha|beta|pre|preview)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
        )
        """

    _regex = re.compile(r"^\s*" + _regex_str + r"\s*$", re.VERBOSE | re.IGNORECASE)

    _operators = {
        "~=": "compatible",
        "==": "equal",
        "!=": "not_equal",
        "<=": "less_than_equal",
        ">=": "greater_than_equal",
        "<": "less_than",
        ">": "greater_than",
        "===": "arbitrary",
    }

    @_require_version_compare
    def _compare_compatible(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool

        # Compatible releases have an equivalent combination of >= and ==. That
        # is that ~=2.2 is equivalent to >=2.2,==2.*. This allows us to
        # implement this in terms of the other specifiers instead of
        # implementing it ourselves. The only thing we need to do is construct
        # the other specifiers.

        # We want everything but the last item in the version, but we want to
        # ignore post and dev releases and we want to treat the pre-release as
        # it's own separate segment.
        prefix = ".".join(
            list(
                itertools.takewhile(
                    lambda x: (not x.startswith("post") and not x.startswith("dev")),
                    _version_split(spec),
                )
            )[:-1]
        )

        # Add the prefix notation to the end of our string
        prefix += ".*"

        return self._get_operator(">=")(prospective, spec) and self._get_operator("==")(
            prospective, prefix
        )

    @_require_version_compare
    def _compare_equal(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool

        # We need special logic to handle prefix matching
        if spec.endswith(".*"):
            # In the case of prefix matching we want to ignore local segment.
            prospective = Version(prospective.public)
            # Split the spec out by dots, and pretend that there is an implicit
            # dot in between a release segment and a pre-release segment.
            split_spec = _version_split(spec[:-2])  # Remove the trailing .*

            # Split the prospective version out by dots, and pretend that there
            # is an implicit dot in between a release segment and a pre-release
            # segment.
            split_prospective = _version_split(str(prospective))

            # Shorten the prospective version to be the same length as the spec
            # so that we can determine if the specifier is a prefix of the
            # prospective version or not.
            shortened_prospective = split_prospective[: len(split_spec)]

            # Pad out our two sides with zeros so that they both equal the same
            # length.
            padded_spec, padded_prospective = _pad_version(
                split_spec, shortened_prospective
            )

            return padded_prospective == padded_spec
        else:
            # Convert our spec string into a Version
            spec_version = Version(spec)

            # If the specifier does not have a local segment, then we want to
            # act as if the prospective version also does not have a local
            # segment.
            if not spec_version.local:
                prospective = Version(prospective.public)

            return prospective == spec_version

    @_require_version_compare
    def _compare_not_equal(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool
        return not self._compare_equal(prospective, spec)

    @_require_version_compare
    def _compare_less_than_equal(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool

        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return Version(prospective.public) <= Version(spec)

    @_require_version_compare
    def _compare_greater_than_equal(self, prospective, spec):
        # type: (ParsedVersion, str) -> bool

        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return Version(prospective.public) >= Version(spec)

    @_require_version_compare
    def _compare_less_than(self, prospective, spec_str):
        # type: (ParsedVersion, str) -> bool

        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec_str)

        # Check to see if the prospective version is less than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective < spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a pre-release version, that we do not accept pre-release
        # versions for the version mentioned in the specifier (e.g. <3.1 should
        # not match 3.1.dev0, but should match 3.0.dev0).
        if not spec.is_prerelease and prospective.is_prerelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # less than the spec version *and* it's not a pre-release of the same
        # version in the spec.
        return True

    @_require_version_compare
    def _compare_greater_than(self, prospective, spec_str):
        # type: (ParsedVersion, str) -> bool

        # Convert our spec to a Version instance, since we'll want to work with
        # it as a version.
        spec = Version(spec_str)

        # Check to see if the prospective version is greater than the spec
        # version. If it's not we can short circuit and just return False now
        # instead of doing extra unneeded work.
        if not prospective > spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a post-release version, that we do not accept
        # post-release versions for the version mentioned in the specifier
        # (e.g. >3.1 should not match 3.0.post0, but should match 3.2.post0).
        if not spec.is_postrelease and prospective.is_postrelease:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # Ensure that we do not allow a local version of the version mentioned
        # in the specifier, which is technically greater than, to match.
        if prospective.local is not None:
            if Version(prospective.base_version) == Version(spec.base_version):
                return False

        # If we've gotten to here, it means that prospective version is both
        # greater than the spec version *and* it's not a pre-release of the
        # same version in the spec.
        return True

    def _compare_arbitrary(self, prospective, spec):
        # type: (Version, str) -> bool
        return str(prospective).lower() == str(spec).lower()

    @property
    def prereleases(self):
        # type: () -> bool

        # If there is an explicit prereleases set for this, then we'll just
        # blindly use that.
        if self._prereleases is not None:
            return self._prereleases

        # Look at all of our specifiers and determine if they are inclusive
        # operators, and if they are if they are including an explicit
        # prerelease.
        operator, version = self._spec
        if operator in ["==", ">=", "<=", "~=", "==="]:
            # The == specifier can include a trailing .*, if it does we
            # want to remove before parsing.
            if operator == "==" and version.endswith(".*"):
                version = version[:-2]

            # Parse the version, and if it is a pre-release than this
            # specifier allows pre-releases.
            if parse(version).is_prerelease:
                return True

        return False

    @prereleases.setter
    def prereleases(self, value):
        # type: (bool) -> None
        self._prereleases = value


_prefix_regex = re.compile(r"^([0-9]+)((?:a|b|c|rc)[0-9]+)$")


def _version_split(version):
    # type: (str) -> List[str]
    result = []  # type: List[str]
    for item in version.split("."):
        match = _prefix_regex.search(item)
        if match:
            result.extend(match.groups())
        else:
            result.append(item)
    return result


def _pad_version(left, right):
    # type: (List[str], List[str]) -> Tuple[List[str], List[str]]
    left_split, right_split = [], []

    # Get the release segment of our versions
    left_split.append(list(itertools.takewhile(lambda x: x.isdigit(), left)))
    right_split.append(list(itertools.takewhile(lambda x: x.isdigit(), right)))

    # Get the rest of our versions
    left_split.append(left[len(left_split[0]) :])
    right_split.append(right[len(right_split[0]) :])

    # Insert our padding
    left_split.insert(1, ["0"] * max(0, len(right_split[0]) - len(left_split[0])))
    right_split.insert(1, ["0"] * max(0, len(left_split[0]) - len(right_split[0])))

    return (list(itertools.chain(*left_split)), list(itertools.chain(*right_split)))


class SpecifierSet(BaseSpecifier):
    def __init__(self, specifiers="", prereleases=None):
        # type: (str, Optional[bool]) -> None

        # Split on , to break each individual specifier into it's own item, and
        # strip each item to remove leading/trailing whitespace.
        split_specifiers = [s.strip() for s in specifiers.split(",") if s.strip()]

        # Parsed each individual specifier, attempting first to make it a
        # Specifier and falling back to a LegacySpecifier.
        parsed = set()
        for specifier in split_specifiers:
            try:
                parsed.add(Specifier(specifier))
            except InvalidSpecifier:
                parsed.add(LegacySpecifier(specifier))

        # Turn our parsed specifiers into a frozen set and save them for later.
        self._specs = frozenset(parsed)

        # Store our prereleases value so we can use it later to determine if
        # we accept prereleases or not.
        self._prereleases = prereleases

    def __repr__(self):
        # type: () -> str
        pre = (
            ", prereleases={0!r}".format(self.prereleases)
            if self._prereleases is not None
            else ""
        )

        return "<SpecifierSet({0!r}{1})>".format(str(self), pre)

    def __str__(self):
        # type: () -> str
        return ",".join(sorted(str(s) for s in self._specs))

    def __hash__(self):
        # type: () -> int
        return hash(self._specs)

    def __and__(self, other):
        # type: (Union[SpecifierSet, str]) -> SpecifierSet
        if isinstance(other, string_types):
            other = SpecifierSet(other)
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        specifier = SpecifierSet()
        specifier._specs = frozenset(self._specs | other._specs)

        if self._prereleases is None and other._prereleases is not None:
            specifier._prereleases = other._prereleases
        elif self._prereleases is not None and other._prereleases is None:
            specifier._prereleases = self._prereleases
        elif self._prereleases == other._prereleases:
            specifier._prereleases = self._prereleases
        else:
            raise ValueError(
                "Cannot combine SpecifierSets with True and False prerelease "
                "overrides."
            )

        return specifier

    def __eq__(self, other):
        # type: (object) -> bool
        if isinstance(other, (string_types, _IndividualSpecifier)):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        return self._specs == other._specs

    def __ne__(self, other):
        # type: (object) -> bool
        if isinstance(other, (string_types, _IndividualSpecifier)):
            other = SpecifierSet(str(other))
        elif not isinstance(other, SpecifierSet):
            return NotImplemented

        return self._specs != other._specs

    def __len__(self):
        # type: () -> int
        return len(self._specs)

    def __iter__(self):
        # type: () -> Iterator[FrozenSet[_IndividualSpecifier]]
        return iter(self._specs)

    @property
    def prereleases(self):
        # type: () -> Optional[bool]

        # If we have been given an explicit prerelease modifier, then we'll
        # pass that through here.
        if self._prereleases is not None:
            return self._prereleases

        # If we don't have any specifiers, and we don't have a forced value,
        # then we'll just return None since we don't know if this should have
        # pre-releases or not.
        if not self._specs:
            return None

        # Otherwise we'll see if any of the given specifiers accept
        # prereleases, if any of them do we'll return True, otherwise False.
        return any(s.prereleases for s in self._specs)

    @prereleases.setter
    def prereleases(self, value):
        # type: (bool) -> None
        self._prereleases = value

    def __contains__(self, item):
        # type: (Union[ParsedVersion, str]) -> bool
        return self.contains(item)

    def contains(self, item, prereleases=None):
        # type: (Union[ParsedVersion, str], Optional[bool]) -> bool

        # Ensure that our item is a Version or LegacyVersion instance.
        if not isinstance(item, (LegacyVersion, Version)):
            item = parse(item)

        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # We can determine if we're going to allow pre-releases by looking to
        # see if any of the underlying items supports them. If none of them do
        # and this item is a pre-release then we do not allow it and we can
        # short circuit that here.
        # Note: This means that 1.0.dev1 would not be contained in something
        #       like >=1.0.devabc however it would be in >=1.0.debabc,>0.0.dev0
        if not prereleases and item.is_prerelease:
            return False

        # We simply dispatch to the underlying specs here to make sure that the
        # given version is contained within all of them.
        # Note: This use of all() here means that an empty set of specifiers
        #       will always return True, this is an explicit design decision.
        return all(s.contains(item, prereleases=prereleases) for s in self._specs)

    def filter(
        self,
        iterable,  # type: Iterable[Union[ParsedVersion, str]]
        prereleases=None,  # type: Optional[bool]
    ):
        # type: (...) -> Iterable[Union[ParsedVersion, str]]

        # Determine if we're forcing a prerelease or not, if we're not forcing
        # one for this particular filter call, then we'll use whatever the
        # SpecifierSet thinks for whether or not we should support prereleases.
        if prereleases is None:
            prereleases = self.prereleases

        # If we have any specifiers, then we want to wrap our iterable in the
        # filter method for each one, this will act as a logical AND amongst
        # each specifier.
        if self._specs:
            for spec in self._specs:
                iterable = spec.filter(iterable, prereleases=bool(prereleases))
            return iterable
        # If we do not have any specifiers, then we need to have a rough filter
        # which will filter out any pre-releases, unless there are no final
        # releases, and which will filter out LegacyVersion in general.
        else:
            filtered = []  # type: List[Union[ParsedVersion, str]]
            found_prereleases = []  # type: List[Union[ParsedVersion, str]]

            for item in iterable:
                # Ensure that we some kind of Version class for this item.
                if not isinstance(item, (LegacyVersion, Version)):
                    parsed_version = parse(item)
                else:
                    parsed_version = item

                # Filter out any item which is parsed as a LegacyVersion
                if isinstance(parsed_version, LegacyVersion):
                    continue

                # Store any item which is a pre-release for later unless we've
                # already found a final version or we are accepting prereleases
                if parsed_version.is_prerelease and not prereleases:
                    if not filtered:
                        found_prereleases.append(item)
                else:
                    filtered.append(item)

            # If we've found no items except for pre-releases, then we'll go
            # ahead and use the pre-releases
            if not filtered and found_prereleases and prereleases is None:
                return found_prereleases

            return filtered
"""------------------------------------------------------------------"""
#!/usr/bin/env python
#
# Copyright (c) 2009 Google Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#    * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Does google-lint on c++ files.

The goal of this script is to identify places in the code that *may*
be in non-compliance with google style.  It does not attempt to fix
up these problems -- the point is to educate.  It does also not
attempt to find all problems, or to ensure that everything it does
find is legitimately a problem.

In particular, we can get very confused by /* and // inside strings!
We do a small hack, which is to ignore //'s with "'s after them on the
same line, but it is far from perfect (in either direction).
"""

import codecs
import copy
import getopt
import math  # for log
import os
import re
import sre_compile
import string
import sys
import unicodedata

try:
  xrange          # Python 2
except NameError:
  xrange = range  # Python 3


_USAGE = """
Syntax: cpplint.py [--verbose=#] [--output=vs7] [--filter=-x,+y,...]
                   [--counting=total|toplevel|detailed] [--root=subdir]
                   [--linelength=digits] [--headers=x,y,...]
                   [--quiet]
        <file> [file] ...

  The style guidelines this tries to follow are those in
    https://google-styleguide.googlecode.com/svn/trunk/cppguide.xml

  Every problem is given a confidence score from 1-5, with 5 meaning we are
  certain of the problem, and 1 meaning it could be a legitimate construct.
  This will miss some errors, and is not a substitute for a code review.

  To suppress false-positive errors of a certain category, add a
  'NOLINT(category)' comment to the line.  NOLINT or NOLINT(*)
  suppresses errors of all categories on that line.

  The files passed in will be linted; at least one file must be provided.
  Default linted extensions are .cc, .cpp, .cu, .cuh and .h.  Change the
  extensions with the --extensions flag.

  Flags:

    output=vs7
      By default, the output is formatted to ease emacs parsing.  Visual Studio
      compatible output (vs7) may also be used.  Other formats are unsupported.

    verbose=#
      Specify a number 0-5 to restrict errors to certain verbosity levels.

    quiet
      Don't print anything if no errors are found.

    filter=-x,+y,...
      Specify a comma-separated list of category-filters to apply: only
      error messages whose category names pass the filters will be printed.
      (Category names are printed with the message and look like
      "[whitespace/indent]".)  Filters are evaluated left to right.
      "-FOO" and "FOO" means "do not print categories that start with FOO".
      "+FOO" means "do print categories that start with FOO".

      Examples: --filter=-whitespace,+whitespace/braces
                --filter=whitespace,runtime/printf,+runtime/printf_format
                --filter=-,+build/include_what_you_use

      To see a list of all the categories used in cpplint, pass no arg:
         --filter=

    counting=total|toplevel|detailed
      The total number of errors found is always printed. If
      'toplevel' is provided, then the count of errors in each of
      the top-level categories like 'build' and 'whitespace' will
      also be printed. If 'detailed' is provided, then a count
      is provided for each category like 'build/class'.

    root=subdir
      The root directory used for deriving header guard CPP variable.
      By default, the header guard CPP variable is calculated as the relative
      path to the directory that contains .git, .hg, or .svn.  When this flag
      is specified, the relative path is calculated from the specified
      directory. If the specified directory does not exist, this flag is
      ignored.

      Examples:
        Assuming that top/src/.git exists (and cwd=top/src), the header guard
        CPP variables for top/src/chrome/browser/ui/browser.h are:

        No flag => CHROME_BROWSER_UI_BROWSER_H_
        --root=chrome => BROWSER_UI_BROWSER_H_
        --root=chrome/browser => UI_BROWSER_H_
        --root=.. => SRC_CHROME_BROWSER_UI_BROWSER_H_

    linelength=digits
      This is the allowed line length for the project. The default value is
      80 characters.

      Examples:
        --linelength=120

    extensions=extension,extension,...
      The allowed file extensions that cpplint will check

      Examples:
        --extensions=hpp,cpp

    headers=x,y,...
      The header extensions that cpplint will treat as .h in checks. Values are
      automatically added to --extensions list.

      Examples:
        --headers=hpp,hxx
        --headers=hpp

    cpplint.py supports per-directory configurations specified in CPPLINT.cfg
    files. CPPLINT.cfg file can contain a number of key=value pairs.
    Currently the following options are supported:

      set noparent
      filter=+filter1,-filter2,...
      exclude_files=regex
      linelength=80
      root=subdir
      headers=x,y,...

    "set noparent" option prevents cpplint from traversing directory tree
    upwards looking for more .cfg files in parent directories. This option
    is usually placed in the top-level project directory.

    The "filter" option is similar in function to --filter flag. It specifies
    message filters in addition to the |_DEFAULT_FILTERS| and those specified
    through --filter command-line flag.

    "exclude_files" allows to specify a regular expression to be matched against
    a file name. If the expression matches, the file is skipped and not run
    through liner.

    "linelength" allows to specify the allowed line length for the project.

    The "root" option is similar in function to the --root flag (see example
    above). Paths are relative to the directory of the CPPLINT.cfg.

    The "headers" option is similar in function to the --headers flag
    (see example above).

    CPPLINT.cfg has an effect on files in the same directory and all
    sub-directories, unless overridden by a nested configuration file.

      Example file:
        filter=-build/include_order,+build/include_alpha
        exclude_files=.*\.cc

    The above example disables build/include_order warning and enables
    build/include_alpha as well as excludes all .cc from being
    processed by linter, in the current directory (where the .cfg
    file is located) and all sub-directories.
"""

# We categorize each error message we print.  Here are the categories.
# We want an explicit list so we can list them all in cpplint --filter=.
# If you add a new error message with a new category, add it to the list
# here!  cpplint_unittest.py should tell you if you forget to do this.
_ERROR_CATEGORIES = [
    'build/class',
    'build/c++11',
    'build/c++14',
    'build/c++tr1',
    'build/deprecated',
    'build/endif_comment',
    'build/explicit_make_pair',
    'build/forward_decl',
    'build/header_guard',
    'build/include',
    'build/include_alpha',
    'build/include_order',
    'build/include_what_you_use',
    'build/namespaces',
    'build/printf_format',
    'build/storage_class',
    'legal/copyright',
    'readability/alt_tokens',
    'readability/braces',
    'readability/casting',
    'readability/check',
    'readability/constructors',
    'readability/fn_size',
    'readability/inheritance',
    'readability/multiline_comment',
    'readability/multiline_string',
    'readability/namespace',
    'readability/nolint',
    'readability/nul',
    'readability/strings',
    'readability/todo',
    'readability/utf8',
    'runtime/arrays',
    'runtime/casting',
    'runtime/explicit',
    'runtime/int',
    'runtime/init',
    'runtime/invalid_increment',
    'runtime/member_string_references',
    'runtime/memset',
    'runtime/indentation_namespace',
    'runtime/operator',
    'runtime/printf',
    'runtime/printf_format',
    'runtime/references',
    'runtime/string',
    'runtime/threadsafe_fn',
    'runtime/vlog',
    'whitespace/blank_line',
    'whitespace/braces',
    'whitespace/comma',
    'whitespace/comments',
    'whitespace/empty_conditional_body',
    'whitespace/empty_if_body',
    'whitespace/empty_loop_body',
    'whitespace/end_of_line',
    'whitespace/ending_newline',
    'whitespace/forcolon',
    'whitespace/indent',
    'whitespace/line_length',
    'whitespace/newline',
    'whitespace/operators',
    'whitespace/parens',
    'whitespace/semicolon',
    'whitespace/tab',
    'whitespace/todo',
    ]

# These error categories are no longer enforced by cpplint, but for backwards-
# compatibility they may still appear in NOLINT comments.
_LEGACY_ERROR_CATEGORIES = [
    'readability/streams',
    'readability/function',
    ]

# The default state of the category filter. This is overridden by the --filter=
# flag. By default all errors are on, so only add here categories that should be
# off by default (i.e., categories that must be enabled by the --filter= flags).
# All entries here should start with a '-' or '+', as in the --filter= flag.
_DEFAULT_FILTERS = ['-build/include_alpha']

# The default list of categories suppressed for C (not C++) files.
_DEFAULT_C_SUPPRESSED_CATEGORIES = [
    'readability/casting',
    ]

# The default list of categories suppressed for Linux Kernel files.
_DEFAULT_KERNEL_SUPPRESSED_CATEGORIES = [
    'whitespace/tab',
    ]

# We used to check for high-bit characters, but after much discussion we
# decided those were OK, as long as they were in UTF-8 and didn't represent
# hard-coded international strings, which belong in a separate i18n file.

# C++ headers
_CPP_HEADERS = frozenset([
    # Legacy
    'algobase.h',
    'algo.h',
    'alloc.h',
    'builtinbuf.h',
    'bvector.h',
    'complex.h',
    'defalloc.h',
    'deque.h',
    'editbuf.h',
    'fstream.h',
    'function.h',
    'hash_map',
    'hash_map.h',
    'hash_set',
    'hash_set.h',
    'hashtable.h',
    'heap.h',
    'indstream.h',
    'iomanip.h',
    'iostream.h',
    'istream.h',
    'iterator.h',
    'list.h',
    'map.h',
    'multimap.h',
    'multiset.h',
    'ostream.h',
    'pair.h',
    'parsestream.h',
    'pfstream.h',
    'procbuf.h',
    'pthread_alloc',
    'pthread_alloc.h',
    'rope',
    'rope.h',
    'ropeimpl.h',
    'set.h',
    'slist',
    'slist.h',
    'stack.h',
    'stdiostream.h',
    'stl_alloc.h',
    'stl_relops.h',
    'streambuf.h',
    'stream.h',
    'strfile.h',
    'strstream.h',
    'tempbuf.h',
    'tree.h',
    'type_traits.h',
    'vector.h',
    # 17.6.1.2 C++ library headers
    'algorithm',
    'array',
    'atomic',
    'bitset',
    'chrono',
    'codecvt',
    'complex',
    'condition_variable',
    'deque',
    'exception',
    'forward_list',
    'fstream',
    'functional',
    'future',
    'initializer_list',
    'iomanip',
    'ios',
    'iosfwd',
    'iostream',
    'istream',
    'iterator',
    'limits',
    'list',
    'locale',
    'map',
    'memory',
    'mutex',
    'new',
    'numeric',
    'ostream',
    'queue',
    'random',
    'ratio',
    'regex',
    'scoped_allocator',
    'set',
    'sstream',
    'stack',
    'stdexcept',
    'streambuf',
    'string',
    'strstream',
    'system_error',
    'thread',
    'tuple',
    'typeindex',
    'typeinfo',
    'type_traits',
    'unordered_map',
    'unordered_set',
    'utility',
    'valarray',
    'vector',
    # 17.6.1.2 C++ headers for C library facilities
    'cassert',
    'ccomplex',
    'cctype',
    'cerrno',
    'cfenv',
    'cfloat',
    'cinttypes',
    'ciso646',
    'climits',
    'clocale',
    'cmath',
    'csetjmp',
    'csignal',
    'cstdalign',
    'cstdarg',
    'cstdbool',
    'cstddef',
    'cstdint',
    'cstdio',
    'cstdlib',
    'cstring',
    'ctgmath',
    'ctime',
    'cuchar',
    'cwchar',
    'cwctype',
    ])

# Type names
_TYPES = re.compile(
    r'^(?:'
    # [dcl.type.simple]
    r'(char(16_t|32_t)?)|wchar_t|'
    r'bool|short|int|long|signed|unsigned|float|double|'
    # [support.types]
    r'(ptrdiff_t|size_t|max_align_t|nullptr_t)|'
    # [cstdint.syn]
    r'(u?int(_fast|_least)?(8|16|32|64)_t)|'
    r'(u?int(max|ptr)_t)|'
    r')$')


# These headers are excluded from [build/include] and [build/include_order]
# checks:
# - Anything not following google file name conventions (containing an
#   uppercase character, such as Python.h or nsStringAPI.h, for example).
# - Lua headers.
_THIRD_PARTY_HEADERS_PATTERN = re.compile(
    r'^(?:[^/]*[A-Z][^/]*\.h|lua\.h|lauxlib\.h|lualib\.h)$')

# Pattern for matching FileInfo.BaseName() against test file name
_TEST_FILE_SUFFIX = r'(_test|_unittest|_regtest)$'

# Pattern that matches only complete whitespace, possibly across multiple lines.
_EMPTY_CONDITIONAL_BODY_PATTERN = re.compile(r'^\s*$', re.DOTALL)

# Assertion macros.  These are defined in base/logging.h and
# testing/base/public/gunit.h.
_CHECK_MACROS = [
    'DCHECK', 'CHECK',
    'EXPECT_TRUE', 'ASSERT_TRUE',
    'EXPECT_FALSE', 'ASSERT_FALSE',
    ]

# Replacement macros for CHECK/DCHECK/EXPECT_TRUE/EXPECT_FALSE
_CHECK_REPLACEMENT = dict([(m, {}) for m in _CHECK_MACROS])

for op, replacement in [('==', 'EQ'), ('!=', 'NE'),
                        ('>=', 'GE'), ('>', 'GT'),
                        ('<=', 'LE'), ('<', 'LT')]:
  _CHECK_REPLACEMENT['DCHECK'][op] = 'DCHECK_%s' % replacement
  _CHECK_REPLACEMENT['CHECK'][op] = 'CHECK_%s' % replacement
  _CHECK_REPLACEMENT['EXPECT_TRUE'][op] = 'EXPECT_%s' % replacement
  _CHECK_REPLACEMENT['ASSERT_TRUE'][op] = 'ASSERT_%s' % replacement

for op, inv_replacement in [('==', 'NE'), ('!=', 'EQ'),
                            ('>=', 'LT'), ('>', 'LE'),
                            ('<=', 'GT'), ('<', 'GE')]:
  _CHECK_REPLACEMENT['EXPECT_FALSE'][op] = 'EXPECT_%s' % inv_replacement
  _CHECK_REPLACEMENT['ASSERT_FALSE'][op] = 'ASSERT_%s' % inv_replacement

# Alternative tokens and their replacements.  For full list, see section 2.5
# Alternative tokens [lex.digraph] in the C++ standard.
#
# Digraphs (such as '%:') are not included here since it's a mess to
# match those on a word boundary.
_ALT_TOKEN_REPLACEMENT = {
    'and': '&&',
    'bitor': '|',
    'or': '||',
    'xor': '^',
    'compl': '~',
    'bitand': '&',
    'and_eq': '&=',
    'or_eq': '|=',
    'xor_eq': '^=',
    'not': '!',
    'not_eq': '!='
    }

# Compile regular expression that matches all the above keywords.  The "[ =()]"
# bit is meant to avoid matching these keywords outside of boolean expressions.
#
# False positives include C-style multi-line comments and multi-line strings
# but those have always been troublesome for cpplint.
_ALT_TOKEN_REPLACEMENT_PATTERN = re.compile(
    r'[ =()](' + ('|'.join(_ALT_TOKEN_REPLACEMENT.keys())) + r')(?=[ (]|$)')


# These constants define types of headers for use with
# _IncludeState.CheckNextIncludeOrder().
_C_SYS_HEADER = 1
_CPP_SYS_HEADER = 2
_LIKELY_MY_HEADER = 3
_POSSIBLE_MY_HEADER = 4
_OTHER_HEADER = 5

# These constants define the current inline assembly state
_NO_ASM = 0       # Outside of inline assembly block
_INSIDE_ASM = 1   # Inside inline assembly block
_END_ASM = 2      # Last line of inline assembly block
_BLOCK_ASM = 3    # The whole block is an inline assembly block

# Match start of assembly blocks
_MATCH_ASM = re.compile(r'^\s*(?:asm|_asm|__asm|__asm__)'
                        r'(?:\s+(volatile|__volatile__))?'
                        r'\s*[{(]')

# Match strings that indicate we're working on a C (not C++) file.
_SEARCH_C_FILE = re.compile(r'\b(?:LINT_C_FILE|'
                            r'vim?:\s*.*(\s*|:)filetype=c(\s*|:|$))')

# Match string that indicates we're working on a Linux Kernel file.
_SEARCH_KERNEL_FILE = re.compile(r'\b(?:LINT_KERNEL_FILE)')

_regexp_compile_cache = {}

# {str, set(int)}: a map from error categories to sets of linenumbers
# on which those errors are expected and should be suppressed.
_error_suppressions = {}

# The root directory used for deriving header guard CPP variable.
# This is set by --root flag.
_root = None
_root_debug = False

# The allowed line length of files.
# This is set by --linelength flag.
_line_length = 80

# The allowed extensions for file names
# This is set by --extensions flag.
_valid_extensions = set(['cc', 'h', 'cpp', 'cu', 'cuh'])

# Treat all headers starting with 'h' equally: .h, .hpp, .hxx etc.
# This is set by --headers flag.
_hpp_headers = set(['h'])

# {str, bool}: a map from error categories to booleans which indicate if the
# category should be suppressed for every line.
_global_error_suppressions = {}

def ProcessHppHeadersOption(val):
  global _hpp_headers
  try:
    _hpp_headers = set(val.split(','))
    # Automatically append to extensions list so it does not have to be set 2 times
    _valid_extensions.update(_hpp_headers)
  except ValueError:
    PrintUsage('Header extensions must be comma seperated list.')

def IsHeaderExtension(file_extension):
  return file_extension in _hpp_headers

def ParseNolintSuppressions(filename, raw_line, linenum, error):
  """Updates the global list of line error-suppressions.

  Parses any NOLINT comments on the current line, updating the global
  error_suppressions store.  Reports an error if the NOLINT comment
  was malformed.

  Args:
    filename: str, the name of the input file.
    raw_line: str, the line of input text, with comments.
    linenum: int, the number of the current line.
    error: function, an error handler.
  """
  matched = Search(r'\bNOLINT(NEXTLINE)?\b(\([^)]+\))?', raw_line)
  if matched:
    if matched.group(1):
      suppressed_line = linenum + 1
    else:
      suppressed_line = linenum
    category = matched.group(2)
    if category in (None, '(*)'):  # => "suppress all"
      _error_suppressions.setdefault(None, set()).add(suppressed_line)
    else:
      if category.startswith('(') and category.endswith(')'):
        category = category[1:-1]
        if category in _ERROR_CATEGORIES:
          _error_suppressions.setdefault(category, set()).add(suppressed_line)
        elif category not in _LEGACY_ERROR_CATEGORIES:
          error(filename, linenum, 'readability/nolint', 5,
                'Unknown NOLINT error category: %s' % category)


def ProcessGlobalSuppresions(lines):
  """Updates the list of global error suppressions.

  Parses any lint directives in the file that have global effect.

  Args:
    lines: An array of strings, each representing a line of the file, with the
           last element being empty if the file is terminated with a newline.
  """
  for line in lines:
    if _SEARCH_C_FILE.search(line):
      for category in _DEFAULT_C_SUPPRESSED_CATEGORIES:
        _global_error_suppressions[category] = True
    if _SEARCH_KERNEL_FILE.search(line):
      for category in _DEFAULT_KERNEL_SUPPRESSED_CATEGORIES:
        _global_error_suppressions[category] = True


def ResetNolintSuppressions():
  """Resets the set of NOLINT suppressions to empty."""
  _error_suppressions.clear()
  _global_error_suppressions.clear()


def IsErrorSuppressedByNolint(category, linenum):
  """Returns true if the specified error category is suppressed on this line.

  Consults the global error_suppressions map populated by
  ParseNolintSuppressions/ProcessGlobalSuppresions/ResetNolintSuppressions.

  Args:
    category: str, the category of the error.
    linenum: int, the current line number.
  Returns:
    bool, True iff the error should be suppressed due to a NOLINT comment or
    global suppression.
  """
  return (_global_error_suppressions.get(category, False) or
          linenum in _error_suppressions.get(category, set()) or
          linenum in _error_suppressions.get(None, set()))


def Match(pattern, s):
  """Matches the string with the pattern, caching the compiled regexp."""
  # The regexp compilation caching is inlined in both Match and Search for
  # performance reasons; factoring it out into a separate function turns out
  # to be noticeably expensive.
  if pattern not in _regexp_compile_cache:
    _regexp_compile_cache[pattern] = sre_compile.compile(pattern)
  return _regexp_compile_cache[pattern].match(s)


def ReplaceAll(pattern, rep, s):
  """Replaces instances of pattern in a string with a replacement.

  The compiled regex is kept in a cache shared by Match and Search.

  Args:
    pattern: regex pattern
    rep: replacement text
    s: search string

  Returns:
    string with replacements made (or original string if no replacements)
  """
  if pattern not in _regexp_compile_cache:
    _regexp_compile_cache[pattern] = sre_compile.compile(pattern)
  return _regexp_compile_cache[pattern].sub(rep, s)


def Search(pattern, s):
  """Searches the string for the pattern, caching the compiled regexp."""
  if pattern not in _regexp_compile_cache:
    _regexp_compile_cache[pattern] = sre_compile.compile(pattern)
  return _regexp_compile_cache[pattern].search(s)


def _IsSourceExtension(s):
  """File extension (excluding dot) matches a source file extension."""
  return s in ('c', 'cc', 'cpp', 'cxx')


class _IncludeState(object):
  """Tracks line numbers for includes, and the order in which includes appear.

  include_list contains list of lists of (header, line number) pairs.
  It's a lists of lists rather than just one flat list to make it
  easier to update across preprocessor boundaries.

  Call CheckNextIncludeOrder() once for each header in the file, passing
  in the type constants defined above. Calls in an illegal order will
  raise an _IncludeError with an appropriate error message.

  """
  # self._section will move monotonically through this set. If it ever
  # needs to move backwards, CheckNextIncludeOrder will raise an error.
  _INITIAL_SECTION = 0
  _MY_H_SECTION = 1
  _C_SECTION = 2
  _CPP_SECTION = 3
  _OTHER_H_SECTION = 4

  _TYPE_NAMES = {
      _C_SYS_HEADER: 'C system header',
      _CPP_SYS_HEADER: 'C++ system header',
      _LIKELY_MY_HEADER: 'header this file implements',
      _POSSIBLE_MY_HEADER: 'header this file may implement',
      _OTHER_HEADER: 'other header',
      }
  _SECTION_NAMES = {
      _INITIAL_SECTION: "... nothing. (This can't be an error.)",
      _MY_H_SECTION: 'a header this file implements',
      _C_SECTION: 'C system header',
      _CPP_SECTION: 'C++ system header',
      _OTHER_H_SECTION: 'other header',
      }

  def __init__(self):
    self.include_list = [[]]
    self.ResetSection('')

  def FindHeader(self, header):
    """Check if a header has already been included.

    Args:
      header: header to check.
    Returns:
      Line number of previous occurrence, or -1 if the header has not
      been seen before.
    """
    for section_list in self.include_list:
      for f in section_list:
        if f[0] == header:
          return f[1]
    return -1

  def ResetSection(self, directive):
    """Reset section checking for preprocessor directive.

    Args:
      directive: preprocessor directive (e.g. "if", "else").
    """
    # The name of the current section.
    self._section = self._INITIAL_SECTION
    # The path of last found header.
    self._last_header = ''

    # Update list of includes.  Note that we never pop from the
    # include list.
    if directive in ('if', 'ifdef', 'ifndef'):
      self.include_list.append([])
    elif directive in ('else', 'elif'):
      self.include_list[-1] = []

  def SetLastHeader(self, header_path):
    self._last_header = header_path

  def CanonicalizeAlphabeticalOrder(self, header_path):
    """Returns a path canonicalized for alphabetical comparison.

    - replaces "-" with "_" so they both cmp the same.
    - removes '-inl' since we don't require them to be after the main header.
    - lowercase everything, just in case.

    Args:
      header_path: Path to be canonicalized.

    Returns:
      Canonicalized path.
    """
    return header_path.replace('-inl.h', '.h').replace('-', '_').lower()

  def IsInAlphabeticalOrder(self, clean_lines, linenum, header_path):
    """Check if a header is in alphabetical order with the previous header.

    Args:
      clean_lines: A CleansedLines instance containing the file.
      linenum: The number of the line to check.
      header_path: Canonicalized header to be checked.

    Returns:
      Returns true if the header is in alphabetical order.
    """
    # If previous section is different from current section, _last_header will
    # be reset to empty string, so it's always less than current header.
    #
    # If previous line was a blank line, assume that the headers are
    # intentionally sorted the way they are.
    if (self._last_header > header_path and
        Match(r'^\s*#\s*include\b', clean_lines.elided[linenum - 1])):
      return False
    return True

  def CheckNextIncludeOrder(self, header_type):
    """Returns a non-empty error message if the next header is out of order.

    This function also updates the internal state to be ready to check
    the next include.

    Args:
      header_type: One of the _XXX_HEADER constants defined above.

    Returns:
      The empty string if the header is in the right order, or an
      error message describing what's wrong.

    """
    error_message = ('Found %s after %s' %
                     (self._TYPE_NAMES[header_type],
                      self._SECTION_NAMES[self._section]))

    last_section = self._section

    if header_type == _C_SYS_HEADER:
      if self._section <= self._C_SECTION:
        self._section = self._C_SECTION
      else:
        self._last_header = ''
        return error_message
    elif header_type == _CPP_SYS_HEADER:
      if self._section <= self._CPP_SECTION:
        self._section = self._CPP_SECTION
      else:
        self._last_header = ''
        return error_message
    elif header_type == _LIKELY_MY_HEADER:
      if self._section <= self._MY_H_SECTION:
        self._section = self._MY_H_SECTION
      else:
        self._section = self._OTHER_H_SECTION
    elif header_type == _POSSIBLE_MY_HEADER:
      if self._section <= self._MY_H_SECTION:
        self._section = self._MY_H_SECTION
      else:
        # This will always be the fallback because we're not sure
        # enough that the header is associated with this file.
        self._section = self._OTHER_H_SECTION
    else:
      assert header_type == _OTHER_HEADER
      self._section = self._OTHER_H_SECTION

    if last_section != self._section:
      self._last_header = ''

    return ''


class _CppLintState(object):
  """Maintains module-wide state.."""

  def __init__(self):
    self.verbose_level = 1  # global setting.
    self.error_count = 0    # global count of reported errors
    # filters to apply when emitting error messages
    self.filters = _DEFAULT_FILTERS[:]
    # backup of filter list. Used to restore the state after each file.
    self._filters_backup = self.filters[:]
    self.counting = 'total'  # In what way are we counting errors?
    self.errors_by_category = {}  # string to int dict storing error counts
    self.quiet = False  # Suppress non-error messagess?

    # output format:
    # "emacs" - format that emacs can parse (default)
    # "vs7" - format that Microsoft Visual Studio 7 can parse
    self.output_format = 'emacs'

  def SetOutputFormat(self, output_format):
    """Sets the output format for errors."""
    self.output_format = output_format

  def SetQuiet(self, quiet):
    """Sets the module's quiet settings, and returns the previous setting."""
    last_quiet = self.quiet
    self.quiet = quiet
    return last_quiet

  def SetVerboseLevel(self, level):
    """Sets the module's verbosity, and returns the previous setting."""
    last_verbose_level = self.verbose_level
    self.verbose_level = level
    return last_verbose_level

  def SetCountingStyle(self, counting_style):
    """Sets the module's counting options."""
    self.counting = counting_style

  def SetFilters(self, filters):
    """Sets the error-message filters.

    These filters are applied when deciding whether to emit a given
    error message.

    Args:
      filters: A string of comma-separated filters (eg "+whitespace/indent").
               Each filter should start with + or -; else we die.

    Raises:
      ValueError: The comma-separated filters did not all start with '+' or '-'.
                  E.g. "-,+whitespace,-whitespace/indent,whitespace/badfilter"
    """
    # Default filters always have less priority than the flag ones.
    self.filters = _DEFAULT_FILTERS[:]
    self.AddFilters(filters)

  def AddFilters(self, filters):
    """ Adds more filters to the existing list of error-message filters. """
    for filt in filters.split(','):
      clean_filt = filt.strip()
      if clean_filt:
        self.filters.append(clean_filt)
    for filt in self.filters:
      if not (filt.startswith('+') or filt.startswith('-')):
        raise ValueError('Every filter in --filters must start with + or -'
                         ' (%s does not)' % filt)

  def BackupFilters(self):
    """ Saves the current filter list to backup storage."""
    self._filters_backup = self.filters[:]

  def RestoreFilters(self):
    """ Restores filters previously backed up."""
    self.filters = self._filters_backup[:]

  def ResetErrorCounts(self):
    """Sets the module's error statistic back to zero."""
    self.error_count = 0
    self.errors_by_category = {}

  def IncrementErrorCount(self, category):
    """Bumps the module's error statistic."""
    self.error_count += 1
    if self.counting in ('toplevel', 'detailed'):
      if self.counting != 'detailed':
        category = category.split('/')[0]
      if category not in self.errors_by_category:
        self.errors_by_category[category] = 0
      self.errors_by_category[category] += 1

  def PrintErrorCounts(self):
    """Print a summary of errors by category, and the total."""
    for category, count in self.errors_by_category.iteritems():
      sys.stderr.write('Category \'%s\' errors found: %d\n' %
                       (category, count))
    sys.stdout.write('Total errors found: %d\n' % self.error_count)

_cpplint_state = _CppLintState()


def _OutputFormat():
  """Gets the module's output format."""
  return _cpplint_state.output_format


def _SetOutputFormat(output_format):
  """Sets the module's output format."""
  _cpplint_state.SetOutputFormat(output_format)

def _Quiet():
  """Return's the module's quiet setting."""
  return _cpplint_state.quiet

def _SetQuiet(quiet):
  """Set the module's quiet status, and return previous setting."""
  return _cpplint_state.SetQuiet(quiet)


def _VerboseLevel():
  """Returns the module's verbosity setting."""
  return _cpplint_state.verbose_level


def _SetVerboseLevel(level):
  """Sets the module's verbosity, and returns the previous setting."""
  return _cpplint_state.SetVerboseLevel(level)


def _SetCountingStyle(level):
  """Sets the module's counting options."""
  _cpplint_state.SetCountingStyle(level)


def _Filters():
  """Returns the module's list of output filters, as a list."""
  return _cpplint_state.filters


def _SetFilters(filters):
  """Sets the module's error-message filters.

  These filters are applied when deciding whether to emit a given
  error message.

  Args:
    filters: A string of comma-separated filters (eg "whitespace/indent").
             Each filter should start with + or -; else we die.
  """
  _cpplint_state.SetFilters(filters)

def _AddFilters(filters):
  """Adds more filter overrides.

  Unlike _SetFilters, this function does not reset the current list of filters
  available.

  Args:
    filters: A string of comma-separated filters (eg "whitespace/indent").
             Each filter should start with + or -; else we die.
  """
  _cpplint_state.AddFilters(filters)

def _BackupFilters():
  """ Saves the current filter list to backup storage."""
  _cpplint_state.BackupFilters()

def _RestoreFilters():
  """ Restores filters previously backed up."""
  _cpplint_state.RestoreFilters()

class _FunctionState(object):
  """Tracks current function name and the number of lines in its body."""

  _NORMAL_TRIGGER = 250  # for --v=0, 500 for --v=1, etc.
  _TEST_TRIGGER = 400    # about 50% more than _NORMAL_TRIGGER.

  def __init__(self):
    self.in_a_function = False
    self.lines_in_function = 0
    self.current_function = ''

  def Begin(self, function_name):
    """Start analyzing function body.

    Args:
      function_name: The name of the function being tracked.
    """
    self.in_a_function = True
    self.lines_in_function = 0
    self.current_function = function_name

  def Count(self):
    """Count line in current function body."""
    if self.in_a_function:
      self.lines_in_function += 1

  def Check(self, error, filename, linenum):
    """Report if too many lines in function body.

    Args:
      error: The function to call with any errors found.
      filename: The name of the current file.
      linenum: The number of the line to check.
    """
    if not self.in_a_function:
      return

    if Match(r'T(EST|est)', self.current_function):
      base_trigger = self._TEST_TRIGGER
    else:
      base_trigger = self._NORMAL_TRIGGER
    trigger = base_trigger * 2**_VerboseLevel()

    if self.lines_in_function > trigger:
      error_level = int(math.log(self.lines_in_function / base_trigger, 2))
      # 50 => 0, 100 => 1, 200 => 2, 400 => 3, 800 => 4, 1600 => 5, ...
      if error_level > 5:
        error_level = 5
      error(filename, linenum, 'readability/fn_size', error_level,
            'Small and focused functions are preferred:'
            ' %s has %d non-comment lines'
            ' (error triggered by exceeding %d lines).'  % (
                self.current_function, self.lines_in_function, trigger))

  def End(self):
    """Stop analyzing function body."""
    self.in_a_function = False


class _IncludeError(Exception):
  """Indicates a problem with the include order in a file."""
  pass


class FileInfo(object):
  """Provides utility functions for filenames.

  FileInfo provides easy access to the components of a file's path
  relative to the project root.
  """

  def __init__(self, filename):
    self._filename = filename

  def FullName(self):
    """Make Windows paths like Unix."""
    return os.path.abspath(self._filename).replace('\\', '/')

  def RepositoryName(self):
    """FullName after removing the local path to the repository.

    If we have a real absolute path name here we can try to do something smart:
    detecting the root of the checkout and truncating /path/to/checkout from
    the name so that we get header guards that don't include things like
    "C:\Documents and Settings\..." or "/home/username/..." in them and thus
    people on different computers who have checked the source out to different
    locations won't see bogus errors.
    """
    fullname = self.FullName()

    if os.path.exists(fullname):
      project_dir = os.path.dirname(fullname)

      if os.path.exists(os.path.join(project_dir, ".svn")):
        # If there's a .svn file in the current directory, we recursively look
        # up the directory tree for the top of the SVN checkout
        root_dir = project_dir
        one_up_dir = os.path.dirname(root_dir)
        while os.path.exists(os.path.join(one_up_dir, ".svn")):
          root_dir = os.path.dirname(root_dir)
          one_up_dir = os.path.dirname(one_up_dir)

        prefix = os.path.commonprefix([root_dir, project_dir])
        return fullname[len(prefix) + 1:]

      # Not SVN <= 1.6? Try to find a git, hg, or svn top level directory by
      # searching up from the current path.
      root_dir = current_dir = os.path.dirname(fullname)
      while current_dir != os.path.dirname(current_dir):
        if (os.path.exists(os.path.join(current_dir, ".git")) or
            os.path.exists(os.path.join(current_dir, ".hg")) or
            os.path.exists(os.path.join(current_dir, ".svn"))):
          root_dir = current_dir
        current_dir = os.path.dirname(current_dir)

      if (os.path.exists(os.path.join(root_dir, ".git")) or
          os.path.exists(os.path.join(root_dir, ".hg")) or
          os.path.exists(os.path.join(root_dir, ".svn"))):
        prefix = os.path.commonprefix([root_dir, project_dir])
        return fullname[len(prefix) + 1:]

    # Don't know what to do; header guard warnings may be wrong...
    return fullname

  def Split(self):
    """Splits the file into the directory, basename, and extension.

    For 'chrome/browser/browser.cc', Split() would
    return ('chrome/browser', 'browser', '.cc')

    Returns:
      A tuple of (directory, basename, extension).
    """

    googlename = self.RepositoryName()
    project, rest = os.path.split(googlename)
    return (project,) + os.path.splitext(rest)

  def BaseName(self):
    """File base name - text after the final slash, before the final period."""
    return self.Split()[1]

  def Extension(self):
    """File extension - text following the final period."""
    return self.Split()[2]

  def NoExtension(self):
    """File has no source file extension."""
    return '/'.join(self.Split()[0:2])

  def IsSource(self):
    """File has a source file extension."""
    return _IsSourceExtension(self.Extension()[1:])


def _ShouldPrintError(category, confidence, linenum):
  """If confidence >= verbose, category passes filter and is not suppressed."""

  # There are three ways we might decide not to print an error message:
  # a "NOLINT(category)" comment appears in the source,
  # the verbosity level isn't high enough, or the filters filter it out.
  if IsErrorSuppressedByNolint(category, linenum):
    return False

  if confidence < _cpplint_state.verbose_level:
    return False

  is_filtered = False
  for one_filter in _Filters():
    if one_filter.startswith('-'):
      if category.startswith(one_filter[1:]):
        is_filtered = True
    elif one_filter.startswith('+'):
      if category.startswith(one_filter[1:]):
        is_filtered = False
    else:
      assert False  # should have been checked for in SetFilter.
  if is_filtered:
    return False

  return True


def Error(filename, linenum, category, confidence, message):
  """Logs the fact we've found a lint error.

  We log where the error was found, and also our confidence in the error,
  that is, how certain we are this is a legitimate style regression, and
  not a misidentification or a use that's sometimes justified.

  False positives can be suppressed by the use of
  "cpplint(category)"  comments on the offending line.  These are
  parsed into _error_suppressions.

  Args:
    filename: The name of the file containing the error.
    linenum: The number of the line containing the error.
    category: A string used to describe the "category" this bug
      falls under: "whitespace", say, or "runtime".  Categories
      may have a hierarchy separated by slashes: "whitespace/indent".
    confidence: A number from 1-5 representing a confidence score for
      the error, with 5 meaning that we are certain of the problem,
      and 1 meaning that it could be a legitimate construct.
    message: The error message.
  """
  if _ShouldPrintError(category, confidence, linenum):
    _cpplint_state.IncrementErrorCount(category)
    if _cpplint_state.output_format == 'vs7':
      sys.stderr.write('%s(%s): error cpplint: [%s] %s [%d]\n' % (
          filename, linenum, category, message, confidence))
    elif _cpplint_state.output_format == 'eclipse':
      sys.stderr.write('%s:%s: warning: %s  [%s] [%d]\n' % (
          filename, linenum, message, category, confidence))
    else:
      sys.stderr.write('%s:%s:  %s  [%s] [%d]\n' % (
          filename, linenum, message, category, confidence))


# Matches standard C++ escape sequences per 2.13.2.3 of the C++ standard.
_RE_PATTERN_CLEANSE_LINE_ESCAPES = re.compile(
    r'\\([abfnrtv?"\\\']|\d+|x[0-9a-fA-F]+)')
# Match a single C style comment on the same line.
_RE_PATTERN_C_COMMENTS = r'/\*(?:[^*]|\*(?!/))*\*/'
# Matches multi-line C style comments.
# This RE is a little bit more complicated than one might expect, because we
# have to take care of space removals tools so we can handle comments inside
# statements better.
# The current rule is: We only clear spaces from both sides when we're at the
# end of the line. Otherwise, we try to remove spaces from the right side,
# if this doesn't work we try on left side but only if there's a non-character
# on the right.
_RE_PATTERN_CLEANSE_LINE_C_COMMENTS = re.compile(
    r'(\s*' + _RE_PATTERN_C_COMMENTS + r'\s*$|' +
    _RE_PATTERN_C_COMMENTS + r'\s+|' +
    r'\s+' + _RE_PATTERN_C_COMMENTS + r'(?=\W)|' +
    _RE_PATTERN_C_COMMENTS + r')')


def IsCppString(line):
  """Does line terminate so, that the next symbol is in string constant.

  This function does not consider single-line nor multi-line comments.

  Args:
    line: is a partial line of code starting from the 0..n.

  Returns:
    True, if next character appended to 'line' is inside a
    string constant.
  """

  line = line.replace(r'\\', 'XX')  # after this, \\" does not match to \"
  return ((line.count('"') - line.count(r'\"') - line.count("'\"'")) & 1) == 1


def CleanseRawStrings(raw_lines):
  """Removes C++11 raw strings from lines.

    Before:
      static const char kData[] = R"(
          multi-line string
          )";

    After:
      static const char kData[] = ""
          (replaced by blank line)
          "";

  Args:
    raw_lines: list of raw lines.

  Returns:
    list of lines with C++11 raw strings replaced by empty strings.
  """

  delimiter = None
  lines_without_raw_strings = []
  for line in raw_lines:
    if delimiter:
      # Inside a raw string, look for the end
      end = line.find(delimiter)
      if end >= 0:
        # Found the end of the string, match leading space for this
        # line and resume copying the original lines, and also insert
        # a "" on the last line.
        leading_space = Match(r'^(\s*)\S', line)
        line = leading_space.group(1) + '""' + line[end + len(delimiter):]
        delimiter = None
      else:
        # Haven't found the end yet, append a blank line.
        line = '""'

    # Look for beginning of a raw string, and replace them with
    # empty strings.  This is done in a loop to handle multiple raw
    # strings on the same line.
    while delimiter is None:
      # Look for beginning of a raw string.
      # See 2.14.15 [lex.string] for syntax.
      #
      # Once we have matched a raw string, we check the prefix of the
      # line to make sure that the line is not part of a single line
      # comment.  It's done this way because we remove raw strings
      # before removing comments as opposed to removing comments
      # before removing raw strings.  This is because there are some
      # cpplint checks that requires the comments to be preserved, but
      # we don't want to check comments that are inside raw strings.
      matched = Match(r'^(.*?)\b(?:R|u8R|uR|UR|LR)"([^\s\\()]*)\((.*)$', line)
      if (matched and
          not Match(r'^([^\'"]|\'(\\.|[^\'])*\'|"(\\.|[^"])*")*//',
                    matched.group(1))):
        delimiter = ')' + matched.group(2) + '"'

        end = matched.group(3).find(delimiter)
        if end >= 0:
          # Raw string ended on same line
          line = (matched.group(1) + '""' +
                  matched.group(3)[end + len(delimiter):])
          delimiter = None
        else:
          # Start of a multi-line raw string
          line = matched.group(1) + '""'
      else:
        break

    lines_without_raw_strings.append(line)

  # TODO(unknown): if delimiter is not None here, we might want to
  # emit a warning for unterminated string.
  return lines_without_raw_strings


def FindNextMultiLineCommentStart(lines, lineix):
  """Find the beginning marker for a multiline comment."""
  while lineix < len(lines):
    if lines[lineix].strip().startswith('/*'):
      # Only return this marker if the comment goes beyond this line
      if lines[lineix].strip().find('*/', 2) < 0:
        return lineix
    lineix += 1
  return len(lines)


def FindNextMultiLineCommentEnd(lines, lineix):
  """We are inside a comment, find the end marker."""
  while lineix < len(lines):
    if lines[lineix].strip().endswith('*/'):
      return lineix
    lineix += 1
  return len(lines)


def RemoveMultiLineCommentsFromRange(lines, begin, end):
  """Clears a range of lines for multi-line comments."""
  # Having // dummy comments makes the lines non-empty, so we will not get
  # unnecessary blank line warnings later in the code.
  for i in range(begin, end):
    lines[i] = '/**/'


def RemoveMultiLineComments(filename, lines, error):
  """Removes multiline (c-style) comments from lines."""
  lineix = 0
  while lineix < len(lines):
    lineix_begin = FindNextMultiLineCommentStart(lines, lineix)
    if lineix_begin >= len(lines):
      return
    lineix_end = FindNextMultiLineCommentEnd(lines, lineix_begin)
    if lineix_end >= len(lines):
      error(filename, lineix_begin + 1, 'readability/multiline_comment', 5,
            'Could not find end of multi-line comment')
      return
    RemoveMultiLineCommentsFromRange(lines, lineix_begin, lineix_end + 1)
    lineix = lineix_end + 1


def CleanseComments(line):
  """Removes //-comments and single-line C-style /* */ comments.

  Args:
    line: A line of C++ source.

  Returns:
    The line with single-line comments removed.
  """
  commentpos = line.find('//')
  if commentpos != -1 and not IsCppString(line[:commentpos]):
    line = line[:commentpos].rstrip()
  # get rid of /* ... */
  return _RE_PATTERN_CLEANSE_LINE_C_COMMENTS.sub('', line)


class CleansedLines(object):
  """Holds 4 copies of all lines with different preprocessing applied to them.

  1) elided member contains lines without strings and comments.
  2) lines member contains lines without comments.
  3) raw_lines member contains all the lines without processing.
  4) lines_without_raw_strings member is same as raw_lines, but with C++11 raw
     strings removed.
  All these members are of <type 'list'>, and of the same length.
  """

  def __init__(self, lines):
    self.elided = []
    self.lines = []
    self.raw_lines = lines
    self.num_lines = len(lines)
    self.lines_without_raw_strings = CleanseRawStrings(lines)
    for linenum in range(len(self.lines_without_raw_strings)):
      self.lines.append(CleanseComments(
          self.lines_without_raw_strings[linenum]))
      elided = self._CollapseStrings(self.lines_without_raw_strings[linenum])
      self.elided.append(CleanseComments(elided))

  def NumLines(self):
    """Returns the number of lines represented."""
    return self.num_lines

  @staticmethod
  def _CollapseStrings(elided):
    """Collapses strings and chars on a line to simple "" or '' blocks.

    We nix strings first so we're not fooled by text like '"http://"'

    Args:
      elided: The line being processed.

    Returns:
      The line with collapsed strings.
    """
    if _RE_PATTERN_INCLUDE.match(elided):
      return elided

    # Remove escaped characters first to make quote/single quote collapsing
    # basic.  Things that look like escaped characters shouldn't occur
    # outside of strings and chars.
    elided = _RE_PATTERN_CLEANSE_LINE_ESCAPES.sub('', elided)

    # Replace quoted strings and digit separators.  Both single quotes
    # and double quotes are processed in the same loop, otherwise
    # nested quotes wouldn't work.
    collapsed = ''
    while True:
      # Find the first quote character
      match = Match(r'^([^\'"]*)([\'"])(.*)$', elided)
      if not match:
        collapsed += elided
        break
      head, quote, tail = match.groups()

      if quote == '"':
        # Collapse double quoted strings
        second_quote = tail.find('"')
        if second_quote >= 0:
          collapsed += head + '""'
          elided = tail[second_quote + 1:]
        else:
          # Unmatched double quote, don't bother processing the rest
          # of the line since this is probably a multiline string.
          collapsed += elided
          break
      else:
        # Found single quote, check nearby text to eliminate digit separators.
        #
        # There is no special handling for floating point here, because
        # the integer/fractional/exponent parts would all be parsed
        # correctly as long as there are digits on both sides of the
        # separator.  So we are fine as long as we don't see something
        # like "0.'3" (gcc 4.9.0 will not allow this literal).
        if Search(r'\b(?:0[bBxX]?|[1-9])[0-9a-fA-F]*$', head):
          match_literal = Match(r'^((?:\'?[0-9a-zA-Z_])*)(.*)$', "'" + tail)
          collapsed += head + match_literal.group(1).replace("'", '')
          elided = match_literal.group(2)
        else:
          second_quote = tail.find('\'')
          if second_quote >= 0:
            collapsed += head + "''"
            elided = tail[second_quote + 1:]
          else:
            # Unmatched single quote
            collapsed += elided
            break

    return collapsed


def FindEndOfExpressionInLine(line, startpos, stack):
  """Find the position just after the end of current parenthesized expression.

  Args:
    line: a CleansedLines line.
    startpos: start searching at this position.
    stack: nesting stack at startpos.

  Returns:
    On finding matching end: (index just after matching end, None)
    On finding an unclosed expression: (-1, None)
    Otherwise: (-1, new stack at end of this line)
  """
  for i in xrange(startpos, len(line)):
    char = line[i]
    if char in '([{':
      # Found start of parenthesized expression, push to expression stack
      stack.append(char)
    elif char == '<':
      # Found potential start of template argument list
      if i > 0 and line[i - 1] == '<':
        # Left shift operator
        if stack and stack[-1] == '<':
          stack.pop()
          if not stack:
            return (-1, None)
      elif i > 0 and Search(r'\boperator\s*$', line[0:i]):
        # operator<, don't add to stack
        continue
      else:
        # Tentative start of template argument list
        stack.append('<')
    elif char in ')]}':
      # Found end of parenthesized expression.
      #
      # If we are currently expecting a matching '>', the pending '<'
      # must have been an operator.  Remove them from expression stack.
      while stack and stack[-1] == '<':
        stack.pop()
      if not stack:
        return (-1, None)
      if ((stack[-1] == '(' and char == ')') or
          (stack[-1] == '[' and char == ']') or
          (stack[-1] == '{' and char == '}')):
        stack.pop()
        if not stack:
          return (i + 1, None)
      else:
        # Mismatched parentheses
        return (-1, None)
    elif char == '>':
      # Found potential end of template argument list.

      # Ignore "->" and operator functions
      if (i > 0 and
          (line[i - 1] == '-' or Search(r'\boperator\s*$', line[0:i - 1]))):
        continue

      # Pop the stack if there is a matching '<'.  Otherwise, ignore
      # this '>' since it must be an operator.
      if stack:
        if stack[-1] == '<':
          stack.pop()
          if not stack:
            return (i + 1, None)
    elif char == ';':
      # Found something that look like end of statements.  If we are currently
      # expecting a '>', the matching '<' must have been an operator, since
      # template argument list should not contain statements.
      while stack and stack[-1] == '<':
        stack.pop()
      if not stack:
        return (-1, None)

  # Did not find end of expression or unbalanced parentheses on this line
  return (-1, stack)


def CloseExpression(clean_lines, linenum, pos):
  """If input points to ( or { or [ or <, finds the position that closes it.

  If lines[linenum][pos] points to a '(' or '{' or '[' or '<', finds the
  linenum/pos that correspond to the closing of the expression.

  TODO(unknown): cpplint spends a fair bit of time matching parentheses.
  Ideally we would want to index all opening and closing parentheses once
  and have CloseExpression be just a simple lookup, but due to preprocessor
  tricks, this is not so easy.

  Args:
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    pos: A position on the line.

  Returns:
    A tuple (line, linenum, pos) pointer *past* the closing brace, or
    (line, len(lines), -1) if we never find a close.  Note we ignore
    strings and comments when matching; and the line we return is the
    'cleansed' line at linenum.
  """

  line = clean_lines.elided[linenum]
  if (line[pos] not in '({[<') or Match(r'<[<=]', line[pos:]):
    return (line, clean_lines.NumLines(), -1)

  # Check first line
  (end_pos, stack) = FindEndOfExpressionInLine(line, pos, [])
  if end_pos > -1:
    return (line, linenum, end_pos)

  # Continue scanning forward
  while stack and linenum < clean_lines.NumLines() - 1:
    linenum += 1
    line = clean_lines.elided[linenum]
    (end_pos, stack) = FindEndOfExpressionInLine(line, 0, stack)
    if end_pos > -1:
      return (line, linenum, end_pos)

  # Did not find end of expression before end of file, give up
  return (line, clean_lines.NumLines(), -1)


def FindStartOfExpressionInLine(line, endpos, stack):
  """Find position at the matching start of current expression.

  This is almost the reverse of FindEndOfExpressionInLine, but note
  that the input position and returned position differs by 1.

  Args:
    line: a CleansedLines line.
    endpos: start searching at this position.
    stack: nesting stack at endpos.

  Returns:
    On finding matching start: (index at matching start, None)
    On finding an unclosed expression: (-1, None)
    Otherwise: (-1, new stack at beginning of this line)
  """
  i = endpos
  while i >= 0:
    char = line[i]
    if char in ')]}':
      # Found end of expression, push to expression stack
      stack.append(char)
    elif char == '>':
      # Found potential end of template argument list.
      #
      # Ignore it if it's a "->" or ">=" or "operator>"
      if (i > 0 and
          (line[i - 1] == '-' or
           Match(r'\s>=\s', line[i - 1:]) or
           Search(r'\boperator\s*$', line[0:i]))):
        i -= 1
      else:
        stack.append('>')
    elif char == '<':
      # Found potential start of template argument list
      if i > 0 and line[i - 1] == '<':
        # Left shift operator
        i -= 1
      else:
        # If there is a matching '>', we can pop the expression stack.
        # Otherwise, ignore this '<' since it must be an operator.
        if stack and stack[-1] == '>':
          stack.pop()
          if not stack:
            return (i, None)
    elif char in '([{':
      # Found start of expression.
      #
      # If there are any unmatched '>' on the stack, they must be
      # operators.  Remove those.
      while stack and stack[-1] == '>':
        stack.pop()
      if not stack:
        return (-1, None)
      if ((char == '(' and stack[-1] == ')') or
          (char == '[' and stack[-1] == ']') or
          (char == '{' and stack[-1] == '}')):
        stack.pop()
        if not stack:
          return (i, None)
      else:
        # Mismatched parentheses
        return (-1, None)
    elif char == ';':
      # Found something that look like end of statements.  If we are currently
      # expecting a '<', the matching '>' must have been an operator, since
      # template argument list should not contain statements.
      while stack and stack[-1] == '>':
        stack.pop()
      if not stack:
        return (-1, None)

    i -= 1

  return (-1, stack)


def ReverseCloseExpression(clean_lines, linenum, pos):
  """If input points to ) or } or ] or >, finds the position that opens it.

  If lines[linenum][pos] points to a ')' or '}' or ']' or '>', finds the
  linenum/pos that correspond to the opening of the expression.

  Args:
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    pos: A position on the line.

  Returns:
    A tuple (line, linenum, pos) pointer *at* the opening brace, or
    (line, 0, -1) if we never find the matching opening brace.  Note
    we ignore strings and comments when matching; and the line we
    return is the 'cleansed' line at linenum.
  """
  line = clean_lines.elided[linenum]
  if line[pos] not in ')}]>':
    return (line, 0, -1)

  # Check last line
  (start_pos, stack) = FindStartOfExpressionInLine(line, pos, [])
  if start_pos > -1:
    return (line, linenum, start_pos)

  # Continue scanning backward
  while stack and linenum > 0:
    linenum -= 1
    line = clean_lines.elided[linenum]
    (start_pos, stack) = FindStartOfExpressionInLine(line, len(line) - 1, stack)
    if start_pos > -1:
      return (line, linenum, start_pos)

  # Did not find start of expression before beginning of file, give up
  return (line, 0, -1)


def CheckForCopyright(filename, lines, error):
  """Logs an error if no Copyright message appears at the top of the file."""

  # We'll say it should occur by line 10. Don't forget there's a
  # dummy line at the front.
  for line in xrange(1, min(len(lines), 11)):
    if re.search(r'Copyright', lines[line], re.I): break
  else:                       # means no copyright line was found
    error(filename, 0, 'legal/copyright', 5,
          'No copyright message found.  '
          'You should have a line: "Copyright [year] <Copyright Owner>"')


def GetIndentLevel(line):
  """Return the number of leading spaces in line.

  Args:
    line: A string to check.

  Returns:
    An integer count of leading spaces, possibly zero.
  """
  indent = Match(r'^( *)\S', line)
  if indent:
    return len(indent.group(1))
  else:
    return 0

def PathSplitToList(path):
  """Returns the path split into a list by the separator.

  Args:
    path: An absolute or relative path (e.g. '/a/b/c/' or '../a')

  Returns:
    A list of path components (e.g. ['a', 'b', 'c]).
  """
  lst = []
  while True:
    (head, tail) = os.path.split(path)
    if head == path: # absolute paths end
      lst.append(head)
      break
    if tail == path: # relative paths end
      lst.append(tail)
      break

    path = head
    lst.append(tail)

  lst.reverse()
  return lst

def GetHeaderGuardCPPVariable(filename):
  """Returns the CPP variable that should be used as a header guard.

  Args:
    filename: The name of a C++ header file.

  Returns:
    The CPP variable that should be used as a header guard in the
    named file.

  """

  # Restores original filename in case that cpplint is invoked from Emacs's
  # flymake.
  filename = re.sub(r'_flymake\.h$', '.h', filename)
  filename = re.sub(r'/\.flymake/([^/]*)$', r'/\1', filename)
  # Replace 'c++' with 'cpp'.
  filename = filename.replace('C++', 'cpp').replace('c++', 'cpp')

  fileinfo = FileInfo(filename)
  file_path_from_root = fileinfo.RepositoryName()

  def FixupPathFromRoot():
    if _root_debug:
      sys.stderr.write("\n_root fixup, _root = '%s', repository name = '%s'\n"
          %(_root, fileinfo.RepositoryName()))

    # Process the file path with the --root flag if it was set.
    if not _root:
      if _root_debug:
        sys.stderr.write("_root unspecified\n")
      return file_path_from_root

    def StripListPrefix(lst, prefix):
      # f(['x', 'y'], ['w, z']) -> None  (not a valid prefix)
      if lst[:len(prefix)] != prefix:
        return None
      # f(['a, 'b', 'c', 'd'], ['a', 'b']) -> ['c', 'd']
      return lst[(len(prefix)):]

    # root behavior:
    #   --root=subdir , lstrips subdir from the header guard
    maybe_path = StripListPrefix(PathSplitToList(file_path_from_root),
                                 PathSplitToList(_root))

    if _root_debug:
      sys.stderr.write("_root lstrip (maybe_path=%s, file_path_from_root=%s," +
          " _root=%s)\n" %(maybe_path, file_path_from_root, _root))

    if maybe_path:
      return os.path.join(*maybe_path)

    #   --root=.. , will prepend the outer directory to the header guard
    full_path = fileinfo.FullName()
    root_abspath = os.path.abspath(_root)

    maybe_path = StripListPrefix(PathSplitToList(full_path),
                                 PathSplitToList(root_abspath))

    if _root_debug:
      sys.stderr.write("_root prepend (maybe_path=%s, full_path=%s, " +
          "root_abspath=%s)\n" %(maybe_path, full_path, root_abspath))

    if maybe_path:
      return os.path.join(*maybe_path)

    if _root_debug:
      sys.stderr.write("_root ignore, returning %s\n" %(file_path_from_root))

    #   --root=FAKE_DIR is ignored
    return file_path_from_root

  file_path_from_root = FixupPathFromRoot()
  return re.sub(r'[^a-zA-Z0-9]', '_', file_path_from_root).upper() + '_'


def CheckForHeaderGuard(filename, clean_lines, error):
  """Checks that the file contains a header guard.

  Logs an error if no #ifndef header guard is present.  For other
  headers, checks that the full pathname is used.

  Args:
    filename: The name of the C++ header file.
    clean_lines: A CleansedLines instance containing the file.
    error: The function to call with any errors found.
  """

  # Don't check for header guards if there are error suppression
  # comments somewhere in this file.
  #
  # Because this is silencing a warning for a nonexistent line, we
  # only support the very specific NOLINT(build/header_guard) syntax,
  # and not the general NOLINT or NOLINT(*) syntax.
  raw_lines = clean_lines.lines_without_raw_strings
  for i in raw_lines:
    if Search(r'//\s*NOLINT\(build/header_guard\)', i):
      return

  cppvar = GetHeaderGuardCPPVariable(filename)

  ifndef = ''
  ifndef_linenum = 0
  define = ''
  endif = ''
  endif_linenum = 0
  for linenum, line in enumerate(raw_lines):
    linesplit = line.split()
    if len(linesplit) >= 2:
      # find the first occurrence of #ifndef and #define, save arg
      if not ifndef and linesplit[0] == '#ifndef':
        # set ifndef to the header guard presented on the #ifndef line.
        ifndef = linesplit[1]
        ifndef_linenum = linenum
      if not define and linesplit[0] == '#define':
        define = linesplit[1]
    # find the last occurrence of #endif, save entire line
    if line.startswith('#endif'):
      endif = line
      endif_linenum = linenum

  if not ifndef or not define or ifndef != define:
    error(filename, 0, 'build/header_guard', 5,
          'No #ifndef header guard found, suggested CPP variable is: %s' %
          cppvar)
    return

  # The guard should be PATH_FILE_H_, but we also allow PATH_FILE_H__
  # for backward compatibility.
  if ifndef != cppvar:
    error_level = 0
    if ifndef != cppvar + '_':
      error_level = 5

    ParseNolintSuppressions(filename, raw_lines[ifndef_linenum], ifndef_linenum,
                            error)
    error(filename, ifndef_linenum, 'build/header_guard', error_level,
          '#ifndef header guard has wrong style, please use: %s' % cppvar)

  # Check for "//" comments on endif line.
  ParseNolintSuppressions(filename, raw_lines[endif_linenum], endif_linenum,
                          error)
  match = Match(r'#endif\s*//\s*' + cppvar + r'(_)?\b', endif)
  if match:
    if match.group(1) == '_':
      # Issue low severity warning for deprecated double trailing underscore
      error(filename, endif_linenum, 'build/header_guard', 0,
            '#endif line should be "#endif  // %s"' % cppvar)
    return

  # Didn't find the corresponding "//" comment.  If this file does not
  # contain any "//" comments at all, it could be that the compiler
  # only wants "/**/" comments, look for those instead.
  no_single_line_comments = True
  for i in xrange(1, len(raw_lines) - 1):
    line = raw_lines[i]
    if Match(r'^(?:(?:\'(?:\.|[^\'])*\')|(?:"(?:\.|[^"])*")|[^\'"])*//', line):
      no_single_line_comments = False
      break

  if no_single_line_comments:
    match = Match(r'#endif\s*/\*\s*' + cppvar + r'(_)?\s*\*/', endif)
    if match:
      if match.group(1) == '_':
        # Low severity warning for double trailing underscore
        error(filename, endif_linenum, 'build/header_guard', 0,
              '#endif line should be "#endif  /* %s */"' % cppvar)
      return

  # Didn't find anything
  error(filename, endif_linenum, 'build/header_guard', 5,
        '#endif line should be "#endif  // %s"' % cppvar)


def CheckHeaderFileIncluded(filename, include_state, error):
  """Logs an error if a .cc file does not include its header."""

  # Do not check test files
  fileinfo = FileInfo(filename)
  if Search(_TEST_FILE_SUFFIX, fileinfo.BaseName()):
    return

  headerfile = filename[0:len(filename) - len(fileinfo.Extension())] + '.h'
  if not os.path.exists(headerfile):
    return
  headername = FileInfo(headerfile).RepositoryName()
  first_include = 0
  for section_list in include_state.include_list:
    for f in section_list:
      if headername in f[0] or f[0] in headername:
        return
      if not first_include:
        first_include = f[1]

  error(filename, first_include, 'build/include', 5,
        '%s should include its header file %s' % (fileinfo.RepositoryName(),
                                                  headername))


def CheckForBadCharacters(filename, lines, error):
  """Logs an error for each line containing bad characters.

  Two kinds of bad characters:

  1. Unicode replacement characters: These indicate that either the file
  contained invalid UTF-8 (likely) or Unicode replacement characters (which
  it shouldn't).  Note that it's possible for this to throw off line
  numbering if the invalid UTF-8 occurred adjacent to a newline.

  2. NUL bytes.  These are problematic for some tools.

  Args:
    filename: The name of the current file.
    lines: An array of strings, each representing a line of the file.
    error: The function to call with any errors found.
  """
  for linenum, line in enumerate(lines):
    if u'\ufffd' in line:
      error(filename, linenum, 'readability/utf8', 5,
            'Line contains invalid UTF-8 (or Unicode replacement character).')
    if '\0' in line:
      error(filename, linenum, 'readability/nul', 5, 'Line contains NUL byte.')


def CheckForNewlineAtEOF(filename, lines, error):
  """Logs an error if there is no newline char at the end of the file.

  Args:
    filename: The name of the current file.
    lines: An array of strings, each representing a line of the file.
    error: The function to call with any errors found.
  """

  # The array lines() was created by adding two newlines to the
  # original file (go figure), then splitting on \n.
  # To verify that the file ends in \n, we just have to make sure the
  # last-but-two element of lines() exists and is empty.
  if len(lines) < 3 or lines[-2]:
    error(filename, len(lines) - 2, 'whitespace/ending_newline', 5,
          'Could not find a newline character at the end of the file.')


def CheckForMultilineCommentsAndStrings(filename, clean_lines, linenum, error):
  """Logs an error if we see /* ... */ or "..." that extend past one line.

  /* ... */ comments are legit inside macros, for one line.
  Otherwise, we prefer // comments, so it's ok to warn about the
  other.  Likewise, it's ok for strings to extend across multiple
  lines, as long as a line continuation character (backslash)
  terminates each line. Although not currently prohibited by the C++
  style guide, it's ugly and unnecessary. We don't do well with either
  in this lint program, so we warn about both.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # Remove all \\ (escaped backslashes) from the line. They are OK, and the
  # second (escaped) slash may trigger later \" detection erroneously.
  line = line.replace('\\\\', '')

  if line.count('/*') > line.count('*/'):
    error(filename, linenum, 'readability/multiline_comment', 5,
          'Complex multi-line /*...*/-style comment found. '
          'Lint may give bogus warnings.  '
          'Consider replacing these with //-style comments, '
          'with #if 0...#endif, '
          'or with more clearly structured multi-line comments.')

  if (line.count('"') - line.count('\\"')) % 2:
    error(filename, linenum, 'readability/multiline_string', 5,
          'Multi-line string ("...") found.  This lint script doesn\'t '
          'do well with such strings, and may give bogus warnings.  '
          'Use C++11 raw strings or concatenation instead.')


# (non-threadsafe name, thread-safe alternative, validation pattern)
#
# The validation pattern is used to eliminate false positives such as:
#  _rand();               // false positive due to substring match.
#  ->rand();              // some member function rand().
#  ACMRandom rand(seed);  // some variable named rand.
#  ISAACRandom rand();    // another variable named rand.
#
# Basically we require the return value of these functions to be used
# in some expression context on the same line by matching on some
# operator before the function name.  This eliminates constructors and
# member function calls.
_UNSAFE_FUNC_PREFIX = r'(?:[-+*/=%^&|(<]\s*|>\s+)'
_THREADING_LIST = (
    ('asctime(', 'asctime_r(', _UNSAFE_FUNC_PREFIX + r'asctime\([^)]+\)'),
    ('ctime(', 'ctime_r(', _UNSAFE_FUNC_PREFIX + r'ctime\([^)]+\)'),
    ('getgrgid(', 'getgrgid_r(', _UNSAFE_FUNC_PREFIX + r'getgrgid\([^)]+\)'),
    ('getgrnam(', 'getgrnam_r(', _UNSAFE_FUNC_PREFIX + r'getgrnam\([^)]+\)'),
    ('getlogin(', 'getlogin_r(', _UNSAFE_FUNC_PREFIX + r'getlogin\(\)'),
    ('getpwnam(', 'getpwnam_r(', _UNSAFE_FUNC_PREFIX + r'getpwnam\([^)]+\)'),
    ('getpwuid(', 'getpwuid_r(', _UNSAFE_FUNC_PREFIX + r'getpwuid\([^)]+\)'),
    ('gmtime(', 'gmtime_r(', _UNSAFE_FUNC_PREFIX + r'gmtime\([^)]+\)'),
    ('localtime(', 'localtime_r(', _UNSAFE_FUNC_PREFIX + r'localtime\([^)]+\)'),
    ('rand(', 'rand_r(', _UNSAFE_FUNC_PREFIX + r'rand\(\)'),
    ('strtok(', 'strtok_r(',
     _UNSAFE_FUNC_PREFIX + r'strtok\([^)]+\)'),
    ('ttyname(', 'ttyname_r(', _UNSAFE_FUNC_PREFIX + r'ttyname\([^)]+\)'),
    )


def CheckPosixThreading(filename, clean_lines, linenum, error):
  """Checks for calls to thread-unsafe functions.

  Much code has been originally written without consideration of
  multi-threading. Also, engineers are relying on their old experience;
  they have learned posix before threading extensions were added. These
  tests guide the engineers to use thread-safe functions (when using
  posix directly).

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]
  for single_thread_func, multithread_safe_func, pattern in _THREADING_LIST:
    # Additional pattern matching check to confirm that this is the
    # function we are looking for
    if Search(pattern, line):
      error(filename, linenum, 'runtime/threadsafe_fn', 2,
            'Consider using ' + multithread_safe_func +
            '...) instead of ' + single_thread_func +
            '...) for improved thread safety.')


def CheckVlogArguments(filename, clean_lines, linenum, error):
  """Checks that VLOG() is only used for defining a logging level.

  For example, VLOG(2) is correct. VLOG(INFO), VLOG(WARNING), VLOG(ERROR), and
  VLOG(FATAL) are not.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]
  if Search(r'\bVLOG\((INFO|ERROR|WARNING|DFATAL|FATAL)\)', line):
    error(filename, linenum, 'runtime/vlog', 5,
          'VLOG() should be used with numeric verbosity level.  '
          'Use LOG() if you want symbolic severity levels.')

# Matches invalid increment: *count++, which moves pointer instead of
# incrementing a value.
_RE_PATTERN_INVALID_INCREMENT = re.compile(
    r'^\s*\*\w+(\+\+|--);')


def CheckInvalidIncrement(filename, clean_lines, linenum, error):
  """Checks for invalid increment *count++.

  For example following function:
  void increment_counter(int* count) {
    *count++;
  }
  is invalid, because it effectively does count++, moving pointer, and should
  be replaced with ++*count, (*count)++ or *count += 1.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]
  if _RE_PATTERN_INVALID_INCREMENT.match(line):
    error(filename, linenum, 'runtime/invalid_increment', 5,
          'Changing pointer instead of value (or unused value of operator*).')


def IsMacroDefinition(clean_lines, linenum):
  if Search(r'^#define', clean_lines[linenum]):
    return True

  if linenum > 0 and Search(r'\\$', clean_lines[linenum - 1]):
    return True

  return False


def IsForwardClassDeclaration(clean_lines, linenum):
  return Match(r'^\s*(\btemplate\b)*.*class\s+\w+;\s*$', clean_lines[linenum])


class _BlockInfo(object):
  """Stores information about a generic block of code."""

  def __init__(self, linenum, seen_open_brace):
    self.starting_linenum = linenum
    self.seen_open_brace = seen_open_brace
    self.open_parentheses = 0
    self.inline_asm = _NO_ASM
    self.check_namespace_indentation = False

  def CheckBegin(self, filename, clean_lines, linenum, error):
    """Run checks that applies to text up to the opening brace.

    This is mostly for checking the text after the class identifier
    and the "{", usually where the base class is specified.  For other
    blocks, there isn't much to check, so we always pass.

    Args:
      filename: The name of the current file.
      clean_lines: A CleansedLines instance containing the file.
      linenum: The number of the line to check.
      error: The function to call with any errors found.
    """
    pass

  def CheckEnd(self, filename, clean_lines, linenum, error):
    """Run checks that applies to text after the closing brace.

    This is mostly used for checking end of namespace comments.

    Args:
      filename: The name of the current file.
      clean_lines: A CleansedLines instance containing the file.
      linenum: The number of the line to check.
      error: The function to call with any errors found.
    """
    pass

  def IsBlockInfo(self):
    """Returns true if this block is a _BlockInfo.

    This is convenient for verifying that an object is an instance of
    a _BlockInfo, but not an instance of any of the derived classes.

    Returns:
      True for this class, False for derived classes.
    """
    return self.__class__ == _BlockInfo


class _ExternCInfo(_BlockInfo):
  """Stores information about an 'extern "C"' block."""

  def __init__(self, linenum):
    _BlockInfo.__init__(self, linenum, True)


class _ClassInfo(_BlockInfo):
  """Stores information about a class."""

  def __init__(self, name, class_or_struct, clean_lines, linenum):
    _BlockInfo.__init__(self, linenum, False)
    self.name = name
    self.is_derived = False
    self.check_namespace_indentation = True
    if class_or_struct == 'struct':
      self.access = 'public'
      self.is_struct = True
    else:
      self.access = 'private'
      self.is_struct = False

    # Remember initial indentation level for this class.  Using raw_lines here
    # instead of elided to account for leading comments.
    self.class_indent = GetIndentLevel(clean_lines.raw_lines[linenum])

    # Try to find the end of the class.  This will be confused by things like:
    #   class A {
    #   } *x = { ...
    #
    # But it's still good enough for CheckSectionSpacing.
    self.last_line = 0
    depth = 0
    for i in range(linenum, clean_lines.NumLines()):
      line = clean_lines.elided[i]
      depth += line.count('{') - line.count('}')
      if not depth:
        self.last_line = i
        break

  def CheckBegin(self, filename, clean_lines, linenum, error):
    # Look for a bare ':'
    if Search('(^|[^:]):($|[^:])', clean_lines.elided[linenum]):
      self.is_derived = True

  def CheckEnd(self, filename, clean_lines, linenum, error):
    # If there is a DISALLOW macro, it should appear near the end of
    # the class.
    seen_last_thing_in_class = False
    for i in xrange(linenum - 1, self.starting_linenum, -1):
      match = Search(
          r'\b(DISALLOW_COPY_AND_ASSIGN|DISALLOW_IMPLICIT_CONSTRUCTORS)\(' +
          self.name + r'\)',
          clean_lines.elided[i])
      if match:
        if seen_last_thing_in_class:
          error(filename, i, 'readability/constructors', 3,
                match.group(1) + ' should be the last thing in the class')
        break

      if not Match(r'^\s*$', clean_lines.elided[i]):
        seen_last_thing_in_class = True

    # Check that closing brace is aligned with beginning of the class.
    # Only do this if the closing brace is indented by only whitespaces.
    # This means we will not check single-line class definitions.
    indent = Match(r'^( *)\}', clean_lines.elided[linenum])
    if indent and len(indent.group(1)) != self.class_indent:
      if self.is_struct:
        parent = 'struct ' + self.name
      else:
        parent = 'class ' + self.name
      error(filename, linenum, 'whitespace/indent', 3,
            'Closing brace should be aligned with beginning of %s' % parent)


class _NamespaceInfo(_BlockInfo):
  """Stores information about a namespace."""

  def __init__(self, name, linenum):
    _BlockInfo.__init__(self, linenum, False)
    self.name = name or ''
    self.check_namespace_indentation = True

  def CheckEnd(self, filename, clean_lines, linenum, error):
    """Check end of namespace comments."""
    line = clean_lines.raw_lines[linenum]

    # Check how many lines is enclosed in this namespace.  Don't issue
    # warning for missing namespace comments if there aren't enough
    # lines.  However, do apply checks if there is already an end of
    # namespace comment and it's incorrect.
    #
    # TODO(unknown): We always want to check end of namespace comments
    # if a namespace is large, but sometimes we also want to apply the
    # check if a short namespace contained nontrivial things (something
    # other than forward declarations).  There is currently no logic on
    # deciding what these nontrivial things are, so this check is
    # triggered by namespace size only, which works most of the time.
    if (linenum - self.starting_linenum < 10
        and not Match(r'^\s*};*\s*(//|/\*).*\bnamespace\b', line)):
      return

    # Look for matching comment at end of namespace.
    #
    # Note that we accept C style "/* */" comments for terminating
    # namespaces, so that code that terminate namespaces inside
    # preprocessor macros can be cpplint clean.
    #
    # We also accept stuff like "// end of namespace <name>." with the
    # period at the end.
    #
    # Besides these, we don't accept anything else, otherwise we might
    # get false negatives when existing comment is a substring of the
    # expected namespace.
    if self.name:
      # Named namespace
      if not Match((r'^\s*};*\s*(//|/\*).*\bnamespace\s+' +
                    re.escape(self.name) + r'[\*/\.\\\s]*$'),
                   line):
        error(filename, linenum, 'readability/namespace', 5,
              'Namespace should be terminated with "// namespace %s"' %
              self.name)
    else:
      # Anonymous namespace
      if not Match(r'^\s*};*\s*(//|/\*).*\bnamespace[\*/\.\\\s]*$', line):
        # If "// namespace anonymous" or "// anonymous namespace (more text)",
        # mention "// anonymous namespace" as an acceptable form
        if Match(r'^\s*}.*\b(namespace anonymous|anonymous namespace)\b', line):
          error(filename, linenum, 'readability/namespace', 5,
                'Anonymous namespace should be terminated with "// namespace"'
                ' or "// anonymous namespace"')
        else:
          error(filename, linenum, 'readability/namespace', 5,
                'Anonymous namespace should be terminated with "// namespace"')


class _PreprocessorInfo(object):
  """Stores checkpoints of nesting stacks when #if/#else is seen."""

  def __init__(self, stack_before_if):
    # The entire nesting stack before #if
    self.stack_before_if = stack_before_if

    # The entire nesting stack up to #else
    self.stack_before_else = []

    # Whether we have already seen #else or #elif
    self.seen_else = False


class NestingState(object):
  """Holds states related to parsing braces."""

  def __init__(self):
    # Stack for tracking all braces.  An object is pushed whenever we
    # see a "{", and popped when we see a "}".  Only 3 types of
    # objects are possible:
    # - _ClassInfo: a class or struct.
    # - _NamespaceInfo: a namespace.
    # - _BlockInfo: some other type of block.
    self.stack = []

    # Top of the previous stack before each Update().
    #
    # Because the nesting_stack is updated at the end of each line, we
    # had to do some convoluted checks to find out what is the current
    # scope at the beginning of the line.  This check is simplified by
    # saving the previous top of nesting stack.
    #
    # We could save the full stack, but we only need the top.  Copying
    # the full nesting stack would slow down cpplint by ~10%.
    self.previous_stack_top = []

    # Stack of _PreprocessorInfo objects.
    self.pp_stack = []

  def SeenOpenBrace(self):
    """Check if we have seen the opening brace for the innermost block.

    Returns:
      True if we have seen the opening brace, False if the innermost
      block is still expecting an opening brace.
    """
    return (not self.stack) or self.stack[-1].seen_open_brace

  def InNamespaceBody(self):
    """Check if we are currently one level inside a namespace body.

    Returns:
      True if top of the stack is a namespace block, False otherwise.
    """
    return self.stack and isinstance(self.stack[-1], _NamespaceInfo)

  def InExternC(self):
    """Check if we are currently one level inside an 'extern "C"' block.

    Returns:
      True if top of the stack is an extern block, False otherwise.
    """
    return self.stack and isinstance(self.stack[-1], _ExternCInfo)

  def InClassDeclaration(self):
    """Check if we are currently one level inside a class or struct declaration.

    Returns:
      True if top of the stack is a class/struct, False otherwise.
    """
    return self.stack and isinstance(self.stack[-1], _ClassInfo)

  def InAsmBlock(self):
    """Check if we are currently one level inside an inline ASM block.

    Returns:
      True if the top of the stack is a block containing inline ASM.
    """
    return self.stack and self.stack[-1].inline_asm != _NO_ASM

  def InTemplateArgumentList(self, clean_lines, linenum, pos):
    """Check if current position is inside template argument list.

    Args:
      clean_lines: A CleansedLines instance containing the file.
      linenum: The number of the line to check.
      pos: position just after the suspected template argument.
    Returns:
      True if (linenum, pos) is inside template arguments.
    """
    while linenum < clean_lines.NumLines():
      # Find the earliest character that might indicate a template argument
      line = clean_lines.elided[linenum]
      match = Match(r'^[^{};=\[\]\.<>]*(.)', line[pos:])
      if not match:
        linenum += 1
        pos = 0
        continue
      token = match.group(1)
      pos += len(match.group(0))

      # These things do not look like template argument list:
      #   class Suspect {
      #   class Suspect x; }
      if token in ('{', '}', ';'): return False

      # These things look like template argument list:
      #   template <class Suspect>
      #   template <class Suspect = default_value>
      #   template <class Suspect[]>
      #   template <class Suspect...>
      if token in ('>', '=', '[', ']', '.'): return True

      # Check if token is an unmatched '<'.
      # If not, move on to the next character.
      if token != '<':
        pos += 1
        if pos >= len(line):
          linenum += 1
          pos = 0
        continue

      # We can't be sure if we just find a single '<', and need to
      # find the matching '>'.
      (_, end_line, end_pos) = CloseExpression(clean_lines, linenum, pos - 1)
      if end_pos < 0:
        # Not sure if template argument list or syntax error in file
        return False
      linenum = end_line
      pos = end_pos
    return False

  def UpdatePreprocessor(self, line):
    """Update preprocessor stack.

    We need to handle preprocessors due to classes like this:
      #ifdef SWIG
      struct ResultDetailsPageElementExtensionPoint {
      #else
      struct ResultDetailsPageElementExtensionPoint : public Extension {
      #endif

    We make the following assumptions (good enough for most files):
    - Preprocessor condition evaluates to true from #if up to first
      #else/#elif/#endif.

    - Preprocessor condition evaluates to false from #else/#elif up
      to #endif.  We still perform lint checks on these lines, but
      these do not affect nesting stack.

    Args:
      line: current line to check.
    """
    if Match(r'^\s*#\s*(if|ifdef|ifndef)\b', line):
      # Beginning of #if block, save the nesting stack here.  The saved
      # stack will allow us to restore the parsing state in the #else case.
      self.pp_stack.append(_PreprocessorInfo(copy.deepcopy(self.stack)))
    elif Match(r'^\s*#\s*(else|elif)\b', line):
      # Beginning of #else block
      if self.pp_stack:
        if not self.pp_stack[-1].seen_else:
          # This is the first #else or #elif block.  Remember the
          # whole nesting stack up to this point.  This is what we
          # keep after the #endif.
          self.pp_stack[-1].seen_else = True
          self.pp_stack[-1].stack_before_else = copy.deepcopy(self.stack)

        # Restore the stack to how it was before the #if
        self.stack = copy.deepcopy(self.pp_stack[-1].stack_before_if)
      else:
        # TODO(unknown): unexpected #else, issue warning?
        pass
    elif Match(r'^\s*#\s*endif\b', line):
      # End of #if or #else blocks.
      if self.pp_stack:
        # If we saw an #else, we will need to restore the nesting
        # stack to its former state before the #else, otherwise we
        # will just continue from where we left off.
        if self.pp_stack[-1].seen_else:
          # Here we can just use a shallow copy since we are the last
          # reference to it.
          self.stack = self.pp_stack[-1].stack_before_else
        # Drop the corresponding #if
        self.pp_stack.pop()
      else:
        # TODO(unknown): unexpected #endif, issue warning?
        pass

  # TODO(unknown): Update() is too long, but we will refactor later.
  def Update(self, filename, clean_lines, linenum, error):
    """Update nesting state with current line.

    Args:
      filename: The name of the current file.
      clean_lines: A CleansedLines instance containing the file.
      linenum: The number of the line to check.
      error: The function to call with any errors found.
    """
    line = clean_lines.elided[linenum]

    # Remember top of the previous nesting stack.
    #
    # The stack is always pushed/popped and not modified in place, so
    # we can just do a shallow copy instead of copy.deepcopy.  Using
    # deepcopy would slow down cpplint by ~28%.
    if self.stack:
      self.previous_stack_top = self.stack[-1]
    else:
      self.previous_stack_top = None

    # Update pp_stack
    self.UpdatePreprocessor(line)

    # Count parentheses.  This is to avoid adding struct arguments to
    # the nesting stack.
    if self.stack:
      inner_block = self.stack[-1]
      depth_change = line.count('(') - line.count(')')
      inner_block.open_parentheses += depth_change

      # Also check if we are starting or ending an inline assembly block.
      if inner_block.inline_asm in (_NO_ASM, _END_ASM):
        if (depth_change != 0 and
            inner_block.open_parentheses == 1 and
            _MATCH_ASM.match(line)):
          # Enter assembly block
          inner_block.inline_asm = _INSIDE_ASM
        else:
          # Not entering assembly block.  If previous line was _END_ASM,
          # we will now shift to _NO_ASM state.
          inner_block.inline_asm = _NO_ASM
      elif (inner_block.inline_asm == _INSIDE_ASM and
            inner_block.open_parentheses == 0):
        # Exit assembly block
        inner_block.inline_asm = _END_ASM

    # Consume namespace declaration at the beginning of the line.  Do
    # this in a loop so that we catch same line declarations like this:
    #   namespace proto2 { namespace bridge { class MessageSet; } }
    while True:
      # Match start of namespace.  The "\b\s*" below catches namespace
      # declarations even if it weren't followed by a whitespace, this
      # is so that we don't confuse our namespace checker.  The
      # missing spaces will be flagged by CheckSpacing.
      namespace_decl_match = Match(r'^\s*namespace\b\s*([:\w]+)?(.*)$', line)
      if not namespace_decl_match:
        break

      new_namespace = _NamespaceInfo(namespace_decl_match.group(1), linenum)
      self.stack.append(new_namespace)

      line = namespace_decl_match.group(2)
      if line.find('{') != -1:
        new_namespace.seen_open_brace = True
        line = line[line.find('{') + 1:]

    # Look for a class declaration in whatever is left of the line
    # after parsing namespaces.  The regexp accounts for decorated classes
    # such as in:
    #   class LOCKABLE API Object {
    #   };
    class_decl_match = Match(
        r'^(\s*(?:template\s*<[\w\s<>,:]*>\s*)?'
        r'(class|struct)\s+(?:[A-Z_]+\s+)*(\w+(?:::\w+)*))'
        r'(.*)$', line)
    if (class_decl_match and
        (not self.stack or self.stack[-1].open_parentheses == 0)):
      # We do not want to accept classes that are actually template arguments:
      #   template <class Ignore1,
      #             class Ignore2 = Default<Args>,
      #             template <Args> class Ignore3>
      #   void Function() {};
      #
      # To avoid template argument cases, we scan forward and look for
      # an unmatched '>'.  If we see one, assume we are inside a
      # template argument list.
      end_declaration = len(class_decl_match.group(1))
      if not self.InTemplateArgumentList(clean_lines, linenum, end_declaration):
        self.stack.append(_ClassInfo(
            class_decl_match.group(3), class_decl_match.group(2),
            clean_lines, linenum))
        line = class_decl_match.group(4)

    # If we have not yet seen the opening brace for the innermost block,
    # run checks here.
    if not self.SeenOpenBrace():
      self.stack[-1].CheckBegin(filename, clean_lines, linenum, error)

    # Update access control if we are inside a class/struct
    if self.stack and isinstance(self.stack[-1], _ClassInfo):
      classinfo = self.stack[-1]
      access_match = Match(
          r'^(.*)\b(public|private|protected|signals)(\s+(?:slots\s*)?)?'
          r':(?:[^:]|$)',
          line)
      if access_match:
        classinfo.access = access_match.group(2)

        # Check that access keywords are indented +1 space.  Skip this
        # check if the keywords are not preceded by whitespaces.
        indent = access_match.group(1)
        if (len(indent) != classinfo.class_indent + 1 and
            Match(r'^\s*$', indent)):
          if classinfo.is_struct:
            parent = 'struct ' + classinfo.name
          else:
            parent = 'class ' + classinfo.name
          slots = ''
          if access_match.group(3):
            slots = access_match.group(3)
          error(filename, linenum, 'whitespace/indent', 3,
                '%s%s: should be indented +1 space inside %s' % (
                    access_match.group(2), slots, parent))

    # Consume braces or semicolons from what's left of the line
    while True:
      # Match first brace, semicolon, or closed parenthesis.
      matched = Match(r'^[^{;)}]*([{;)}])(.*)$', line)
      if not matched:
        break

      token = matched.group(1)
      if token == '{':
        # If namespace or class hasn't seen a opening brace yet, mark
        # namespace/class head as complete.  Push a new block onto the
        # stack otherwise.
        if not self.SeenOpenBrace():
          self.stack[-1].seen_open_brace = True
        elif Match(r'^extern\s*"[^"]*"\s*\{', line):
          self.stack.append(_ExternCInfo(linenum))
        else:
          self.stack.append(_BlockInfo(linenum, True))
          if _MATCH_ASM.match(line):
            self.stack[-1].inline_asm = _BLOCK_ASM

      elif token == ';' or token == ')':
        # If we haven't seen an opening brace yet, but we already saw
        # a semicolon, this is probably a forward declaration.  Pop
        # the stack for these.
        #
        # Similarly, if we haven't seen an opening brace yet, but we
        # already saw a closing parenthesis, then these are probably
        # function arguments with extra "class" or "struct" keywords.
        # Also pop these stack for these.
        if not self.SeenOpenBrace():
          self.stack.pop()
      else:  # token == '}'
        # Perform end of block checks and pop the stack.
        if self.stack:
          self.stack[-1].CheckEnd(filename, clean_lines, linenum, error)
          self.stack.pop()
      line = matched.group(2)

  def InnermostClass(self):
    """Get class info on the top of the stack.

    Returns:
      A _ClassInfo object if we are inside a class, or None otherwise.
    """
    for i in range(len(self.stack), 0, -1):
      classinfo = self.stack[i - 1]
      if isinstance(classinfo, _ClassInfo):
        return classinfo
    return None

  def CheckCompletedBlocks(self, filename, error):
    """Checks that all classes and namespaces have been completely parsed.

    Call this when all lines in a file have been processed.
    Args:
      filename: The name of the current file.
      error: The function to call with any errors found.
    """
    # Note: This test can result in false positives if #ifdef constructs
    # get in the way of brace matching. See the testBuildClass test in
    # cpplint_unittest.py for an example of this.
    for obj in self.stack:
      if isinstance(obj, _ClassInfo):
        error(filename, obj.starting_linenum, 'build/class', 5,
              'Failed to find complete declaration of class %s' %
              obj.name)
      elif isinstance(obj, _NamespaceInfo):
        error(filename, obj.starting_linenum, 'build/namespaces', 5,
              'Failed to find complete declaration of namespace %s' %
              obj.name)


def CheckForNonStandardConstructs(filename, clean_lines, linenum,
                                  nesting_state, error):
  r"""Logs an error if we see certain non-ANSI constructs ignored by gcc-2.

  Complain about several constructs which gcc-2 accepts, but which are
  not standard C++.  Warning about these in lint is one way to ease the
  transition to new compilers.
  - put storage class first (e.g. "static const" instead of "const static").
  - "%lld" instead of %qd" in printf-type functions.
  - "%1$d" is non-standard in printf-type functions.
  - "\%" is an undefined character escape sequence.
  - text after #endif is not allowed.
  - invalid inner-style forward declaration.
  - >? and <? operators, and their >?= and <?= cousins.

  Additionally, check for constructor/destructor style violations and reference
  members, as it is very convenient to do so while checking for
  gcc-2 compliance.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    nesting_state: A NestingState instance which maintains information about
                   the current stack of nested blocks being parsed.
    error: A callable to which errors are reported, which takes 4 arguments:
           filename, line number, error level, and message
  """

  # Remove comments from the line, but leave in strings for now.
  line = clean_lines.lines[linenum]

  if Search(r'printf\s*\(.*".*%[-+ ]?\d*q', line):
    error(filename, linenum, 'runtime/printf_format', 3,
          '%q in format strings is deprecated.  Use %ll instead.')

  if Search(r'printf\s*\(.*".*%\d+\$', line):
    error(filename, linenum, 'runtime/printf_format', 2,
          '%N$ formats are unconventional.  Try rewriting to avoid them.')

  # Remove escaped backslashes before looking for undefined escapes.
  line = line.replace('\\\\', '')

  if Search(r'("|\').*\\(%|\[|\(|{)', line):
    error(filename, linenum, 'build/printf_format', 3,
          '%, [, (, and { are undefined character escapes.  Unescape them.')

  # For the rest, work with both comments and strings removed.
  line = clean_lines.elided[linenum]

  if Search(r'\b(const|volatile|void|char|short|int|long'
            r'|float|double|signed|unsigned'
            r'|schar|u?int8|u?int16|u?int32|u?int64)'
            r'\s+(register|static|extern|typedef)\b',
            line):
    error(filename, linenum, 'build/storage_class', 5,
          'Storage-class specifier (static, extern, typedef, etc) should be '
          'at the beginning of the declaration.')

  if Match(r'\s*#\s*endif\s*[^/\s]+', line):
    error(filename, linenum, 'build/endif_comment', 5,
          'Uncommented text after #endif is non-standard.  Use a comment.')

  if Match(r'\s*class\s+(\w+\s*::\s*)+\w+\s*;', line):
    error(filename, linenum, 'build/forward_decl', 5,
          'Inner-style forward declarations are invalid.  Remove this line.')

  if Search(r'(\w+|[+-]?\d+(\.\d*)?)\s*(<|>)\?=?\s*(\w+|[+-]?\d+)(\.\d*)?',
            line):
    error(filename, linenum, 'build/deprecated', 3,
          '>? and <? (max and min) operators are non-standard and deprecated.')

  if Search(r'^\s*const\s*string\s*&\s*\w+\s*;', line):
    # TODO(unknown): Could it be expanded safely to arbitrary references,
    # without triggering too many false positives? The first
    # attempt triggered 5 warnings for mostly benign code in the regtest, hence
    # the restriction.
    # Here's the original regexp, for the reference:
    # type_name = r'\w+((\s*::\s*\w+)|(\s*<\s*\w+?\s*>))?'
    # r'\s*const\s*' + type_name + '\s*&\s*\w+\s*;'
    error(filename, linenum, 'runtime/member_string_references', 2,
          'const string& members are dangerous. It is much better to use '
          'alternatives, such as pointers or simple constants.')

  # Everything else in this function operates on class declarations.
  # Return early if the top of the nesting stack is not a class, or if
  # the class head is not completed yet.
  classinfo = nesting_state.InnermostClass()
  if not classinfo or not classinfo.seen_open_brace:
    return

  # The class may have been declared with namespace or classname qualifiers.
  # The constructor and destructor will not have those qualifiers.
  base_classname = classinfo.name.split('::')[-1]

  # Look for single-argument constructors that aren't marked explicit.
  # Technically a valid construct, but against style.
  explicit_constructor_match = Match(
      r'\s+(?:(?:inline|constexpr)\s+)*(explicit\s+)?'
      r'(?:(?:inline|constexpr)\s+)*%s\s*'
      r'\(((?:[^()]|\([^()]*\))*)\)'
      % re.escape(base_classname),
      line)

  if explicit_constructor_match:
    is_marked_explicit = explicit_constructor_match.group(1)

    if not explicit_constructor_match.group(2):
      constructor_args = []
    else:
      constructor_args = explicit_constructor_match.group(2).split(',')

    # collapse arguments so that commas in template parameter lists and function
    # argument parameter lists don't split arguments in two
    i = 0
    while i < len(constructor_args):
      constructor_arg = constructor_args[i]
      while (constructor_arg.count('<') > constructor_arg.count('>') or
             constructor_arg.count('(') > constructor_arg.count(')')):
        constructor_arg += ',' + constructor_args[i + 1]
        del constructor_args[i + 1]
      constructor_args[i] = constructor_arg
      i += 1

    defaulted_args = [arg for arg in constructor_args if '=' in arg]
    noarg_constructor = (not constructor_args or  # empty arg list
                         # 'void' arg specifier
                         (len(constructor_args) == 1 and
                          constructor_args[0].strip() == 'void'))
    onearg_constructor = ((len(constructor_args) == 1 and  # exactly one arg
                           not noarg_constructor) or
                          # all but at most one arg defaulted
                          (len(constructor_args) >= 1 and
                           not noarg_constructor and
                           len(defaulted_args) >= len(constructor_args) - 1))
    initializer_list_constructor = bool(
        onearg_constructor and
        Search(r'\bstd\s*::\s*initializer_list\b', constructor_args[0]))
    copy_constructor = bool(
        onearg_constructor and
        Match(r'(const\s+)?%s(\s*<[^>]*>)?(\s+const)?\s*(?:<\w+>\s*)?&'
              % re.escape(base_classname), constructor_args[0].strip()))

    if (not is_marked_explicit and
        onearg_constructor and
        not initializer_list_constructor and
        not copy_constructor):
      if defaulted_args:
        error(filename, linenum, 'runtime/explicit', 5,
              'Constructors callable with one argument '
              'should be marked explicit.')
      else:
        error(filename, linenum, 'runtime/explicit', 5,
              'Single-parameter constructors should be marked explicit.')
    elif is_marked_explicit and not onearg_constructor:
      if noarg_constructor:
        error(filename, linenum, 'runtime/explicit', 5,
              'Zero-parameter constructors should not be marked explicit.')


def CheckSpacingForFunctionCall(filename, clean_lines, linenum, error):
  """Checks for the correctness of various spacing around function calls.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # Since function calls often occur inside if/for/while/switch
  # expressions - which have their own, more liberal conventions - we
  # first see if we should be looking inside such an expression for a
  # function call, to which we can apply more strict standards.
  fncall = line    # if there's no control flow construct, look at whole line
  for pattern in (r'\bif\s*\((.*)\)\s*{',
                  r'\bfor\s*\((.*)\)\s*{',
                  r'\bwhile\s*\((.*)\)\s*[{;]',
                  r'\bswitch\s*\((.*)\)\s*{'):
    match = Search(pattern, line)
    if match:
      fncall = match.group(1)    # look inside the parens for function calls
      break

  # Except in if/for/while/switch, there should never be space
  # immediately inside parens (eg "f( 3, 4 )").  We make an exception
  # for nested parens ( (a+b) + c ).  Likewise, there should never be
  # a space before a ( when it's a function argument.  I assume it's a
  # function argument when the char before the whitespace is legal in
  # a function name (alnum + _) and we're not starting a macro. Also ignore
  # pointers and references to arrays and functions coz they're too tricky:
  # we use a very simple way to recognize these:
  # " (something)(maybe-something)" or
  # " (something)(maybe-something," or
  # " (something)[something]"
  # Note that we assume the contents of [] to be short enough that
  # they'll never need to wrap.
  if (  # Ignore control structures.
      not Search(r'\b(if|for|while|switch|return|new|delete|catch|sizeof)\b',
                 fncall) and
      # Ignore pointers/references to functions.
      not Search(r' \([^)]+\)\([^)]*(\)|,$)', fncall) and
      # Ignore pointers/references to arrays.
      not Search(r' \([^)]+\)\[[^\]]+\]', fncall)):
    if Search(r'\w\s*\(\s(?!\s*\\$)', fncall):      # a ( used for a fn call
      error(filename, linenum, 'whitespace/parens', 4,
            'Extra space after ( in function call')
    elif Search(r'\(\s+(?!(\s*\\)|\()', fncall):
      error(filename, linenum, 'whitespace/parens', 2,
            'Extra space after (')
    if (Search(r'\w\s+\(', fncall) and
        not Search(r'_{0,2}asm_{0,2}\s+_{0,2}volatile_{0,2}\s+\(', fncall) and
        not Search(r'#\s*define|typedef|using\s+\w+\s*=', fncall) and
        not Search(r'\w\s+\((\w+::)*\*\w+\)\(', fncall) and
        not Search(r'\bcase\s+\(', fncall)):
      # TODO(unknown): Space after an operator function seem to be a common
      # error, silence those for now by restricting them to highest verbosity.
      if Search(r'\boperator_*\b', line):
        error(filename, linenum, 'whitespace/parens', 0,
              'Extra space before ( in function call')
      else:
        error(filename, linenum, 'whitespace/parens', 4,
              'Extra space before ( in function call')
    # If the ) is followed only by a newline or a { + newline, assume it's
    # part of a control statement (if/while/etc), and don't complain
    if Search(r'[^)]\s+\)\s*[^{\s]', fncall):
      # If the closing parenthesis is preceded by only whitespaces,
      # try to give a more descriptive error message.
      if Search(r'^\s+\)', fncall):
        error(filename, linenum, 'whitespace/parens', 2,
              'Closing ) should be moved to the previous line')
      else:
        error(filename, linenum, 'whitespace/parens', 2,
              'Extra space before )')


def IsBlankLine(line):
  """Returns true if the given line is blank.

  We consider a line to be blank if the line is empty or consists of
  only white spaces.

  Args:
    line: A line of a string.

  Returns:
    True, if the given line is blank.
  """
  return not line or line.isspace()


def CheckForNamespaceIndentation(filename, nesting_state, clean_lines, line,
                                 error):
  is_namespace_indent_item = (
      len(nesting_state.stack) > 1 and
      nesting_state.stack[-1].check_namespace_indentation and
      isinstance(nesting_state.previous_stack_top, _NamespaceInfo) and
      nesting_state.previous_stack_top == nesting_state.stack[-2])

  if ShouldCheckNamespaceIndentation(nesting_state, is_namespace_indent_item,
                                     clean_lines.elided, line):
    CheckItemIndentationInNamespace(filename, clean_lines.elided,
                                    line, error)


def CheckForFunctionLengths(filename, clean_lines, linenum,
                            function_state, error):
  """Reports for long function bodies.

  For an overview why this is done, see:
  https://google-styleguide.googlecode.com/svn/trunk/cppguide.xml#Write_Short_Functions

  Uses a simplistic algorithm assuming other style guidelines
  (especially spacing) are followed.
  Only checks unindented functions, so class members are unchecked.
  Trivial bodies are unchecked, so constructors with huge initializer lists
  may be missed.
  Blank/comment lines are not counted so as to avoid encouraging the removal
  of vertical space and comments just to get through a lint check.
  NOLINT *on the last line of a function* disables this check.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    function_state: Current function name and lines in body so far.
    error: The function to call with any errors found.
  """
  lines = clean_lines.lines
  line = lines[linenum]
  joined_line = ''

  starting_func = False
  regexp = r'(\w(\w|::|\*|\&|\s)*)\('  # decls * & space::name( ...
  match_result = Match(regexp, line)
  if match_result:
    # If the name is all caps and underscores, figure it's a macro and
    # ignore it, unless it's TEST or TEST_F.
    function_name = match_result.group(1).split()[-1]
    if function_name == 'TEST' or function_name == 'TEST_F' or (
        not Match(r'[A-Z_]+$', function_name)):
      starting_func = True

  if starting_func:
    body_found = False
    for start_linenum in xrange(linenum, clean_lines.NumLines()):
      start_line = lines[start_linenum]
      joined_line += ' ' + start_line.lstrip()
      if Search(r'(;|})', start_line):  # Declarations and trivial functions
        body_found = True
        break                              # ... ignore
      elif Search(r'{', start_line):
        body_found = True
        function = Search(r'((\w|:)*)\(', line).group(1)
        if Match(r'TEST', function):    # Handle TEST... macros
          parameter_regexp = Search(r'(\(.*\))', joined_line)
          if parameter_regexp:             # Ignore bad syntax
            function += parameter_regexp.group(1)
        else:
          function += '()'
        function_state.Begin(function)
        break
    if not body_found:
      # No body for the function (or evidence of a non-function) was found.
      error(filename, linenum, 'readability/fn_size', 5,
            'Lint failed to find start of function body.')
  elif Match(r'^\}\s*$', line):  # function end
    function_state.Check(error, filename, linenum)
    function_state.End()
  elif not Match(r'^\s*$', line):
    function_state.Count()  # Count non-blank/non-comment lines.


_RE_PATTERN_TODO = re.compile(r'^//(\s*)TODO(\(.+?\))?:?(\s|$)?')


def CheckComment(line, filename, linenum, next_line_start, error):
  """Checks for common mistakes in comments.

  Args:
    line: The line in question.
    filename: The name of the current file.
    linenum: The number of the line to check.
    next_line_start: The first non-whitespace column of the next line.
    error: The function to call with any errors found.
  """
  commentpos = line.find('//')
  if commentpos != -1:
    # Check if the // may be in quotes.  If so, ignore it
    if re.sub(r'\\.', '', line[0:commentpos]).count('"') % 2 == 0:
      # Allow one space for new scopes, two spaces otherwise:
      if (not (Match(r'^.*{ *//', line) and next_line_start == commentpos) and
          ((commentpos >= 1 and
            line[commentpos-1] not in string.whitespace) or
           (commentpos >= 2 and
            line[commentpos-2] not in string.whitespace))):
        error(filename, linenum, 'whitespace/comments', 2,
              'At least two spaces is best between code and comments')

      # Checks for common mistakes in TODO comments.
      comment = line[commentpos:]
      match = _RE_PATTERN_TODO.match(comment)
      if match:
        # One whitespace is correct; zero whitespace is handled elsewhere.
        leading_whitespace = match.group(1)
        if len(leading_whitespace) > 1:
          error(filename, linenum, 'whitespace/todo', 2,
                'Too many spaces before TODO')

        username = match.group(2)
        if not username:
          error(filename, linenum, 'readability/todo', 2,
                'Missing username in TODO; it should look like '
                '"// TODO(my_username): Stuff."')

        middle_whitespace = match.group(3)
        # Comparisons made explicit for correctness -- pylint: disable=g-explicit-bool-comparison
        if middle_whitespace != ' ' and middle_whitespace != '':
          error(filename, linenum, 'whitespace/todo', 2,
                'TODO(my_username) should be followed by a space')

      # If the comment contains an alphanumeric character, there
      # should be a space somewhere between it and the // unless
      # it's a /// or //! Doxygen comment.
      if (Match(r'//[^ ]*\w', comment) and
          not Match(r'(///|//\!)(\s+|$)', comment)):
        error(filename, linenum, 'whitespace/comments', 4,
              'Should have a space between // and comment')


def CheckSpacing(filename, clean_lines, linenum, nesting_state, error):
  """Checks for the correctness of various spacing issues in the code.

  Things we check for: spaces around operators, spaces after
  if/for/while/switch, no spaces around parens in function calls, two
  spaces between code and comment, don't start a block with a blank
  line, don't end a function with a blank line, don't add a blank line
  after public/protected/private, don't have too many blank lines in a row.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    nesting_state: A NestingState instance which maintains information about
                   the current stack of nested blocks being parsed.
    error: The function to call with any errors found.
  """

  # Don't use "elided" lines here, otherwise we can't check commented lines.
  # Don't want to use "raw" either, because we don't want to check inside C++11
  # raw strings,
  raw = clean_lines.lines_without_raw_strings
  line = raw[linenum]

  # Before nixing comments, check if the line is blank for no good
  # reason.  This includes the first line after a block is opened, and
  # blank lines at the end of a function (ie, right before a line like '}'
  #
  # Skip all the blank line checks if we are immediately inside a
  # namespace body.  In other words, don't issue blank line warnings
  # for this block:
  #   namespace {
  #
  #   }
  #
  # A warning about missing end of namespace comments will be issued instead.
  #
  # Also skip blank line checks for 'extern "C"' blocks, which are formatted
  # like namespaces.
  if (IsBlankLine(line) and
      not nesting_state.InNamespaceBody() and
      not nesting_state.InExternC()):
    elided = clean_lines.elided
    prev_line = elided[linenum - 1]
    prevbrace = prev_line.rfind('{')
    # TODO(unknown): Don't complain if line before blank line, and line after,
    #                both start with alnums and are indented the same amount.
    #                This ignores whitespace at the start of a namespace block
    #                because those are not usually indented.
    if prevbrace != -1 and prev_line[prevbrace:].find('}') == -1:
      # OK, we have a blank line at the start of a code block.  Before we
      # complain, we check if it is an exception to the rule: The previous
      # non-empty line has the parameters of a function header that are indented
      # 4 spaces (because they did not fit in a 80 column line when placed on
      # the same line as the function name).  We also check for the case where
      # the previous line is indented 6 spaces, which may happen when the
      # initializers of a constructor do not fit into a 80 column line.
      exception = False
      if Match(r' {6}\w', prev_line):  # Initializer list?
        # We are looking for the opening column of initializer list, which
        # should be indented 4 spaces to cause 6 space indentation afterwards.
        search_position = linenum-2
        while (search_position >= 0
               and Match(r' {6}\w', elided[search_position])):
          search_position -= 1
        exception = (search_position >= 0
                     and elided[search_position][:5] == '    :')
      else:
        # Search for the function arguments or an initializer list.  We use a
        # simple heuristic here: If the line is indented 4 spaces; and we have a
        # closing paren, without the opening paren, followed by an opening brace
        # or colon (for initializer lists) we assume that it is the last line of
        # a function header.  If we have a colon indented 4 spaces, it is an
        # initializer list.
        exception = (Match(r' {4}\w[^\(]*\)\s*(const\s*)?(\{\s*$|:)',
                           prev_line)
                     or Match(r' {4}:', prev_line))

      if not exception:
        error(filename, linenum, 'whitespace/blank_line', 2,
              'Redundant blank line at the start of a code block '
              'should be deleted.')
    # Ignore blank lines at the end of a block in a long if-else
    # chain, like this:
    #   if (condition1) {
    #     // Something followed by a blank line
    #
    #   } else if (condition2) {
    #     // Something else
    #   }
    if linenum + 1 < clean_lines.NumLines():
      next_line = raw[linenum + 1]
      if (next_line
          and Match(r'\s*}', next_line)
          and next_line.find('} else ') == -1):
        error(filename, linenum, 'whitespace/blank_line', 3,
              'Redundant blank line at the end of a code block '
              'should be deleted.')

    matched = Match(r'\s*(public|protected|private):', prev_line)
    if matched:
      error(filename, linenum, 'whitespace/blank_line', 3,
            'Do not leave a blank line after "%s:"' % matched.group(1))

  # Next, check comments
  next_line_start = 0
  if linenum + 1 < clean_lines.NumLines():
    next_line = raw[linenum + 1]
    next_line_start = len(next_line) - len(next_line.lstrip())
  CheckComment(line, filename, linenum, next_line_start, error)

  # get rid of comments and strings
  line = clean_lines.elided[linenum]

  # You shouldn't have spaces before your brackets, except maybe after
  # 'delete []' or 'return []() {};'
  if Search(r'\w\s+\[', line) and not Search(r'(?:delete|return)\s+\[', line):
    error(filename, linenum, 'whitespace/braces', 5,
          'Extra space before [')

  # In range-based for, we wanted spaces before and after the colon, but
  # not around "::" tokens that might appear.
  if (Search(r'for *\(.*[^:]:[^: ]', line) or
      Search(r'for *\(.*[^: ]:[^:]', line)):
    error(filename, linenum, 'whitespace/forcolon', 2,
          'Missing space around colon in range-based for loop')


def CheckOperatorSpacing(filename, clean_lines, linenum, error):
  """Checks for horizontal spacing around operators.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # Don't try to do spacing checks for operator methods.  Do this by
  # replacing the troublesome characters with something else,
  # preserving column position for all other characters.
  #
  # The replacement is done repeatedly to avoid false positives from
  # operators that call operators.
  while True:
    match = Match(r'^(.*\boperator\b)(\S+)(\s*\(.*)$', line)
    if match:
      line = match.group(1) + ('_' * len(match.group(2))) + match.group(3)
    else:
      break

  # We allow no-spaces around = within an if: "if ( (a=Foo()) == 0 )".
  # Otherwise not.  Note we only check for non-spaces on *both* sides;
  # sometimes people put non-spaces on one side when aligning ='s among
  # many lines (not that this is behavior that I approve of...)
  if ((Search(r'[\w.]=', line) or
       Search(r'=[\w.]', line))
      and not Search(r'\b(if|while|for) ', line)
      # Operators taken from [lex.operators] in C++11 standard.
      and not Search(r'(>=|<=|==|!=|&=|\^=|\|=|\+=|\*=|\/=|\%=)', line)
      and not Search(r'operator=', line)):
    error(filename, linenum, 'whitespace/operators', 4,
          'Missing spaces around =')

  # It's ok not to have spaces around binary operators like + - * /, but if
  # there's too little whitespace, we get concerned.  It's hard to tell,
  # though, so we punt on this one for now.  TODO.

  # You should always have whitespace around binary operators.
  #
  # Check <= and >= first to avoid false positives with < and >, then
  # check non-include lines for spacing around < and >.
  #
  # If the operator is followed by a comma, assume it's be used in a
  # macro context and don't do any checks.  This avoids false
  # positives.
  #
  # Note that && is not included here.  This is because there are too
  # many false positives due to RValue references.
  match = Search(r'[^<>=!\s](==|!=|<=|>=|\|\|)[^<>=!\s,;\)]', line)
  if match:
    error(filename, linenum, 'whitespace/operators', 3,
          'Missing spaces around %s' % match.group(1))
  elif not Match(r'#.*include', line):
    # Look for < that is not surrounded by spaces.  This is only
    # triggered if both sides are missing spaces, even though
    # technically should should flag if at least one side is missing a
    # space.  This is done to avoid some false positives with shifts.
    match = Match(r'^(.*[^\s<])<[^\s=<,]', line)
    if match:
      (_, _, end_pos) = CloseExpression(
          clean_lines, linenum, len(match.group(1)))
      if end_pos <= -1:
        error(filename, linenum, 'whitespace/operators', 3,
              'Missing spaces around <')

    # Look for > that is not surrounded by spaces.  Similar to the
    # above, we only trigger if both sides are missing spaces to avoid
    # false positives with shifts.
    match = Match(r'^(.*[^-\s>])>[^\s=>,]', line)
    if match:
      (_, _, start_pos) = ReverseCloseExpression(
          clean_lines, linenum, len(match.group(1)))
      if start_pos <= -1:
        error(filename, linenum, 'whitespace/operators', 3,
              'Missing spaces around >')

  # We allow no-spaces around << when used like this: 10<<20, but
  # not otherwise (particularly, not when used as streams)
  #
  # We also allow operators following an opening parenthesis, since
  # those tend to be macros that deal with operators.
  match = Search(r'(operator|[^\s(<])(?:L|UL|LL|ULL|l|ul|ll|ull)?<<([^\s,=<])', line)
  if (match and not (match.group(1).isdigit() and match.group(2).isdigit()) and
      not (match.group(1) == 'operator' and match.group(2) == ';')):
    error(filename, linenum, 'whitespace/operators', 3,
          'Missing spaces around <<')

  # We allow no-spaces around >> for almost anything.  This is because
  # C++11 allows ">>" to close nested templates, which accounts for
  # most cases when ">>" is not followed by a space.
  #
  # We still warn on ">>" followed by alpha character, because that is
  # likely due to ">>" being used for right shifts, e.g.:
  #   value >> alpha
  #
  # When ">>" is used to close templates, the alphanumeric letter that
  # follows would be part of an identifier, and there should still be
  # a space separating the template type and the identifier.
  #   type<type<type>> alpha
  match = Search(r'>>[a-zA-Z_]', line)
  if match:
    error(filename, linenum, 'whitespace/operators', 3,
          'Missing spaces around >>')

  # There shouldn't be space around unary operators
  match = Search(r'(!\s|~\s|[\s]--[\s;]|[\s]\+\+[\s;])', line)
  if match:
    error(filename, linenum, 'whitespace/operators', 4,
          'Extra space for operator %s' % match.group(1))


def CheckParenthesisSpacing(filename, clean_lines, linenum, error):
  """Checks for horizontal spacing around parentheses.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # No spaces after an if, while, switch, or for
  match = Search(r' (if\(|for\(|while\(|switch\()', line)
  if match:
    error(filename, linenum, 'whitespace/parens', 5,
          'Missing space before ( in %s' % match.group(1))

  # For if/for/while/switch, the left and right parens should be
  # consistent about how many spaces are inside the parens, and
  # there should either be zero or one spaces inside the parens.
  # We don't want: "if ( foo)" or "if ( foo   )".
  # Exception: "for ( ; foo; bar)" and "for (foo; bar; )" are allowed.
  match = Search(r'\b(if|for|while|switch)\s*'
                 r'\(([ ]*)(.).*[^ ]+([ ]*)\)\s*{\s*$',
                 line)
  if match:
    if len(match.group(2)) != len(match.group(4)):
      if not (match.group(3) == ';' and
              len(match.group(2)) == 1 + len(match.group(4)) or
              not match.group(2) and Search(r'\bfor\s*\(.*; \)', line)):
        error(filename, linenum, 'whitespace/parens', 5,
              'Mismatching spaces inside () in %s' % match.group(1))
    if len(match.group(2)) not in [0, 1]:
      error(filename, linenum, 'whitespace/parens', 5,
            'Should have zero or one spaces inside ( and ) in %s' %
            match.group(1))


def CheckCommaSpacing(filename, clean_lines, linenum, error):
  """Checks for horizontal spacing near commas and semicolons.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  raw = clean_lines.lines_without_raw_strings
  line = clean_lines.elided[linenum]

  # You should always have a space after a comma (either as fn arg or operator)
  #
  # This does not apply when the non-space character following the
  # comma is another comma, since the only time when that happens is
  # for empty macro arguments.
  #
  # We run this check in two passes: first pass on elided lines to
  # verify that lines contain missing whitespaces, second pass on raw
  # lines to confirm that those missing whitespaces are not due to
  # elided comments.
  if (Search(r',[^,\s]', ReplaceAll(r'\boperator\s*,\s*\(', 'F(', line)) and
      Search(r',[^,\s]', raw[linenum])):
    error(filename, linenum, 'whitespace/comma', 3,
          'Missing space after ,')

  # You should always have a space after a semicolon
  # except for few corner cases
  # TODO(unknown): clarify if 'if (1) { return 1;}' is requires one more
  # space after ;
  if Search(r';[^\s};\\)/]', line):
    error(filename, linenum, 'whitespace/semicolon', 3,
          'Missing space after ;')


def _IsType(clean_lines, nesting_state, expr):
  """Check if expression looks like a type name, returns true if so.

  Args:
    clean_lines: A CleansedLines instance containing the file.
    nesting_state: A NestingState instance which maintains information about
                   the current stack of nested blocks being parsed.
    expr: The expression to check.
  Returns:
    True, if token looks like a type.
  """
  # Keep only the last token in the expression
  last_word = Match(r'^.*(\b\S+)$', expr)
  if last_word:
    token = last_word.group(1)
  else:
    token = expr

  # Match native types and stdint types
  if _TYPES.match(token):
    return True

  # Try a bit harder to match templated types.  Walk up the nesting
  # stack until we find something that resembles a typename
  # declaration for what we are looking for.
  typename_pattern = (r'\b(?:typename|class|struct)\s+' + re.escape(token) +
                      r'\b')
  block_index = len(nesting_state.stack) - 1
  while block_index >= 0:
    if isinstance(nesting_state.stack[block_index], _NamespaceInfo):
      return False

    # Found where the opening brace is.  We want to scan from this
    # line up to the beginning of the function, minus a few lines.
    #   template <typename Type1,  // stop scanning here
    #             ...>
    #   class C
    #     : public ... {  // start scanning here
    last_line = nesting_state.stack[block_index].starting_linenum

    next_block_start = 0
    if block_index > 0:
      next_block_start = nesting_state.stack[block_index - 1].starting_linenum
    first_line = last_line
    while first_line >= next_block_start:
      if clean_lines.elided[first_line].find('template') >= 0:
        break
      first_line -= 1
    if first_line < next_block_start:
      # Didn't find any "template" keyword before reaching the next block,
      # there are probably no template things to check for this block
      block_index -= 1
      continue

    # Look for typename in the specified range
    for i in xrange(first_line, last_line + 1, 1):
      if Search(typename_pattern, clean_lines.elided[i]):
        return True
    block_index -= 1

  return False


def CheckBracesSpacing(filename, clean_lines, linenum, nesting_state, error):
  """Checks for horizontal spacing near commas.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    nesting_state: A NestingState instance which maintains information about
                   the current stack of nested blocks being parsed.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # Except after an opening paren, or after another opening brace (in case of
  # an initializer list, for instance), you should have spaces before your
  # braces when they are delimiting blocks, classes, namespaces etc.
  # And since you should never have braces at the beginning of a line,
  # this is an easy test.  Except that braces used for initialization don't
  # follow the same rule; we often don't want spaces before those.
  match = Match(r'^(.*[^ ({>]){', line)

  if match:
    # Try a bit harder to check for brace initialization.  This
    # happens in one of the following forms:
    #   Constructor() : initializer_list_{} { ... }
    #   Constructor{}.MemberFunction()
    #   Type variable{};
    #   FunctionCall(type{}, ...);
    #   LastArgument(..., type{});
    #   LOG(INFO) << type{} << " ...";
    #   map_of_type[{...}] = ...;
    #   ternary = expr ? new type{} : nullptr;
    #   OuterTemplate<InnerTemplateConstructor<Type>{}>
    #
    # We check for the character following the closing brace, and
    # silence the warning if it's one of those listed above, i.e.
    # "{.;,)<>]:".
    #
    # To account for nested initializer list, we allow any number of
    # closing braces up to "{;,)<".  We can't simply silence the
    # warning on first sight of closing brace, because that would
    # cause false negatives for things that are not initializer lists.
    #   Silence this:         But not this:
    #     Outer{                if (...) {
    #       Inner{...}            if (...){  // Missing space before {
    #     };                    }
    #
    # There is a false negative with this approach if people inserted
    # spurious semicolons, e.g. "if (cond){};", but we will catch the
    # spurious semicolon with a separate check.
    leading_text = match.group(1)
    (endline, endlinenum, endpos) = CloseExpression(
        clean_lines, linenum, len(match.group(1)))
    trailing_text = ''
    if endpos > -1:
      trailing_text = endline[endpos:]
    for offset in xrange(endlinenum + 1,
                         min(endlinenum + 3, clean_lines.NumLines() - 1)):
      trailing_text += clean_lines.elided[offset]
    # We also suppress warnings for `uint64_t{expression}` etc., as the style
    # guide recommends brace initialization for integral types to avoid
    # overflow/truncation.
    if (not Match(r'^[\s}]*[{.;,)<>\]:]', trailing_text)
        and not _IsType(clean_lines, nesting_state, leading_text)):
      error(filename, linenum, 'whitespace/braces', 5,
            'Missing space before {')

  # Make sure '} else {' has spaces.
  if Search(r'}else', line):
    error(filename, linenum, 'whitespace/braces', 5,
          'Missing space before else')

  # You shouldn't have a space before a semicolon at the end of the line.
  # There's a special case for "for" since the style guide allows space before
  # the semicolon there.
  if Search(r':\s*;\s*$', line):
    error(filename, linenum, 'whitespace/semicolon', 5,
          'Semicolon defining empty statement. Use {} instead.')
  elif Search(r'^\s*;\s*$', line):
    error(filename, linenum, 'whitespace/semicolon', 5,
          'Line contains only semicolon. If this should be an empty statement, '
          'use {} instead.')
  elif (Search(r'\s+;\s*$', line) and
        not Search(r'\bfor\b', line)):
    error(filename, linenum, 'whitespace/semicolon', 5,
          'Extra space before last semicolon. If this should be an empty '
          'statement, use {} instead.')


def IsDecltype(clean_lines, linenum, column):
  """Check if the token ending on (linenum, column) is decltype().

  Args:
    clean_lines: A CleansedLines instance containing the file.
    linenum: the number of the line to check.
    column: end column of the token to check.
  Returns:
    True if this token is decltype() expression, False otherwise.
  """
  (text, _, start_col) = ReverseCloseExpression(clean_lines, linenum, column)
  if start_col < 0:
    return False
  if Search(r'\bdecltype\s*$', text[0:start_col]):
    return True
  return False


def CheckSectionSpacing(filename, clean_lines, class_info, linenum, error):
  """Checks for additional blank line issues related to sections.

  Currently the only thing checked here is blank line before protected/private.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    class_info: A _ClassInfo objects.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  # Skip checks if the class is small, where small means 25 lines or less.
  # 25 lines seems like a good cutoff since that's the usual height of
  # terminals, and any class that can't fit in one screen can't really
  # be considered "small".
  #
  # Also skip checks if we are on the first line.  This accounts for
  # classes that look like
  #   class Foo { public: ... };
  #
  # If we didn't find the end of the class, last_line would be zero,
  # and the check will be skipped by the first condition.
  if (class_info.last_line - class_info.starting_linenum <= 24 or
      linenum <= class_info.starting_linenum):
    return

  matched = Match(r'\s*(public|protected|private):', clean_lines.lines[linenum])
  if matched:
    # Issue warning if the line before public/protected/private was
    # not a blank line, but don't do this if the previous line contains
    # "class" or "struct".  This can happen two ways:
    #  - We are at the beginning of the class.
    #  - We are forward-declaring an inner class that is semantically
    #    private, but needed to be public for implementation reasons.
    # Also ignores cases where the previous line ends with a backslash as can be
    # common when defining classes in C macros.
    prev_line = clean_lines.lines[linenum - 1]
    if (not IsBlankLine(prev_line) and
        not Search(r'\b(class|struct)\b', prev_line) and
        not Search(r'\\$', prev_line)):
      # Try a bit harder to find the beginning of the class.  This is to
      # account for multi-line base-specifier lists, e.g.:
      #   class Derived
      #       : public Base {
      end_class_head = class_info.starting_linenum
      for i in range(class_info.starting_linenum, linenum):
        if Search(r'\{\s*$', clean_lines.lines[i]):
          end_class_head = i
          break
      if end_class_head < linenum - 1:
        error(filename, linenum, 'whitespace/blank_line', 3,
              '"%s:" should be preceded by a blank line' % matched.group(1))


def GetPreviousNonBlankLine(clean_lines, linenum):
  """Return the most recent non-blank line and its line number.

  Args:
    clean_lines: A CleansedLines instance containing the file contents.
    linenum: The number of the line to check.

  Returns:
    A tuple with two elements.  The first element is the contents of the last
    non-blank line before the current line, or the empty string if this is the
    first non-blank line.  The second is the line number of that line, or -1
    if this is the first non-blank line.
  """

  prevlinenum = linenum - 1
  while prevlinenum >= 0:
    prevline = clean_lines.elided[prevlinenum]
    if not IsBlankLine(prevline):     # if not a blank line...
      return (prevline, prevlinenum)
    prevlinenum -= 1
  return ('', -1)


def CheckBraces(filename, clean_lines, linenum, error):
  """Looks for misplaced braces (e.g. at the end of line).

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """

  line = clean_lines.elided[linenum]        # get rid of comments and strings

  if Match(r'\s*{\s*$', line):
    # We allow an open brace to start a line in the case where someone is using
    # braces in a block to explicitly create a new scope, which is commonly used
    # to control the lifetime of stack-allocated variables.  Braces are also
    # used for brace initializers inside function calls.  We don't detect this
    # perfectly: we just don't complain if the last non-whitespace character on
    # the previous non-blank line is ',', ';', ':', '(', '{', or '}', or if the
    # previous line starts a preprocessor block. We also allow a brace on the
    # following line if it is part of an array initialization and would not fit
    # within the 80 character limit of the preceding line.
    prevline = GetPreviousNonBlankLine(clean_lines, linenum)[0]
    if (not Search(r'[,;:}{(]\s*$', prevline) and
        not Match(r'\s*#', prevline) and
        not (GetLineWidth(prevline) > _line_length - 2 and '[]' in prevline)):
      error(filename, linenum, 'whitespace/braces', 4,
            '{ should almost always be at the end of the previous line')

  # An else clause should be on the same line as the preceding closing brace.
  if Match(r'\s*else\b\s*(?:if\b|\{|$)', line):
    prevline = GetPreviousNonBlankLine(clean_lines, linenum)[0]
    if Match(r'\s*}\s*$', prevline):
      error(filename, linenum, 'whitespace/newline', 4,
            'An else should appear on the same line as the preceding }')

  # If braces come on one side of an else, they should be on both.
  # However, we have to worry about "else if" that spans multiple lines!
  if Search(r'else if\s*\(', line):       # could be multi-line if
    brace_on_left = bool(Search(r'}\s*else if\s*\(', line))
    # find the ( after the if
    pos = line.find('else if')
    pos = line.find('(', pos)
    if pos > 0:
      (endline, _, endpos) = CloseExpression(clean_lines, linenum, pos)
      brace_on_right = endline[endpos:].find('{') != -1
      if brace_on_left != brace_on_right:    # must be brace after if
        error(filename, linenum, 'readability/braces', 5,
              'If an else has a brace on one side, it should have it on both')
  elif Search(r'}\s*else[^{]*$', line) or Match(r'[^}]*else\s*{', line):
    error(filename, linenum, 'readability/braces', 5,
          'If an else has a brace on one side, it should have it on both')

  # Likewise, an else should never have the else clause on the same line
  if Search(r'\belse [^\s{]', line) and not Search(r'\belse if\b', line):
    error(filename, linenum, 'whitespace/newline', 4,
          'Else clause should never be on same line as else (use 2 lines)')

  # In the same way, a do/while should never be on one line
  if Match(r'\s*do [^\s{]', line):
    error(filename, linenum, 'whitespace/newline', 4,
          'do/while clauses should not be on a single line')

  # Check single-line if/else bodies. The style guide says 'curly braces are not
  # required for single-line statements'. We additionally allow multi-line,
  # single statements, but we reject anything with more than one semicolon in
  # it. This means that the first semicolon after the if should be at the end of
  # its line, and the line after that should have an indent level equal to or
  # lower than the if. We also check for ambiguous if/else nesting without
  # braces.
  if_else_match = Search(r'\b(if\s*\(|else\b)', line)
  if if_else_match and not Match(r'\s*#', line):
    if_indent = GetIndentLevel(line)
    endline, endlinenum, endpos = line, linenum, if_else_match.end()
    if_match = Search(r'\bif\s*\(', line)
    if if_match:
      # This could be a multiline if condition, so find the end first.
      pos = if_match.end() - 1
      (endline, endlinenum, endpos) = CloseExpression(clean_lines, linenum, pos)
    # Check for an opening brace, either directly after the if or on the next
    # line. If found, this isn't a single-statement conditional.
    if (not Match(r'\s*{', endline[endpos:])
        and not (Match(r'\s*$', endline[endpos:])
                 and endlinenum < (len(clean_lines.elided) - 1)
                 and Match(r'\s*{', clean_lines.elided[endlinenum + 1]))):
      while (endlinenum < len(clean_lines.elided)
             and ';' not in clean_lines.elided[endlinenum][endpos:]):
        endlinenum += 1
        endpos = 0
      if endlinenum < len(clean_lines.elided):
        endline = clean_lines.elided[endlinenum]
        # We allow a mix of whitespace and closing braces (e.g. for one-liner
        # methods) and a single \ after the semicolon (for macros)
        endpos = endline.find(';')
        if not Match(r';[\s}]*(\\?)$', endline[endpos:]):
          # Semicolon isn't the last character, there's something trailing.
          # Output a warning if the semicolon is not contained inside
          # a lambda expression.
          if not Match(r'^[^{};]*\[[^\[\]]*\][^{}]*\{[^{}]*\}\s*\)*[;,]\s*$',
                       endline):
            error(filename, linenum, 'readability/braces', 4,
                  'If/else bodies with multiple statements require braces')
        elif endlinenum < len(clean_lines.elided) - 1:
          # Make sure the next line is dedented
          next_line = clean_lines.elided[endlinenum + 1]
          next_indent = GetIndentLevel(next_line)
          # With ambiguous nested if statements, this will error out on the
          # if that *doesn't* match the else, regardless of whether it's the
          # inner one or outer one.
          if (if_match and Match(r'\s*else\b', next_line)
              and next_indent != if_indent):
            error(filename, linenum, 'readability/braces', 4,
                  'Else clause should be indented at the same level as if. '
                  'Ambiguous nested if/else chains require braces.')
          elif next_indent > if_indent:
            error(filename, linenum, 'readability/braces', 4,
                  'If/else bodies with multiple statements require braces')


def CheckTrailingSemicolon(filename, clean_lines, linenum, error):
  """Looks for redundant trailing semicolon.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """

  line = clean_lines.elided[linenum]

  # Block bodies should not be followed by a semicolon.  Due to C++11
  # brace initialization, there are more places where semicolons are
  # required than not, so we use a whitelist approach to check these
  # rather than a blacklist.  These are the places where "};" should
  # be replaced by just "}":
  # 1. Some flavor of block following closing parenthesis:
  #    for (;;) {};
  #    while (...) {};
  #    switch (...) {};
  #    Function(...) {};
  #    if (...) {};
  #    if (...) else if (...) {};
  #
  # 2. else block:
  #    if (...) else {};
  #
  # 3. const member function:
  #    Function(...) const {};
  #
  # 4. Block following some statement:
  #    x = 42;
  #    {};
  #
  # 5. Block at the beginning of a function:
  #    Function(...) {
  #      {};
  #    }
  #
  #    Note that naively checking for the preceding "{" will also match
  #    braces inside multi-dimensional arrays, but this is fine since
  #    that expression will not contain semicolons.
  #
  # 6. Block following another block:
  #    while (true) {}
  #    {};
  #
  # 7. End of namespaces:
  #    namespace {};
  #
  #    These semicolons seems far more common than other kinds of
  #    redundant semicolons, possibly due to people converting classes
  #    to namespaces.  For now we do not warn for this case.
  #
  # Try matching case 1 first.
  match = Match(r'^(.*\)\s*)\{', line)
  if match:
    # Matched closing parenthesis (case 1).  Check the token before the
    # matching opening parenthesis, and don't warn if it looks like a
    # macro.  This avoids these false positives:
    #  - macro that defines a base class
    #  - multi-line macro that defines a base class
    #  - macro that defines the whole class-head
    #
    # But we still issue warnings for macros that we know are safe to
    # warn, specifically:
    #  - TEST, TEST_F, TEST_P, MATCHER, MATCHER_P
    #  - TYPED_TEST
    #  - INTERFACE_DEF
    #  - EXCLUSIVE_LOCKS_REQUIRED, SHARED_LOCKS_REQUIRED, LOCKS_EXCLUDED:
    #
    # We implement a whitelist of safe macros instead of a blacklist of
    # unsafe macros, even though the latter appears less frequently in
    # google code and would have been easier to implement.  This is because
    # the downside for getting the whitelist wrong means some extra
    # semicolons, while the downside for getting the blacklist wrong
    # would result in compile errors.
    #
    # In addition to macros, we also don't want to warn on
    #  - Compound literals
    #  - Lambdas
    #  - alignas specifier with anonymous structs
    #  - decltype
    closing_brace_pos = match.group(1).rfind(')')
    opening_parenthesis = ReverseCloseExpression(
        clean_lines, linenum, closing_brace_pos)
    if opening_parenthesis[2] > -1:
      line_prefix = opening_parenthesis[0][0:opening_parenthesis[2]]
      macro = Search(r'\b([A-Z_][A-Z0-9_]*)\s*$', line_prefix)
      func = Match(r'^(.*\])\s*$', line_prefix)
      if ((macro and
           macro.group(1) not in (
               'TEST', 'TEST_F', 'MATCHER', 'MATCHER_P', 'TYPED_TEST',
               'EXCLUSIVE_LOCKS_REQUIRED', 'SHARED_LOCKS_REQUIRED',
               'LOCKS_EXCLUDED', 'INTERFACE_DEF')) or
          (func and not Search(r'\boperator\s*\[\s*\]', func.group(1))) or
          Search(r'\b(?:struct|union)\s+alignas\s*$', line_prefix) or
          Search(r'\bdecltype$', line_prefix) or
          Search(r'\s+=\s*$', line_prefix)):
        match = None
    if (match and
        opening_parenthesis[1] > 1 and
        Search(r'\]\s*$', clean_lines.elided[opening_parenthesis[1] - 1])):
      # Multi-line lambda-expression
      match = None

  else:
    # Try matching cases 2-3.
    match = Match(r'^(.*(?:else|\)\s*const)\s*)\{', line)
    if not match:
      # Try matching cases 4-6.  These are always matched on separate lines.
      #
      # Note that we can't simply concatenate the previous line to the
      # current line and do a single match, otherwise we may output
      # duplicate warnings for the blank line case:
      #   if (cond) {
      #     // blank line
      #   }
      prevline = GetPreviousNonBlankLine(clean_lines, linenum)[0]
      if prevline and Search(r'[;{}]\s*$', prevline):
        match = Match(r'^(\s*)\{', line)

  # Check matching closing brace
  if match:
    (endline, endlinenum, endpos) = CloseExpression(
        clean_lines, linenum, len(match.group(1)))
    if endpos > -1 and Match(r'^\s*;', endline[endpos:]):
      # Current {} pair is eligible for semicolon check, and we have found
      # the redundant semicolon, output warning here.
      #
      # Note: because we are scanning forward for opening braces, and
      # outputting warnings for the matching closing brace, if there are
      # nested blocks with trailing semicolons, we will get the error
      # messages in reversed order.

      # We need to check the line forward for NOLINT
      raw_lines = clean_lines.raw_lines
      ParseNolintSuppressions(filename, raw_lines[endlinenum-1], endlinenum-1,
                              error)
      ParseNolintSuppressions(filename, raw_lines[endlinenum], endlinenum,
                              error)

      error(filename, endlinenum, 'readability/braces', 4,
            "You don't need a ; after a }")


def CheckEmptyBlockBody(filename, clean_lines, linenum, error):
  """Look for empty loop/conditional body with only a single semicolon.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """

  # Search for loop keywords at the beginning of the line.  Because only
  # whitespaces are allowed before the keywords, this will also ignore most
  # do-while-loops, since those lines should start with closing brace.
  #
  # We also check "if" blocks here, since an empty conditional block
  # is likely an error.
  line = clean_lines.elided[linenum]
  matched = Match(r'\s*(for|while|if)\s*\(', line)
  if matched:
    # Find the end of the conditional expression.
    (end_line, end_linenum, end_pos) = CloseExpression(
        clean_lines, linenum, line.find('('))

    # Output warning if what follows the condition expression is a semicolon.
    # No warning for all other cases, including whitespace or newline, since we
    # have a separate check for semicolons preceded by whitespace.
    if end_pos >= 0 and Match(r';', end_line[end_pos:]):
      if matched.group(1) == 'if':
        error(filename, end_linenum, 'whitespace/empty_conditional_body', 5,
              'Empty conditional bodies should use {}')
      else:
        error(filename, end_linenum, 'whitespace/empty_loop_body', 5,
              'Empty loop bodies should use {} or continue')

    # Check for if statements that have completely empty bodies (no comments)
    # and no else clauses.
    if end_pos >= 0 and matched.group(1) == 'if':
      # Find the position of the opening { for the if statement.
      # Return without logging an error if it has no brackets.
      opening_linenum = end_linenum
      opening_line_fragment = end_line[end_pos:]
      # Loop until EOF or find anything that's not whitespace or opening {.
      while not Search(r'^\s*\{', opening_line_fragment):
        if Search(r'^(?!\s*$)', opening_line_fragment):
          # Conditional has no brackets.
          return
        opening_linenum += 1
        if opening_linenum == len(clean_lines.elided):
          # Couldn't find conditional's opening { or any code before EOF.
          return
        opening_line_fragment = clean_lines.elided[opening_linenum]
      # Set opening_line (opening_line_fragment may not be entire opening line).
      opening_line = clean_lines.elided[opening_linenum]

      # Find the position of the closing }.
      opening_pos = opening_line_fragment.find('{')
      if opening_linenum == end_linenum:
        # We need to make opening_pos relative to the start of the entire line.
        opening_pos += end_pos
      (closing_line, closing_linenum, closing_pos) = CloseExpression(
          clean_lines, opening_linenum, opening_pos)
      if closing_pos < 0:
        return

      # Now construct the body of the conditional. This consists of the portion
      # of the opening line after the {, all lines until the closing line,
      # and the portion of the closing line before the }.
      if (clean_lines.raw_lines[opening_linenum] !=
          CleanseComments(clean_lines.raw_lines[opening_linenum])):
        # Opening line ends with a comment, so conditional isn't empty.
        return
      if closing_linenum > opening_linenum:
        # Opening line after the {. Ignore comments here since we checked above.
        body = list(opening_line[opening_pos+1:])
        # All lines until closing line, excluding closing line, with comments.
        body.extend(clean_lines.raw_lines[opening_linenum+1:closing_linenum])
        # Closing line before the }. Won't (and can't) have comments.
        body.append(clean_lines.elided[closing_linenum][:closing_pos-1])
        body = '\n'.join(body)
      else:
        # If statement has brackets and fits on a single line.
        body = opening_line[opening_pos+1:closing_pos-1]

      # Check if the body is empty
      if not _EMPTY_CONDITIONAL_BODY_PATTERN.search(body):
        return
      # The body is empty. Now make sure there's not an else clause.
      current_linenum = closing_linenum
      current_line_fragment = closing_line[closing_pos:]
      # Loop until EOF or find anything that's not whitespace or else clause.
      while Search(r'^\s*$|^(?=\s*else)', current_line_fragment):
        if Search(r'^(?=\s*else)', current_line_fragment):
          # Found an else clause, so don't log an error.
          return
        current_linenum += 1
        if current_linenum == len(clean_lines.elided):
          break
        current_line_fragment = clean_lines.elided[current_linenum]

      # The body is empty and there's no else clause until EOF or other code.
      error(filename, end_linenum, 'whitespace/empty_if_body', 4,
            ('If statement had no body and no else clause'))


def FindCheckMacro(line):
  """Find a replaceable CHECK-like macro.

  Args:
    line: line to search on.
  Returns:
    (macro name, start position), or (None, -1) if no replaceable
    macro is found.
  """
  for macro in _CHECK_MACROS:
    i = line.find(macro)
    if i >= 0:
      # Find opening parenthesis.  Do a regular expression match here
      # to make sure that we are matching the expected CHECK macro, as
      # opposed to some other macro that happens to contain the CHECK
      # substring.
      matched = Match(r'^(.*\b' + macro + r'\s*)\(', line)
      if not matched:
        continue
      return (macro, len(matched.group(1)))
  return (None, -1)


def CheckCheck(filename, clean_lines, linenum, error):
  """Checks the use of CHECK and EXPECT macros.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """

  # Decide the set of replacement macros that should be suggested
  lines = clean_lines.elided
  (check_macro, start_pos) = FindCheckMacro(lines[linenum])
  if not check_macro:
    return

  # Find end of the boolean expression by matching parentheses
  (last_line, end_line, end_pos) = CloseExpression(
      clean_lines, linenum, start_pos)
  if end_pos < 0:
    return

  # If the check macro is followed by something other than a
  # semicolon, assume users will log their own custom error messages
  # and don't suggest any replacements.
  if not Match(r'\s*;', last_line[end_pos:]):
    return

  if linenum == end_line:
    expression = lines[linenum][start_pos + 1:end_pos - 1]
  else:
    expression = lines[linenum][start_pos + 1:]
    for i in xrange(linenum + 1, end_line):
      expression += lines[i]
    expression += last_line[0:end_pos - 1]

  # Parse expression so that we can take parentheses into account.
  # This avoids false positives for inputs like "CHECK((a < 4) == b)",
  # which is not replaceable by CHECK_LE.
  lhs = ''
  rhs = ''
  operator = None
  while expression:
    matched = Match(r'^\s*(<<|<<=|>>|>>=|->\*|->|&&|\|\||'
                    r'==|!=|>=|>|<=|<|\()(.*)$', expression)
    if matched:
      token = matched.group(1)
      if token == '(':
        # Parenthesized operand
        expression = matched.group(2)
        (end, _) = FindEndOfExpressionInLine(expression, 0, ['('])
        if end < 0:
          return  # Unmatched parenthesis
        lhs += '(' + expression[0:end]
        expression = expression[end:]
      elif token in ('&&', '||'):
        # Logical and/or operators.  This means the expression
        # contains more than one term, for example:
        #   CHECK(42 < a && a < b);
        #
        # These are not replaceable with CHECK_LE, so bail out early.
        return
      elif token in ('<<', '<<=', '>>', '>>=', '->*', '->'):
        # Non-relational operator
        lhs += token
        expression = matched.group(2)
      else:
        # Relational operator
        operator = token
        rhs = matched.group(2)
        break
    else:
      # Unparenthesized operand.  Instead of appending to lhs one character
      # at a time, we do another regular expression match to consume several
      # characters at once if possible.  Trivial benchmark shows that this
      # is more efficient when the operands are longer than a single
      # character, which is generally the case.
      matched = Match(r'^([^-=!<>()&|]+)(.*)$', expression)
      if not matched:
        matched = Match(r'^(\s*\S)(.*)$', expression)
        if not matched:
          break
      lhs += matched.group(1)
      expression = matched.group(2)

  # Only apply checks if we got all parts of the boolean expression
  if not (lhs and operator and rhs):
    return

  # Check that rhs do not contain logical operators.  We already know
  # that lhs is fine since the loop above parses out && and ||.
  if rhs.find('&&') > -1 or rhs.find('||') > -1:
    return

  # At least one of the operands must be a constant literal.  This is
  # to avoid suggesting replacements for unprintable things like
  # CHECK(variable != iterator)
  #
  # The following pattern matches decimal, hex integers, strings, and
  # characters (in that order).
  lhs = lhs.strip()
  rhs = rhs.strip()
  match_constant = r'^([-+]?(\d+|0[xX][0-9a-fA-F]+)[lLuU]{0,3}|".*"|\'.*\')$'
  if Match(match_constant, lhs) or Match(match_constant, rhs):
    # Note: since we know both lhs and rhs, we can provide a more
    # descriptive error message like:
    #   Consider using CHECK_EQ(x, 42) instead of CHECK(x == 42)
    # Instead of:
    #   Consider using CHECK_EQ instead of CHECK(a == b)
    #
    # We are still keeping the less descriptive message because if lhs
    # or rhs gets long, the error message might become unreadable.
    error(filename, linenum, 'readability/check', 2,
          'Consider using %s instead of %s(a %s b)' % (
              _CHECK_REPLACEMENT[check_macro][operator],
              check_macro, operator))


def CheckAltTokens(filename, clean_lines, linenum, error):
  """Check alternative keywords being used in boolean expressions.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # Avoid preprocessor lines
  if Match(r'^\s*#', line):
    return

  # Last ditch effort to avoid multi-line comments.  This will not help
  # if the comment started before the current line or ended after the
  # current line, but it catches most of the false positives.  At least,
  # it provides a way to workaround this warning for people who use
  # multi-line comments in preprocessor macros.
  #
  # TODO(unknown): remove this once cpplint has better support for
  # multi-line comments.
  if line.find('/*') >= 0 or line.find('*/') >= 0:
    return

  for match in _ALT_TOKEN_REPLACEMENT_PATTERN.finditer(line):
    error(filename, linenum, 'readability/alt_tokens', 2,
          'Use operator %s instead of %s' % (
              _ALT_TOKEN_REPLACEMENT[match.group(1)], match.group(1)))


def GetLineWidth(line):
  """Determines the width of the line in column positions.

  Args:
    line: A string, which may be a Unicode string.

  Returns:
    The width of the line in column positions, accounting for Unicode
    combining characters and wide characters.
  """
  if isinstance(line, unicode):
    width = 0
    for uc in unicodedata.normalize('NFC', line):
      if unicodedata.east_asian_width(uc) in ('W', 'F'):
        width += 2
      elif not unicodedata.combining(uc):
        width += 1
    return width
  else:
    return len(line)


def CheckStyle(filename, clean_lines, linenum, file_extension, nesting_state,
               error):
  """Checks rules from the 'C++ style rules' section of cppguide.html.

  Most of these rules are hard to test (naming, comment style), but we
  do what we can.  In particular we check for 2-space indents, line lengths,
  tab usage, spaces inside code, etc.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    file_extension: The extension (without the dot) of the filename.
    nesting_state: A NestingState instance which maintains information about
                   the current stack of nested blocks being parsed.
    error: The function to call with any errors found.
  """

  # Don't use "elided" lines here, otherwise we can't check commented lines.
  # Don't want to use "raw" either, because we don't want to check inside C++11
  # raw strings,
  raw_lines = clean_lines.lines_without_raw_strings
  line = raw_lines[linenum]
  prev = raw_lines[linenum - 1] if linenum > 0 else ''

  if line.find('\t') != -1:
    error(filename, linenum, 'whitespace/tab', 1,
          'Tab found; better to use spaces')

  # One or three blank spaces at the beginning of the line is weird; it's
  # hard to reconcile that with 2-space indents.
  # NOTE: here are the conditions rob pike used for his tests.  Mine aren't
  # as sophisticated, but it may be worth becoming so:  RLENGTH==initial_spaces
  # if(RLENGTH > 20) complain = 0;
  # if(match($0, " +(error|private|public|protected):")) complain = 0;
  # if(match(prev, "&& *$")) complain = 0;
  # if(match(prev, "\\|\\| *$")) complain = 0;
  # if(match(prev, "[\",=><] *$")) complain = 0;
  # if(match($0, " <<")) complain = 0;
  # if(match(prev, " +for \\(")) complain = 0;
  # if(prevodd && match(prevprev, " +for \\(")) complain = 0;
  scope_or_label_pattern = r'\s*\w+\s*:\s*\\?$'
  classinfo = nesting_state.InnermostClass()
  initial_spaces = 0
  cleansed_line = clean_lines.elided[linenum]
  while initial_spaces < len(line) and line[initial_spaces] == ' ':
    initial_spaces += 1
  # There are certain situations we allow one space, notably for
  # section labels, and also lines containing multi-line raw strings.
  # We also don't check for lines that look like continuation lines
  # (of lines ending in double quotes, commas, equals, or angle brackets)
  # because the rules for how to indent those are non-trivial.
  if (not Search(r'[",=><] *$', prev) and
      (initial_spaces == 1 or initial_spaces == 3) and
      not Match(scope_or_label_pattern, cleansed_line) and
      not (clean_lines.raw_lines[linenum] != line and
           Match(r'^\s*""', line))):
    error(filename, linenum, 'whitespace/indent', 3,
          'Weird number of spaces at line-start.  '
          'Are you using a 2-space indent?')

  if line and line[-1].isspace():
    error(filename, linenum, 'whitespace/end_of_line', 4,
          'Line ends in whitespace.  Consider deleting these extra spaces.')

  # Check if the line is a header guard.
  is_header_guard = False
  if IsHeaderExtension(file_extension):
    cppvar = GetHeaderGuardCPPVariable(filename)
    if (line.startswith('#ifndef %s' % cppvar) or
        line.startswith('#define %s' % cppvar) or
        line.startswith('#endif  // %s' % cppvar)):
      is_header_guard = True
  # #include lines and header guards can be long, since there's no clean way to
  # split them.
  #
  # URLs can be long too.  It's possible to split these, but it makes them
  # harder to cut&paste.
  #
  # The "$Id:...$" comment may also get very long without it being the
  # developers fault.
  if (not line.startswith('#include') and not is_header_guard and
      not Match(r'^\s*//.*http(s?)://\S*$', line) and
      not Match(r'^\s*//\s*[^\s]*$', line) and
      not Match(r'^// \$Id:.*#[0-9]+ \$$', line)):
    line_width = GetLineWidth(line)
    if line_width > _line_length:
      error(filename, linenum, 'whitespace/line_length', 2,
            'Lines should be <= %i characters long' % _line_length)

  if (cleansed_line.count(';') > 1 and
      # for loops are allowed two ;'s (and may run over two lines).
      cleansed_line.find('for') == -1 and
      (GetPreviousNonBlankLine(clean_lines, linenum)[0].find('for') == -1 or
       GetPreviousNonBlankLine(clean_lines, linenum)[0].find(';') != -1) and
      # It's ok to have many commands in a switch case that fits in 1 line
      not ((cleansed_line.find('case ') != -1 or
            cleansed_line.find('default:') != -1) and
           cleansed_line.find('break;') != -1)):
    error(filename, linenum, 'whitespace/newline', 0,
          'More than one command on the same line')

  # Some more style checks
  CheckBraces(filename, clean_lines, linenum, error)
  CheckTrailingSemicolon(filename, clean_lines, linenum, error)
  CheckEmptyBlockBody(filename, clean_lines, linenum, error)
  CheckSpacing(filename, clean_lines, linenum, nesting_state, error)
  CheckOperatorSpacing(filename, clean_lines, linenum, error)
  CheckParenthesisSpacing(filename, clean_lines, linenum, error)
  CheckCommaSpacing(filename, clean_lines, linenum, error)
  CheckBracesSpacing(filename, clean_lines, linenum, nesting_state, error)
  CheckSpacingForFunctionCall(filename, clean_lines, linenum, error)
  CheckCheck(filename, clean_lines, linenum, error)
  CheckAltTokens(filename, clean_lines, linenum, error)
  classinfo = nesting_state.InnermostClass()
  if classinfo:
    CheckSectionSpacing(filename, clean_lines, classinfo, linenum, error)


_RE_PATTERN_INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]*)[>"].*$')
# Matches the first component of a filename delimited by -s and _s. That is:
#  _RE_FIRST_COMPONENT.match('foo').group(0) == 'foo'
#  _RE_FIRST_COMPONENT.match('foo.cc').group(0) == 'foo'
#  _RE_FIRST_COMPONENT.match('foo-bar_baz.cc').group(0) == 'foo'
#  _RE_FIRST_COMPONENT.match('foo_bar-baz.cc').group(0) == 'foo'
_RE_FIRST_COMPONENT = re.compile(r'^[^-_.]+')


def _DropCommonSuffixes(filename):
  """Drops common suffixes like _test.cc or -inl.h from filename.

  For example:
    >>> _DropCommonSuffixes('foo/foo-inl.h')
    'foo/foo'
    >>> _DropCommonSuffixes('foo/bar/foo.cc')
    'foo/bar/foo'
    >>> _DropCommonSuffixes('foo/foo_internal.h')
    'foo/foo'
    >>> _DropCommonSuffixes('foo/foo_unusualinternal.h')
    'foo/foo_unusualinternal'

  Args:
    filename: The input filename.

  Returns:
    The filename with the common suffix removed.
  """
  for suffix in ('test.cc', 'regtest.cc', 'unittest.cc',
                 'inl.h', 'impl.h', 'internal.h'):
    if (filename.endswith(suffix) and len(filename) > len(suffix) and
        filename[-len(suffix) - 1] in ('-', '_')):
      return filename[:-len(suffix) - 1]
  return os.path.splitext(filename)[0]


def _ClassifyInclude(fileinfo, include, is_system):
  """Figures out what kind of header 'include' is.

  Args:
    fileinfo: The current file cpplint is running over. A FileInfo instance.
    include: The path to a #included file.
    is_system: True if the #include used <> rather than "".

  Returns:
    One of the _XXX_HEADER constants.

  For example:
    >>> _ClassifyInclude(FileInfo('foo/foo.cc'), 'stdio.h', True)
    _C_SYS_HEADER
    >>> _ClassifyInclude(FileInfo('foo/foo.cc'), 'string', True)
    _CPP_SYS_HEADER
    >>> _ClassifyInclude(FileInfo('foo/foo.cc'), 'foo/foo.h', False)
    _LIKELY_MY_HEADER
    >>> _ClassifyInclude(FileInfo('foo/foo_unknown_extension.cc'),
    ...                  'bar/foo_other_ext.h', False)
    _POSSIBLE_MY_HEADER
    >>> _ClassifyInclude(FileInfo('foo/foo.cc'), 'foo/bar.h', False)
    _OTHER_HEADER
  """
  # This is a list of all standard c++ header files, except
  # those already checked for above.
  is_cpp_h = include in _CPP_HEADERS

  if is_system:
    if is_cpp_h:
      return _CPP_SYS_HEADER
    else:
      return _C_SYS_HEADER

  # If the target file and the include we're checking share a
  # basename when we drop common extensions, and the include
  # lives in . , then it's likely to be owned by the target file.
  target_dir, target_base = (
      os.path.split(_DropCommonSuffixes(fileinfo.RepositoryName())))
  include_dir, include_base = os.path.split(_DropCommonSuffixes(include))
  if target_base == include_base and (
      include_dir == target_dir or
      include_dir == os.path.normpath(target_dir + '/../public')):
    return _LIKELY_MY_HEADER

  # If the target and include share some initial basename
  # component, it's possible the target is implementing the
  # include, so it's allowed to be first, but we'll never
  # complain if it's not there.
  target_first_component = _RE_FIRST_COMPONENT.match(target_base)
  include_first_component = _RE_FIRST_COMPONENT.match(include_base)
  if (target_first_component and include_first_component and
      target_first_component.group(0) ==
      include_first_component.group(0)):
    return _POSSIBLE_MY_HEADER

  return _OTHER_HEADER



def CheckIncludeLine(filename, clean_lines, linenum, include_state, error):
  """Check rules that are applicable to #include lines.

  Strings on #include lines are NOT removed from elided line, to make
  certain tasks easier. However, to prevent false positives, checks
  applicable to #include lines in CheckLanguage must be put here.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    include_state: An _IncludeState instance in which the headers are inserted.
    error: The function to call with any errors found.
  """
  fileinfo = FileInfo(filename)
  line = clean_lines.lines[linenum]

  # "include" should use the new style "foo/bar.h" instead of just "bar.h"
  # Only do this check if the included header follows google naming
  # conventions.  If not, assume that it's a 3rd party API that
  # requires special include conventions.
  #
  # We also make an exception for Lua headers, which follow google
  # naming convention but not the include convention.
  match = Match(r'#include\s*"([^/]+\.h)"', line)
  if match and not _THIRD_PARTY_HEADERS_PATTERN.match(match.group(1)):
    error(filename, linenum, 'build/include', 4,
          'Include the directory when naming .h files')

  # we shouldn't include a file more than once. actually, there are a
  # handful of instances where doing so is okay, but in general it's
  # not.
  match = _RE_PATTERN_INCLUDE.search(line)
  if match:
    include = match.group(2)
    is_system = (match.group(1) == '<')
    duplicate_line = include_state.FindHeader(include)
    if duplicate_line >= 0:
      error(filename, linenum, 'build/include', 4,
            '"%s" already included at %s:%s' %
            (include, filename, duplicate_line))
    elif (include.endswith('.cc') and
          os.path.dirname(fileinfo.RepositoryName()) != os.path.dirname(include)):
      error(filename, linenum, 'build/include', 4,
            'Do not include .cc files from other packages')
    elif not _THIRD_PARTY_HEADERS_PATTERN.match(include):
      include_state.include_list[-1].append((include, linenum))

      # We want to ensure that headers appear in the right order:
      # 1) for foo.cc, foo.h  (preferred location)
      # 2) c system files
      # 3) cpp system files
      # 4) for foo.cc, foo.h  (deprecated location)
      # 5) other google headers
      #
      # We classify each include statement as one of those 5 types
      # using a number of techniques. The include_state object keeps
      # track of the highest type seen, and complains if we see a
      # lower type after that.
      error_message = include_state.CheckNextIncludeOrder(
          _ClassifyInclude(fileinfo, include, is_system))
      if error_message:
        error(filename, linenum, 'build/include_order', 4,
              '%s. Should be: %s.h, c system, c++ system, other.' %
              (error_message, fileinfo.BaseName()))
      canonical_include = include_state.CanonicalizeAlphabeticalOrder(include)
      if not include_state.IsInAlphabeticalOrder(
          clean_lines, linenum, canonical_include):
        error(filename, linenum, 'build/include_alpha', 4,
              'Include "%s" not in alphabetical order' % include)
      include_state.SetLastHeader(canonical_include)



def _GetTextInside(text, start_pattern):
  r"""Retrieves all the text between matching open and close parentheses.

  Given a string of lines and a regular expression string, retrieve all the text
  following the expression and between opening punctuation symbols like
  (, [, or {, and the matching close-punctuation symbol. This properly nested
  occurrences of the punctuations, so for the text like
    printf(a(), b(c()));
  a call to _GetTextInside(text, r'printf\(') will return 'a(), b(c())'.
  start_pattern must match string having an open punctuation symbol at the end.

  Args:
    text: The lines to extract text. Its comments and strings must be elided.
           It can be single line and can span multiple lines.
    start_pattern: The regexp string indicating where to start extracting
                   the text.
  Returns:
    The extracted text.
    None if either the opening string or ending punctuation could not be found.
  """
  # TODO(unknown): Audit cpplint.py to see what places could be profitably
  # rewritten to use _GetTextInside (and use inferior regexp matching today).

  # Give opening punctuations to get the matching close-punctuations.
  matching_punctuation = {'(': ')', '{': '}', '[': ']'}
  closing_punctuation = set(matching_punctuation.itervalues())

  # Find the position to start extracting text.
  match = re.search(start_pattern, text, re.M)
  if not match:  # start_pattern not found in text.
    return None
  start_position = match.end(0)

  assert start_position > 0, (
      'start_pattern must ends with an opening punctuation.')
  assert text[start_position - 1] in matching_punctuation, (
      'start_pattern must ends with an opening punctuation.')
  # Stack of closing punctuations we expect to have in text after position.
  punctuation_stack = [matching_punctuation[text[start_position - 1]]]
  position = start_position
  while punctuation_stack and position < len(text):
    if text[position] == punctuation_stack[-1]:
      punctuation_stack.pop()
    elif text[position] in closing_punctuation:
      # A closing punctuation without matching opening punctuations.
      return None
    elif text[position] in matching_punctuation:
      punctuation_stack.append(matching_punctuation[text[position]])
    position += 1
  if punctuation_stack:
    # Opening punctuations left without matching close-punctuations.
    return None
  # punctuations match.
  return text[start_position:position - 1]


# Patterns for matching call-by-reference parameters.
#
# Supports nested templates up to 2 levels deep using this messy pattern:
#   < (?: < (?: < [^<>]*
#               >
#           |   [^<>] )*
#         >
#     |   [^<>] )*
#   >
_RE_PATTERN_IDENT = r'[_a-zA-Z]\w*'  # =~ [[:alpha:]][[:alnum:]]*
_RE_PATTERN_TYPE = (
    r'(?:const\s+)?(?:typename\s+|class\s+|struct\s+|union\s+|enum\s+)?'
    r'(?:\w|'
    r'\s*<(?:<(?:<[^<>]*>|[^<>])*>|[^<>])*>|'
    r'::)+')
# A call-by-reference parameter ends with '& identifier'.
_RE_PATTERN_REF_PARAM = re.compile(
    r'(' + _RE_PATTERN_TYPE + r'(?:\s*(?:\bconst\b|[*]))*\s*'
    r'&\s*' + _RE_PATTERN_IDENT + r')\s*(?:=[^,()]+)?[,)]')
# A call-by-const-reference parameter either ends with 'const& identifier'
# or looks like 'const type& identifier' when 'type' is atomic.
_RE_PATTERN_CONST_REF_PARAM = (
    r'(?:.*\s*\bconst\s*&\s*' + _RE_PATTERN_IDENT +
    r'|const\s+' + _RE_PATTERN_TYPE + r'\s*&\s*' + _RE_PATTERN_IDENT + r')')
# Stream types.
_RE_PATTERN_REF_STREAM_PARAM = (
    r'(?:.*stream\s*&\s*' + _RE_PATTERN_IDENT + r')')


def CheckLanguage(filename, clean_lines, linenum, file_extension,
                  include_state, nesting_state, error):
  """Checks rules from the 'C++ language rules' section of cppguide.html.

  Some of these rules are hard to test (function overloading, using
  uint32 inappropriately), but we do the best we can.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    file_extension: The extension (without the dot) of the filename.
    include_state: An _IncludeState instance in which the headers are inserted.
    nesting_state: A NestingState instance which maintains information about
                   the current stack of nested blocks being parsed.
    error: The function to call with any errors found.
  """
  # If the line is empty or consists of entirely a comment, no need to
  # check it.
  line = clean_lines.elided[linenum]
  if not line:
    return

  match = _RE_PATTERN_INCLUDE.search(line)
  if match:
    CheckIncludeLine(filename, clean_lines, linenum, include_state, error)
    return

  # Reset include state across preprocessor directives.  This is meant
  # to silence warnings for conditional includes.
  match = Match(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b', line)
  if match:
    include_state.ResetSection(match.group(1))

  # Make Windows paths like Unix.
  fullname = os.path.abspath(filename).replace('\\', '/')

  # Perform other checks now that we are sure that this is not an include line
  CheckCasts(filename, clean_lines, linenum, error)
  CheckGlobalStatic(filename, clean_lines, linenum, error)
  CheckPrintf(filename, clean_lines, linenum, error)

  if IsHeaderExtension(file_extension):
    # TODO(unknown): check that 1-arg constructors are explicit.
    #                How to tell it's a constructor?
    #                (handled in CheckForNonStandardConstructs for now)
    # TODO(unknown): check that classes declare or disable copy/assign
    #                (level 1 error)
    pass

  # Check if people are using the verboten C basic types.  The only exception
  # we regularly allow is "unsigned short port" for port.
  if Search(r'\bshort port\b', line):
    if not Search(r'\bunsigned short port\b', line):
      error(filename, linenum, 'runtime/int', 4,
            'Use "unsigned short" for ports, not "short"')
  else:
    match = Search(r'\b(short|long(?! +double)|long long)\b', line)
    if match:
      error(filename, linenum, 'runtime/int', 4,
            'Use int16/int64/etc, rather than the C type %s' % match.group(1))

  # Check if some verboten operator overloading is going on
  # TODO(unknown): catch out-of-line unary operator&:
  #   class X {};
  #   int operator&(const X& x) { return 42; }  // unary operator&
  # The trick is it's hard to tell apart from binary operator&:
  #   class Y { int operator&(const Y& x) { return 23; } }; // binary operator&
  if Search(r'\boperator\s*&\s*\(\s*\)', line):
    error(filename, linenum, 'runtime/operator', 4,
          'Unary operator& is dangerous.  Do not use it.')

  # Check for suspicious usage of "if" like
  # } if (a == b) {
  if Search(r'\}\s*if\s*\(', line):
    error(filename, linenum, 'readability/braces', 4,
          'Did you mean "else if"? If not, start a new line for "if".')

  # Check for potential format string bugs like printf(foo).
  # We constrain the pattern not to pick things like DocidForPrintf(foo).
  # Not perfect but it can catch printf(foo.c_str()) and printf(foo->c_str())
  # TODO(unknown): Catch the following case. Need to change the calling
  # convention of the whole function to process multiple line to handle it.
  #   printf(
  #       boy_this_is_a_really_long_variable_that_cannot_fit_on_the_prev_line);
  printf_args = _GetTextInside(line, r'(?i)\b(string)?printf\s*\(')
  if printf_args:
    match = Match(r'([\w.\->()]+)$', printf_args)
    if match and match.group(1) != '__VA_ARGS__':
      function_name = re.search(r'\b((?:string)?printf)\s*\(',
                                line, re.I).group(1)
      error(filename, linenum, 'runtime/printf', 4,
            'Potential format string bug. Do %s("%%s", %s) instead.'
            % (function_name, match.group(1)))

  # Check for potential memset bugs like memset(buf, sizeof(buf), 0).
  match = Search(r'memset\s*\(([^,]*),\s*([^,]*),\s*0\s*\)', line)
  if match and not Match(r"^''|-?[0-9]+|0x[0-9A-Fa-f]$", match.group(2)):
    error(filename, linenum, 'runtime/memset', 4,
          'Did you mean "memset(%s, 0, %s)"?'
          % (match.group(1), match.group(2)))

  if Search(r'\busing namespace\b', line):
    error(filename, linenum, 'build/namespaces', 5,
          'Do not use namespace using-directives.  '
          'Use using-declarations instead.')

  # Detect variable-length arrays.
  match = Match(r'\s*(.+::)?(\w+) [a-z]\w*\[(.+)];', line)
  if (match and match.group(2) != 'return' and match.group(2) != 'delete' and
      match.group(3).find(']') == -1):
    # Split the size using space and arithmetic operators as delimiters.
    # If any of the resulting tokens are not compile time constants then
    # report the error.
    tokens = re.split(r'\s|\+|\-|\*|\/|<<|>>]', match.group(3))
    is_const = True
    skip_next = False
    for tok in tokens:
      if skip_next:
        skip_next = False
        continue

      if Search(r'sizeof\(.+\)', tok): continue
      if Search(r'arraysize\(\w+\)', tok): continue

      tok = tok.lstrip('(')
      tok = tok.rstrip(')')
      if not tok: continue
      if Match(r'\d+', tok): continue
      if Match(r'0[xX][0-9a-fA-F]+', tok): continue
      if Match(r'k[A-Z0-9]\w*', tok): continue
      if Match(r'(.+::)?k[A-Z0-9]\w*', tok): continue
      if Match(r'(.+::)?[A-Z][A-Z0-9_]*', tok): continue
      # A catch all for tricky sizeof cases, including 'sizeof expression',
      # 'sizeof(*type)', 'sizeof(const type)', 'sizeof(struct StructName)'
      # requires skipping the next token because we split on ' ' and '*'.
      if tok.startswith('sizeof'):
        skip_next = True
        continue
      is_const = False
      break
    if not is_const:
      error(filename, linenum, 'runtime/arrays', 1,
            'Do not use variable-length arrays.  Use an appropriately named '
            "('k' followed by CamelCase) compile-time constant for the size.")

  # Check for use of unnamed namespaces in header files.  Registration
  # macros are typically OK, so we allow use of "namespace {" on lines
  # that end with backslashes.
  if (IsHeaderExtension(file_extension)
      and Search(r'\bnamespace\s*{', line)
      and line[-1] != '\\'):
    error(filename, linenum, 'build/namespaces', 4,
          'Do not use unnamed namespaces in header files.  See '
          'https://google-styleguide.googlecode.com/svn/trunk/cppguide.xml#Namespaces'
          ' for more information.')


def CheckGlobalStatic(filename, clean_lines, linenum, error):
  """Check for unsafe global or static objects.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # Match two lines at a time to support multiline declarations
  if linenum + 1 < clean_lines.NumLines() and not Search(r'[;({]', line):
    line += clean_lines.elided[linenum + 1].strip()

  # Check for people declaring static/global STL strings at the top level.
  # This is dangerous because the C++ language does not guarantee that
  # globals with constructors are initialized before the first access, and
  # also because globals can be destroyed when some threads are still running.
  # TODO(unknown): Generalize this to also find static unique_ptr instances.
  # TODO(unknown): File bugs for clang-tidy to find these.
  match = Match(
      r'((?:|static +)(?:|const +))(?::*std::)?string( +const)? +'
      r'([a-zA-Z0-9_:]+)\b(.*)',
      line)

  # Remove false positives:
  # - String pointers (as opposed to values).
  #    string *pointer
  #    const string *pointer
  #    string const *pointer
  #    string *const pointer
  #
  # - Functions and template specializations.
  #    string Function<Type>(...
  #    string Class<Type>::Method(...
  #
  # - Operators.  These are matched separately because operator names
  #   cross non-word boundaries, and trying to match both operators
  #   and functions at the same time would decrease accuracy of
  #   matching identifiers.
  #    string Class::operator*()
  if (match and
      not Search(r'\bstring\b(\s+const)?\s*[\*\&]\s*(const\s+)?\w', line) and
      not Search(r'\boperator\W', line) and
      not Match(r'\s*(<.*>)?(::[a-zA-Z0-9_]+)*\s*\(([^"]|$)', match.group(4))):
    if Search(r'\bconst\b', line):
      error(filename, linenum, 'runtime/string', 4,
            'For a static/global string constant, use a C style string '
            'instead: "%schar%s %s[]".' %
            (match.group(1), match.group(2) or '', match.group(3)))
    else:
      error(filename, linenum, 'runtime/string', 4,
            'Static/global string variables are not permitted.')

  if (Search(r'\b([A-Za-z0-9_]*_)\(\1\)', line) or
      Search(r'\b([A-Za-z0-9_]*_)\(CHECK_NOTNULL\(\1\)\)', line)):
    error(filename, linenum, 'runtime/init', 4,
          'You seem to be initializing a member variable with itself.')


def CheckPrintf(filename, clean_lines, linenum, error):
  """Check for printf related issues.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # When snprintf is used, the second argument shouldn't be a literal.
  match = Search(r'snprintf\s*\(([^,]*),\s*([0-9]*)\s*,', line)
  if match and match.group(2) != '0':
    # If 2nd arg is zero, snprintf is used to calculate size.
    error(filename, linenum, 'runtime/printf', 3,
          'If you can, use sizeof(%s) instead of %s as the 2nd arg '
          'to snprintf.' % (match.group(1), match.group(2)))

  # Check if some verboten C functions are being used.
  if Search(r'\bsprintf\s*\(', line):
    error(filename, linenum, 'runtime/printf', 5,
          'Never use sprintf. Use snprintf instead.')
  match = Search(r'\b(strcpy|strcat)\s*\(', line)
  if match:
    error(filename, linenum, 'runtime/printf', 4,
          'Almost always, snprintf is better than %s' % match.group(1))


def IsDerivedFunction(clean_lines, linenum):
  """Check if current line contains an inherited function.

  Args:
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
  Returns:
    True if current line contains a function with "override"
    virt-specifier.
  """
  # Scan back a few lines for start of current function
  for i in xrange(linenum, max(-1, linenum - 10), -1):
    match = Match(r'^([^()]*\w+)\(', clean_lines.elided[i])
    if match:
      # Look for "override" after the matching closing parenthesis
      line, _, closing_paren = CloseExpression(
          clean_lines, i, len(match.group(1)))
      return (closing_paren >= 0 and
              Search(r'\boverride\b', line[closing_paren:]))
  return False


def IsOutOfLineMethodDefinition(clean_lines, linenum):
  """Check if current line contains an out-of-line method definition.

  Args:
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
  Returns:
    True if current line contains an out-of-line method definition.
  """
  # Scan back a few lines for start of current function
  for i in xrange(linenum, max(-1, linenum - 10), -1):
    if Match(r'^([^()]*\w+)\(', clean_lines.elided[i]):
      return Match(r'^[^()]*\w+::\w+\(', clean_lines.elided[i]) is not None
  return False


def IsInitializerList(clean_lines, linenum):
  """Check if current line is inside constructor initializer list.

  Args:
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
  Returns:
    True if current line appears to be inside constructor initializer
    list, False otherwise.
  """
  for i in xrange(linenum, 1, -1):
    line = clean_lines.elided[i]
    if i == linenum:
      remove_function_body = Match(r'^(.*)\{\s*$', line)
      if remove_function_body:
        line = remove_function_body.group(1)

    if Search(r'\s:\s*\w+[({]', line):
      # A lone colon tend to indicate the start of a constructor
      # initializer list.  It could also be a ternary operator, which
      # also tend to appear in constructor initializer lists as
      # opposed to parameter lists.
      return True
    if Search(r'\}\s*,\s*$', line):
      # A closing brace followed by a comma is probably the end of a
      # brace-initialized member in constructor initializer list.
      return True
    if Search(r'[{};]\s*$', line):
      # Found one of the following:
      # - A closing brace or semicolon, probably the end of the previous
      #   function.
      # - An opening brace, probably the start of current class or namespace.
      #
      # Current line is probably not inside an initializer list since
      # we saw one of those things without seeing the starting colon.
      return False

  # Got to the beginning of the file without seeing the start of
  # constructor initializer list.
  return False


def CheckForNonConstReference(filename, clean_lines, linenum,
                              nesting_state, error):
  """Check for non-const references.

  Separate from CheckLanguage since it scans backwards from current
  line, instead of scanning forward.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    nesting_state: A NestingState instance which maintains information about
                   the current stack of nested blocks being parsed.
    error: The function to call with any errors found.
  """
  # Do nothing if there is no '&' on current line.
  line = clean_lines.elided[linenum]
  if '&' not in line:
    return

  # If a function is inherited, current function doesn't have much of
  # a choice, so any non-const references should not be blamed on
  # derived function.
  if IsDerivedFunction(clean_lines, linenum):
    return

  # Don't warn on out-of-line method definitions, as we would warn on the
  # in-line declaration, if it isn't marked with 'override'.
  if IsOutOfLineMethodDefinition(clean_lines, linenum):
    return

  # Long type names may be broken across multiple lines, usually in one
  # of these forms:
  #   LongType
  #       ::LongTypeContinued &identifier
  #   LongType::
  #       LongTypeContinued &identifier
  #   LongType<
  #       ...>::LongTypeContinued &identifier
  #
  # If we detected a type split across two lines, join the previous
  # line to current line so that we can match const references
  # accordingly.
  #
  # Note that this only scans back one line, since scanning back
  # arbitrary number of lines would be expensive.  If you have a type
  # that spans more than 2 lines, please use a typedef.
  if linenum > 1:
    previous = None
    if Match(r'\s*::(?:[\w<>]|::)+\s*&\s*\S', line):
      # previous_line\n + ::current_line
      previous = Search(r'\b((?:const\s*)?(?:[\w<>]|::)+[\w<>])\s*$',
                        clean_lines.elided[linenum - 1])
    elif Match(r'\s*[a-zA-Z_]([\w<>]|::)+\s*&\s*\S', line):
      # previous_line::\n + current_line
      previous = Search(r'\b((?:const\s*)?(?:[\w<>]|::)+::)\s*$',
                        clean_lines.elided[linenum - 1])
    if previous:
      line = previous.group(1) + line.lstrip()
    else:
      # Check for templated parameter that is split across multiple lines
      endpos = line.rfind('>')
      if endpos > -1:
        (_, startline, startpos) = ReverseCloseExpression(
            clean_lines, linenum, endpos)
        if startpos > -1 and startline < linenum:
          # Found the matching < on an earlier line, collect all
          # pieces up to current line.
          line = ''
          for i in xrange(startline, linenum + 1):
            line += clean_lines.elided[i].strip()

  # Check for non-const references in function parameters.  A single '&' may
  # found in the following places:
  #   inside expression: binary & for bitwise AND
  #   inside expression: unary & for taking the address of something
  #   inside declarators: reference parameter
  # We will exclude the first two cases by checking that we are not inside a
  # function body, including one that was just introduced by a trailing '{'.
  # TODO(unknown): Doesn't account for 'catch(Exception& e)' [rare].
  if (nesting_state.previous_stack_top and
      not (isinstance(nesting_state.previous_stack_top, _ClassInfo) or
           isinstance(nesting_state.previous_stack_top, _NamespaceInfo))):
    # Not at toplevel, not within a class, and not within a namespace
    return

  # Avoid initializer lists.  We only need to scan back from the
  # current line for something that starts with ':'.
  #
  # We don't need to check the current line, since the '&' would
  # appear inside the second set of parentheses on the current line as
  # opposed to the first set.
  if linenum > 0:
    for i in xrange(linenum - 1, max(0, linenum - 10), -1):
      previous_line = clean_lines.elided[i]
      if not Search(r'[),]\s*$', previous_line):
        break
      if Match(r'^\s*:\s+\S', previous_line):
        return

  # Avoid preprocessors
  if Search(r'\\\s*$', line):
    return

  # Avoid constructor initializer lists
  if IsInitializerList(clean_lines, linenum):
    return

  # We allow non-const references in a few standard places, like functions
  # called "swap()" or iostream operators like "<<" or ">>".  Do not check
  # those function parameters.
  #
  # We also accept & in static_assert, which looks like a function but
  # it's actually a declaration expression.
  whitelisted_functions = (r'(?:[sS]wap(?:<\w:+>)?|'
                           r'operator\s*[<>][<>]|'
                           r'static_assert|COMPILE_ASSERT'
                           r')\s*\(')
  if Search(whitelisted_functions, line):
    return
  elif not Search(r'\S+\([^)]*$', line):
    # Don't see a whitelisted function on this line.  Actually we
    # didn't see any function name on this line, so this is likely a
    # multi-line parameter list.  Try a bit harder to catch this case.
    for i in xrange(2):
      if (linenum > i and
          Search(whitelisted_functions, clean_lines.elided[linenum - i - 1])):
        return

  decls = ReplaceAll(r'{[^}]*}', ' ', line)  # exclude function body
  for parameter in re.findall(_RE_PATTERN_REF_PARAM, decls):
    if (not Match(_RE_PATTERN_CONST_REF_PARAM, parameter) and
        not Match(_RE_PATTERN_REF_STREAM_PARAM, parameter)):
      error(filename, linenum, 'runtime/references', 2,
            'Is this a non-const reference? '
            'If so, make const or use a pointer: ' +
            ReplaceAll(' *<', '<', parameter))


def CheckCasts(filename, clean_lines, linenum, error):
  """Various cast related checks.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  # Check to see if they're using an conversion function cast.
  # I just try to capture the most common basic types, though there are more.
  # Parameterless conversion functions, such as bool(), are allowed as they are
  # probably a member operator declaration or default constructor.
  match = Search(
      r'(\bnew\s+(?:const\s+)?|\S<\s*(?:const\s+)?)?\b'
      r'(int|float|double|bool|char|int32|uint32|int64|uint64)'
      r'(\([^)].*)', line)
  expecting_function = ExpectingFunctionArgs(clean_lines, linenum)
  if match and not expecting_function:
    matched_type = match.group(2)

    # matched_new_or_template is used to silence two false positives:
    # - New operators
    # - Template arguments with function types
    #
    # For template arguments, we match on types immediately following
    # an opening bracket without any spaces.  This is a fast way to
    # silence the common case where the function type is the first
    # template argument.  False negative with less-than comparison is
    # avoided because those operators are usually followed by a space.
    #
    #   function<double(double)>   // bracket + no space = false positive
    #   value < double(42)         // bracket + space = true positive
    matched_new_or_template = match.group(1)

    # Avoid arrays by looking for brackets that come after the closing
    # parenthesis.
    if Match(r'\([^()]+\)\s*\[', match.group(3)):
      return

    # Other things to ignore:
    # - Function pointers
    # - Casts to pointer types
    # - Placement new
    # - Alias declarations
    matched_funcptr = match.group(3)
    if (matched_new_or_template is None and
        not (matched_funcptr and
             (Match(r'\((?:[^() ]+::\s*\*\s*)?[^() ]+\)\s*\(',
                    matched_funcptr) or
              matched_funcptr.startswith('(*)'))) and
        not Match(r'\s*using\s+\S+\s*=\s*' + matched_type, line) and
        not Search(r'new\(\S+\)\s*' + matched_type, line)):
      error(filename, linenum, 'readability/casting', 4,
            'Using deprecated casting style.  '
            'Use static_cast<%s>(...) instead' %
            matched_type)

  if not expecting_function:
    CheckCStyleCast(filename, clean_lines, linenum, 'static_cast',
                    r'\((int|float|double|bool|char|u?int(16|32|64))\)', error)

  # This doesn't catch all cases. Consider (const char * const)"hello".
  #
  # (char *) "foo" should always be a const_cast (reinterpret_cast won't
  # compile).
  if CheckCStyleCast(filename, clean_lines, linenum, 'const_cast',
                     r'\((char\s?\*+\s?)\)\s*"', error):
    pass
  else:
    # Check pointer casts for other than string constants
    CheckCStyleCast(filename, clean_lines, linenum, 'reinterpret_cast',
                    r'\((\w+\s?\*+\s?)\)', error)

  # In addition, we look for people taking the address of a cast.  This
  # is dangerous -- casts can assign to temporaries, so the pointer doesn't
  # point where you think.
  #
  # Some non-identifier character is required before the '&' for the
  # expression to be recognized as a cast.  These are casts:
  #   expression = &static_cast<int*>(temporary());
  #   function(&(int*)(temporary()));
  #
  # This is not a cast:
  #   reference_type&(int* function_param);
  match = Search(
      r'(?:[^\w]&\(([^)*][^)]*)\)[\w(])|'
      r'(?:[^\w]&(static|dynamic|down|reinterpret)_cast\b)', line)
  if match:
    # Try a better error message when the & is bound to something
    # dereferenced by the casted pointer, as opposed to the casted
    # pointer itself.
    parenthesis_error = False
    match = Match(r'^(.*&(?:static|dynamic|down|reinterpret)_cast\b)<', line)
    if match:
      _, y1, x1 = CloseExpression(clean_lines, linenum, len(match.group(1)))
      if x1 >= 0 and clean_lines.elided[y1][x1] == '(':
        _, y2, x2 = CloseExpression(clean_lines, y1, x1)
        if x2 >= 0:
          extended_line = clean_lines.elided[y2][x2:]
          if y2 < clean_lines.NumLines() - 1:
            extended_line += clean_lines.elided[y2 + 1]
          if Match(r'\s*(?:->|\[)', extended_line):
            parenthesis_error = True

    if parenthesis_error:
      error(filename, linenum, 'readability/casting', 4,
            ('Are you taking an address of something dereferenced '
             'from a cast?  Wrapping the dereferenced expression in '
             'parentheses will make the binding more obvious'))
    else:
      error(filename, linenum, 'runtime/casting', 4,
            ('Are you taking an address of a cast?  '
             'This is dangerous: could be a temp var.  '
             'Take the address before doing the cast, rather than after'))


def CheckCStyleCast(filename, clean_lines, linenum, cast_type, pattern, error):
  """Checks for a C-style cast by looking for the pattern.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    cast_type: The string for the C++ cast to recommend.  This is either
      reinterpret_cast, static_cast, or const_cast, depending.
    pattern: The regular expression used to find C-style casts.
    error: The function to call with any errors found.

  Returns:
    True if an error was emitted.
    False otherwise.
  """
  line = clean_lines.elided[linenum]
  match = Search(pattern, line)
  if not match:
    return False

  # Exclude lines with keywords that tend to look like casts
  context = line[0:match.start(1) - 1]
  if Match(r'.*\b(?:sizeof|alignof|alignas|[_A-Z][_A-Z0-9]*)\s*$', context):
    return False

  # Try expanding current context to see if we one level of
  # parentheses inside a macro.
  if linenum > 0:
    for i in xrange(linenum - 1, max(0, linenum - 5), -1):
      context = clean_lines.elided[i] + context
  if Match(r'.*\b[_A-Z][_A-Z0-9]*\s*\((?:\([^()]*\)|[^()])*$', context):
    return False

  # operator++(int) and operator--(int)
  if context.endswith(' operator++') or context.endswith(' operator--'):
    return False

  # A single unnamed argument for a function tends to look like old style cast.
  # If we see those, don't issue warnings for deprecated casts.
  remainder = line[match.end(0):]
  if Match(r'^\s*(?:;|const\b|throw\b|final\b|override\b|[=>{),]|->)',
           remainder):
    return False

  # At this point, all that should be left is actual casts.
  error(filename, linenum, 'readability/casting', 4,
        'Using C-style cast.  Use %s<%s>(...) instead' %
        (cast_type, match.group(1)))

  return True


def ExpectingFunctionArgs(clean_lines, linenum):
  """Checks whether where function type arguments are expected.

  Args:
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.

  Returns:
    True if the line at 'linenum' is inside something that expects arguments
    of function types.
  """
  line = clean_lines.elided[linenum]
  return (Match(r'^\s*MOCK_(CONST_)?METHOD\d+(_T)?\(', line) or
          (linenum >= 2 and
           (Match(r'^\s*MOCK_(?:CONST_)?METHOD\d+(?:_T)?\((?:\S+,)?\s*$',
                  clean_lines.elided[linenum - 1]) or
            Match(r'^\s*MOCK_(?:CONST_)?METHOD\d+(?:_T)?\(\s*$',
                  clean_lines.elided[linenum - 2]) or
            Search(r'\bstd::m?function\s*\<\s*$',
                   clean_lines.elided[linenum - 1]))))


_HEADERS_CONTAINING_TEMPLATES = (
    ('<deque>', ('deque',)),
    ('<functional>', ('unary_function', 'binary_function',
                      'plus', 'minus', 'multiplies', 'divides', 'modulus',
                      'negate',
                      'equal_to', 'not_equal_to', 'greater', 'less',
                      'greater_equal', 'less_equal',
                      'logical_and', 'logical_or', 'logical_not',
                      'unary_negate', 'not1', 'binary_negate', 'not2',
                      'bind1st', 'bind2nd',
                      'pointer_to_unary_function',
                      'pointer_to_binary_function',
                      'ptr_fun',
                      'mem_fun_t', 'mem_fun', 'mem_fun1_t', 'mem_fun1_ref_t',
                      'mem_fun_ref_t',
                      'const_mem_fun_t', 'const_mem_fun1_t',
                      'const_mem_fun_ref_t', 'const_mem_fun1_ref_t',
                      'mem_fun_ref',
                     )),
    ('<limits>', ('numeric_limits',)),
    ('<list>', ('list',)),
    ('<map>', ('map', 'multimap',)),
    ('<memory>', ('allocator', 'make_shared', 'make_unique', 'shared_ptr',
                  'unique_ptr', 'weak_ptr')),
    ('<queue>', ('queue', 'priority_queue',)),
    ('<set>', ('set', 'multiset',)),
    ('<stack>', ('stack',)),
    ('<string>', ('char_traits', 'basic_string',)),
    ('<tuple>', ('tuple',)),
    ('<unordered_map>', ('unordered_map', 'unordered_multimap')),
    ('<unordered_set>', ('unordered_set', 'unordered_multiset')),
    ('<utility>', ('pair',)),
    ('<vector>', ('vector',)),

    # gcc extensions.
    # Note: std::hash is their hash, ::hash is our hash
    ('<hash_map>', ('hash_map', 'hash_multimap',)),
    ('<hash_set>', ('hash_set', 'hash_multiset',)),
    ('<slist>', ('slist',)),
    )

_HEADERS_MAYBE_TEMPLATES = (
    ('<algorithm>', ('copy', 'max', 'min', 'min_element', 'sort',
                     'transform',
                    )),
    ('<utility>', ('forward', 'make_pair', 'move', 'swap')),
    )

_RE_PATTERN_STRING = re.compile(r'\bstring\b')

_re_pattern_headers_maybe_templates = []
for _header, _templates in _HEADERS_MAYBE_TEMPLATES:
  for _template in _templates:
    # Match max<type>(..., ...), max(..., ...), but not foo->max, foo.max or
    # type::max().
    _re_pattern_headers_maybe_templates.append(
        (re.compile(r'[^>.]\b' + _template + r'(<.*?>)?\([^\)]'),
            _template,
            _header))

# Other scripts may reach in and modify this pattern.
_re_pattern_templates = []
for _header, _templates in _HEADERS_CONTAINING_TEMPLATES:
  for _template in _templates:
    _re_pattern_templates.append(
        (re.compile(r'(\<|\b)' + _template + r'\s*\<'),
         _template + '<>',
         _header))


def FilesBelongToSameModule(filename_cc, filename_h):
  """Check if these two filenames belong to the same module.

  The concept of a 'module' here is a as follows:
  foo.h, foo-inl.h, foo.cc, foo_test.cc and foo_unittest.cc belong to the
  same 'module' if they are in the same directory.
  some/path/public/xyzzy and some/path/internal/xyzzy are also considered
  to belong to the same module here.

  If the filename_cc contains a longer path than the filename_h, for example,
  '/absolute/path/to/base/sysinfo.cc', and this file would include
  'base/sysinfo.h', this function also produces the prefix needed to open the
  header. This is used by the caller of this function to more robustly open the
  header file. We don't have access to the real include paths in this context,
  so we need this guesswork here.

  Known bugs: tools/base/bar.cc and base/bar.h belong to the same module
  according to this implementation. Because of this, this function gives
  some false positives. This should be sufficiently rare in practice.

  Args:
    filename_cc: is the path for the .cc file
    filename_h: is the path for the header path

  Returns:
    Tuple with a bool and a string:
    bool: True if filename_cc and filename_h belong to the same module.
    string: the additional prefix needed to open the header file.
  """

  fileinfo = FileInfo(filename_cc)
  if not fileinfo.IsSource():
    return (False, '')
  filename_cc = filename_cc[:-len(fileinfo.Extension())]
  matched_test_suffix = Search(_TEST_FILE_SUFFIX, fileinfo.BaseName())
  if matched_test_suffix:
    filename_cc = filename_cc[:-len(matched_test_suffix.group(1))]
  filename_cc = filename_cc.replace('/public/', '/')
  filename_cc = filename_cc.replace('/internal/', '/')

  if not filename_h.endswith('.h'):
    return (False, '')
  filename_h = filename_h[:-len('.h')]
  if filename_h.endswith('-inl'):
    filename_h = filename_h[:-len('-inl')]
  filename_h = filename_h.replace('/public/', '/')
  filename_h = filename_h.replace('/internal/', '/')

  files_belong_to_same_module = filename_cc.endswith(filename_h)
  common_path = ''
  if files_belong_to_same_module:
    common_path = filename_cc[:-len(filename_h)]
  return files_belong_to_same_module, common_path


def UpdateIncludeState(filename, include_dict, io=codecs):
  """Fill up the include_dict with new includes found from the file.

  Args:
    filename: the name of the header to read.
    include_dict: a dictionary in which the headers are inserted.
    io: The io factory to use to read the file. Provided for testability.

  Returns:
    True if a header was successfully added. False otherwise.
  """
  headerfile = None
  try:
    headerfile = io.open(filename, 'r', 'utf8', 'replace')
  except IOError:
    return False
  linenum = 0
  for line in headerfile:
    linenum += 1
    clean_line = CleanseComments(line)
    match = _RE_PATTERN_INCLUDE.search(clean_line)
    if match:
      include = match.group(2)
      include_dict.setdefault(include, linenum)
  return True


def CheckForIncludeWhatYouUse(filename, clean_lines, include_state, error,
                              io=codecs):
  """Reports for missing stl includes.

  This function will output warnings to make sure you are including the headers
  necessary for the stl containers and functions that you use. We only give one
  reason to include a header. For example, if you use both equal_to<> and
  less<> in a .h file, only one (the latter in the file) of these will be
  reported as a reason to include the <functional>.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    include_state: An _IncludeState instance.
    error: The function to call with any errors found.
    io: The IO factory to use to read the header file. Provided for unittest
        injection.
  """
  required = {}  # A map of header name to linenumber and the template entity.
                 # Example of required: { '<functional>': (1219, 'less<>') }

  for linenum in xrange(clean_lines.NumLines()):
    line = clean_lines.elided[linenum]
    if not line or line[0] == '#':
      continue

    # String is special -- it is a non-templatized type in STL.
    matched = _RE_PATTERN_STRING.search(line)
    if matched:
      # Don't warn about strings in non-STL namespaces:
      # (We check only the first match per line; good enough.)
      prefix = line[:matched.start()]
      if prefix.endswith('std::') or not prefix.endswith('::'):
        required['<string>'] = (linenum, 'string')

    for pattern, template, header in _re_pattern_headers_maybe_templates:
      if pattern.search(line):
        required[header] = (linenum, template)

    # The following function is just a speed up, no semantics are changed.
    if not '<' in line:  # Reduces the cpu time usage by skipping lines.
      continue

    for pattern, template, header in _re_pattern_templates:
      matched = pattern.search(line)
      if matched:
        # Don't warn about IWYU in non-STL namespaces:
        # (We check only the first match per line; good enough.)
        prefix = line[:matched.start()]
        if prefix.endswith('std::') or not prefix.endswith('::'):
          required[header] = (linenum, template)

  # The policy is that if you #include something in foo.h you don't need to
  # include it again in foo.cc. Here, we will look at possible includes.
  # Let's flatten the include_state include_list and copy it into a dictionary.
  include_dict = dict([item for sublist in include_state.include_list
                       for item in sublist])

  # Did we find the header for this file (if any) and successfully load it?
  header_found = False

  # Use the absolute path so that matching works properly.
  abs_filename = FileInfo(filename).FullName()

  # For Emacs's flymake.
  # If cpplint is invoked from Emacs's flymake, a temporary file is generated
  # by flymake and that file name might end with '_flymake.cc'. In that case,
  # restore original file name here so that the corresponding header file can be
  # found.
  # e.g. If the file name is 'foo_flymake.cc', we should search for 'foo.h'
  # instead of 'foo_flymake.h'
  abs_filename = re.sub(r'_flymake\.cc$', '.cc', abs_filename)

  # include_dict is modified during iteration, so we iterate over a copy of
  # the keys.
  header_keys = include_dict.keys()
  for header in header_keys:
    (same_module, common_path) = FilesBelongToSameModule(abs_filename, header)
    fullpath = common_path + header
    if same_module and UpdateIncludeState(fullpath, include_dict, io):
      header_found = True

  # If we can't find the header file for a .cc, assume it's because we don't
  # know where to look. In that case we'll give up as we're not sure they
  # didn't include it in the .h file.
  # TODO(unknown): Do a better job of finding .h files so we are confident that
  # not having the .h file means there isn't one.
  if filename.endswith('.cc') and not header_found:
    return

  # All the lines have been processed, report the errors found.
  for required_header_unstripped in required:
    template = required[required_header_unstripped][1]
    if required_header_unstripped.strip('<>"') not in include_dict:
      error(filename, required[required_header_unstripped][0],
            'build/include_what_you_use', 4,
            'Add #include ' + required_header_unstripped + ' for ' + template)


_RE_PATTERN_EXPLICIT_MAKEPAIR = re.compile(r'\bmake_pair\s*<')


def CheckMakePairUsesDeduction(filename, clean_lines, linenum, error):
  """Check that make_pair's template arguments are deduced.

  G++ 4.6 in C++11 mode fails badly if make_pair's template arguments are
  specified explicitly, and such use isn't intended in any case.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]
  match = _RE_PATTERN_EXPLICIT_MAKEPAIR.search(line)
  if match:
    error(filename, linenum, 'build/explicit_make_pair',
          4,  # 4 = high confidence
          'For C++11-compatibility, omit template arguments from make_pair'
          ' OR use pair directly OR if appropriate, construct a pair directly')


def CheckRedundantVirtual(filename, clean_lines, linenum, error):
  """Check if line contains a redundant "virtual" function-specifier.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  # Look for "virtual" on current line.
  line = clean_lines.elided[linenum]
  virtual = Match(r'^(.*)(\bvirtual\b)(.*)$', line)
  if not virtual: return

  # Ignore "virtual" keywords that are near access-specifiers.  These
  # are only used in class base-specifier and do not apply to member
  # functions.
  if (Search(r'\b(public|protected|private)\s+$', virtual.group(1)) or
      Match(r'^\s+(public|protected|private)\b', virtual.group(3))):
    return

  # Ignore the "virtual" keyword from virtual base classes.  Usually
  # there is a column on the same line in these cases (virtual base
  # classes are rare in google3 because multiple inheritance is rare).
  if Match(r'^.*[^:]:[^:].*$', line): return

  # Look for the next opening parenthesis.  This is the start of the
  # parameter list (possibly on the next line shortly after virtual).
  # TODO(unknown): doesn't work if there are virtual functions with
  # decltype() or other things that use parentheses, but csearch suggests
  # that this is rare.
  end_col = -1
  end_line = -1
  start_col = len(virtual.group(2))
  for start_line in xrange(linenum, min(linenum + 3, clean_lines.NumLines())):
    line = clean_lines.elided[start_line][start_col:]
    parameter_list = Match(r'^([^(]*)\(', line)
    if parameter_list:
      # Match parentheses to find the end of the parameter list
      (_, end_line, end_col) = CloseExpression(
          clean_lines, start_line, start_col + len(parameter_list.group(1)))
      break
    start_col = 0

  if end_col < 0:
    return  # Couldn't find end of parameter list, give up

  # Look for "override" or "final" after the parameter list
  # (possibly on the next few lines).
  for i in xrange(end_line, min(end_line + 3, clean_lines.NumLines())):
    line = clean_lines.elided[i][end_col:]
    match = Search(r'\b(override|final)\b', line)
    if match:
      error(filename, linenum, 'readability/inheritance', 4,
            ('"virtual" is redundant since function is '
             'already declared as "%s"' % match.group(1)))

    # Set end_col to check whole lines after we are done with the
    # first line.
    end_col = 0
    if Search(r'[^\w]\s*$', line):
      break


def CheckRedundantOverrideOrFinal(filename, clean_lines, linenum, error):
  """Check if line contains a redundant "override" or "final" virt-specifier.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  # Look for closing parenthesis nearby.  We need one to confirm where
  # the declarator ends and where the virt-specifier starts to avoid
  # false positives.
  line = clean_lines.elided[linenum]
  declarator_end = line.rfind(')')
  if declarator_end >= 0:
    fragment = line[declarator_end:]
  else:
    if linenum > 1 and clean_lines.elided[linenum - 1].rfind(')') >= 0:
      fragment = line
    else:
      return

  # Check that at most one of "override" or "final" is present, not both
  if Search(r'\boverride\b', fragment) and Search(r'\bfinal\b', fragment):
    error(filename, linenum, 'readability/inheritance', 4,
          ('"override" is redundant since function is '
           'already declared as "final"'))




# Returns true if we are at a new block, and it is directly
# inside of a namespace.
def IsBlockInNameSpace(nesting_state, is_forward_declaration):
  """Checks that the new block is directly in a namespace.

  Args:
    nesting_state: The _NestingState object that contains info about our state.
    is_forward_declaration: If the class is a forward declared class.
  Returns:
    Whether or not the new block is directly in a namespace.
  """
  if is_forward_declaration:
    if len(nesting_state.stack) >= 1 and (
        isinstance(nesting_state.stack[-1], _NamespaceInfo)):
      return True
    else:
      return False

  return (len(nesting_state.stack) > 1 and
          nesting_state.stack[-1].check_namespace_indentation and
          isinstance(nesting_state.stack[-2], _NamespaceInfo))


def ShouldCheckNamespaceIndentation(nesting_state, is_namespace_indent_item,
                                    raw_lines_no_comments, linenum):
  """This method determines if we should apply our namespace indentation check.

  Args:
    nesting_state: The current nesting state.
    is_namespace_indent_item: If we just put a new class on the stack, True.
      If the top of the stack is not a class, or we did not recently
      add the class, False.
    raw_lines_no_comments: The lines without the comments.
    linenum: The current line number we are processing.

  Returns:
    True if we should apply our namespace indentation check. Currently, it
    only works for classes and namespaces inside of a namespace.
  """

  is_forward_declaration = IsForwardClassDeclaration(raw_lines_no_comments,
                                                     linenum)

  if not (is_namespace_indent_item or is_forward_declaration):
    return False

  # If we are in a macro, we do not want to check the namespace indentation.
  if IsMacroDefinition(raw_lines_no_comments, linenum):
    return False

  return IsBlockInNameSpace(nesting_state, is_forward_declaration)


# Call this method if the line is directly inside of a namespace.
# If the line above is blank (excluding comments) or the start of
# an inner namespace, it cannot be indented.
def CheckItemIndentationInNamespace(filename, raw_lines_no_comments, linenum,
                                    error):
  line = raw_lines_no_comments[linenum]
  if Match(r'^\s+', line):
    error(filename, linenum, 'runtime/indentation_namespace', 4,
          'Do not indent within a namespace')


def ProcessLine(filename, file_extension, clean_lines, line,
                include_state, function_state, nesting_state, error,
                extra_check_functions=[]):
  """Processes a single line in the file.

  Args:
    filename: Filename of the file that is being processed.
    file_extension: The extension (dot not included) of the file.
    clean_lines: An array of strings, each representing a line of the file,
                 with comments stripped.
    line: Number of line being processed.
    include_state: An _IncludeState instance in which the headers are inserted.
    function_state: A _FunctionState instance which counts function lines, etc.
    nesting_state: A NestingState instance which maintains information about
                   the current stack of nested blocks being parsed.
    error: A callable to which errors are reported, which takes 4 arguments:
           filename, line number, error level, and message
    extra_check_functions: An array of additional check functions that will be
                           run on each source line. Each function takes 4
                           arguments: filename, clean_lines, line, error
  """
  raw_lines = clean_lines.raw_lines
  ParseNolintSuppressions(filename, raw_lines[line], line, error)
  nesting_state.Update(filename, clean_lines, line, error)
  CheckForNamespaceIndentation(filename, nesting_state, clean_lines, line,
                               error)
  if nesting_state.InAsmBlock(): return
  CheckForFunctionLengths(filename, clean_lines, line, function_state, error)
  CheckForMultilineCommentsAndStrings(filename, clean_lines, line, error)
  CheckStyle(filename, clean_lines, line, file_extension, nesting_state, error)
  CheckLanguage(filename, clean_lines, line, file_extension, include_state,
                nesting_state, error)
  CheckForNonConstReference(filename, clean_lines, line, nesting_state, error)
  CheckForNonStandardConstructs(filename, clean_lines, line,
                                nesting_state, error)
  CheckVlogArguments(filename, clean_lines, line, error)
  CheckPosixThreading(filename, clean_lines, line, error)
  CheckInvalidIncrement(filename, clean_lines, line, error)
  CheckMakePairUsesDeduction(filename, clean_lines, line, error)
  CheckRedundantVirtual(filename, clean_lines, line, error)
  CheckRedundantOverrideOrFinal(filename, clean_lines, line, error)
  for check_fn in extra_check_functions:
    check_fn(filename, clean_lines, line, error)

def FlagCxx11Features(filename, clean_lines, linenum, error):
  """Flag those c++11 features that we only allow in certain places.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  include = Match(r'\s*#\s*include\s+[<"]([^<"]+)[">]', line)

  # Flag unapproved C++ TR1 headers.
  if include and include.group(1).startswith('tr1/'):
    error(filename, linenum, 'build/c++tr1', 5,
          ('C++ TR1 headers such as <%s> are unapproved.') % include.group(1))

  # Flag unapproved C++11 headers.
  if include and include.group(1) in ('cfenv',
                                      'condition_variable',
                                      'fenv.h',
                                      'future',
                                      'mutex',
                                      'thread',
                                      'chrono',
                                      'ratio',
                                      'regex',
                                      'system_error',
                                     ):
    error(filename, linenum, 'build/c++11', 5,
          ('<%s> is an unapproved C++11 header.') % include.group(1))

  # The only place where we need to worry about C++11 keywords and library
  # features in preprocessor directives is in macro definitions.
  if Match(r'\s*#', line) and not Match(r'\s*#\s*define\b', line): return

  # These are classes and free functions.  The classes are always
  # mentioned as std::*, but we only catch the free functions if
  # they're not found by ADL.  They're alphabetical by header.
  for top_name in (
      # type_traits
      'alignment_of',
      'aligned_union',
      ):
    if Search(r'\bstd::%s\b' % top_name, line):
      error(filename, linenum, 'build/c++11', 5,
            ('std::%s is an unapproved C++11 class or function.  Send c-style '
             'an example of where it would make your code more readable, and '
             'they may let you use it.') % top_name)


def FlagCxx14Features(filename, clean_lines, linenum, error):
  """Flag those C++14 features that we restrict.

  Args:
    filename: The name of the current file.
    clean_lines: A CleansedLines instance containing the file.
    linenum: The number of the line to check.
    error: The function to call with any errors found.
  """
  line = clean_lines.elided[linenum]

  include = Match(r'\s*#\s*include\s+[<"]([^<"]+)[">]', line)

  # Flag unapproved C++14 headers.
  if include and include.group(1) in ('scoped_allocator', 'shared_mutex'):
    error(filename, linenum, 'build/c++14', 5,
          ('<%s> is an unapproved C++14 header.') % include.group(1))


def ProcessFileData(filename, file_extension, lines, error,
                    extra_check_functions=[]):
  """Performs lint checks and reports any errors to the given error function.

  Args:
    filename: Filename of the file that is being processed.
    file_extension: The extension (dot not included) of the file.
    lines: An array of strings, each representing a line of the file, with the
           last element being empty if the file is terminated with a newline.
    error: A callable to which errors are reported, which takes 4 arguments:
           filename, line number, error level, and message
    extra_check_functions: An array of additional check functions that will be
                           run on each source line. Each function takes 4
                           arguments: filename, clean_lines, line, error
  """
  lines = (['// marker so line numbers and indices both start at 1'] + lines +
           ['// marker so line numbers end in a known way'])

  include_state = _IncludeState()
  function_state = _FunctionState()
  nesting_state = NestingState()

  ResetNolintSuppressions()

  CheckForCopyright(filename, lines, error)
  ProcessGlobalSuppresions(lines)
  RemoveMultiLineComments(filename, lines, error)
  clean_lines = CleansedLines(lines)

  if IsHeaderExtension(file_extension):
    CheckForHeaderGuard(filename, clean_lines, error)

  for line in xrange(clean_lines.NumLines()):
    ProcessLine(filename, file_extension, clean_lines, line,
                include_state, function_state, nesting_state, error,
                extra_check_functions)
    FlagCxx11Features(filename, clean_lines, line, error)
  nesting_state.CheckCompletedBlocks(filename, error)

  CheckForIncludeWhatYouUse(filename, clean_lines, include_state, error)

  # Check that the .cc file has included its header if it exists.
  if _IsSourceExtension(file_extension):
    CheckHeaderFileIncluded(filename, include_state, error)

  # We check here rather than inside ProcessLine so that we see raw
  # lines rather than "cleaned" lines.
  CheckForBadCharacters(filename, lines, error)

  CheckForNewlineAtEOF(filename, lines, error)

def ProcessConfigOverrides(filename):
  """ Loads the configuration files and processes the config overrides.

  Args:
    filename: The name of the file being processed by the linter.

  Returns:
    False if the current |filename| should not be processed further.
  """

  abs_filename = os.path.abspath(filename)
  cfg_filters = []
  keep_looking = True
  while keep_looking:
    abs_path, base_name = os.path.split(abs_filename)
    if not base_name:
      break  # Reached the root directory.

    cfg_file = os.path.join(abs_path, "CPPLINT.cfg")
    abs_filename = abs_path
    if not os.path.isfile(cfg_file):
      continue

    try:
      with open(cfg_file) as file_handle:
        for line in file_handle:
          line, _, _ = line.partition('#')  # Remove comments.
          if not line.strip():
            continue

          name, _, val = line.partition('=')
          name = name.strip()
          val = val.strip()
          if name == 'set noparent':
            keep_looking = False
          elif name == 'filter':
            cfg_filters.append(val)
          elif name == 'exclude_files':
            # When matching exclude_files pattern, use the base_name of
            # the current file name or the directory name we are processing.
            # For example, if we are checking for lint errors in /foo/bar/baz.cc
            # and we found the .cfg file at /foo/CPPLINT.cfg, then the config
            # file's "exclude_files" filter is meant to be checked against "bar"
            # and not "baz" nor "bar/baz.cc".
            if base_name:
              pattern = re.compile(val)
              if pattern.match(base_name):
                if _cpplint_state.quiet:
                  # Suppress "Ignoring file" warning when using --quiet.
                  return False
                sys.stderr.write('Ignoring "%s": file excluded by "%s". '
                                 'File path component "%s" matches '
                                 'pattern "%s"\n' %
                                 (filename, cfg_file, base_name, val))
                return False
          elif name == 'linelength':
            global _line_length
            try:
                _line_length = int(val)
            except ValueError:
                sys.stderr.write('Line length must be numeric.')
          elif name == 'root':
            global _root
            # root directories are specified relative to CPPLINT.cfg dir.
            _root = os.path.join(os.path.dirname(cfg_file), val)
          elif name == 'headers':
            ProcessHppHeadersOption(val)
          else:
            sys.stderr.write(
                'Invalid configuration option (%s) in file %s\n' %
                (name, cfg_file))

    except IOError:
      sys.stderr.write(
          "Skipping config file '%s': Can't open for reading\n" % cfg_file)
      keep_looking = False

  # Apply all the accumulated filters in reverse order (top-level directory
  # config options having the least priority).
  for filter in reversed(cfg_filters):
     _AddFilters(filter)

  return True


def ProcessFile(filename, vlevel, extra_check_functions=[]):
  """Does google-lint on a single file.

  Args:
    filename: The name of the file to parse.

    vlevel: The level of errors to report.  Every error of confidence
    >= verbose_level will be reported.  0 is a good default.

    extra_check_functions: An array of additional check functions that will be
                           run on each source line. Each function takes 4
                           arguments: filename, clean_lines, line, error
  """

  _SetVerboseLevel(vlevel)
  _BackupFilters()
  old_errors = _cpplint_state.error_count

  if not ProcessConfigOverrides(filename):
    _RestoreFilters()
    return

  lf_lines = []
  crlf_lines = []
  try:
    # Support the UNIX convention of using "-" for stdin.  Note that
    # we are not opening the file with universal newline support
    # (which codecs doesn't support anyway), so the resulting lines do
    # contain trailing '\r' characters if we are reading a file that
    # has CRLF endings.
    # If after the split a trailing '\r' is present, it is removed
    # below.
    if filename == '-':
      lines = codecs.StreamReaderWriter(sys.stdin,
                                        codecs.getreader('utf8'),
                                        codecs.getwriter('utf8'),
                                        'replace').read().split('\n')
    else:
      lines = codecs.open(filename, 'r', 'utf8', 'replace').read().split('\n')

    # Remove trailing '\r'.
    # The -1 accounts for the extra trailing blank line we get from split()
    for linenum in range(len(lines) - 1):
      if lines[linenum].endswith('\r'):
        lines[linenum] = lines[linenum].rstrip('\r')
        crlf_lines.append(linenum + 1)
      else:
        lf_lines.append(linenum + 1)

  except IOError:
    sys.stderr.write(
        "Skipping input '%s': Can't open for reading\n" % filename)
    _RestoreFilters()
    return

  # Note, if no dot is found, this will give the entire filename as the ext.
  file_extension = filename[filename.rfind('.') + 1:]

  # When reading from stdin, the extension is unknown, so no cpplint tests
  # should rely on the extension.
  if filename != '-' and file_extension not in _valid_extensions:
    sys.stderr.write('Ignoring %s; not a valid file name '
                     '(%s)\n' % (filename, ', '.join(_valid_extensions)))
  else:
    ProcessFileData(filename, file_extension, lines, Error,
                    extra_check_functions)

    # If end-of-line sequences are a mix of LF and CR-LF, issue
    # warnings on the lines with CR.
    #
    # Don't issue any warnings if all lines are uniformly LF or CR-LF,
    # since critique can handle these just fine, and the style guide
    # doesn't dictate a particular end of line sequence.
    #
    # We can't depend on os.linesep to determine what the desired
    # end-of-line sequence should be, since that will return the
    # server-side end-of-line sequence.
    if lf_lines and crlf_lines:
      # Warn on every line with CR.  An alternative approach might be to
      # check whether the file is mostly CRLF or just LF, and warn on the
      # minority, we bias toward LF here since most tools prefer LF.
      for linenum in crlf_lines:
        Error(filename, linenum, 'whitespace/newline', 1,
              'Unexpected \\r (^M) found; better to use only \\n')

  # Suppress printing anything if --quiet was passed unless the error
  # count has increased after processing this file.
  if not _cpplint_state.quiet or old_errors != _cpplint_state.error_count:
    sys.stdout.write('Done processing %s\n' % filename)
  _RestoreFilters()


def PrintUsage(message):
  """Prints a brief usage string and exits, optionally with an error message.

  Args:
    message: The optional error message.
  """
  sys.stderr.write(_USAGE)
  if message:
    sys.exit('\nFATAL ERROR: ' + message)
  else:
    sys.exit(1)


def PrintCategories():
  """Prints a list of all the error-categories used by error messages.

  These are the categories used to filter messages via --filter.
  """
  sys.stderr.write(''.join('  %s\n' % cat for cat in _ERROR_CATEGORIES))
  sys.exit(0)


def ParseArguments(args):
  """Parses the command line arguments.

  This may set the output format and verbosity level as side-effects.

  Args:
    args: The command line arguments:

  Returns:
    The list of filenames to lint.
  """
  try:
    (opts, filenames) = getopt.getopt(args, '', ['help', 'output=', 'verbose=',
                                                 'counting=',
                                                 'filter=',
                                                 'root=',
                                                 'linelength=',
                                                 'extensions=',
                                                 'headers=',
                                                 'quiet'])
  except getopt.GetoptError:
    PrintUsage('Invalid arguments.')

  verbosity = _VerboseLevel()
  output_format = _OutputFormat()
  filters = ''
  quiet = _Quiet()
  counting_style = ''

  for (opt, val) in opts:
    if opt == '--help':
      PrintUsage(None)
    elif opt == '--output':
      if val not in ('emacs', 'vs7', 'eclipse'):
        PrintUsage('The only allowed output formats are emacs, vs7 and eclipse.')
      output_format = val
    elif opt == '--quiet':
      quiet = True
    elif opt == '--verbose':
      verbosity = int(val)
    elif opt == '--filter':
      filters = val
      if not filters:
        PrintCategories()
    elif opt == '--counting':
      if val not in ('total', 'toplevel', 'detailed'):
        PrintUsage('Valid counting options are total, toplevel, and detailed')
      counting_style = val
    elif opt == '--root':
      global _root
      _root = val
    elif opt == '--linelength':
      global _line_length
      try:
          _line_length = int(val)
      except ValueError:
          PrintUsage('Line length must be digits.')
    elif opt == '--extensions':
      global _valid_extensions
      try:
          _valid_extensions = set(val.split(','))
      except ValueError:
          PrintUsage('Extensions must be comma seperated list.')
    elif opt == '--headers':
      ProcessHppHeadersOption(val)

  if not filenames:
    PrintUsage('No files were specified.')

  _SetOutputFormat(output_format)
  _SetQuiet(quiet)
  _SetVerboseLevel(verbosity)
  _SetFilters(filters)
  _SetCountingStyle(counting_style)

  return filenames


def main():
  filenames = ParseArguments(sys.argv[1:])

  # Change stderr to write with replacement characters so we don't die
  # if we try to print something containing non-ASCII characters.
  sys.stderr = codecs.StreamReaderWriter(sys.stderr,
                                         codecs.getreader('utf8'),
                                         codecs.getwriter('utf8'),
                                         'replace')

  _cpplint_state.ResetErrorCounts()
  for filename in filenames:
    ProcessFile(filename, _cpplint_state.verbose_level)
  # If --quiet is passed, suppress printing error count unless there are errors.
  if not _cpplint_state.quiet or _cpplint_state.error_count > 0:
    _cpplint_state.PrintErrorCounts()

  sys.exit(_cpplint_state.error_count > 0)


if __name__ == '__main__':
  main()

  """-----------------------------------------------------------------------------------------------"""# This is a port of the Vista SDK "FolderView" sample, and associated
# notes at http://shellrevealed.com/blogs/shellblog/archive/2007/03/15/Shell-Namespace-Extension_3A00_-Creating-and-Using-the-System-Folder-View-Object.aspx
# A key difference to shell_view.py is that this version uses the default
# IShellView provided by the shell (via SHCreateShellFolderView) rather
# than our own.
# XXX - sadly, it doesn't work quite like the original sample.  Oh well,
# another day...
import sys
import os
import pickle
import random
import win32api
import winxpgui as win32gui  # the needs vista, let alone xp!
import win32con
import winerror
import commctrl
import pythoncom
from win32com.util import IIDToInterfaceName
from win32com.server.exception import COMException
from win32com.server.util import wrap as _wrap
from win32com.server.util import NewEnum as _NewEnum
from win32com.shell import shell, shellcon
from win32com.axcontrol import axcontrol  # IObjectWithSite
from win32com.propsys import propsys

GUID = pythoncom.MakeIID

# If set, output spews to the win32traceutil collector...
debug = 0
# wrap a python object in a COM pointer
def wrap(ob, iid=None):
    return _wrap(ob, iid, useDispatcher=(debug > 0))


def NewEnum(seq, iid):
    return _NewEnum(seq, iid=iid, useDispatcher=(debug > 0))


# The sample makes heavy use of "string ids" (ie, integer IDs defined in .h
# files, loaded at runtime from a (presumably localized) DLL.  We cheat.
_sids = {}  # strings, indexed bystring_id,


def LoadString(sid):
    return _sids[sid]


# fn to create a unique string ID
_last_ids = 0


def _make_ids(s):
    global _last_ids
    _last_ids += 1
    _sids[_last_ids] = s
    return _last_ids


# These strings are what the user sees and would be localized.
# XXX - its possible that the shell might persist these values, so
# this scheme wouldn't really be suitable in a real ap.
IDS_UNSPECIFIED = _make_ids("unspecified")
IDS_SMALL = _make_ids("small")
IDS_MEDIUM = _make_ids("medium")
IDS_LARGE = _make_ids("large")
IDS_CIRCLE = _make_ids("circle")
IDS_TRIANGLE = _make_ids("triangle")
IDS_RECTANGLE = _make_ids("rectangle")
IDS_POLYGON = _make_ids("polygon")
IDS_DISPLAY = _make_ids("Display")
IDS_DISPLAY_TT = _make_ids("Display the item.")
IDS_SETTINGS = _make_ids("Settings")
IDS_SETTING1 = _make_ids("Setting 1")
IDS_SETTING2 = _make_ids("Setting 2")
IDS_SETTING3 = _make_ids("Setting 3")
IDS_SETTINGS_TT = _make_ids("Modify settings.")
IDS_SETTING1_TT = _make_ids("Modify setting 1.")
IDS_SETTING2_TT = _make_ids("Modify setting 2.")
IDS_SETTING3_TT = _make_ids("Modify setting 3.")
IDS_LESSTHAN5 = _make_ids("Less Than 5")
IDS_5ORGREATER = _make_ids("Five or Greater")
del _make_ids, _last_ids

# Other misc resource stuff
IDI_ICON1 = 100
IDI_SETTINGS = 101

# The sample defines a number of "category ids".  Each one gets
# its own GUID.
CAT_GUID_NAME = GUID("{de094c9d-c65a-11dc-ba21-005056c00008}")
CAT_GUID_SIZE = GUID("{de094c9e-c65a-11dc-ba21-005056c00008}")
CAT_GUID_SIDES = GUID("{de094c9f-c65a-11dc-ba21-005056c00008}")
CAT_GUID_LEVEL = GUID("{de094ca0-c65a-11dc-ba21-005056c00008}")
# The next category guid is NOT based on a column (see
# ViewCategoryProvider::EnumCategories()...)
CAT_GUID_VALUE = "{de094ca1-c65a-11dc-ba21-005056c00008}"

GUID_Display = GUID("{4d6c2fdd-c689-11dc-ba21-005056c00008}")
GUID_Settings = GUID("{4d6c2fde-c689-11dc-ba21-005056c00008}")
GUID_Setting1 = GUID("{4d6c2fdf-c689-11dc-ba21-005056c00008}")
GUID_Setting2 = GUID("{4d6c2fe0-c689-11dc-ba21-005056c00008}")
GUID_Setting3 = GUID("{4d6c2fe1-c689-11dc-ba21-005056c00008}")

# Hrm - not sure what to do about the std keys.
# Probably need a simple parser for propkey.h
PKEY_ItemNameDisplay = ("{B725F130-47EF-101A-A5F1-02608C9EEBAC}", 10)
PKEY_PropList_PreviewDetails = ("{C9944A21-A406-48FE-8225-AEC7E24C211B}", 8)

# Not sure what the "3" here refers to - docs say PID_FIRST_USABLE (2) be
# used.  Presumably it is the 'propID' value in the .propdesc file!
# note that the following GUIDs are also references in the .propdesc file
PID_SOMETHING = 3
# These are our 'private' PKEYs
# Col 2, name="Sample.AreaSize"
PKEY_Sample_AreaSize = ("{d6f5e341-c65c-11dc-ba21-005056c00008}", PID_SOMETHING)
# Col 3, name="Sample.NumberOfSides"
PKEY_Sample_NumberOfSides = ("{d6f5e342-c65c-11dc-ba21-005056c00008}", PID_SOMETHING)
# Col 4, name="Sample.DirectoryLevel"
PKEY_Sample_DirectoryLevel = ("{d6f5e343-c65c-11dc-ba21-005056c00008}", PID_SOMETHING)

# We construct a PIDL from a pickle of a dict - turn it back into a
# dict (we should *never* be called with a PIDL that the last elt is not
# ours, so it is safe to assume we created it (assume->"ass" = "u" + "me" :)
def pidl_to_item(pidl):
    # Note that only the *last* elt in the PIDL is certainly ours,
    # but it contains everything we need encoded as a dict.
    return pickle.loads(pidl[-1])


# Start of msdn sample port...
# make_item_enum replaces the sample's entire EnumIDList.cpp :)
def make_item_enum(level, flags):
    pidls = []
    nums = """zero one two three four five size seven eight nine ten""".split()
    for i, name in enumerate(nums):
        size = random.randint(0, 255)
        sides = 1
        while sides in [1, 2]:
            sides = random.randint(0, 5)
        is_folder = (i % 2) != 0
        # check the flags say to include it.
        # (This seems strange; if you ask the same folder for, but appear
        skip = False
        if not (flags & shellcon.SHCONTF_STORAGE):
            if is_folder:
                skip = not (flags & shellcon.SHCONTF_FOLDERS)
            else:
                skip = not (flags & shellcon.SHCONTF_NONFOLDERS)
        if not skip:
            data = dict(
                name=name, size=size, sides=sides, level=level, is_folder=is_folder
            )
            pidls.append([pickle.dumps(data)])
    return NewEnum(pidls, shell.IID_IEnumIDList)


# start of Utils.cpp port
def DisplayItem(shell_item_array, hwnd_parent=0):
    # Get the first ShellItem and display its name
    if shell_item_array is None:
        msg = "You must select something!"
    else:
        si = shell_item_array.GetItemAt(0)
        name = si.GetDisplayName(shellcon.SIGDN_NORMALDISPLAY)
        msg = "%d items selected, first is %r" % (shell_item_array.GetCount(), name)
    win32gui.MessageBox(hwnd_parent, msg, "Hello", win32con.MB_OK)


# end of Utils.cpp port

# start of sample's FVCommands.cpp port
class Command:
    def __init__(self, guid, ids, ids_tt, idi, flags, callback, children):
        self.guid = guid
        self.ids = ids
        self.ids_tt = ids_tt
        self.idi = idi
        self.flags = flags
        self.callback = callback
        self.children = children
        assert not children or isinstance(children[0], Command)

    def tuple(self):
        return (
            self.guid,
            self.ids,
            self.ids_tt,
            self.idi,
            self.flags,
            self.callback,
            self.children,
        )


# command callbacks - called back directly by us - see ExplorerCommand.Invoke
def onDisplay(items, bindctx):
    DisplayItem(items)


def onSetting1(items, bindctx):
    win32gui.MessageBox(0, LoadString(IDS_SETTING1), "Hello", win32con.MB_OK)


def onSetting2(items, bindctx):
    win32gui.MessageBox(0, LoadString(IDS_SETTING2), "Hello", win32con.MB_OK)


def onSetting3(items, bindctx):
    win32gui.MessageBox(0, LoadString(IDS_SETTING3), "Hello", win32con.MB_OK)


taskSettings = [
    Command(
        GUID_Setting1, IDS_SETTING1, IDS_SETTING1_TT, IDI_SETTINGS, 0, onSetting1, None
    ),
    Command(
        GUID_Setting2, IDS_SETTING2, IDS_SETTING2_TT, IDI_SETTINGS, 0, onSetting2, None
    ),
    Command(
        GUID_Setting3, IDS_SETTING3, IDS_SETTING3_TT, IDI_SETTINGS, 0, onSetting3, None
    ),
]

tasks = [
    Command(GUID_Display, IDS_DISPLAY, IDS_DISPLAY_TT, IDI_ICON1, 0, onDisplay, None),
    Command(
        GUID_Settings,
        IDS_SETTINGS,
        IDS_SETTINGS_TT,
        IDI_SETTINGS,
        shellcon.ECF_HASSUBCOMMANDS,
        None,
        taskSettings,
    ),
]


class ExplorerCommandProvider:
    _com_interfaces_ = [shell.IID_IExplorerCommandProvider]
    _public_methods_ = shellcon.IExplorerCommandProvider_Methods

    def GetCommands(self, site, iid):
        items = [wrap(ExplorerCommand(t)) for t in tasks]
        return NewEnum(items, shell.IID_IEnumExplorerCommand)


class ExplorerCommand:
    _com_interfaces_ = [shell.IID_IExplorerCommand]
    _public_methods_ = shellcon.IExplorerCommand_Methods

    def __init__(self, cmd):
        self.cmd = cmd

    # The sample also appears to ignore the pidl args!?
    def GetTitle(self, pidl):
        return LoadString(self.cmd.ids)

    def GetToolTip(self, pidl):
        return LoadString(self.cmd.ids_tt)

    def GetIcon(self, pidl):
        # Return a string of the usual "dll,resource_id" format
        # todo - just return any ".ico that comes with python" + ",0" :)
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetState(self, shell_items, slow_ok):
        return shellcon.ECS_ENABLED

    def GetFlags(self):
        return self.cmd.flags

    def GetCanonicalName(self):
        return self.cmd.guid

    def Invoke(self, items, bind_ctx):
        # If no function defined - just return S_OK
        if self.cmd.callback:
            self.cmd.callback(items, bind_ctx)
        else:
            print("No callback for command ", LoadString(self.cmd.ids))

    def EnumSubCommands(self):
        if not self.cmd.children:
            return None
        items = [wrap(ExplorerCommand(c)) for c in self.cmd.children]
        return NewEnum(items, shell.IID_IEnumExplorerCommand)


# end of sample's FVCommands.cpp port

# start of sample's Category.cpp port
class FolderViewCategorizer:
    _com_interfaces_ = [shell.IID_ICategorizer]
    _public_methods_ = shellcon.ICategorizer_Methods

    description = None  # subclasses should set their own

    def __init__(self, shell_folder):
        self.sf = shell_folder

    #  Determines the relative order of two items in their item identifier lists.
    def CompareCategory(self, flags, cat1, cat2):
        return cat1 - cat2

    #  Retrieves the name of a categorizer, such as "Group By Device
    #  Type", that can be displayed in the user interface.
    def GetDescription(self, cch):
        return self.description

    # Retrieves information about a category, such as the default
    # display and the text to display in the user interface.
    def GetCategoryInfo(self, catid):
        # Note: this isn't always appropriate!  See overrides below
        return 0, str(catid)  # ????


class FolderViewCategorizer_Name(FolderViewCategorizer):
    description = "Alphabetical"

    def GetCategory(self, pidls):
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_ItemNameDisplay)
            ret.append(val)
        return ret


class FolderViewCategorizer_Size(FolderViewCategorizer):
    description = "Group By Size"

    def GetCategory(self, pidls):
        ret = []
        for pidl in pidls:
            # Why don't we just get the size of the PIDL?
            val = self.sf.GetDetailsEx(pidl, PKEY_Sample_AreaSize)
            val = int(val)  # it probably came in a VT_BSTR variant
            if val < 255 // 3:
                cid = IDS_SMALL
            elif val < 2 * 255 // 3:
                cid = IDS_MEDIUM
            else:
                cid = IDS_LARGE
            ret.append(cid)
        return ret

    def GetCategoryInfo(self, catid):
        return 0, LoadString(catid)


class FolderViewCategorizer_Sides(FolderViewCategorizer):
    description = "Group By Sides"

    def GetCategory(self, pidls):
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_ItemNameDisplay)
            if val == 0:
                cid = IDS_CIRCLE
            elif val == 3:
                cid = IDS_TRIANGLE
            elif val == 4:
                cid = IDS_RECTANGLE
            elif val == 5:
                cid = IDS_POLYGON
            else:
                cid = IDS_UNSPECIFIED
            ret.append(cid)
        return ret

    def GetCategoryInfo(self, catid):
        return 0, LoadString(catid)


class FolderViewCategorizer_Value(FolderViewCategorizer):
    description = "Group By Value"

    def GetCategory(self, pidls):
        ret = []
        for pidl in pidls:
            val = self.sf.GetDetailsEx(pidl, PKEY_ItemNameDisplay)
            if val in "one two three four".split():
                ret.append(IDS_LESSTHAN5)
            else:
                ret.append(IDS_5ORGREATER)
        return ret

    def GetCategoryInfo(self, catid):
        return 0, LoadString(catid)


class FolderViewCategorizer_Level(FolderViewCategorizer):
    description = "Group By Value"

    def GetCategory(self, pidls):
        return [
            self.sf.GetDetailsEx(pidl, PKEY_Sample_DirectoryLevel) for pidl in pidls
        ]


class ViewCategoryProvider:
    _com_interfaces_ = [shell.IID_ICategoryProvider]
    _public_methods_ = shellcon.ICategoryProvider_Methods

    def __init__(self, shell_folder):
        self.shell_folder = shell_folder

    def CanCategorizeOnSCID(self, pkey):
        return pkey in [
            PKEY_ItemNameDisplay,
            PKEY_Sample_AreaSize,
            PKEY_Sample_NumberOfSides,
            PKEY_Sample_DirectoryLevel,
        ]

    #  Creates a category object.
    def CreateCategory(self, guid, iid):
        if iid == shell.IID_ICategorizer:
            if guid == CAT_GUID_NAME:
                klass = FolderViewCategorizer_Name
            elif guid == CAT_GUID_SIDES:
                klass = FolderViewCategorizer_Sides
            elif guid == CAT_GUID_SIZE:
                klass = FolderViewCategorizer_Size
            elif guid == CAT_GUID_VALUE:
                klass = FolderViewCategorizer_Value
            elif guid == CAT_GUID_LEVEL:
                klass = FolderViewCategorizer_Level
            else:
                raise COMException(hresult=winerror.E_INVALIDARG)
            return wrap(klass(self.shell_folder))
        raise COMException(hresult=winerror.E_NOINTERFACE)

    #  Retrieves the enumerator for the categories.
    def EnumCategories(self):
        # These are additional categories beyond the columns
        seq = [CAT_GUID_VALUE]
        return NewEnum(seq, pythoncom.IID_IEnumGUID)

    #  Retrieves a globally unique identifier (GUID) that represents
    #  the categorizer to use for the specified Shell column.
    def GetCategoryForSCID(self, scid):
        if scid == PKEY_ItemNameDisplay:
            guid = CAT_GUID_NAME
        elif scid == PKEY_Sample_AreaSize:
            guid = CAT_GUID_SIZE
        elif scid == PKEY_Sample_NumberOfSides:
            guid = CAT_GUID_SIDES
        elif scid == PKEY_Sample_DirectoryLevel:
            guid = CAT_GUID_LEVEL
        elif scid == pythoncom.IID_NULL:
            # This can be called with a NULL
            # format ID. This will happen if you have a category,
            # not based on a column, that gets stored in the
            # property bag. When a return is made to this item,
            # it will call this function with a NULL format id.
            guid = CAT_GUID_VALUE
        else:
            raise COMException(hresult=winerror.E_INVALIDARG)
        return guid

    #  Retrieves the name of the specified category. This is where
    #  additional categories that appear under the column
    #  related categories in the UI, get their display names.
    def GetCategoryName(self, guid, cch):
        if guid == CAT_GUID_VALUE:
            return "Value"
        raise COMException(hresult=winerror.E_FAIL)

    #  Enables the folder to override the default grouping.
    def GetDefaultCategory(self):
        return CAT_GUID_LEVEL, (pythoncom.IID_NULL, 0)


# end of sample's Category.cpp port

# start of sample's ContextMenu.cpp port
MENUVERB_DISPLAY = 0

folderViewImplContextMenuIDs = [
    (
        "display",
        MENUVERB_DISPLAY,
        0,
    ),
]


class ContextMenu:
    _reg_progid_ = "Python.ShellFolderSample.ContextMenu"
    _reg_desc_ = "Python FolderView Context Menu"
    _reg_clsid_ = "{fed40039-021f-4011-87c5-6188b9979764}"
    _com_interfaces_ = [
        shell.IID_IShellExtInit,
        shell.IID_IContextMenu,
        axcontrol.IID_IObjectWithSite,
    ]
    _public_methods_ = (
        shellcon.IContextMenu_Methods
        + shellcon.IShellExtInit_Methods
        + ["GetSite", "SetSite"]
    )
    _context_menu_type_ = "PythonFolderViewSampleType"

    def __init__(self):
        self.site = None
        self.dataobj = None

    def Initialize(self, folder, dataobj, hkey):
        self.dataobj = dataobj

    def QueryContextMenu(self, hMenu, indexMenu, idCmdFirst, idCmdLast, uFlags):
        s = LoadString(IDS_DISPLAY)
        win32gui.InsertMenu(
            hMenu, indexMenu, win32con.MF_BYPOSITION, idCmdFirst + MENUVERB_DISPLAY, s
        )
        indexMenu += 1
        # other verbs could go here...

        # indicate that we added one verb.
        return 1

    def InvokeCommand(self, ci):
        mask, hwnd, verb, params, dir, nShow, hotkey, hicon = ci
        # this seems very convuluted, but its what the sample does :)
        for verb_name, verb_id, flag in folderViewImplContextMenuIDs:
            if isinstance(verb, int):
                matches = verb == verb_id
            else:
                matches = verb == verb_name
            if matches:
                break
        else:
            assert False, ci  # failed to find our ID
        if verb_id == MENUVERB_DISPLAY:
            sia = shell.SHCreateShellItemArrayFromDataObject(self.dataobj)
            DisplayItem(hwnd, sia)
        else:
            assert False, ci  # Got some verb we weren't expecting?

    def GetCommandString(self, cmd, typ):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetSite(self, site):
        self.site = site

    def GetSite(self, iid):
        return self.site


# end of sample's ContextMenu.cpp port


# start of sample's ShellFolder.cpp port
class ShellFolder:
    _com_interfaces_ = [
        shell.IID_IBrowserFrameOptions,
        pythoncom.IID_IPersist,
        shell.IID_IPersistFolder,
        shell.IID_IPersistFolder2,
        shell.IID_IShellFolder,
        shell.IID_IShellFolder2,
    ]

    _public_methods_ = (
        shellcon.IBrowserFrame_Methods
        + shellcon.IPersistFolder2_Methods
        + shellcon.IShellFolder2_Methods
    )

    _reg_progid_ = "Python.ShellFolderSample.Folder2"
    _reg_desc_ = "Python FolderView sample"
    _reg_clsid_ = "{bb8c24ad-6aaa-4cec-ac5e-c429d5f57627}"

    max_levels = 5

    def __init__(self, level=0):
        self.current_level = level
        self.pidl = None  # set when Initialize is called

    def ParseDisplayName(self, hwnd, reserved, displayName, attr):
        # print "ParseDisplayName", displayName
        raise COMException(hresult=winerror.E_NOTIMPL)

    def EnumObjects(self, hwndOwner, flags):
        if self.current_level >= self.max_levels:
            return None
        return make_item_enum(self.current_level + 1, flags)

    def BindToObject(self, pidl, bc, iid):
        tail = pidl_to_item(pidl)
        # assert tail['is_folder'], "BindToObject should only be called on folders?"
        # *sob*
        # No point creating object just to have QI fail.
        if iid not in ShellFolder._com_interfaces_:
            raise COMException(hresult=winerror.E_NOTIMPL)
        child = ShellFolder(self.current_level + 1)
        # hrmph - not sure what multiple PIDLs here mean?
        #        assert len(pidl)==1, pidl # expecting just relative child PIDL
        child.Initialize(self.pidl + pidl)
        return wrap(child, iid)

    def BindToStorage(self, pidl, bc, iid):
        return self.BindToObject(pidl, bc, iid)

    def CompareIDs(self, param, id1, id2):
        return 0  # XXX - todo - implement this!

    def CreateViewObject(self, hwnd, iid):
        if iid == shell.IID_IShellView:
            com_folder = wrap(self)
            return shell.SHCreateShellFolderView(com_folder)
        elif iid == shell.IID_ICategoryProvider:
            return wrap(ViewCategoryProvider(self))
        elif iid == shell.IID_IContextMenu:
            ws = wrap(self)
            dcm = (hwnd, None, self.pidl, ws, None)
            return shell.SHCreateDefaultContextMenu(dcm, iid)
        elif iid == shell.IID_IExplorerCommandProvider:
            return wrap(ExplorerCommandProvider())
        else:
            raise COMException(hresult=winerror.E_NOINTERFACE)

    def GetAttributesOf(self, pidls, attrFlags):
        assert len(pidls) == 1, "sample only expects 1 too!"
        assert len(pidls[0]) == 1, "expect relative pidls!"
        item = pidl_to_item(pidls[0])
        flags = 0
        if item["is_folder"]:
            flags |= shellcon.SFGAO_FOLDER
        if item["level"] < self.max_levels:
            flags |= shellcon.SFGAO_HASSUBFOLDER
        return flags

    #  Retrieves an OLE interface that can be used to carry out
    #  actions on the specified file objects or folders.
    def GetUIObjectOf(self, hwndOwner, pidls, iid, inout):
        assert len(pidls) == 1, "oops - arent expecting more than one!"
        assert len(pidls[0]) == 1, "assuming relative pidls!"
        item = pidl_to_item(pidls[0])
        if iid == shell.IID_IContextMenu:
            ws = wrap(self)
            dcm = (hwndOwner, None, self.pidl, ws, pidls)
            return shell.SHCreateDefaultContextMenu(dcm, iid)
        elif iid == shell.IID_IExtractIconW:
            dxi = shell.SHCreateDefaultExtractIcon()
            # dxi is IDefaultExtractIconInit
            if item["is_folder"]:
                dxi.SetNormalIcon("shell32.dll", 4)
            else:
                dxi.SetNormalIcon("shell32.dll", 1)
            # just return the dxi - let Python QI for IID_IExtractIconW
            return dxi

        elif iid == pythoncom.IID_IDataObject:
            return shell.SHCreateDataObject(self.pidl, pidls, None, iid)

        elif iid == shell.IID_IQueryAssociations:
            elts = []
            if item["is_folder"]:
                elts.append((shellcon.ASSOCCLASS_FOLDER, None, None))
            elts.append(
                (shellcon.ASSOCCLASS_PROGID_STR, None, ContextMenu._context_menu_type_)
            )
            return shell.AssocCreateForClasses(elts, iid)

        raise COMException(hresult=winerror.E_NOINTERFACE)

    #  Retrieves the display name for the specified file object or subfolder.
    def GetDisplayNameOf(self, pidl, flags):
        item = pidl_to_item(pidl)
        if flags & shellcon.SHGDN_FORPARSING:
            if flags & shellcon.SHGDN_INFOLDER:
                return item["name"]
            else:
                if flags & shellcon.SHGDN_FORADDRESSBAR:
                    sigdn = shellcon.SIGDN_DESKTOPABSOLUTEEDITING
                else:
                    sigdn = shellcon.SIGDN_DESKTOPABSOLUTEPARSING
                parent = shell.SHGetNameFromIDList(self.pidl, sigdn)
                return parent + "\\" + item["name"]
        else:
            return item["name"]

    def SetNameOf(self, hwndOwner, pidl, new_name, flags):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetClassID(self):
        return self._reg_clsid_

    #  IPersistFolder method
    def Initialize(self, pidl):
        self.pidl = pidl

    #  IShellFolder2 methods
    def EnumSearches(self):
        raise COMException(hresult=winerror.E_NOINTERFACE)

    #  Retrieves the default sorting and display columns.
    def GetDefaultColumn(self, dwres):
        # result is (sort, display)
        return 0, 0

    #  Retrieves the default state for a specified column.
    def GetDefaultColumnState(self, iCol):
        if iCol < 3:
            return shellcon.SHCOLSTATE_ONBYDEFAULT | shellcon.SHCOLSTATE_TYPE_STR
        raise COMException(hresult=winerror.E_INVALIDARG)

    #  Requests the GUID of the default search object for the folder.
    def GetDefaultSearchGUID(self):
        raise COMException(hresult=winerror.E_NOTIMPL)

    #  Helper function for getting the display name for a column.
    def _GetColumnDisplayName(self, pidl, pkey):
        item = pidl_to_item(pidl)
        is_folder = item["is_folder"]
        if pkey == PKEY_ItemNameDisplay:
            val = item["name"]
        elif pkey == PKEY_Sample_AreaSize and not is_folder:
            val = "%d Sq. Ft." % item["size"]
        elif pkey == PKEY_Sample_NumberOfSides and not is_folder:
            val = str(item["sides"])  # not sure why str()
        elif pkey == PKEY_Sample_DirectoryLevel:
            val = str(item["level"])
        else:
            val = ""
        return val

    #  Retrieves detailed information, identified by a
    #  property set ID (FMTID) and property ID (PID),
    #  on an item in a Shell folder.
    def GetDetailsEx(self, pidl, pkey):
        item = pidl_to_item(pidl)
        is_folder = item["is_folder"]
        if not is_folder and pkey == PKEY_PropList_PreviewDetails:
            return "prop:Sample.AreaSize;Sample.NumberOfSides;Sample.DirectoryLevel"
        return self._GetColumnDisplayName(pidl, pkey)

    #  Retrieves detailed information, identified by a
    #  column index, on an item in a Shell folder.
    def GetDetailsOf(self, pidl, iCol):
        key = self.MapColumnToSCID(iCol)
        if pidl is None:
            data = [
                (commctrl.LVCFMT_LEFT, "Name"),
                (commctrl.LVCFMT_CENTER, "Size"),
                (commctrl.LVCFMT_CENTER, "Sides"),
                (commctrl.LVCFMT_CENTER, "Level"),
            ]
            if iCol >= len(data):
                raise COMException(hresult=winerror.E_FAIL)
            fmt, val = data[iCol]
        else:
            fmt = 0  # ?
            val = self._GetColumnDisplayName(pidl, key)
        cxChar = 24
        return fmt, cxChar, val

    #  Converts a column name to the appropriate
    #  property set ID (FMTID) and property ID (PID).
    def MapColumnToSCID(self, iCol):
        data = [
            PKEY_ItemNameDisplay,
            PKEY_Sample_AreaSize,
            PKEY_Sample_NumberOfSides,
            PKEY_Sample_DirectoryLevel,
        ]
        if iCol >= len(data):
            raise COMException(hresult=winerror.E_FAIL)
        return data[iCol]

    #  IPersistFolder2 methods
    #  Retrieves the PIDLIST_ABSOLUTE for the folder object.
    def GetCurFolder(self):
        # The docs say this is OK, but I suspect its a problem in this case :)
        # assert self.pidl, "haven't been initialized?"
        return self.pidl


# end of sample's ShellFolder.cpp port


def get_schema_fname():
    me = win32api.GetFullPathName(__file__)
    sc = os.path.splitext(me)[0] + ".propdesc"
    assert os.path.isfile(sc), sc
    return sc


def DllRegisterServer():
    import winreg

    if sys.getwindowsversion()[0] < 6:
        print("This sample only works on Vista")
        sys.exit(1)

    key = winreg.CreateKey(
        winreg.HKEY_LOCAL_MACHINE,
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\"
        "Explorer\\Desktop\\Namespace\\" + ShellFolder._reg_clsid_,
    )
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ShellFolder._reg_desc_)
    # And special shell keys under our CLSID
    key = winreg.CreateKey(
        winreg.HKEY_CLASSES_ROOT, "CLSID\\" + ShellFolder._reg_clsid_ + "\\ShellFolder"
    )
    # 'Attributes' is an int stored as a binary! use struct
    attr = (
        shellcon.SFGAO_FOLDER | shellcon.SFGAO_HASSUBFOLDER | shellcon.SFGAO_BROWSABLE
    )
    import struct

    s = struct.pack("i", attr)
    winreg.SetValueEx(key, "Attributes", 0, winreg.REG_BINARY, s)
    # register the context menu handler under the FolderViewSampleType type.
    keypath = "%s\\shellex\\ContextMenuHandlers\\%s" % (
        ContextMenu._context_menu_type_,
        ContextMenu._reg_desc_,
    )
    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, keypath)
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ContextMenu._reg_clsid_)
    propsys.PSRegisterPropertySchema(get_schema_fname())
    print(ShellFolder._reg_desc_, "registration complete.")


def DllUnregisterServer():
    import winreg

    paths = [
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Desktop\\Namespace\\"
        + ShellFolder._reg_clsid_,
        "%s\\shellex\\ContextMenuHandlers\\%s"
        % (ContextMenu._context_menu_type_, ContextMenu._reg_desc_),
    ]
    for path in paths:
        try:
            winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, path)
        except WindowsError as details:
            import errno

            if details.errno != errno.ENOENT:
                print("FAILED to remove %s: %s" % (path, details))

    propsys.PSUnregisterPropertySchema(get_schema_fname())
    print(ShellFolder._reg_desc_, "unregistration complete.")


if __name__ == "__main__":
    from win32com.server import register

    register.UseCommandLine(
        ShellFolder,
        ContextMenu,
        debug=debug,
        finalize_register=DllRegisterServer,
        finalize_unregister=DllUnregisterServer,
    )

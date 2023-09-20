# python3.7
"""Contains the implementation of the text2image generator.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat, reduce

from third_party.stylegan3_official_ops import bias_act
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan3_official_ops import conv2d_gradfix
from .utils.ops import all_gather
# import torch.utils.checkpoint as checkpoint

__all__ = ['Text2ImageGenerator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256]

# pylint: disable=missing-function-docstring
# pylint: disable=arguments-renamed

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    return func(*inputs)


class Text2ImageGenerator(nn.Module):
    """Defines the generator network of a text to images.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].
    Settings for the mapping network:

    (1) z_dim: Dimension of the input latent space, Z. (default: 128)
    (2) w_dim: Dimension of the output latent space, W. (default: 1024)
    (3) repeat_w: Repeat w-code for different layers. (default: True)
    (4) normalize_z: Whether to normalize the z-code. (default: True)
    (5) mapping_layers: Number of layers of the mapping network. (default: 4)
    (6) mapping_fmaps: Number of hidden channels of the mapping network.
        (default: 1024)
    (7) mapping_use_wscale: Whether to use weight scaling for the mapping
        network. (default: True)
    (8) mapping_wscale_gain: The factor to control weight scaling for the
        mapping network. (default: 1.0)
    (9) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)

    Settings for conditional generation (class label):

    (1) use_class_label: Whether to use class label condition. (default False)
    (2) label_dim:  Dimension of the additional label for conditional
        generation. In one-hot conditioning case, it is equal to the number of
        classes. If set to 0, conditioning training will be disabled.
        (default: 0)
    (3) embedding_dim: Dimension of the embedding space, if needed.
        (default: 512)
    (4) embedding_bias: Whether to add bias to embedding learning.
        (default: True)
    (5) embedding_use_wscale: Whether to use weight scaling for embedding
        learning. (default: True)
    (6) embedding_wscale_gian: The factor to control weight scaling for
        embedding. (default: 1.0)
    (7) embedding_lr_mul: Learning rate multiplier for the embedding
        learning. (default: 1.0)
    (8) normalize_class_embed: Whether to normalize the class embedding.
        (default: True)

    Settings for conditional generation (text):

    (1) use_text_cond: Whether to use text condition. (default True)
    (2) num_layers_text_enc: Number of transformer blocks will be used in the
        trainable text encoder (default: 4).
    (3) weight_name: Weight name for the pre-trained clip model.
    (4) dropout: Dropout ratio for in the project out layer in transformer.
        (default: 0.)
    (5) text_use_wscale: Whether to use weight scaling for the trainable text
        network. (default: True)
    (6) text_wscale_gain: The factor to control weight scaling for the
        trainable text network. (default: 1.0)
    (7) text_lr_mul: Learning rate multiplier for the trainable text network.
        (default: 0.01)
    (8) use_transformer_clip: Whether to use the clip model from `transformers`
        library, otherwise, using clip from  `open_clip`. (default: True)
    (9) trainable_head: Whether to use trainable head in text network.
        (default: True)
    (10) context_dim: Dimension of the text embedding. (default: 768)
    (11) normalize_text_cond: Whether to normalize the text embedding.
        (default: True)

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image. (default: -1)
    (2) init_res: The initial resolution to start with convolution. (default: 4)
    (3) image_channels: Number of channels of the output image. (default: 3)
    (4) num_adaptive_kernels: Number of adaptive conv kernels for each
        resolution blocks. (default: {4:1, 8:1, 16:2, 32:4, 64:8})
    (5) num_block_per_res: Number of synthesis blocks per resolution.
        (default: {4:3, 8:3, 16: 3, 32: 2, 64: 2})
    (6) attn_resolutions: Attention add on which resolution.
        (default: (8, 16, 32))
    (7) attn_depth: Attention depth for each resolution. (default: {8:2,
        16:2, 32:1})
    (8) attn_ch_factor: Attention dimension multiplier. (default 1.0)
    (9) attn_gain: Residual gain on attention. (default: 0.3)
    (10) zero_out: Whether to zero the weights for the attention project out
        layer. (default: True)
    (11) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (12) fourier_feat: Whether to use a fourier feature in the first
        convolutional layer. (default: True)
    (13) demodulate: Whether to perform style demodulation. (default: True)
    (14) use_wscale: Whether to use weight scaling. (default: True)
    (15) wscale_gain: The factor to control weight scaling. (default: 1.0)
    (16) lr_mul: Learning rate multiplier for the synthesis network.
         (default: 1.0)
    (17) fmaps_base: Factor to control number of feature maps for each layer.
         (default: 32 << 9)
    (18) fmaps_max: Maximum number of feature maps in each layer.
          (default: 1600)
    (19) filter_kernel: Kernel used for filtering (e.g., downsampling).
         (default: (1, 3, 3, 1))
    (20) conv_clamp: A threshold to clamp the output of convolution layers to
         avoid overflow under FP16 training. (default: None)
    (21) num_heads: Number of head in multi-head attention. (default 8)
    (22) head_dim: Dimension of each head. (default 64)
    (23) l2_attention: Whether or not to use l2 distance to compute the
         similarity. (default True)
    (24) use_checkpoint: if True, use gradient checkpointing on this module.
        (default False) NOTE: In current GAN training, we can not use this
        function since the second derivative need the intermediate results.
    (25) eps: A small value to avoid divide overflow. (default: 1e-8)
    (26) mtm: Whether to apply modulated transformation module. (default: False)
    (27) num_experts: Number of experts for MoE. (default: {4:-1,8:-1,16:-1,32:-1,64:-1})

    Runtime settings:

    (1) w_moving_decay: Decay factor for updating `w_avg`, which is used for
        training only. Set `None` to disable. (default: None)
    (2) sync_w_avg: Synchronizing the stats of `w_avg` across replicas. If set
        as `True`, the stats will be more accurate, yet the speed maybe a little
        bit slower. (default: False)
    (3) trunc_psi: Truncation psi, set `None` to disable. (default: None)
    (4) trunc_layers: Number of layers to perform truncation. (default: None)
    (5) impl: Implementation mode of some particular ops, e.g., `filtering`,
        `bias_act`, etc. `cuda` means using the official CUDA implementation
        from StyleGAN2, while `ref` means using the native PyTorch ops.
        (default: `cuda`)
    """

    def __init__(self,
                 # Settings for mapping network.
                 z_dim=128,
                 w_dim=1024,
                 repeat_w=True,
                 normalize_z=True,
                 mapping_layers=4,
                 mapping_fmaps=1024,
                 mapping_use_wscale=True,
                 mapping_wscale_gain=1.0,
                 mapping_lr_mul=0.01,
                 # Settings for conditional generation (class label).
                 use_class_label=False,
                 label_dim=0,
                 embedding_dim=512,
                 embedding_bias=True,
                 embedding_use_wscale=True,
                 embedding_wscale_gain=1.0,
                 embedding_lr_mul=1.0,
                 normalize_class_embed=True,
                 # Settings for conditional generation (text).
                 use_text_cond=True,
                 use_w_cond=False,
                 num_layers_text_enc=4,
                 dropout=0.0,
                 text_use_wscale=True,
                 text_wscale_gain=1.0,
                 text_lr_mul=0.01,
                 clip_out_dim=768,
                 context_dim=768,
                 normalize_text_cond=True,
                 # Settings for synthesis network.
                 resolution=-1,
                 init_res=4,
                 image_channels=3,
                 num_adaptive_kernels={},
                 num_block_per_res={},
                 attn_resolutions=(8, 16, 32),
                 attn_depth={},
                 attn_ch_factor=1.0,
                 mask_self=False,
                 residual_gain=1.0,
                 attn_gain=1.0,
                 text_head_gain=1.0,
                 zero_out=True,
                 final_tanh=False,
                 fourier_feat=True,
                 demodulate=True,
                 use_wscale=True,
                 wscale_gain=1.0,
                 lr_mul=1.0,
                 fmaps_base=32 << 9,
                 fmaps_max=1600,
                 filter_kernel=(1, 3, 3, 1),
                 conv_clamp=None,
                 num_heads=8,
                 head_dim=None,
                 l2_attention=True,
                 tie=True,
                 scale_in=True,
                 include_ff=True,
                 glu=False,
                 use_checkpoint=False,
                 checkpoint_res=None,
                 eps=1e-8,
                 mtm=False,
                 num_experts={},
                 ms_training_res=['4', '8', '16', '32', '64'],
                 skip_connection=False):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `architecture`
                is not supported.
        """
        super().__init__()
        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')

        assert ((use_text_cond and not use_class_label) or
                (not use_text_cond and use_class_label)), ('Condition '
                'information can not from both the text and class label.')

        # assert ((label_dim > 0 and not context_dim) or
        #         (label_dim <= 0 and context_dim)), ('Class label '
        #         'and context dimension should be just give one.')

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.repeat_w = repeat_w
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_use_wscale = mapping_use_wscale
        self.mapping_wscale_gain = mapping_wscale_gain
        self.mapping_lr_mul = mapping_lr_mul

        self.use_class_label=use_class_label
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_wscale_gain = embedding_wscale_gain
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_class_embed = normalize_class_embed

        self.use_text_cond = use_text_cond
        self.use_w_cond = use_w_cond
        self.num_layers_text_enc = num_layers_text_enc
        self.dropout = dropout
        self.text_use_wscale = text_use_wscale
        self.text_wscale_gain = text_wscale_gain
        self.text_lr_mul = text_lr_mul
        self.context_dim = context_dim
        self.normalize_text_cond = normalize_text_cond

        self.resolution = resolution
        self.init_res = init_res
        self.image_channels = image_channels
        self.num_adaptive_kernel = num_adaptive_kernels
        self.num_block_per_res = num_block_per_res
        self.attn_depth = attn_depth
        if attn_resolutions is None:
            self.attn_resolutions = list(attn_depth.keys())
        else:
            self.attn_resolutions = attn_resolutions

        self.mask_self = mask_self
        self.residual_gain = residual_gain
        self.attn_ch_factor = attn_ch_factor
        self.attn_gain = attn_gain
        self.zero_out = zero_out
        self.final_tanh = final_tanh
        self.fourier_feat = fourier_feat
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.conv_clamp = conv_clamp
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_checkpoint = use_checkpoint
        if checkpoint_res is None:
            self.checkpoint_res = self.attn_resolutions
        else:
            self.checkpoint_res = checkpoint_res
        self.l2_attention = l2_attention
        self.tie = tie
        self.scale_in = scale_in
        self.include_ff = include_ff
        self.glu = glu
        self.eps = eps
        self.mtm = mtm
        self.num_experts = num_experts
        self.ms_training_res = ms_training_res
        self.skip_connection = skip_connection

        # Dimension of latent space, which is convenient for sampling.
        self.latent_dim = (z_dim,)

        # Number of synthesis (convolutional) layers.
        num_block_per_res_keys = num_block_per_res.keys()
        self.num_layers = 0
        self.layer_id_to_res = {}
        for res_ in num_block_per_res_keys:
            if res_ in self.attn_resolutions:
                self.num_layers += num_block_per_res[res_] * 2
            else:
                self.num_layers += num_block_per_res[res_]
            self.layer_id_to_res[self.num_layers - 1] = res_
        print(f'Number of layers: {self.num_layers}')
        print(f'layer_id_to_res: {self.layer_id_to_res}')

        if use_text_cond:
            self.text_head = TextHead(in_dim=clip_out_dim,
                                      context_dim=context_dim,
                                      num_heads=num_heads,
                                      num_layers=num_layers_text_enc,
                                      dropout=dropout,
                                      use_wscale=text_use_wscale,
                                      wscale_gain=text_wscale_gain,
                                      lr_mul=text_lr_mul,
                                      use_checkpoint=use_checkpoint,
                                      mlp_ratio=4,
                                      l2_attention=l2_attention,
                                      scale_in=scale_in,
                                      mask_self=mask_self,
                                      attn_gain=text_head_gain)

        self.mapping = MappingNetwork(
                           input_dim=z_dim,
                           output_dim=w_dim,
                           num_outputs=self.num_layers,
                           repeat_output=repeat_w,
                           normalize_z=normalize_z,
                           num_layers=mapping_layers,
                           hidden_dim=mapping_fmaps,
                           use_wscale=mapping_use_wscale,
                           wscale_gain=mapping_wscale_gain,
                           lr_mul=mapping_lr_mul,
                           use_class_label=use_class_label,
                           label_dim=label_dim,
                           embedding_dim=embedding_dim,
                           embedding_bias=embedding_bias,
                           embedding_use_wscale=embedding_use_wscale,
                           embedding_wscale_gain=embedding_wscale_gain,
                           embedding_lr_mul=embedding_lr_mul,
                           normalize_class_embed=normalize_class_embed,
                           use_text_cond=use_text_cond,
                           context_dim=context_dim,
                           normalize_text_cond=normalize_text_cond,
                           eps=eps)

        # This is used for truncation trick.
        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(self.num_layers * w_dim))

        self.synthesis = SynthesisNetwork(
                            resolution=resolution,
                            init_res=init_res,
                            w_dim=w_dim,
                            num_adaptive_kernels=num_adaptive_kernels,
                            num_block_per_res=num_block_per_res,
                            image_channels=image_channels,
                            attn_resolutions=self.attn_resolutions,
                            attn_depth=attn_depth,
                            attn_ch_factor=attn_ch_factor,
                            layer_id_to_res=self.layer_id_to_res,
                            fourier_feat=fourier_feat,
                            demodulate=demodulate,
                            use_wscale=use_wscale,
                            wscale_gain=wscale_gain,
                            lr_mul=lr_mul,
                            fmaps_base=fmaps_base,
                            fmaps_max=fmaps_max,
                            filter_kernel=filter_kernel,
                            conv_clamp=conv_clamp,
                            use_text_cond=use_text_cond,
                            use_w_cond=use_w_cond,
                            context_dim=context_dim,
                            num_heads=num_heads,
                            head_dim=head_dim,
                            use_checkpoint=use_checkpoint,
                            checkpoint_res=self.checkpoint_res,
                            l2_attention=l2_attention,
                            tie=tie,
                            scale_in=scale_in,
                            include_ff=include_ff,
                            glu=glu,
                            mask_self=mask_self,
                            residual_gain=residual_gain,
                            attn_gain=attn_gain,
                            final_tanh=final_tanh,
                            zero_out=zero_out,
                            dropout=dropout,
                            eps=eps,
                            mtm=mtm,
                            num_experts=num_experts,
                            ms_training_res=ms_training_res,
                            skip_connection=skip_connection)

    def forward(self,
                z,
                label=None,
                context=None,
                eot_ind=None,
                w_moving_decay=None,
                sync_w_avg=False,
                trunc_psi=None,
                trunc_layers=None,
                impl='cuda'):

        if self.use_text_cond:
            global_text, local_text = self.text_head(context, eot_ind=eot_ind)
            mapping_results = self.mapping(z,
                                           label=None,
                                           context=global_text)
        else:
            local_text = None
            mapping_results = self.mapping(z,
                                           label=label,
                                           context=None)

        w = mapping_results['w']
        if self.training and w_moving_decay is not None:
            if sync_w_avg:
                batch_w_avg = all_gather(w.detach()).mean(dim=0)
            else:
                batch_w_avg = w.detach().mean(dim=0)
            self.w_avg.copy_(batch_w_avg.lerp(self.w_avg, w_moving_decay))

        wp = mapping_results.pop('wp')
        if not self.training:
            trunc_psi = 1.0 if trunc_psi is None else trunc_psi
            trunc_layers = 0 if trunc_layers is None else trunc_layers
            if trunc_psi < 1.0 and trunc_layers > 0:
                w_avg = self.w_avg.reshape(1, -1, self.w_dim)[:, :trunc_layers]
                wp[:, :trunc_layers] = w_avg.lerp(
                    wp[:, :trunc_layers], trunc_psi)

        synthesis_results = self.synthesis(wp, context=local_text)

        return {**mapping_results, **synthesis_results}


class TextHead(nn.Module):
    """Define the trainable text head.

    Trainable head is build at the top of a text encoder (i.g., CLIP text
    encoder). Namely, the text description is firstly send to a pre-trained text
    encoder to extract the features. Then this trainable head is used as an
    adaptor between the pre-trained text encoder and learned generator. Besides,
    the features of the trainable head is categorized into two groups. One is
    global features and the other is the local features, The last token (`end of
    text` or `EOT`) of embedding serves as the global feature, while the
    remaining tokens are considered as the local features. Specifically, the
    output of the trainable head has the shape of [B, C, dim], where B is the
    batch size, C is the max token length (usually C equals to 77), and dim is
    the embedding dimension. If an input description has token length `len` and
    features `feats`. Then, the global feature is feats[:, len, :] and the
    local feature is torch.cat([feats[1, :len-1, :], feats[1, len:, :]], 1).
    """

    def __init__(self,
                 in_dim,
                 context_dim,
                 num_heads,
                 num_layers,
                 dropout,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 use_checkpoint,
                 mlp_ratio,
                 l2_attention,
                 scale_in,
                 mask_self,
                 attn_gain):
        super().__init__()
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.use_checkpoint = use_checkpoint
        self.mlp_ratio = mlp_ratio
        self.l2_attention = l2_attention
        self.scale_in = scale_in
        self.mask_self = mask_self
        self.align_ch = in_dim != context_dim
        self.attn_gain = attn_gain

        if self.align_ch:
            self.align_dim = DenseLayer(in_channels=in_dim,
                                        out_channels=context_dim,
                                        add_bias=True,
                                        init_bias=1.0,
                                        use_wscale=use_wscale,
                                        wscale_gain=wscale_gain,
                                        lr_mul=lr_mul,
                                        activation_type='linear')

        self.transformer = TextTransformer(width=context_dim,
                                           num_heads=num_heads,
                                           num_layers=num_layers,
                                           dropout=dropout,
                                           add_bias=True,
                                           init_bias=0.0,
                                           use_wscale=use_wscale,
                                           wscale_gain=wscale_gain,
                                           lr_mul=lr_mul,
                                           use_checkpoint=use_checkpoint,
                                           mlp_ratio=4,
                                           l2_attention=l2_attention,
                                           scale_in=scale_in,
                                           mask_self=mask_self,
                                           attn_gain=attn_gain)

    def forward(self, text_feats, eot_ind):
        if self.align_ch:
            text_feats = self.align_dim(text_feats)
        bs, emb_length, dim = text_feats.shape
        device = text_feats.device

        text_feats = text_feats.permute(1, 0, 2)
        text_feats = self.transformer(text_feats)
        text_feats = text_feats.permute(1, 0, 2)
        length_ind = eot_ind[:, None, None].repeat(1, 1, dim).to(device)
        mask = torch.ones_like(text_feats, device=device)
        zero_fill = torch.zeros_like(text_feats, device=device)
        mask = torch.scatter(mask, dim=1, index=length_ind, src=zero_fill)
        local_feats = text_feats[mask.bool()].view(bs, emb_length-1, dim)
        global_feats = torch.gather(text_feats, dim=1, index=length_ind)
        # global_feats = text_feats[torch.arange(text_feats.shape[0]), eot_ind]
        # global_feats = global_feats.unsqueeze(1)

        return global_feats[:, 0], local_feats


class MappingNetwork(nn.Module):
    """Implements the latent space mapping network.

    Basically, this network executes several dense layers in sequence, and also
    support add text embedding if needed.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_outputs,
                 repeat_output,
                 normalize_z,
                 num_layers,
                 hidden_dim,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 use_class_label,
                 label_dim,
                 embedding_dim,
                 embedding_bias,
                 embedding_use_wscale,
                 embedding_wscale_gain,
                 embedding_lr_mul,
                 normalize_class_embed,
                 use_text_cond,
                 context_dim,
                 normalize_text_cond,
                 eps):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_outputs = num_outputs
        self.repeat_output = repeat_output
        self.normalize_z = normalize_z
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul

        self.use_class_label = use_class_label
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_wscale_gain = embedding_wscale_gain
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_class_embed = normalize_class_embed

        self.use_text_cond = use_text_cond
        self.context_dim = context_dim
        self.normalize_text_cond = normalize_text_cond

        self.eps = eps

        self.norm = PixelNormLayer(dim=1, eps=eps)

        if self.use_text_cond:
            input_dim = input_dim + context_dim

        if self.use_class_label:
            input_dim = input_dim + embedding_dim
            self.embedding = DenseLayer(in_channels=label_dim,
                                        out_channels=embedding_dim,
                                        add_bias=embedding_bias,
                                        init_bias=0.0,
                                        use_wscale=embedding_use_wscale,
                                        wscale_gain=embedding_wscale_gain,
                                        lr_mul=embedding_lr_mul,
                                        activation_type='linear')

        if num_outputs is not None and not repeat_output:
            output_dim = output_dim * num_outputs

        for i in range(num_layers):
            in_channels = (input_dim if i == 0 else hidden_dim)
            out_channels = (output_dim if i == (num_layers - 1) else hidden_dim)
            self.add_module(f'dense{i}',
                            DenseLayer(in_channels=in_channels,
                                       out_channels=out_channels,
                                       add_bias=True,
                                       init_bias=0.0,
                                       use_wscale=use_wscale,
                                       wscale_gain=wscale_gain,
                                       lr_mul=lr_mul,
                                       activation_type='lrelu'))

    def forward(self, z, label=None, context=None):
        if z.ndim != 2 or z.shape[1] != self.input_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, input_dim], where '
                             f'`input_dim` equals to {self.input_dim}!\n'
                             f'But `{z.shape}` is received!')

        if self.use_text_cond and context is None:
            raise ValueError(f'Text condition is needed! but got {context}')
        if self.use_class_label and label is None:
            raise ValueError(f'Class condition is needed! but got {label}')

        if self.normalize_z:
            z = self.norm(z)

        if self.use_text_cond:
            if (context.ndim != 2 or
                context.shape != (z.shape[0], self.context_dim)):
                raise ValueError(f'Input text condition should be with shape '
                                 f'[batch_size, context_dim], where '
                                 f'`batch_size` equals to that of '
                                 f'latent codes ({z.shape[0]}) and '
                                 f'`context_dim` equals to {self.context_dim}! '
                                 f'But `{context.shape}` is received!')
            if self.normalize_text_cond:
                context = self.norm(context)
            w = torch.cat((z, context), dim=1)
        elif self.use_class_label:
            if label.ndim != 2 or label.shape != (z.shape[0], self.label_dim):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_dim], where '
                                 f'`batch_size` equals to that of '
                                 f'latent codes ({z.shape[0]}) and '
                                 f'`label_dim` equals to {self.label_dim}!'
                                 f'But `{label.shape}` is received!')
            label = label.to(dtype=torch.float32)
            embedding = self.embedding(label)
            if self.normalize_class_embed:
                embedding = self.norm(embedding)
            w = torch.cat((z, embedding), dim=1)
        else:
            w = z

        for i in range(self.num_layers):
            w = getattr(self, f'dense{i}')(w)

        wp = None
        if self.num_outputs is not None:
            if self.repeat_output:
                wp = w.unsqueeze(1).repeat((1, self.num_outputs, 1))
            else:
                wp = w.reshape(-1, self.num_outputs, self.output_dim)

        results = {
            'z': z,
            'global_text': context,
            'label': label,
            'w': w,
            'wp': wp,
        }
        return results


class SynthesisNetwork(nn.Module):
    """Implements the image synthesis network.

    Basically, this network executes several convolutional layers in sequence.
    """
    def __init__(self,
                 resolution,
                 init_res,
                 w_dim,
                 num_adaptive_kernels,
                 num_block_per_res,
                 image_channels,
                 attn_resolutions,
                 attn_depth,
                 attn_ch_factor,
                 layer_id_to_res,
                 fourier_feat,
                 demodulate,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 fmaps_base,
                 fmaps_max,
                 filter_kernel,
                 conv_clamp,
                 use_text_cond,
                 use_w_cond,
                 context_dim,
                 num_heads,
                 head_dim,
                 use_checkpoint,
                 checkpoint_res,
                 l2_attention,
                 tie,
                 scale_in,
                 include_ff,
                 glu,
                 mask_self,
                 residual_gain,
                 attn_gain,
                 final_tanh,
                 zero_out,
                 dropout,
                 eps,
                 mtm,
                 num_experts,
                 ms_training_res,
                 skip_connection):
        super().__init__()

        self.init_res = init_res
        self.init_res_log2 = int(np.log2(init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(resolution))
        self.w_dim = w_dim
        self.num_adaptive_kernels = num_adaptive_kernels
        self.num_block_per_res = num_block_per_res
        self.attn_ch_factor = attn_ch_factor
        self.image_channels = image_channels
        self.attn_resolutions = attn_resolutions
        self.attn_depth = attn_depth
        self.fourier_feat = fourier_feat
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.final_tanh = final_tanh
        self.conv_clamp = conv_clamp
        self.eps = eps
        self.mtm = mtm
        self.num_experts = num_experts
        self.ms_training_res = ms_training_res
        self.skip_connection = skip_connection

        # hyper-parameter on the attention.
        self.use_text_cond = use_text_cond
        self.use_w_cond = use_w_cond
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_checkpoint = use_checkpoint
        self.checkpoint_res = checkpoint_res
        self.l2_attention = l2_attention
        self.tie = tie
        self.scale_in = scale_in
        self.include_ff = include_ff
        self.glu = glu
        self.attn_gain = attn_gain
        self.zero_out = zero_out
        self.dropout = dropout

        self.res_level = []
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            self.res_level.append(2**res_log2)

        assert len(num_adaptive_kernels) == len(self.res_level), (f'For each '
                f'resolution level, must assign a corresponding number, but '
                f'got resolution: {self.res_level}, and number of kernel list: '
                f'{self.num_kernels}')
        assert len(self.res_level) >= len(attn_resolutions), (f'Attention '
                f'resolution list must less than or equal to the resolution '
                f'level, but get resolution: {self.res_level}, and attention '
                f'resolution: {self.attn_resolutions}')

        self.layer_id_to_res = layer_id_to_res

        self.net = nn.ModuleList()
        for res in self.res_level:
            in_channels = self.get_nf(res // 2)
            out_channels = self.get_nf(res)
            num_blocks = num_block_per_res[str(res)]

            # Layer with initial resolution, i.e., 4
            if res == self.init_res:
                for i in range(num_blocks):
                    if i == 0:  # First block at resolution 4x4.
                        # First conv layer
                        if fourier_feat:
                            layer0 = FourierInputLayer(w_dim=w_dim,
                                                       size=res,
                                                       channels=in_channels,
                                                       sampling_rate=res,
                                                       bandwidth=1)
                        else:
                            layer0 = ConstInputLayer(init_res=res,
                                                     channels=in_channels)

                        # Second conv layer
                        layer1 = AdaptiveModulateConvLayer(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    modulate_dim=w_dim,
                                    kernel_size=3,
                                    num_kernels=num_adaptive_kernels[str(res)],
                                    add_bias=True,
                                    scale_factor=1,
                                    filter_kernel=None,
                                    demodulate=demodulate,
                                    use_wscale=use_wscale,
                                    wscale_gain=wscale_gain,
                                    lr_mul=lr_mul,
                                    activation_type='lrelu',
                                    conv_clamp=conv_clamp,
                                    eps=eps,
                                    mtm=mtm)
                        self.net.append(ResolutionSequential(*[layer0, layer1]))
                    else: # The rest blocks at resolution 4x4.
                        layer = ResidualConvBlock(
                                    in_channels=out_channels,
                                    out_channels=out_channels,
                                    modulate_dim=w_dim,
                                    num_kernels=num_adaptive_kernels[str(res)],
                                    upsample=False,
                                    residual_gain=residual_gain,
                                    filter_kernel=filter_kernel,
                                    demodulate=demodulate,
                                    use_wscale=use_wscale,
                                    wscale_gain=wscale_gain,
                                    lr_mul=lr_mul,
                                    conv_clamp=conv_clamp,
                                    eps=eps,
                                    mtm=mtm)
                        self.net.append(ResolutionSequential(layer))

                    if str(res) in attn_resolutions:
                        layer = ResidualAttnBlock(
                                            depth=attn_depth[str(res)],
                                            in_channels=out_channels,
                                            res=res,
                                            context_dim=context_dim,
                                            attn_ch_factor=attn_ch_factor,
                                            num_heads=num_heads,
                                            head_dim=head_dim,
                                            w_dim=w_dim,
                                            group_channels=32,
                                            demodulate=demodulate,
                                            use_text_cond=use_text_cond,
                                            use_w_cond=use_w_cond,
                                            use_wscale=use_wscale,
                                            wscale_gain=wscale_gain,
                                            lr_mul=lr_mul,
                                            conv_clamp=conv_clamp,
                                            use_checkpoint=use_checkpoint,
                                            checkpoint_res=checkpoint_res,
                                            l2_attention=l2_attention,
                                            tie=tie,
                                            scale_in=scale_in,
                                            include_ff=include_ff,
                                            mask_self=mask_self,
                                            attn_gain=attn_gain,
                                            dropout=dropout,
                                            mlp_ratio=4,
                                            add_bias=True,
                                            init_bias=0.0,
                                            glu=glu,
                                            zero_out=zero_out,
                                            eps=eps,
                                            num_experts=num_experts[str(res)])
                        self.net.append(ResolutionSequential(layer))

            else: # Layers with rest resolution i.e., res >= 8
                for i in range(num_blocks):
                    if i == 0:
                        upsample = True
                    else:
                        upsample = False

                    layer = ResidualConvBlock(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                modulate_dim=w_dim,
                                num_kernels=num_adaptive_kernels[str(res)],
                                upsample=upsample,
                                residual_gain=residual_gain,
                                filter_kernel=filter_kernel,
                                demodulate=demodulate,
                                use_wscale=use_wscale,
                                wscale_gain=wscale_gain,
                                lr_mul=lr_mul,
                                conv_clamp=conv_clamp,
                                eps=eps,
                                mtm=mtm and res < 32)
                    self.net.append(ResolutionSequential(layer))

                    if str(res) in attn_resolutions:
                        layer = ResidualAttnBlock(
                                          depth=attn_depth[str(res)],
                                          in_channels=out_channels,
                                          res=res,
                                          context_dim=context_dim,
                                          attn_ch_factor=attn_ch_factor,
                                          num_heads=num_heads,
                                          head_dim=head_dim,
                                          w_dim=w_dim,
                                          group_channels=32,
                                          demodulate=demodulate,
                                          use_text_cond=use_text_cond,
                                          use_w_cond=use_w_cond,
                                          use_wscale=use_wscale,
                                          wscale_gain=wscale_gain,
                                          lr_mul=lr_mul,
                                          conv_clamp=conv_clamp,
                                          use_checkpoint=use_checkpoint,
                                          checkpoint_res=checkpoint_res,
                                          l2_attention=l2_attention,
                                          tie=tie,
                                          scale_in=scale_in,
                                          include_ff=include_ff,
                                          mask_self=mask_self,
                                          attn_gain=attn_gain,
                                          dropout=dropout,
                                          mlp_ratio=4,
                                          add_bias=True,
                                          init_bias=0.0,
                                          glu=glu,
                                          zero_out=zero_out,
                                          eps=eps,
                                          num_experts=num_experts[str(res)])
                        self.net.append(ResolutionSequential(layer))
                    in_channels = out_channels
            if use_text_cond: # Multi-scale output for text input.
                layer_name = f'output_res_{res}x{res}'
                self.add_module(
                    layer_name,
                    AdaptiveModulateConvLayer(
                                    in_channels=out_channels,
                                    out_channels=image_channels,
                                    modulate_dim=w_dim,
                                    kernel_size=1,
                                    num_kernels=num_adaptive_kernels[str(res)],
                                    add_bias=True,
                                    scale_factor=1,
                                    filter_kernel=None,
                                    demodulate=False,
                                    use_wscale=use_wscale,
                                    wscale_gain=wscale_gain,
                                    lr_mul=lr_mul,
                                    activation_type='linear',
                                    conv_clamp=conv_clamp,
                                    eps=eps))
            else: # Single output for the class label input.
                if res == self.resolution:
                    layer_name = f'output_res_{res}x{res}'
                    self.add_module(
                        layer_name,
                        AdaptiveModulateConvLayer(
                                    in_channels=out_channels,
                                    out_channels=image_channels,
                                    modulate_dim=w_dim,
                                    kernel_size=1,
                                    num_kernels=num_adaptive_kernels[str(res)],
                                    add_bias=True,
                                    scale_factor=1,
                                    filter_kernel=None,
                                    demodulate=False,
                                    use_wscale=use_wscale,
                                    wscale_gain=wscale_gain,
                                    lr_mul=lr_mul,
                                    activation_type='linear',
                                    conv_clamp=conv_clamp,
                                    eps=eps))

        # Used for upsampling output images for each resolution block for sum.
        if self.skip_connection:
            self.register_buffer(
                'filter', upfirdn2d.setup_filter(filter_kernel))


    def get_nf(self, res):
        """Gets number of feature maps according to the given resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)


    def forward(self, wp, context):
        results = {'wp': wp,
                   'local_context': context}
        h = None
        img = None
        moe_losses = 0.0
        for idx, module in enumerate(self.net):
            h, moe_loss = module(h, wp[:, idx], context)
            moe_losses += moe_loss
            if idx in self.layer_id_to_res:
                res = self.layer_id_to_res[idx]
                if res in self.ms_training_res:
                    if self.use_text_cond:
                        out_layer = getattr(self, f'output_res_{res}x{res}')
                        y, _ = out_layer(h, wp[:, idx])
                        if self.final_tanh:
                            y = torch.tanh(y)
                        if self.skip_connection:
                            y = y.to(torch.float32)
                            if res == str(self.init_res):
                                img = y
                            else:
                                img = y + upfirdn2d.upsample2d(img, self.filter)
                        else:
                            img = y
                        results[f'img_res_{res}x{res}'] = img
                    else:
                        if int(res) == self.resolution:
                            out_layer = getattr(self, f'output_res_{res}x{res}')
                            img, _ = out_layer(h, wp[:, idx])
                            if self.final_tanh:
                                img = torch.tanh(img)
                            results[f'img_res_{res}x{res}'] = img
                    if int(res) == max([int(each) for each in self.ms_training_res]):
                        break
        results['image'] = img
        results['moe_loss'] = moe_losses
        return results


class ResolutionSequential(nn.Sequential):
    """A sequential module that passes modulate code `w` and `text` as
       the condition.
    """

    def forward(self, x, w, context=None):
        moe_loss = 0.0
        for layer in self:
            if isinstance(layer, (FourierInputLayer, ConstInputLayer)):
                x = layer(w)
            elif isinstance(layer, (ResidualConvBlock,
                                    AdaptiveModulateConvLayer)):
                x, _ = layer(x, w)
            elif isinstance(layer, ResidualAttnBlock):
                x, _, moe_loss = layer(x, w, context)
            else:
                x = layer(x)
        return x, moe_loss


class ResidualConvBlock(nn.Module):
    """Residual block that that consists of two convolutional layers. If the

    dimension of the input and output features changed, an extra conv layer will
    be added.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 modulate_dim,
                 num_kernels,
                 upsample,
                 residual_gain,
                 filter_kernel,
                 demodulate,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 conv_clamp,
                 eps,
                 mtm):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of projection in.
            out_channels: Number of channels of projection out.
            w_dim: Dimension of W space for style modulation.
            num_kernels: Number of adaptive kernels.
            upsample: Whether or not to do the upsampling.
            residual_gain: residual gain for the residual connection.
            filter_kernel: Kernel used for filtering.
            demodulate: Whether to perform style demodulation.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            conv_clamp: A threshold to clamp the output of convolution layers to
                avoid overflow under FP16 training.
            eps: A small value to avoid divide overflow.
            mtm: Whether to apply modulated tranformation module
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modulate_dim = modulate_dim
        self.num_kernels = num_kernels
        self.upsample = upsample
        self.residual_gain = residual_gain
        self.filter_kernel = filter_kernel
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.conv_clamp = conv_clamp
        self.eps = eps
        self.mtm = mtm

        if upsample:
            scale_factor = 2
        else:
            scale_factor = 1
            filter_kernel = None

        self.conv1 = AdaptiveModulateConvLayer(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    modulate_dim=modulate_dim,
                                    kernel_size=3,
                                    num_kernels=num_kernels,
                                    add_bias=True,
                                    scale_factor=scale_factor,
                                    filter_kernel=filter_kernel,
                                    demodulate=demodulate,
                                    use_wscale=use_wscale,
                                    wscale_gain=wscale_gain,
                                    lr_mul=lr_mul,
                                    activation_type='lrelu',
                                    conv_clamp=conv_clamp,
                                    eps=eps)

        self.conv2 = AdaptiveModulateConvLayer(
                                    in_channels=out_channels,
                                    out_channels=out_channels,
                                    modulate_dim=modulate_dim,
                                    kernel_size=3,
                                    num_kernels=num_kernels,
                                    add_bias=True,
                                    scale_factor=1,
                                    filter_kernel=None,
                                    demodulate=demodulate,
                                    use_wscale=use_wscale,
                                    wscale_gain=wscale_gain,
                                    lr_mul=lr_mul,
                                    activation_type='lrelu',
                                    conv_clamp=conv_clamp,
                                    eps=eps,
                                    mtm=mtm) # using mtm

        if scale_factor == 1 and in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = ConvLayer(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1,
                                             add_bias=False,
                                             scale_factor=scale_factor,
                                             filter_kernel=filter_kernel,
                                             use_wscale=use_wscale,
                                             wscale_gain=wscale_gain,
                                             lr_mul=lr_mul,
                                             activation_type='linear',
                                             conv_clamp=None)

    def forward(self, x, modulate_code):
        h, _ = self.conv1(x, modulate_code)
        h, style = self.conv2(h, modulate_code)
        residual = self.skip_connection(x)
        return (h + residual) * self.residual_gain, style


class ConstInputLayer(nn.Module):
    """Implements the input layer to start convolution with.

    Basically, this block starts from a const input, which is with shape
    `(channels, init_res, init_res)`.
    """

    def __init__(self, init_res, channels):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channels, init_res, init_res))

    def forward(self, w):
        x = self.const.repeat(w.shape[0], 1, 1, 1)
        return x


class FourierInputLayer(nn.Module):
    """Implements the input layer with Fourier features."""
    def __init__(self, w_dim, size, channels, sampling_rate, bandwidth):
        super().__init__()

        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = DenseLayer(w_dim, 4, init_weight=0, init_bias=[1,0,0,0], add_bias=True, use_wscale=True, wscale_gain=1.0, lr_mul=1.0, activation_type='linear')

        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)


    def extra_repr(self):
        return (f'w_dim={self.w_dim}, '
                f'channels={self.channels}, '
                f'sampling_rate={self.sampling_rate}, '
                f'bandwidth={self.bandwidth}, '
                f'size={self.size}, ')

    def forward(self, w):

        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        #amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        #x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        return x.contiguous()


class ResidualAttnBlock(nn.Module):
    """Implements the attention used in the generator.

    Attention is composed by several `AttentionBlock`
    """
    def __init__(self,
                 depth,
                 in_channels,
                 res,
                 context_dim,
                 attn_ch_factor,
                 num_heads,
                 head_dim,
                 w_dim,
                 group_channels,
                 demodulate,
                 use_text_cond,
                 use_w_cond,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 conv_clamp,
                 use_checkpoint,
                 checkpoint_res,
                 l2_attention,
                 tie,
                 scale_in,
                 include_ff,
                 mask_self,
                 attn_gain,
                 dropout,
                 mlp_ratio,
                 add_bias,
                 init_bias,
                 glu,
                 zero_out,
                 eps,
                 num_experts):
        """Initializes with layer settings.

        Args:
            depth: Number of attention block used in this layer.
            in_channels: Number of channels of projection out.
            res:
            context_dim: Number of channels of the context tensor.
            attn_ch_factor: Attention dimension multiplier
            num_heads: Number of head of multi-head attention.
            head_dim: Dimension of each attention head.
            w_dim: Dimension of W space for style modulation.
            group_channel: Group number of group norm.
            demodulate: Whether to perform style demodulation.
            use_text_cond: Whether to involve the text condition.
            use_w_cond: Whether to involve the wp as the condition.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            conv_clamp: A threshold to clamp the output of convolution layers to
                avoid overflow under FP16 training.
            use_checkpoint: if True, use gradient checkpointing on this module.
            checkpoint_res:
            l2_attention: Whether or not to use l2 distance to compute the
                similarity.
            tie:
            scale_in:
            include_ff:
            mask_self:
            attn_gain: residual gain for the self attention.
            dropout: Dropout ratio.
            mlp_ratio:
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            glu: use glu/geglu as activation, true: glu, false: geglu.
            zero_out: Whether or not to init zero weight at the out layer.
            eps: A small value to avoid divide overflow.
            num_experts: Number of experts for MoE
        """
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.res = res
        self.context_dim = context_dim
        self.attn_ch_factor = attn_ch_factor
        self.num_heads = num_heads
        if head_dim is None:
            self.head_dim = in_channels // num_heads
            assert self.head_dim * num_heads == in_channels
        else:
            self.head_dim = head_dim
        self.w_dim = w_dim
        self.group_channels = group_channels
        self.demodulate = demodulate
        self.use_text_cond = use_text_cond
        self.use_w_cond = use_w_cond
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.conv_clamp = conv_clamp
        self.use_checkpoint = use_checkpoint
        self.checkpoint_res = checkpoint_res
        self.l2_attention = l2_attention
        self.tie = tie
        self.scale_in = scale_in
        self.include_ff = include_ff
        self.attn_gain = attn_gain
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.glu = glu
        self.zero_out = zero_out
        self.eps = eps
        self.num_experts = num_experts

        inner_dim = int(num_heads * self.head_dim)

        self.norm = GroupNorm(group_channels, in_channels)

        self.proj_in = ConvLayer(in_channels=in_channels,
                                 out_channels=inner_dim,
                                 kernel_size=1,
                                 add_bias=False,
                                 scale_factor=1,
                                 filter_kernel=None,
                                 use_wscale=use_wscale,
                                 wscale_gain=wscale_gain,
                                 lr_mul=lr_mul,
                                 activation_type='linear',
                                 conv_clamp=conv_clamp)

        self.attn_blocks = nn.ModuleList(
            [AttentionBlock(in_channels=inner_dim,
                            res=res,
                            context_dim=context_dim,
                            num_heads=num_heads,
                            head_dim=self.head_dim,
                            w_dim=w_dim,
                            demodulate=demodulate,
                            use_text_cond=use_text_cond,
                            use_w_cond=use_w_cond,
                            attn_gain=attn_gain,
                            use_wscale=use_wscale,
                            wscale_gain=wscale_gain,
                            lr_mul=lr_mul,
                            conv_clamp=conv_clamp,
                            use_checkpoint=use_checkpoint,
                            checkpoint_res=checkpoint_res,
                            l2_attention=l2_attention,
                            tie=tie,
                            scale_in=scale_in,
                            include_ff=include_ff,
                            mask_self=mask_self,
                            dropout=dropout,
                            mlp_ratio=mlp_ratio,
                            add_bias=add_bias,
                            init_bias=init_bias,
                            glu=glu,
                            eps=eps,
                            num_experts=num_experts)
                for _ in range(depth)])

        self.proj_out = ConvLayer(in_channels=inner_dim,
                                  out_channels=in_channels,
                                  kernel_size=1,
                                  add_bias=False,
                                  scale_factor=1,
                                  filter_kernel=None,
                                  use_wscale=use_wscale,
                                  wscale_gain=wscale_gain,
                                  lr_mul=lr_mul,
                                  activation_type='linear',
                                  conv_clamp=conv_clamp)
        if zero_out:
            self.proj_out = zero_module(self.proj_out)

    def extra_repr(self):
        return (f'depth={self.depth}, '
                f'in_channels={self.in_channels}, '
                f'num_heads={self.num_heads}, '
                f'context_dim={self.context_dim}, '
                f'head_dim={self.head_dim}, '
                f'group_channels={self.group_channels}, '
                f'w_dim={self.w_dim}')

    def forward(self, x, w, context=None):
        moe_losses = 0.0
        h = self.norm(x)
        h = self.proj_in(h)
        for block in self.attn_blocks:
            h, style, moe_loss = block(h, w, context=context)
            moe_losses += moe_loss
        h = self.proj_out(h)
        return (x + h) * self.attn_gain, style, moe_losses


class FeedForward(nn.Module):
    """Implements of the feed-forward layer in attention layer.

    """
    def __init__(self,
                 in_dim,
                 mlp_ratio,
                 dropout,
                 w_dim,
                 add_bias,
                 init_bias,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 glu,
                 num_experts=-1):
        """Initializes with layer settings.

        Args:
            in_dim: Number of channels of the input tensor.
            mlp_ratio: A factor used in increasing the dimension of MLP.
            dropout: Dropout ratio.
            w_dim: Dimension of W space for style modulation.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            glu: True: Use glu, otherwise, use geglu.
            num_experts: Number of involved experts. Disable in default.
        """
        super().__init__()
        self.in_dim = in_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.glu = glu
        self.num_experts = num_experts

        if self.num_experts < 2: # one ffn layer
            if glu:
                self.project_in = nn.Sequential(
                        DenseLayer(in_channels=in_dim,
                                   out_channels=in_dim * mlp_ratio,
                                   add_bias=add_bias,
                                   init_bias=init_bias,
                                   use_wscale=use_wscale,
                                   wscale_gain=wscale_gain,
                                   lr_mul=lr_mul,
                                   activation_type='linear'),
                        nn.GELU())
            else:
                self.project_in = GEGLU(in_channels=in_dim,
                                        out_channels=in_dim * mlp_ratio,
                                        add_bias=add_bias,
                                        init_bias=init_bias,
                                        use_wscale=use_wscale,
                                        wscale_gain=wscale_gain,
                                        lr_mul=lr_mul)

            self.proj_out = DenseLayer(in_channels=in_dim * mlp_ratio,
                                       out_channels=in_dim,
                                       add_bias=add_bias,
                                       init_bias=init_bias,
                                       use_wscale=use_wscale,
                                       wscale_gain=wscale_gain,
                                       lr_mul=lr_mul,
                                       activation_type='linear')

            self.net = nn.Sequential(
                            self.project_in,
                            nn.Dropout(dropout),
                            self.proj_out)
        else:
            self.net = nn.ModuleList([])
            # build each expert
            for _ in range(self.num_experts):
                if glu:
                    project_in = nn.Sequential(
                            DenseLayer(in_channels=in_dim,
                                       out_channels=in_dim * mlp_ratio,
                                       add_bias=add_bias,
                                       init_bias=init_bias,
                                       use_wscale=use_wscale,
                                       wscale_gain=wscale_gain,
                                       lr_mul=lr_mul,
                                       activation_type='linear'),
                            nn.GELU())
                else:
                    project_in = GEGLU(in_channels=in_dim,
                                            out_channels=in_dim * mlp_ratio,
                                            add_bias=add_bias,
                                            init_bias=init_bias,
                                            use_wscale=use_wscale,
                                            wscale_gain=wscale_gain,
                                            lr_mul=lr_mul)

                proj_out = DenseLayer(in_channels=in_dim * mlp_ratio,
                                           out_channels=in_dim,
                                           add_bias=add_bias,
                                           init_bias=init_bias,
                                           use_wscale=use_wscale,
                                           wscale_gain=wscale_gain,
                                           lr_mul=lr_mul,
                                           activation_type='linear')

                net = nn.Sequential(
                                project_in,
                                nn.Dropout(dropout),
                                proj_out)
                self.net.append(net)
            # build routing layer
            self.routing = AdaptiveModulateConvLayer(
                                    in_channels=in_dim,
                                    out_channels=self.num_experts,
                                    modulate_dim=w_dim,
                                    kernel_size=1,
                                    num_kernels=1,
                                    add_bias=add_bias,
                                    scale_factor=1,
                                    filter_kernel=None,
                                    demodulate=True,
                                    use_wscale=use_wscale,
                                    wscale_gain=wscale_gain,
                                    lr_mul=lr_mul,
                                    activation_type='lrelu',
                                    conv_clamp=None,
                                    eps=1e-8,
                                    mtm=False)
            # hyper-parameters for MoE
            self.jitter_noise = 0.0 # currently we disable the representation jittering
            self.expert_capacity = 262144 # 512*512

    def router(self, hidden_states, w):
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        original_dtype = hidden_states.dtype
        assert(original_dtype==torch.float32)

        if self.jitter_noise > 0: # question: diable this jittering when inference??
            # Get the lower and upper bound of the uniform distribution
            # Adapted from: https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
            distrib_lower_bound = 1.0 - self.jitter_noise
            distrib_upper_bound = 1.0 + self.jitter_noise

            uniform_distrib = torch.rand(hidden_states.shape, device=hidden_states.device, dtype=original_dtype)
            uniform_distrib = uniform_distrib * (distrib_lower_bound - distrib_upper_bound)

            uniform_distrib = uniform_distrib + distrib_upper_bound
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= uniform_distrib

        # Shape: [num_groups, tokens_per_group, num_experts]
        height = int(np.sqrt(hidden_states.shape[1]))
        width = int(np.sqrt(hidden_states.shape[1]))
        assert(height*width==hidden_states.shape[1])
        hidden_states = rearrange(hidden_states, 'b (h w) c -> b c h w', h=height, w=width)
        router_logits = self.routing(hidden_states, w)[0]
        router_logits = rearrange(router_logits, 'b c h w -> b (h w) c')


        # Apply Softmax and cast back to the original `dtype`
        router_probs = nn.functional.softmax(router_logits, dim=-1)


        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

    def cal_moe_loss(self, router_logits, expert_index):
        z_loss = router_z_loss_func(router_logits)
        router_probs = nn.Softmax(dim=-1)(router_logits)
        aux_loss = load_balancing_loss_func(router_probs, expert_index)
        loss = 0.001 * z_loss + 0.001 * aux_loss
        return loss

    def forward(self, x, w=None):
        if self.num_experts < 2:
            return self.net(x), 0.0

        # Step 1: Get the router_mask from the router as wel as the probabilities
        router_mask, router_probs, router_logits = self.router(x, w)
        expert_index = torch.argmax(router_mask, dim=-1)

        # The routers introduced might not always map all the tokens, to a router, which means that some hidden states
        # can be unchanged from one layer to another. That is why the hidden states are cloned before updating only the seleced ones.
        next_states = x.clone()
        for idx, expert in enumerate(self.net):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(x[token_indices])

        x = router_probs * next_states
        moe_loss = self.cal_moe_loss(router_logits, expert_index)
        return x, moe_loss


class AttentionBlock(nn.Module):
    """Implements the attention block used in the `Attention`.

    Attention block composed by self-attention and cross-attention.
    Self-attention allows spatial positions to attend to each other, while the
    cross-attention is used to involve the text condition information. Note that
    cross-attention only be involved when perform the text to image generation.
    """
    def __init__(self,
                 in_channels,
                 res,
                 context_dim,
                 num_heads,
                 head_dim,
                 w_dim,
                 demodulate,
                 use_text_cond,
                 use_w_cond,
                 attn_gain,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 conv_clamp,
                 use_checkpoint,
                 checkpoint_res,
                 l2_attention,
                 tie,
                 scale_in,
                 include_ff,
                 mask_self,
                 dropout,
                 mlp_ratio,
                 add_bias,
                 init_bias,
                 glu,
                 eps,
                 num_experts):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of projection out.
            res:
            context_dim: Number of channels of the context tensor.
            num_heads: Number of head of multi-head attention.
            head_dim: Dimension of each head.
            w_dim: Dimension of W space for style modulation.
            demodulate: Whether to perform style demodulation.
            use_text_cond: Whether to involve the text condition.
            use_w_cond:
            attn_gain:
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            conv_clamp: A threshold to clamp the output of convolution layers to
                avoid overflow under FP16 training.
            use_checkpoint: if True, use gradient checkpointing on this module.
            checkpoint_res:
            l2_attention: Whether or not to use l2 distance to compute the
                similarity.
            dropout: Dropout ratio.
            mlp_ratio: Ratio for the attention multipler.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            glu: use glu/geglu as activation, true: glu, false: geglu.
            eps: A small value to avoid divide overflow.
            num_experts: Numer of experts for MoE
        """
        super().__init__()
        self.in_channels = in_channels
        self.res = res
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.w_dim = w_dim
        self.demodulate = demodulate
        self.use_text_cond = use_text_cond
        self.use_w_cond = use_w_cond
        self.attn_gain = attn_gain
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.conv_clamp = conv_clamp
        self.use_checkpoint = use_checkpoint
        self.checkpoint_res = checkpoint_res
        self.l2_attention = l2_attention
        self.tie = tie
        self.scale_in = scale_in
        self.include_ff = include_ff
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.glu = glu
        self.eps = eps
        self.num_experts = num_experts

        if self.use_text_cond or self.use_w_cond:
            assert context_dim, ('Must given the context dimension when '
                                 '`use_text_cond` or `use_w_cond` is true.')

        self.norm1 = LayerNorm(in_channels)
        self.self_attn = SelfAttentionLayer(in_channels=in_channels,
                                            num_heads=num_heads,
                                            head_dim=head_dim,
                                            w_dim=w_dim,
                                            demodulate=demodulate,
                                            use_wscale=use_wscale,
                                            wscale_gain=wscale_gain,
                                            lr_mul=lr_mul,
                                            use_checkpoint=use_checkpoint,
                                            checkpoint_res=checkpoint_res,
                                            mask_self=mask_self,
                                            l2_attention=l2_attention,
                                            tie=tie,
                                            scale_in=scale_in,
                                            eps=eps)
        if use_text_cond or use_w_cond:
            self.norm2 = LayerNorm(in_channels)
            if use_w_cond:
                self.align_dim = DenseLayer(in_channels=w_dim,
                                            out_channels=context_dim,
                                            add_bias=True,
                                            init_bias=1.0,
                                            use_wscale=use_wscale,
                                            wscale_gain=wscale_gain,
                                            lr_mul=lr_mul,
                                            activation_type='linear')

            self.cross_attn = CrossAttentionLayer(query_dim=in_channels,
                                                  context_dim=context_dim,
                                                  num_heads=num_heads,
                                                  head_dim=head_dim,
                                                  dropout=0.,
                                                  use_wscale=use_wscale,
                                                  wscale_gain=wscale_gain,
                                                  lr_mul=lr_mul,
                                                  use_checkpoint=use_checkpoint,
                                                  checkpoint_res=checkpoint_res,
                                                  l2_attention=l2_attention,
                                                  scale_in=scale_in,
                                                  mask_self=mask_self)
        if include_ff:
            self.norm3 = LayerNorm(in_channels)
            self.ff = FeedForward(in_dim=in_channels,
                                mlp_ratio=mlp_ratio,
                                dropout=0.,
                                w_dim=w_dim,
                                add_bias=add_bias,
                                init_bias=init_bias,
                                use_wscale=use_wscale,
                                wscale_gain=wscale_gain,
                                lr_mul=lr_mul,
                                glu=glu,
                                num_experts=num_experts)

    def extra_repr(self) -> str:
        return (f'in_channels={self.in_channels}, '
                f'num_heads={self.num_heads}, '
                f'context_dim={self.context_dim}, '
                f'head_dim={self.head_dim}, '
                f'w_dim={self.w_dim}')

    def forward(self, x, w, context=None):
        moe_loss = 0.0
        residual = x
        _, _, height, width = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)
        x, style = self.self_attn(x, w)
        #x = (x + residual) * self.attn_gain
        x = x + residual

        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.use_text_cond or self.use_w_cond:
            contexts = []
            if self.use_text_cond:
                assert context is not None
                contexts.append(context)
            if self.use_w_cond:
                w_ = self.align_dim(w)
                contexts.append(w_[:, None])
            contexts = torch.cat(contexts, dim=1)
            #x = (self.cross_attn(self.norm2(x), contexts) + x) * self.attn_gain
            x = self.cross_attn(self.norm2(x), contexts) + x

        if self.include_ff:
            out, moe_loss = self.ff(self.norm3(x), w)
            #x = (out + x) * self.attn_gain
            x = out + x

        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)

        return x, style, moe_loss


class SelfAttentionLayer(nn.Module):
    """
    Implements the self-attention layer.

    Supporting the similarity computed using dot product or l2 norm.

    TODO: Scale down the l2 distance logit to match the unit normal distribution
    when init weight, reduce residual gain from attention.
    """

    def __init__(
        self,
        in_channels,
        num_heads,
        head_dim,
        w_dim,
        demodulate,
        use_wscale,
        wscale_gain,
        lr_mul,
        use_checkpoint,
        checkpoint_res,
        mask_self,
        l2_attention,
        tie,
        scale_in,
        eps):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of projection out.
            num_heads: Number of head of multi-head attention.
            head_dim: Dimension of the each head.
            w_dim: Dimension of W space for style modulation.
            demodulate: Whether to perform style demodulation.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            use_checkpoint: if True, use gradient checkpointing on this module.
            checkpoint_res:
            mask_self: Whether to omit the attention to self.
            l2_attention: Whether or not to use l2 distance to compute the
                similarity.
            eps: A small value to avoid divide overflow.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.w_dim = w_dim
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul

        self.use_checkpoint = use_checkpoint
        self.checkpoint_res = checkpoint_res
        self.l2_attention = l2_attention
        self.tie = tie
        self.scale_in = scale_in
        self.mask_self = mask_self
        self.eps = eps

        self.scale = self.head_dim ** -0.5

        if self.l2_attention:
            if self.tie:
                self.to_qk = DenseLayer(in_channels=in_channels,
                                        out_channels=in_channels,
                                        add_bias=False,
                                        init_bias=0.,
                                        use_wscale=use_wscale,
                                        wscale_gain=wscale_gain,
                                        lr_mul=lr_mul,
                                        activation_type='linear')
            else:
                self.to_qk = DenseLayer(in_channels=in_channels,
                                        out_channels=in_channels * 2,
                                        add_bias=False,
                                        init_bias=0.,
                                        use_wscale=use_wscale,
                                        wscale_gain=wscale_gain,
                                        lr_mul=lr_mul,
                                        activation_type='linear')

            self.to_v = DenseLayer(in_channels=in_channels,
                                   out_channels=in_channels,
                                   add_bias=False,
                                   init_bias=0.,
                                   use_wscale=use_wscale,
                                   wscale_gain=wscale_gain,
                                   lr_mul=lr_mul,
                                   activation_type='linear')
        else:
            self.to_qkv = DenseLayer(in_channels=in_channels,
                                     out_channels=in_channels * 3,
                                     add_bias=False,
                                     init_bias=0.,
                                     use_wscale=use_wscale,
                                     wscale_gain=wscale_gain,
                                     lr_mul=lr_mul,
                                     activation_type='linear')

        self.attention = QKVAttention(n_heads=self.num_heads,
                                      scale=self.scale,
                                      l2_attention=l2_attention,
                                      scale_in=scale_in,
                                      mask_self=mask_self)

        self.proj_out = AdaptiveModulateConvLayer(in_channels=in_channels,
                                                  out_channels=in_channels,
                                                  modulate_dim=w_dim,
                                                  kernel_size=1,
                                                  num_kernels=1,
                                                  add_bias=True,
                                                  scale_factor=1,
                                                  filter_kernel=None,
                                                  demodulate=demodulate,
                                                  use_wscale=use_wscale,
                                                  wscale_gain=wscale_gain,
                                                  lr_mul=lr_mul,
                                                  activation_type='linear',
                                                  conv_clamp=None,
                                                  eps=eps)

    def extra_repr(self):
        return (f'in_channels={self.in_channels}, '
                f'num_heads={self.num_heads}, '
                f'head_dim={self.head_dim}, '
                f'w_dim={self.w_dim}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'lr_mul={self.lr_mul:.3f}')

    def forward(self, x, w):
        right_res = str(x.shape[-1]) in self.checkpoint_res
        do_checkpoint = right_res and self.use_checkpoint
        return checkpoint(self._forward,
                          (x, w),
                          self.parameters(),
                          do_checkpoint)

    def _forward(self, x, w):
        _, _, height, width = x.shape
        h = rearrange(x, 'b c h w -> b (h w) c')
        if self.l2_attention:
            qk = self.to_qk(h) # b, x*y, c (num_heads * head_dim)
            qk = rearrange(qk, 'b x (h d) -> (b h) x d', h=self.num_heads)
            v = self.to_v(h)
            v = rearrange(v, 'b x (h d) -> (b h) x d', h=self.num_heads)
            if self.tie:
                qkv = torch.cat([qk, qk, v], dim=2)
            else:
                qkv = torch.cat([qk, v], dim=2)
        else:
            qkv = self.to_qkv(h)
            qkv = rearrange(qkv, 'b x (h d) -> (b h) x d', h=self.num_heads)
        out = self.attention(qkv)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y',
                        x=height, y=width, h=self.num_heads)
        out, style = self.proj_out(out, w)

        return out, style


class QKVAttention(nn.Module):
    """
    QKV attention module that support similarity computed using dot product and

    L2 norm.
    """
    def __init__(self,
                 n_heads,
                 scale,
                 l2_attention=True,
                 scale_in=True,
                 mask_self=False,
                 mask_value='-inf'):
        """Initializes with layer settings.

        Args:
            n_head: Number of heads for the multi-head attention.
            scale: The scale for the attention.
            l2_attention: Whether or not use l2 distance to compute similarity.
            mask_self: Whether to omit the attention to self.
            mask_value: Default value to fill in the distance matrix.
        """
        super().__init__()
        self.n_heads = n_heads
        self.scale = scale
        self.l2_attention = l2_attention
        self.scale_in = scale_in
        self.mask_value = mask_value
        self.mask_self = mask_self

    def extra_repr(self) -> str:
        return (f'num_heads={self.n_heads}, '
                f'scale={self.scale}, '
                f'mask_val={self.mask_value}')

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (HW) x C*3] tensor of Qs, Ks, and Vs.
        :return: an [N x (HW) x C] tensor after attention.
        """
        q, k, v = qkv.chunk(3, dim=2)

        if self.scale_in:
            q = q * self.scale
            k = k * self.scale

        if self.l2_attention:
            attn_logits = - torch.cdist(q, k, p=2)
            if self.mask_self:
                mask = torch.eye(attn_logits.shape[1],
                                 device=attn_logits.device,
                                 dtype=torch.bool)
                attn_logits = attn_logits.masked_fill(mask,
                                                      float(self.mask_value))
        else:
            attn_logits = torch.einsum('b t c, b s c -> b t s', q, k)

        if not self.scale_in:
            attn_logits = attn_logits * self.scale

        attn = torch.softmax(attn_logits, dim=-1)
        a = torch.einsum('b t s, b s c -> b t c', attn, v)
        return a


class CrossAttentionLayer(nn.Module):
    """Implements the cross-attention layer.

    NOTE: This layer support compute similarity using l2 distance. However, the
    query and key can not share the same weight as ViTGAN dose since query comes
    from image feature while key comes from text condition.

    TODO: Need to make sure whether this layer has a residual connection.
    """
    def __init__(self,
                 query_dim,
                 context_dim,
                 num_heads,
                 head_dim,
                 dropout,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 use_checkpoint,
                 checkpoint_res,
                 l2_attention,
                 scale_in,
                 mask_self):
        """Initializes with layer settings.

        Args:
            query_dim: Number of channels of the query tensor.
            context_dim: Number of channels of the context tensor.
            num_heads: Number of attention heads.
            head_dim: Dimension of each head.
            dropout: Dropout ratio.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            use_checkpoint: if True, use gradient checkpointing on this module.
            checkpoint_res:
            l2_attention: Whether or not to use l2 distance to compute the
                similarity.
            scale_in:
            mask_self:
        """
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.use_checkpoint = use_checkpoint
        self.checkpoint_res = checkpoint_res
        self.l2_attention = l2_attention
        self.scale_in = scale_in
        self.mask_self = mask_self

        inner_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        self.to_q = DenseLayer(in_channels=query_dim,
                               out_channels=inner_dim,
                               add_bias=False,
                               init_bias=0.,
                               use_wscale=use_wscale,
                               wscale_gain=wscale_gain,
                               lr_mul=lr_mul,
                               activation_type='linear')

        self.to_kv = DenseLayer(in_channels=context_dim,
                                out_channels=inner_dim * 2,
                                add_bias=False,
                                init_bias=0.,
                                use_wscale=use_wscale,
                                wscale_gain=wscale_gain,
                                lr_mul=lr_mul,
                                activation_type='linear')

        self.proj_out = nn.Sequential(
            DenseLayer(in_channels=inner_dim,
                       out_channels=query_dim,
                       add_bias=True,
                       init_bias=0.,
                       use_wscale=use_wscale,
                       wscale_gain=wscale_gain,
                       lr_mul=lr_mul,
                       activation_type='linear'),
            nn.Dropout(dropout)
        )

    def extra_repr(self):
        return (f'query_dim={self.query_dim}, '
                f'context_dim={self.context_dim}, '
                f'num_heads={self.num_heads}, '
                f'head_dim={self.head_dim}, '
                f'dropout={self.dropout}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'lr_mul={self.lr_mul:.3f}')

    def forward(self, x, context):
        return checkpoint(self._forward,
                          (x, context),
                          self.parameters(),
                          self.use_checkpoint)

    def _forward(self, x, context):
        q = self.to_q(x) # b, x*y, c (num_heads * head_dim)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=2)
        q, k, v = map(lambda t: rearrange(t, 'b x (h d) -> (b h) x d',
                                          h=self.num_heads), (q, k, v))
        if self.scale_in:
            q = q * self.scale
            k = k * self.scale

        if self.l2_attention:
            attn_logits =  - torch.cdist(q, k, p=2)
            if self.mask_self:
                mask = torch.eye(attn_logits.shape[1],
                                 device=attn_logits.device,
                                 dtype=torch.bool)
                attn_logits = attn_logits.masked_fill(mask, float('-inf'))
        else:
            attn_logits = torch.einsum('b i d, b j d -> b i j', q, k)

        if not self.scale_in:
            attn_logits = attn_logits * self.scale

        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)
        out = self.proj_out(out)

        return out


class ResidualSparseAttnBlock(nn.Module):
    """Implements the sparse attention used in the generator.

    Attention is composed by several `SparseAttentionBlock`
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


class ResidualDownsampleAttnBlock(nn.Module):
    """Implements the downsampling attention used in the generator.

    Attention is composed by several `DownsampleAttentionBlock`
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass


class ResidualLongNetAttnBlock(nn.Module):
    """Implements the longnet attention used in the generator.

    Attention is composed by several `LongNetAttentionBlock`
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass



class MultiHeadAttention(nn.Module):
    """Implements the multi-head attention layer with linear layer."""
    def __init__(self,
                 in_dim,
                 num_heads,
                 dropout,
                 add_bias,
                 init_bias,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 use_checkpoint,
                 l2_attention,
                 scale_in,
                 mask_self):
        """Initializes with layer settings.

        Args:
            in_dim: Number of channels of the input tensor.
            num_heads: Number of attention heads.
            dropout: Dropout ratio.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            use_checkpoint: if True, use gradient checkpointing on this module.
            l2_attention: Whether or not to use l2 distance to compute the
                similarity.
            scale_in:
            mask_self:
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.use_checkpoint = use_checkpoint
        self.l2_attention = l2_attention
        self.scale_in = scale_in
        self.mask_self = mask_self

        self.head_dim = in_dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.to_qkv = DenseLayer(in_channels=in_dim,
                                 out_channels=in_dim * 3,
                                 add_bias=add_bias,
                                 init_bias=init_bias,
                                 use_wscale=use_wscale,
                                 wscale_gain=wscale_gain,
                                 lr_mul=lr_mul,
                                 activation_type='linear')

        self.to_out = nn.Sequential(
            DenseLayer(in_channels=in_dim,
                       out_channels=in_dim,
                       add_bias=add_bias,
                       init_bias=init_bias,
                       use_wscale=use_wscale,
                       wscale_gain=wscale_gain,
                       lr_mul=lr_mul,
                       activation_type='linear'),
            nn.Dropout(dropout)
        )

    def extra_repr(self):
        return (f'in_dim={self.in_dim}, '
                f'num_heads={self.num_heads}, '
                f'head_dim={self.head_dim}, '
                f'init_bias={self.init_bias}, '
                f'dropout={self.dropout}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'l2_attantion={self.l2_attention}, '
                f'scale_in={self.scale_in}, '
                f'mask_self={self.mask_self}')

    def forward(self, x):
        return checkpoint(self._forward,
                          (x,),
                          self.parameters(),
                          self.use_checkpoint)

    def _forward(self, x):
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d',
                                          h=self.num_heads), (q, k, v))
        if self.scale_in:
            q = q * self.scale
            k = k * self.scale

        if self.l2_attention:
            attn_logits = - torch.cdist(q, k, p=2)
            if self.mask_self:
                mask = torch.eye(attn_logits.shape[1],
                                 device=attn_logits.device,
                                 dtype=torch.bool)
                attn_logits = attn_logits.masked_fill(mask, float('-inf'))
        else:
            attn_logits = torch.einsum('b i d, b j d -> b i j', q, k)

        if not self.scale_in:
            attn_logits = attn_logits * self.scale_in

        # attention, what we cannot get enough of
        attn = torch.softmax(attn_logits, dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)
        out = self.to_out(out)

        return out


class ResidualAttentionBlock(nn.Module):
    """Residual attention block in transformer."""
    def __init__(self,
                 in_dim,
                 num_heads,
                 dropout,
                 add_bias,
                 init_bias,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 use_checkpoint,
                 mlp_ratio,
                 l2_attention,
                 scale_in,
                 mask_self,
                 attn_gain):
        """Initializes with layer settings.

        Args:
            in_dim: Number of channels of the input tensor.
            num_heads: Number of attention heads.
            dropout: Dropout ratio.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            use_checkpoint: if True, use gradient checkpointing on this module.
            mlp_ratio: A factor used in increasing the dimension of MLP.
            l2_attention:
            scale_in:
            mask_self:
            attn_gain
        """
        super().__init__()
        self.attn_gain = attn_gain

        self.attn = MultiHeadAttention(in_dim=in_dim,
                                       num_heads=num_heads,
                                       dropout=dropout,
                                       add_bias=add_bias,
                                       init_bias=init_bias,
                                       use_wscale=use_wscale,
                                       wscale_gain=wscale_gain,
                                       lr_mul=lr_mul,
                                       use_checkpoint=use_checkpoint,
                                       l2_attention=l2_attention,
                                       scale_in=scale_in,
                                       mask_self=mask_self)

        self.c_fc = DenseLayer(in_channels=in_dim,
                               out_channels=in_dim * mlp_ratio,
                               add_bias=add_bias,
                               init_bias=init_bias,
                               use_wscale=use_wscale,
                               wscale_gain=wscale_gain,
                               lr_mul=lr_mul,
                               activation_type='linear')

        self.c_proj = DenseLayer(in_channels=in_dim * mlp_ratio,
                                 out_channels=in_dim,
                                 add_bias=add_bias,
                                 init_bias=init_bias,
                                 use_wscale=use_wscale,
                                 wscale_gain=wscale_gain,
                                 lr_mul=lr_mul,
                                 activation_type='linear')

        self.ln_1 = LayerNorm(in_dim)
        self.ln_2 = LayerNorm(in_dim)

        self.mlp = nn.Sequential(OrderedDict([('c_fc', self.c_fc),
                                              ('gelu', QuickGELU()),
                                              ('c_proj', self.c_proj)]))

    def forward(self, x):
        x = (x + self.attn(self.ln_1(x))) * self.attn_gain
        x = (x + self.mlp(self.ln_2(x))) * self.attn_gain
        return x


class TextTransformer(nn.Module):
    """TextTransformer implementation."""
    def __init__(self,
                 width,
                 num_heads,
                 num_layers,
                 dropout,
                 add_bias,
                 init_bias,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 use_checkpoint,
                 mlp_ratio,
                 l2_attention,
                 scale_in,
                 mask_self,
                 attn_gain):
        """Initializes with layer settings.

        Args:
            width: Total dimension of the model.
            num_heads: Number of parallel attention heads.
            num_layers: Number of transformer blocks.
            dropout: Dropout ratio.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            use_checkpoint: if True, use gradient checkpoint
            activation_type: Type of activation.
            mlp_ratio: Ratio used in increases the dimension of MLP.
            l2_attention:
            scale_in:
            mask_self:
        """
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.use_checkpoint = use_checkpoint
        self.mlp_ratio = mlp_ratio
        self.l2_attention = l2_attention
        self.scale_in = scale_in
        self.mask_self = mask_self
        self.attn_gain = attn_gain

        net_blocks = []
        for _ in range(self.num_layers):
            net_blocks.append(
                ResidualAttentionBlock(in_dim=width,
                                       num_heads=num_heads,
                                       dropout=dropout,
                                       add_bias=add_bias,
                                       init_bias=init_bias,
                                       use_wscale=use_wscale,
                                       wscale_gain=wscale_gain,
                                       lr_mul=lr_mul,
                                       use_checkpoint=use_checkpoint,
                                       mlp_ratio=mlp_ratio,
                                       l2_attention=l2_attention,
                                       scale_in=scale_in,
                                       mask_self=mask_self,
                                       attn_gain=attn_gain))

        self.res_blocks = nn.Sequential(*net_blocks)

    def extra_repr(self) -> str:
        return (f'width={self.width}, '
                f'num_heads={self.num_heads}, '
                f'num_layers={self.num_layers}, '
                f'dropout={self.dropout}, '
                f'bias={self.add_bias}, '
                f'init_bias={self.init_bias}, '
                f'use_wscale={self.use_wscale}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'mlp_ration={self.mlp_ratio}')

    def forward(self, x):
        if self.num_layers == 0:
            return x
        return self.res_blocks(x)


class AdaptiveModulateConvLayer(nn.Module):
    """Implements the adaptive convolutional layer with style modulation.

    When num_kernels equals to 1, it is the same as the conventional
    ModulateConvLayer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 modulate_dim,
                 kernel_size,
                 num_kernels,
                 add_bias,
                 scale_factor,
                 filter_kernel,
                 demodulate,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 activation_type,
                 conv_clamp,
                 eps,
                 mtm=False):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            modulate_dim: Dimension of the input modulate code.
            kernel_size: Size of the convolutional kernels.
            num_kernels: Number of convolution filters used.
            add_bias: Whether to add bias onto the convolutional result.
            scale_factor: Scale factor for upsampling.
            filter_kernel: Kernel used for filtering.
            demodulate: Whether to perform style demodulation.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            activation_type: Type of activation.
            conv_clamp: A threshold to clamp the output of convolution layers to
                avoid overflow under FP16 training.
            eps: A small value to avoid divide overflow.
            mtm: Whether to apply modulated tranformation module
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modulate_dim = modulate_dim
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.add_bias = add_bias
        self.scale_factor = scale_factor
        self.filter_kernel = filter_kernel
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.activation_type = activation_type
        self.conv_clamp = conv_clamp
        self.eps = eps
        self.mtm = mtm
        self.adaptive = self.num_kernels > 1

        # Set up weight.
        weight_shape = (num_kernels, out_channels, in_channels,
                        kernel_size, kernel_size)
        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                    torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        # Set up bias.
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None
        self.act_gain = bias_act.activation_funcs[activation_type].def_gain

        # Set up style.
        self.style = DenseLayer(in_channels=modulate_dim,
                                out_channels=in_channels,
                                add_bias=True,
                                init_bias=1.0,
                                use_wscale=use_wscale,
                                wscale_gain=wscale_gain,
                                lr_mul=lr_mul,
                                activation_type='linear')
        if self.adaptive:
            # Set up weight selection
            self.filter_weight = DenseLayer(in_channels=modulate_dim,
                                            out_channels=num_kernels,
                                            add_bias=True,
                                            init_bias=1.0,
                                            use_wscale=use_wscale,
                                            wscale_gain=wscale_gain,
                                            lr_mul=lr_mul,
                                            activation_type='linear')
        if scale_factor > 1:
            assert filter_kernel is not None
            self.register_buffer(
                'filter', upfirdn2d.setup_filter(filter_kernel))
            fh, fw = self.filter.shape
            self.filter_padding = (
                kernel_size // 2 + (fw + scale_factor - 1) // 2,
                kernel_size // 2 + (fw - scale_factor) // 2,
                kernel_size // 2 + (fh + scale_factor - 1) // 2,
                kernel_size // 2 + (fh - scale_factor) // 2)

        if self.mtm:
            self.mtm_style = DenseLayer(in_channels=modulate_dim,
                                        out_channels=in_channels,
                                        add_bias=True,
                                        init_bias=1.0,
                                        use_wscale=use_wscale,
                                        wscale_gain=wscale_gain,
                                        lr_mul=lr_mul,
                                        activation_type='linear')
            mtm_out_channels = 3 * kernel_size * kernel_size
            self.mtm_weight = nn.Parameter(torch.randn([mtm_out_channels, in_channels, kernel_size, kernel_size]))
            self.mtm_bias = nn.Parameter(torch.zeros([1, mtm_out_channels, 1, 1]))
            self.mtm_scales = nn.Parameter(torch.zeros([1, mtm_out_channels, 1, 1]))

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'modulate_dim={self.modulate_dim}, '
                f'ksize={self.kernel_size}, '
                f'num_kernel={self.num_kernels}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'upsample={self.scale_factor}, '
                f'upsample_filter={self.filter_kernel}, '
                f'demodulate={self.demodulate}, '
                f'act={self.activation_type}, '
                f'mtm={self.mtm}, ')

    def get_offsets(self, x, modulate_code):
        dtype = x.dtype
        N, _, H, W = x.shape
        mtm_weight = self.mtm_weight

        out_ch, in_ch, kh, kw = mtm_weight.shape
        mtm_style = self.mtm_style(modulate_code)
        wscale = 1.0 / np.sqrt(in_ch * kh * kw)
        if not self.demodulate:
            _style = mtm_style * wscale  # Equivalent to scaling weight.
        else:
            _style = mtm_style

        # Pre-normalize inputs.
        if self.demodulate:
            mtm_weight = mtm_weight.unsqueeze(0)
            mtm_weight = (mtm_weight *
                      mtm_weight.square().mean(dim=(2, 3, 4), keepdim=True).rsqrt())
            _style = _style * _style.square().mean().rsqrt()

        mtm_weight = mtm_weight * _style.reshape(N, 1, in_ch, 1, 1)  # modulation

        if self.demodulate:
            decoef = (mtm_weight.square().sum(dim=(2, 3, 4)) + self.eps).rsqrt()
            mtm_weight = mtm_weight * decoef.reshape(N, out_ch, 1, 1, 1)  # demodulation

        # Fuse `conv` and `style modulation` as one op, using group convolution.
        x = x.reshape(1, N * in_ch, H, W)
        w = mtm_weight.reshape(N * out_ch, in_ch, kh, kw).to(dtype)
        offset = conv2d_gradfix.conv2d(
                x, w, stride=1, padding=kh//2, groups=N)
        offset = offset.reshape(N, -1, *offset.shape[2:])
        mtm_bias = self.mtm_bias.to(dtype)
        mtm_scales = self.mtm_scales.to(dtype)
        offset = offset * mtm_scales + mtm_bias
        offset_x, offset_y, offsets_mask = torch.chunk(offset, 3, dim=1)
        offsets = torch.cat([offset_x, offset_y], 1)
        offsets_mask = offsets_mask.sigmoid() * 2 # range in [0, 2]
        return offsets, offsets_mask


    def forward(self, x, modulate_code):
        dtype = x.dtype
        N, C, H, W = x.shape

        # calculate offsets
        if self.mtm:
            offsets, offsets_mask = self.get_offsets(x, modulate_code)

        if self.adaptive:
            # Affine on `modulate_code` to get the weight for each filter.
            filter_weights  = self.filter_weight(modulate_code)
            filter_weights = torch.softmax(filter_weights, dim=-1)
            filter_weights = filter_weights[:, :, None, None, None, None]
            weight = repeat(self.weight, '... -> b ...', b=N)
            weight = reduce(weight * filter_weights, 'b n ... -> b ...', 'sum')
        else:
            weight = self.weight

        _, out_ch, in_ch, kh, kw = weight.shape
        assert in_ch == C

        # Affine on `modulate_code`.
        style = self.style(modulate_code)
        if not self.demodulate:
            _style = style * self.wscale  # Equivalent to scaling weight.
        else:
            _style = style

        # Pre-normalize inputs.
        if self.demodulate:
            weight = (weight *
                      weight.square().mean(dim=(2, 3, 4), keepdim=True).rsqrt())
            _style = _style * _style.square().mean().rsqrt()

        weight = weight * _style.reshape(N, 1, in_ch, 1, 1)  # modulation

        if self.demodulate:
            decoef = (weight.square().sum(dim=(2, 3, 4)) + self.eps).rsqrt()
            weight = weight * decoef.reshape(N, out_ch, 1, 1, 1)  # demodulation

        # Fuse `conv` and `style modulation` as one op, using group convolution.
        x = x.reshape(1, N * in_ch, H, W)
        w = weight.reshape(N * out_ch, in_ch, kh, kw).to(dtype)

        if self.scale_factor == 1:  # Native convolution without upsampling.
            padding = self.kernel_size // 2
            if self.mtm: # Perform modulated transformation module
                offsets = offsets.reshape(1, -1, *offsets.shape[2:])
                offsets_mask = offsets_mask.reshape(1, -1, *offsets_mask.shape[2:])
                x = torchvision.ops.deform_conv2d(x, offsets, w,
                    mask=offsets_mask, padding=padding)
            else:
                x = conv2d_gradfix.conv2d(
                    x, w, stride=1, padding=padding, groups=N)
        else:  # Convolution with upsampling.
            up = self.scale_factor
            f = self.filter
            # When kernel size = 1, use filtering function for upsampling.
            if self.kernel_size == 1:
                padding = self.filter_padding
                x = conv2d_gradfix.conv2d(
                    x, w, stride=1, padding=0, groups=N)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=up, padding=padding, gain=up ** 2)
            # When kernel size != 1, use stride convolution for upsampling.
            else:
                # Following codes are borrowed from
                # https://github.com/NVlabs/stylegan2-ada-pytorch
                px0, px1, py0, py1 = self.filter_padding
                px0 = px0 - (kw - 1)
                px1 = px1 - (kw - up)
                py0 = py0 - (kh - 1)
                py1 = py1 - (kh - up)
                pxt = max(min(-px0, -px1), 0)
                pyt = max(min(-py0, -py1), 0)
                if N == 1:
                    w = w.transpose(0, 1)
                else:
                    w = w.reshape(N, out_ch, in_ch, kh, kw)
                    w = w.transpose(1, 2)
                    w = w.reshape(N * in_ch, out_ch, kh, kw)
                padding = (pyt, pxt)
                x = conv2d_gradfix.conv_transpose2d(
                    x, w, stride=up, padding=padding, groups=N)
                padding = (px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=1, padding=padding, gain=up ** 2)

        x = x.reshape(N, out_ch, x.shape[2], x.shape[3])

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype)
            if self.bscale != 1.0:
                bias = bias * self.bscale

        if self.activation_type == 'linear':  # Shortcut for output layer.
            x = bias_act.bias_act(
                x, bias, act='linear', clamp=self.conv_clamp)
        else:
            act_gain = self.act_gain
            act_clamp = None
            if self.conv_clamp is not None:
                act_clamp = self.conv_clamp
            x = bias_act.bias_act(x, bias,
                                  act=self.activation_type,
                                  gain=act_gain,
                                  clamp=act_clamp)

        assert x.dtype == dtype
        assert style.dtype == torch.float32
        return x, style


class ConvLayer(nn.Module):
    """Implements the convolutional layer.

    If upsampling is needed (i.e., `scale_factor = 2`), the feature map will
    be filtered with `filter_kernel` after convolution. This layer will only be
    used for skip connection in `resnet` architecture.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 add_bias,
                 scale_factor,
                 filter_kernel,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 activation_type,
                 conv_clamp):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            kernel_size: Size of the convolutional kernels.
            add_bias: Whether to add bias onto the convolutional result.
            scale_factor: Scale factor for upsampling.
            filter_kernel: Kernel used for filtering.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            activation_type: Type of activation.
            conv_clamp: A threshold to clamp the output of convolution layers to
                avoid overflow under FP16 training.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.add_bias = add_bias
        self.scale_factor = scale_factor
        self.filter_kernel = filter_kernel
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.activation_type = activation_type
        self.conv_clamp = conv_clamp

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None
        self.act_gain = bias_act.activation_funcs[activation_type].def_gain

        if scale_factor > 1:
            assert filter_kernel is not None
            self.register_buffer(
                'filter', upfirdn2d.setup_filter(filter_kernel))
            fh, fw = self.filter.shape
            self.filter_padding = (
                kernel_size // 2 + (fw + scale_factor - 1) // 2,
                kernel_size // 2 + (fw - scale_factor) // 2,
                kernel_size // 2 + (fh + scale_factor - 1) // 2,
                kernel_size // 2 + (fh - scale_factor) // 2)

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'ksize={self.kernel_size}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'upsample={self.scale_factor}, '
                f'upsample_filter={self.filter_kernel}, '
                f'act={self.activation_type}, '
                f'clamp={self.conv_clamp}')

    def forward(self, x):
        dtype = x.dtype

        weight = self.weight
        if self.wscale != 1.0:
            weight = weight * self.wscale
        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype)
            if self.bscale != 1.0:
                bias = bias * self.bscale

        if self.scale_factor == 1:  # Native convolution without upsampling.
            padding = self.kernel_size // 2
            x = conv2d_gradfix.conv2d(
                x, weight.to(dtype), stride=1, padding=padding)
        else:  # Convolution with upsampling.
            up = self.scale_factor
            f = self.filter
            # When kernel size = 1, use filtering function for upsampling.
            if self.kernel_size == 1:
                padding = self.filter_padding
                x = conv2d_gradfix.conv2d(
                    x, weight.to(dtype), stride=1, padding=0)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=up, padding=padding, gain=up ** 2)
            # When kernel size != 1, use transpose convolution for upsampling.
            else:
                # Following codes are borrowed from
                # https://github.com/NVlabs/stylegan2-ada-pytorch
                px0, px1, py0, py1 = self.filter_padding
                kh, kw = weight.shape[2:]
                px0 = px0 - (kw - 1)
                px1 = px1 - (kw - up)
                py0 = py0 - (kh - 1)
                py1 = py1 - (kh - up)
                pxt = max(min(-px0, -px1), 0)
                pyt = max(min(-py0, -py1), 0)
                weight = weight.transpose(0, 1)
                padding = (pyt, pxt)
                x = conv2d_gradfix.conv_transpose2d(
                    x, weight.to(dtype), stride=up, padding=padding)
                padding = (px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=1, padding=padding, gain=up ** 2)

        act_gain = self.act_gain
        act_clamp = None
        if self.conv_clamp is not None:
            act_clamp = self.conv_clamp
        x = bias_act.bias_act(x, bias,
                              act=self.activation_type,
                              gain=act_gain,
                              clamp=act_clamp)

        assert x.dtype == dtype
        return x


class QuickGELU(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate.

    See: https://github.com/hendrycks/GELUs.
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class GEGLU(nn.Module):
    """ A variant of the gated linear unit activation function
        from https://arxiv.org/abs/2002.05202.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias,
                 init_bias,
                 use_wscale,
                 wscale_gain,
                 lr_mul):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul

        self.proj = DenseLayer(in_channels=in_channels,
                               out_channels=out_channels * 2,
                               add_bias=add_bias,
                               init_bias=init_bias,
                               use_wscale=use_wscale,
                               wscale_gain=wscale_gain,
                               lr_mul=lr_mul,
                               activation_type='linear')

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, dim, eps):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self):
        return f'dim={self.dim}, epsilon={self.eps}'

    def forward(self, x):
        scale = (x.square().mean(dim=self.dim, keepdim=True) + self.eps).rsqrt()
        return x * scale


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(self, embedding_dim):
        super().__init__(embedding_dim)
        self.embedding_dim = embedding_dim

    def extra_repr(self):
        return f'embedding_dim={self.embedding_dim}'

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class GroupNorm(nn.GroupNorm):
    """Implement of the group norm 32"""
    def __init__(self, num_groups, num_channels):
        super().__init__(num_groups, num_channels)
        self.num_groups = num_groups
        self.num_channels = num_channels

    def extra_repr(self):
        return (f'num_group={self.num_groups}, '
                f'num_channels={self.num_channels}')

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class DenseLayer(nn.Module):
    """Implements the dense layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias,
                 init_bias,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 activation_type,
                 init_weight=1.0):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            activation_type: Type of activation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.activation_type = activation_type

        weight_shape = (out_channels, in_channels)
        wscale = wscale_gain / np.sqrt(in_channels)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) * init_weight / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale * init_weight / lr_mul)
            self.wscale = lr_mul

        if add_bias:
            init_bias = np.broadcast_to(np.asarray(init_bias, dtype=np.float32), [out_channels])
            self.bias = nn.Parameter(torch.from_numpy(init_bias / lr_mul))
            self.bscale = lr_mul
        else:
            self.bias = None

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'init_bias={self.init_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'act={self.activation_type}')

    def forward(self, x):
        dtype = x.dtype

        weight = self.weight.to(dtype) * self.wscale
        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype)
            if self.bscale != 1.0:
                bias = bias * self.bscale

        # Fast pass for two dimension input.
        if x.dim() == 2 and bias is not None:
            x = torch.addmm(bias.unsqueeze(0), x, weight.t())
            if self.activation_type != 'linear':
                x = bias_act.bias_act(x,
                                      b=None,
                                      dim=1,
                                      act=self.activation_type)
        else:
            x = x.matmul(weight.t())
            x = bias_act.bias_act(x, bias,
                                  dim=2,
                                  act=self.activation_type)

        assert x.dtype == dtype
        return x


class CheckpointFunction(torch.autograd.Function):
    """CheckpointFunction """
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True)
                             for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


# list two loss for MoE training
def router_z_loss_func(router_logits):
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [batch_size, sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def load_balancing_loss_func(router_probs, expert_indices):
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)

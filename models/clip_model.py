# python3.7
"""Contains the implementation of text embedding code used in our text2image
generation. We used the pre-trained CLIP model, and the paper can be found
here: https://arxiv.org/abs/2103.00020. Also, part of codes are borrow from
open_clip (https://github.com/mlfoundations/open_clip).
"""
from itertools import repeat
import collections.abc
from dataclasses import dataclass
import os
from collections import OrderedDict
from typing import Tuple, Union, Optional, Callable, Sequence, List
import json
import re
from pathlib import Path
from copy import deepcopy
import numpy as np
from pkg_resources import packaging
from PIL import Image

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from huggingface_hub import hf_hub_download

from utils.misc import download_url
from utils.misc import get_cache_dir
from third_party.clip_official.simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

__all__ = ['CLIPModel']

# pylint: disable=arguments-renamed
# pylint: disable=line-too-long
# pylint: disable=global-statement
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def _pcfg(url='', hf_hub='', mean=None, std=None):
    return dict(
        url=url,
        hf_hub=hf_hub,
        mean=mean,
        std=std,
    )

_VITB32 = dict(
    openai=_pcfg(
        'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'),
    laion400m_e31=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt'),
    laion400m_e32=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt'),
    laion2b_e16=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth'),
    laion2b_s34b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-laion2B-s34B-b79K/'),
    # DataComp-M models
    datacomp_m_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K/'),
    commonpool_m_clip_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K/'),
    commonpool_m_laion_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K/'),
    commonpool_m_image_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K/'),
    commonpool_m_text_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K/'),
    commonpool_m_basic_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K/'),
    commonpool_m_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K/'),
    # DataComp-S models
    datacomp_s_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K/'),
    commonpool_s_clip_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K/'),
    commonpool_s_laion_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K/'),
    commonpool_s_image_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K/'),
    commonpool_s_text_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K/'),
    commonpool_s_basic_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K/'),
    commonpool_s_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K/'),
)

_VITB32_quickgelu = dict(
    openai=_pcfg(
        'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'),
    laion400m_e31=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt'),
    laion400m_e32=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt'),
)

_VITB16 = dict(
    openai=_pcfg(
        'https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt'),
    laion400m_e31=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-00efa78f.pt'),
    laion400m_e32=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt'),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-laion2B-s34B-b88K/'),
    # DataComp-L models
    datacomp_l_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K/'),
    commonpool_l_clip_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K/'),
    commonpool_l_laion_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K/'),
    commonpool_l_image_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K/'),
    commonpool_l_text_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K/'),
    commonpool_l_basic_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K/'),
    commonpool_l_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K/'),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt'),
    laion400m_e32=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e32-699c4b84.pt'),
)

_VITL14 = dict(
    openai=_pcfg(
        'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'),
    laion400m_e31=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt'),
    laion400m_e32=_pcfg(
        'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt'),
    laion2b_s32b_b82k=_pcfg(
        hf_hub='laion/CLIP-ViT-L-14-laion2B-s32B-b82K/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/'),
    commonpool_xl_clip_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K/'),
    commonpool_xl_laion_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K/'),
    commonpool_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K/'),
)

_VITL14_336 = dict(
    openai=_pcfg(
        'https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt'),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-H-14-laion2B-s32B-b79K/'),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s12B-b42K/'),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s34B-b88K/'),
)

_VITbigG14 = dict(
    laion2b_s39b_b160k=_pcfg(hf_hub='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/'),
)

_coca_VITB32 = dict(
    laion2b_s13b_b90k=_pcfg(hf_hub='laion/CoCa-ViT-B-32-laion2B-s13B-b90k/'),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(hf_hub='laion/mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k/')
)

_coca_VITL14 = dict(
    laion2b_s13b_b90k=_pcfg(hf_hub='laion/CoCa-ViT-L-14-laion2B-s13B-b90k/'),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(hf_hub='laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/')
)


_PRETRAINED = {
    'ViT-B-32': _VITB32,
    'ViT-B-32-quickgelu': _VITB32_quickgelu,
    'ViT-B-16': _VITB16,
    'ViT-B-16-plus-240': _VITB16_PLUS_240,
    'ViT-L-14': _VITL14,
    'ViT-L-14-336': _VITL14_336,
    'ViT-H-14': _VITH14,
    'ViT-g-14': _VITg14,
    'ViT-bigG-14': _VITbigG14,
    'coca_ViT-B-32': _coca_VITB32,
    'coca_ViT-L-14': _coca_VITL14,
}

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / 'clip_configs/']
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

_tokenizer = _Tokenizer()

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_pretrained_models_by_tag(tag: str):
    """ return all models having the specified pretrain tag """
    models = []
    tag = _clean_tag(tag)
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models


def list_pretrained_tags_by_model(model: str):
    """ return all pretrain tags for the specified model architecture """
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def download_pretrained(cfg,
                        path=None,
                        force_hf_hub=False,
                        revision=None):
    target = ''
    if not cfg:
        return target

    url = cfg.get('url', '')
    hf_hub = cfg.get('hf_hub', '')
    if hf_hub and force_hf_hub:
        # use HF hub even if url exists
        url = ''

    if url:
        expected_sha256 = None
        if 'mlfoundations' in url:
            expected_sha256 = os.path.splitext(url)[0].split('-')[-1]
        target, hash_check = download_url(url, path=path, sha256=expected_sha256)
        if hash_check is False:
            print(f'Hash check failed! The remote file from URL '
                  f'`{url}` may be changed, or the downloading is '
                  f'interrupted. The loaded perceptual model may '
                  f'have unexpected behavior.')
    elif hf_hub:
        # we assume the hf_hub entries in pretrained config combine model_id + filename in
        # 'org/model_name/filename.pt' form. To specify just the model id w/o filename and
        # use 'open_clip_pytorch_model.bin' default, there must be a trailing slash 'org/model_name/'.
        model_id, filename = os.path.split(hf_hub)
        if filename:
            target = download_hf_model(model_id,
                                       path=path,
                                       filename=filename,
                                       revision=revision)
        else:
            target = download_hf_model(model_id,
                                       path=path,
                                       revision=revision)

    return target


def _clean_tag(tag: str):
    # normalize pretrained tags
    return tag.lower().replace('-', '_')


def get_pretrained_cfg(model, tag):
    if model not in _PRETRAINED:
        return {}
    model_pretrained = _PRETRAINED[model]
    return model_pretrained.get(_clean_tag(tag), {})


def get_pretrained_url(model, tag):
    cfg = get_pretrained_cfg(model, _clean_tag(tag))
    return cfg.get('url', '')


def list_openai_models():
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag('openai')


def _convert_image_to_rgb(image):
    return image.convert('RGB')


def _transfor_ori(img_size):
    return Compose([
        Resize(img_size, interpolation=BICUBIC),
        CenterCrop(img_size),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def _transfor_mod(img_size):
    return Compose([
        Resize(img_size, interpolation=BICUBIC),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])



def download_hf_model(model_id,
                      path=None,
                      filename='open_clip_pytorch_model.bin',
                      library_name='open_clip',
                      version='2.20.0',
                      revision='85b38ccd4a98128c6e4f7428c45ab8d71660ab9c'):
    """Download file from Hugging Face.

    Args:
        model_id: A user or an organization name and a repo name separated
            by a `/`.
        path: Path (directory) to save the downloaded file. If set as `None`,
            the cache directory will be used. Please see `get_cache_dir()` for
            more details. (default: None)
        filename: The name of the file in the repo.
        library_name: The name of the library to which the object corresponds.
        version: The version of the library.
        revision: An optional Git revision id which can be a branch name, a tag,
            or a commit hash.

    Returns:
        Local path (string) of file or if networking is off, last version of
        file cached on disk.
    """
    # Handle file path.
    if path is None:
        path = get_cache_dir()
    save_path = hf_hub_download(model_id,
                                filename,
                                revision=revision,
                                library_name=library_name,
                                library_version=version,
                                cache_dir=path)
    return save_path

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16
      (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32),
                         self.normalized_shape,
                         self.weight,
                         self.bias,
                         self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x,
                         self.normalized_shape,
                         self.weight,
                         self.bias,
                         self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=np.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float('-inf'))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, mlp_width)),
            ('gelu', act_layer()),
            ('c_proj', nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None

        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            attentional_pool: bool = False,
            n_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            input_patchnorm: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False
    ):
        super().__init__()
        self.output_tokens = output_tokens
        # self.image_size = image_size
        # self.patch_size = patch_size
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.global_average_pool = global_average_pool
        if attentional_pool:
            self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
            self.ln_post = norm_layer(output_dim)
            self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward(self, x: torch.Tensor):

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens

        return pooled


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            embed_cls: bool = False,
            pad_id: int = 0,
            output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float('-inf'))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']], 'in_proj_bias', 'bias_k', 'bias_v']:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, 'text_projection', None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, 'proj', None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(np.sqrt(len(pos_emb_img))))

    print('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU
    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        input_patchnorm=vision_cfg.input_patchnorm,
        global_average_pool=vision_cfg.global_average_pool,
        attentional_pool=vision_cfg.attentional_pool,
        n_queries=vision_cfg.n_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        output_tokens=text_cfg.output_tokens,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return text


def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(state_dict,
                                       quick_gelu=True,
                                       cast_dtype=torch.float16,):
    vit = 'visual.proj' in state_dict

    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round((state_dict['visual.positional_embedding'].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split('.')[2] for k in state_dict if k.startswith(f'visual.layer{b}'))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round((state_dict['visual.attnpool.positional_embedding'].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict['visual.attnpool.positional_embedding'].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split('.')[2] for k in state_dict if k.startswith('transformer.resblocks')))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ['input_resolution', 'context_length', 'vocab_size']:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def load_openai_model(name,
                      precision=None,
                      device=None,
                      cache_dir=None):
    """Load CLIP model from OpenAI.

    Args:
        name: A model name listed by `clip.available_models()`, or the path to
              a model checkpoint containing the state_dict.
        precision: Model precision, if None defaults to 'fp32'
            if device == 'cpu' else 'fp16'.
        device: The device to put the loaded model.
        cache_dir : The directory to cache the downloaded model weights.

    Returns:
        model: The CLIP model
        preprocess:  A torchvision transform that converts a PIL image into a
            tensor that the returned model can take as its input.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'
    model_url = get_pretrained_url(name, 'openai')
    if model_url:
        expected_sha256 = None
        if 'openaipublic' in model_url:
            expected_sha256 = model_url.split('/')[-2]
        model_path, hash_check = download_url(model_url,
                                              path=cache_dir,
                                              sha256=expected_sha256)
        if hash_check is False:
            print(f'Hash check failed! The remote file from URL '
                  f'`{model_url}` may be changed, or the downloading is '
                  f'interrupted. The loaded perceptual model may '
                  f'have unexpected behavior.')
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f'Model {name} not found; available models '
                           f'= {list_openai_models()}')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        state_dict = torch.load(model_path, map_location='cpu')

    # Build a non-jit model from the OpenAI jitted model state dict
    cast_dtype = get_cast_dtype(precision)
    try:
        model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
    except KeyError:
        sd = {k[7:]: v for k, v in state_dict['state_dict'].items()}
        model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

    # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
    model = model.to(device)
    # FIXME support pure fp16/bf16 precision modes
    if precision != 'fp16':
        model.float()
        if precision == 'bf16':
            # for bf16, convert back to low-precision
            convert_weights_to_lp(model, dtype=torch.bfloat16)

    # add mean / std attributes for consistency with OpenCLIP models
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD
    return model


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize=True):
        features = self.visual(image)
        if normalize:
            features = F.normalize(features, dim=-1)
        return  features

    def encode_text(self, text, normalize=True):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_ = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x_[torch.arange(x_.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if normalize:
            x = F.normalize(x, dim=-1)
        return x, x_

    def forward(self, image=None, text=None):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        pooled_out, out = self.encode_text(text, normalize=True) if text is not None else None
        return {
                'image_features': image_features,
                'text_features_pooled': pooled_out,
                'text_features': out,
                'logit_scale': self.logit_scale.exp()
            }


class CLIPModel(nn.Module):
    """CLIP model used for extracting the image and text features."""
    def __init__(self,
                 model_name='ViT-L/14',
                 pretrained='openai',
                 max_length=77,
                 precision='fp32',
                 cache_dir=None,
                 device='cuda',
                 jit=False,
                 force_quick_gelu=False,
                 force_patch_dropout=None,
                 force_image_size=None,
                 output_dict=None,
                 require_pretrained=False,
                 freeze_clip=True,
                 revision='85b38ccd4a98128c6e4f7428c45ab8d71660ab9c'):
        super().__init__()

        self.device = device
        self.freeze_clip = freeze_clip
        self.max_length = max_length

        has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
        if has_hf_hub_prefix:
            model_id = model_name[len(HF_HUB_PREFIX):]
            self.checkpoint_path = download_hf_model(model_id,
                                                     path=cache_dir,
                                                     revision=revision)
            config_path = download_hf_model(model_id,
                                            path=cache_dir,
                                            filename='open_clip_config.json',
                                            revision=revision)
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.pretrained_cfg = config['preprocess_cfg']
            self.model_cfg = config['model_cfg']
        else:
            # for callers using old naming with / in ViT names
            model_name = model_name.replace('/', '-')
            self.checkpoint_path = None
            self.pretrained_cfg = {}
            self.model_cfg = None

        if pretrained and pretrained.lower() == 'openai':
            print(f'Loading pretrained {model_name} from OpenAI.')
            self.model = load_openai_model(model_name,
                                           precision=precision,
                                           device=device,
                                           cache_dir=cache_dir)
        else:
            self.model_cfg = self.model_cfg or get_model_config(model_name)
            if self.model_cfg is not None:
                print(f'Loaded {model_name} model config.')
            else:
                print(f'Model config for {model_name} not found; '
                      f'available models {list_models()}.')
                raise RuntimeError(f'Model config for {model_name} not found.')

            if force_quick_gelu:
                # override for use of QuickGELU on non-OpenAI transformer models
                self.model_cfg['quick_gelu'] = True

            if force_patch_dropout is not None:
                # override the default patch dropout value
                self.model_cfg['vision_cfg']['patch_dropout'] = force_patch_dropout

            if force_image_size is not None:
                # override model config's image size
                self.model_cfg['vision_cfg']['image_size'] = force_image_size

            # cast_dtype set for fp16 and bf16 (manual mixed-precision),
            # not set for 'amp' or 'pure' modes
            cast_dtype = get_cast_dtype(precision)
            self.model = CLIP(**self.model_cfg, cast_dtype=cast_dtype)
            if precision in ('fp16', 'bf16'):
                dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
                # manual mixed precision that matches original OpenAI behaviour
                self.model.to(device=device)
                convert_weights_to_lp(self.model, dtype=dtype)
            elif precision in ('pure_fp16', 'pure_bf16'):
                dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
                self.model.to(device=device, dtype=dtype)
            else:
                self.model.to(device=device)

            pretrained_loaded = False
            if pretrained:
                checkpoint_path = ''
                pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
                if pretrained_cfg:
                    checkpoint_path = download_pretrained(pretrained_cfg,
                                                          path=cache_dir,
                                                          revision=revision)
                elif os.path.exists(pretrained):
                    checkpoint_path = pretrained

                if checkpoint_path:
                    print(f'Loading pretrained {model_name} weights ({pretrained}).')
                    load_checkpoint(self.model, checkpoint_path)
                else:
                    error_str = (
                        f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                        f'Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                    print(error_str)
                    raise RuntimeError(error_str)
                pretrained_loaded = True
            elif has_hf_hub_prefix:
                print(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(self.model, checkpoint_path)
                pretrained_loaded = True

            if require_pretrained and not pretrained_loaded:
                # callers of create_model_from_pretrained always expect pretrained weights
                raise RuntimeError(
                    f'Pretrained weights were required for (model: '
                    f'{model_name}, pretrained: {pretrained}) but not loaded.')

            # set image / mean metadata from pretrained_cfg if available, or use default
            self.model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
            self.model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

        self.preprocess = _transfor_ori(self.model.visual.image_size[0])
        self.preprocess_mod = _transfor_mod(self.model.visual.image_size[0])

        if output_dict and hasattr(self.model, 'output_dict'):
            self.model.output_dict = True

        if jit:
            self.model = torch.jit.script(self.model)

        if freeze_clip:
            self.freeze()

        self.model.to(self.device)

    def freeze(self):
        """Freeze the parameters of the clip."""
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def tokenize(texts, context_length=77, truncate=True):
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = _tokenizer.encoder['<|startoftext|>']
        eot_token = _tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse('1.8.0'):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f'Input {texts[i]} is too long for context length {context_length}')
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


    def encode_image(self, images, normalize=True, is_preprocess=False):
        if is_preprocess:
            images = self.preprocess(images)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)
        image_features = self.model.encode_image(images, normalize=normalize)
        return image_features


    def encode_text(self, text, normalize=True, is_tokenize=False):
        if is_tokenize:
            if not isinstance(text, list):
                text = [text]
            tokens = self.tokenize(text, context_length=self.max_length)
        else:
            tokens = text
        eot_ind = tokens.argmax(dim=-1)
        tokens = tokens.to(self.device)
        pooled_out, out = self.model.encode_text(tokens, normalize=normalize)
        return pooled_out, out, eot_ind


    def forward(self, text, image):
        results = self.model.forward(image=image, text=text)
        return results

    def clip_loss(self, text, image, real_image=None, img_clip_loss_weight=0.0):
        image = self.preprocess_mod(image)
        img_feats = self.encode_image(image, is_preprocess=False)
        txt_feats, _, _ = self.encode_text(text, is_tokenize=True)
        img_feats = torch.cat(GatherLayer.apply(img_feats), dim=0)
        txt_feats = torch.cat(GatherLayer.apply(txt_feats), dim=0)
        if real_image is not None:
            with torch.no_grad():
                real_image = self.preprocess_mod(real_image)
                real_img_feats = self.encode_image(real_image, is_preprocess=False)
                real_img_feats = concat_all_gather(real_img_feats)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale
        logit_scale = logit_scale.exp()
        logits_per_image = logit_scale * img_feats @ txt_feats.t()
        logits_per_text = logits_per_image.t()
        # logits_per_image = torch.diagonal(logits_per_image, 0)
        # similarity = 1 - logits_per_image / 100

        batch_size = img_feats.shape[0]
        labels = torch.arange(batch_size, device=img_feats.device).long()
        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2

        if real_image is not None:
            logits = logit_scale * img_feats @ real_img_feats.t()
            img_clip_loss = F.cross_entropy(logits, labels)
            total_loss = total_loss + img_clip_loss_weight * img_clip_loss
        return total_loss


    def clip_score(self, text, image):
        image = self.preprocess_mod(image)
        tokens = self.tokenize(text).to(self.device)
        results = self(text=tokens, image=image)
        img_features = results['image_features']
        txt_features = results['text_features_pooled']
        score = F.relu((img_features * txt_features).sum(axis=-1))
        return score

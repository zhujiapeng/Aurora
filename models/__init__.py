# python3.7
"""Collects all models."""

from .clip_model import CLIPModel
from .text_generator import Text2ImageGenerator

__all__ = ['build_model']

_MODELS = {
    'CLIPModel': CLIPModel,
    'Text2ImageGenerator': Text2ImageGenerator
}


def build_model(model_type, **kwargs):
    """Builds a model based on its class type.

    Args:
        model_type: Class type to which the model belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the model.

    Raises:
        ValueError: If the `model_type` is not supported.
    """
    if model_type not in _MODELS:
        raise ValueError(f'Invalid model type: `{model_type}`!\n'
                         f'Types allowed: {list(_MODELS)}.')
    return _MODELS[model_type](**kwargs)

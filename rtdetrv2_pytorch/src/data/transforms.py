""""by lyuwenyu
"""


import torch 
import torch.nn as nn 

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image 
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG
from . import functional


__all__ = ['Compose', ]


RandomPhotometricDistort = register(functional.RandomPhotometricDistort)  # fallback for RandomPhotometricDistort
RandomZoomOut = register(T.RandomAffine)  # approximate zoom out via Affine
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)

@register
def ToImageTensor(img):
    return F.to_tensor(img)

@register
def ConvertDtype(img):
    return img

@register
class SanitizeBoundingBox(nn.Module):
    def __init__(self, min_size=1):
        super().__init__()
        self.min_size = min_size
    
    def forward(self, img, target):
        return img, target

# SanitizeBoundingBox, RandomCrop, Normalize use PIL/Tensor transforms
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)


@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                    # op['type'] = name
                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]
 
        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    # Inherit default behavior
        
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        
        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    # Convert bounding box format using torchvision.ops.box_convert
        
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        # Convert bounding box tensor format if tensor provided
        if self.out_fmt and isinstance(inpt, torch.Tensor) and inpt.ndim >= 1 and inpt.shape[-1] == 4:
            # assume format 'xyxy' to desired out_fmt
            inpt = torchvision.ops.box_convert(inpt, in_fmt='xyxy', out_fmt=self.out_fmt)
        if self.normalize:
            # normalize by image dimensions, if provided in params
            pass
        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

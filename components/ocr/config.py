import torch
from PIL import Image
from .models import parseq
from torchvision import transforms as T

transforms = []
transforms.extend([
            T.Resize(parseq.hparams.img_size, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
img_transform = T.Compose(transforms)
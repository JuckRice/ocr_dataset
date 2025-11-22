from __future__ import annotations
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def resize_pad_to_patch(img: Image.Image, img_height: int, max_width: int, patch_size: int = 14) -> Image.Image:
    """
    Resize image to fixed height (keep aspect ratio), then pad right/bottom so H,W are multiples of patch_size.
    If width exceeds max_width after resizing, clip on the right (no sliding window is performed).
    """
    w, h = img.size
    if h != img_height:
        new_w = int(round(w * (img_height / h)))
        img = img.resize((new_w, img_height), Image.BICUBIC)

    w, h = img.size
    # Clamp to max_width
    if w > max_width:
        img = img.crop((0, 0, max_width, h))
        w = max_width

    # Pad so that H and W are divisible by patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    pad_h = (patch_size - (h % patch_size)) % patch_size
    if pad_w or pad_h:
        img = ImageOps.expand(img, (0, 0, pad_w, pad_h), fill=0)
    return img

def to_tensor_norm(img: Image.Image):
    x = TF.to_tensor(img)  # [C,H,W] in [0,1]
    x = TF.normalize(x, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    return x

def default_train_transform(img: Image.Image, img_height: int, max_width: int, patch_size: int = 14):
    img = img.convert("RGB")
    img = resize_pad_to_patch(img, img_height=img_height, max_width=max_width, patch_size=patch_size)
    return to_tensor_norm(img)

def default_val_transform(img: Image.Image, img_height: int, max_width: int, patch_size: int = 14):
    img = img.convert("RGB")
    img = resize_pad_to_patch(img, img_height=img_height, max_width=max_width, patch_size=patch_size)
    return to_tensor_norm(img)

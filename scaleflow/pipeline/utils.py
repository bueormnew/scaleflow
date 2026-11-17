"""
Utility functions for ScaleFlow pipeline: image conversions, grid creation, and saving.
"""

from pathlib import Path
from typing import List, Optional, Union
import math
import torch
from PIL import Image


def tensor_to_pil(
    tensor: torch.Tensor,
    value_range: str = "symmetric",
    squeeze: bool = True,
) -> Union[Image.Image, List[Image.Image]]:
    """
    Convert a PyTorch tensor (B, C, H, W) or (C, H, W) to PIL Image(s).

    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W).
        value_range: "symmetric" for [-1, 1] or "unit" for [0, 1].
        squeeze: If True, returns a single PIL Image when batch size is 1 or 3D tensor.
                 If False, always returns a list of PIL Images for 4D tensor.

    Returns:
        A single PIL Image if squeeze=True and batch size is 1 (or 3D tensor),
        or a list of PIL Images otherwise.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    t = tensor.detach().cpu().float()

    single_image = False
    if t.ndim == 3:
        # (C, H, W)
        t = t.unsqueeze(0)
        single_image = True
    elif t.ndim == 4:
        # (B, C, H, W)
        if squeeze and t.shape[0] == 1:
            single_image = True
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor.shape}")

    if value_range == "symmetric":
        # [-1, 1] -> [0, 1]
        t = (t + 1.0) / 2.0
    elif value_range == "unit":
        # [0, 1]
        pass
    else:
        raise ValueError(f"Unknown value_range: {value_range}. Use 'symmetric' or 'unit'.")

    # Clamp to [0, 1]
    t = torch.clamp(t, 0.0, 1.0)

    # Convert to uint8 (B, H, W, C)
    t = (t * 255.0).round().to(torch.uint8)
    t = t.permute(0, 2, 3, 1).numpy()

    images: List[Image.Image] = []
    for i in range(t.shape[0]):
        arr = t[i]
        if arr.shape[2] == 1:
            arr = arr.squeeze(2)
            images.append(Image.fromarray(arr, mode="L"))
        elif arr.shape[2] == 3:
            images.append(Image.fromarray(arr, mode="RGB"))
        elif arr.shape[2] == 4:
            images.append(Image.fromarray(arr, mode="RGBA"))
        else:
            raise ValueError(f"Unsupported number of channels: {arr.shape[2]}")

    if single_image:
        return images[0]
    return images


def pil_to_tensor(
    images: Union[Image.Image, List[Image.Image]],
    value_range: str = "symmetric",
) -> torch.Tensor:
    """
    Convert a PIL Image or list of PIL Images to a PyTorch tensor (B, C, H, W).

    Args:
        images: A PIL Image or list of PIL Images.
        value_range: "symmetric" for [-1, 1] or "unit" for [0, 1].

    Returns:
        Tensor of shape (B, C, H, W).
    """
    if isinstance(images, Image.Image):
        image_list = [images]
    elif isinstance(images, list):
        image_list = images
    else:
        raise TypeError(f"Expected PIL Image or list of PIL Images, got {type(images)}")

    tensors = []
    for img in image_list:
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Extract pixel data cleanly without deprecation warnings
        if hasattr(img, "get_flattened_data"):
            data = img.get_flattened_data()
            arr = torch.tensor(list(data), dtype=torch.float32).reshape(img.height, img.width, 3)
        else:
            try:
                import numpy as np
                arr = torch.from_numpy(np.array(img, dtype=np.float32))
            except ImportError:
                arr = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(img.height, img.width, 3)
        t = arr.permute(2, 0, 1) / 255.0  # in [0, 1]
        if value_range == "symmetric":
            t = t * 2.0 - 1.0  # in [-1, 1]
        tensors.append(t)

    return torch.stack(tensors, dim=0)


def make_image_grid(
    images: Union[List[Image.Image], torch.Tensor],
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    padding: int = 2,
    pad_value: int = 255,
) -> Image.Image:
    """
    Arrange a collection of images into a single grid PIL Image.

    Args:
        images: List of PIL Images, or 4D torch.Tensor.
        rows: Number of rows in grid (optional).
        cols: Number of columns in grid (optional).
        padding: Padding in pixels between images.
        pad_value: Background color for padding (0=black, 255=white).

    Returns:
        A combined PIL Image representing the grid.
    """
    if isinstance(images, torch.Tensor):
        pil_images = tensor_to_pil(images, squeeze=False)
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]
    else:
        pil_images = list(images)

    if not pil_images:
        raise ValueError("Cannot create grid from empty image list")

    n = len(pil_images)
    if cols is None and rows is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    elif cols is not None and rows is None:
        rows = math.ceil(n / cols)
    elif rows is not None and cols is None:
        cols = math.ceil(n / rows)

    w, h = pil_images[0].size
    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), color=(pad_value, pad_value, pad_value))

    for idx, img in enumerate(pil_images):
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.BILINEAR)
        r = idx // cols
        c = idx % cols
        x = padding + c * (w + padding)
        y = padding + r * (h + padding)
        grid.paste(img, (x, y))

    return grid


def save_image(
    image: Union[Image.Image, torch.Tensor, List[Image.Image]],
    path: Union[str, Path],
    format: Optional[str] = None,
) -> None:
    """
    Save an image or batch of images (as a grid) to the specified path.

    Args:
        image: PIL Image, torch.Tensor, or list of PIL Images.
        path: Target file path.
        format: Optional image format (e.g., 'PNG', 'JPEG'). Inferred from extension if None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, Image.Image):
        image.save(path, format=format)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 4 and image.shape[0] > 1:
            grid = make_image_grid(image)
            grid.save(path, format=format)
        else:
            pil_img = tensor_to_pil(image, squeeze=True)
            if isinstance(pil_img, list):
                pil_img = pil_img[0]
            pil_img.save(path, format=format)
    elif isinstance(image, list):
        if len(image) == 1:
            image[0].save(path, format=format)
        else:
            grid = make_image_grid(image)
            grid.save(path, format=format)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

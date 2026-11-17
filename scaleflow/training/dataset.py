"""Dataset and DataLoader utilities for ScaleFlow multiscale training.

Supports:
- Loading from an image directory with paired text files (.txt), or metadata files (json/jsonl/csv).
- Synthetic image-caption generation for testing and rapid development.
- Automatic multiscale image pyramid downsampling (8x8 ... 1024x1024).
- Collate functions and DataLoader creation helpers.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import csv
import json
import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image, ImageDraw

from scaleflow.training.flow_matching import DEFAULT_SCALE_RESOLUTIONS, build_image_pyramid


class ScaleFlowDataset(Dataset):
    """Text-Image dataset supporting image folders, captions files, and synthetic data.
    
    Can load:
    1. Paired files: folder with images (.png, .jpg, etc.) and matching .txt captions.
    2. Metadata file: json, jsonl, or csv containing mapping of image filenames to captions.
    3. Synthetic mode: on-the-fly generated images with procedural geometric captions.
    """

    def __init__(
        self,
        image_dir: Optional[Union[str, Path]] = None,
        captions_file: Optional[Union[str, Path]] = None,
        synthetic: bool = False,
        num_synthetic_samples: int = 100,
        image_size: int = 256,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Any] = None,
        max_seq_len: int = 64,
        return_pyramid: bool = False,
        scale_resolutions: Optional[Dict[int, int]] = None,
        seed: int = 42,
    ) -> None:
        """Initialize ScaleFlowDataset.
        
        Args:
            image_dir: Path to directory containing images.
            captions_file: Path to captions metadata file (json, jsonl, csv, or txt).
            synthetic: If True, generate synthetic image-text pairs.
            num_synthetic_samples: Number of samples when synthetic=True.
            image_size: Target square resolution (H=W).
            transform: Optional torchvision transform applied to PIL images.
            tokenizer: Optional tokenizer (e.g. BPETokenizer) to tokenize captions.
            max_seq_len: Maximum token sequence length if tokenizer is provided.
            return_pyramid: Whether __getitem__ returns the multi-resolution pyramid dict.
            scale_resolutions: Resolutions for pyramid if return_pyramid=True.
            seed: Random seed for synthetic dataset reproducibility.
        """
        super().__init__()
        self.image_dir = Path(image_dir) if image_dir else None
        self.captions_file = Path(captions_file) if captions_file else None
        self.synthetic = synthetic
        self.num_synthetic_samples = num_synthetic_samples
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.return_pyramid = return_pyramid
        self.scale_resolutions = scale_resolutions or DEFAULT_SCALE_RESOLUTIONS
        self.seed = seed

        # Default transform: Resize -> CenterCrop -> ToTensor -> Normalize to [-1, 1]
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        self.samples: List[Dict[str, Any]] = []

        if self.synthetic:
            self._init_synthetic_samples()
        else:
            self._init_disk_samples()

    def _init_synthetic_samples(self) -> None:
        """Pre-generate procedural synthetic metadata."""
        rng = random.Random(self.seed)
        colors = [
            ("red", (230, 50, 50)),
            ("blue", (50, 80, 230)),
            ("green", (50, 200, 70)),
            ("yellow", (240, 220, 40)),
            ("purple", (160, 60, 220)),
            ("orange", (240, 140, 30)),
            ("cyan", (30, 210, 210)),
            ("magenta", (220, 40, 180)),
        ]
        shapes = ["circle", "square", "cross", "diamond", "triangle"]
        backgrounds = [
            ("black", (10, 10, 10)),
            ("white", (245, 245, 245)),
            ("gray", (128, 128, 128)),
            ("dark blue", (15, 20, 50)),
        ]

        self.samples = []
        for i in range(self.num_synthetic_samples):
            color_name, color_rgb = rng.choice(colors)
            shape = rng.choice(shapes)
            bg_name, bg_rgb = rng.choice(backgrounds)

            caption = f"a {color_name} {shape} on a {bg_name} background"
            self.samples.append({
                "synthetic": True,
                "color_rgb": color_rgb,
                "shape": shape,
                "bg_rgb": bg_rgb,
                "caption": caption,
                "index": i,
            })

    def _generate_synthetic_image(self, item: Dict[str, Any]) -> Image.Image:
        """Generate a procedural PIL Image based on synthetic item description."""
        size = self.image_size
        img = Image.new("RGB", (size, size), color=item["bg_rgb"])
        draw = ImageDraw.Draw(img)

        color = item["color_rgb"]
        shape = item["shape"]
        margin = size // 4
        x0, y0 = margin, margin
        x1, y1 = size - margin, size - margin

        if shape == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=color)
        elif shape == "square":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        elif shape == "diamond":
            mid_x = size // 2
            mid_y = size // 2
            points = [(mid_x, y0), (x1, mid_y), (mid_x, y1), (x0, mid_y)]
            draw.polygon(points, fill=color)
        elif shape == "triangle":
            mid_x = size // 2
            points = [(mid_x, y0), (x1, y1), (x0, y1)]
            draw.polygon(points, fill=color)
        elif shape == "cross":
            thick = size // 8
            mid_x = size // 2
            mid_y = size // 2
            draw.rectangle([mid_x - thick // 2, y0, mid_x + thick // 2, y1], fill=color)
            draw.rectangle([x0, mid_y - thick // 2, x1, mid_y + thick // 2], fill=color)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=color)

        return img

    def _init_disk_samples(self) -> None:
        """Index disk image files and associated captions."""
        if self.captions_file and self.captions_file.exists():
            ext = self.captions_file.suffix.lower()
            if ext == ".json":
                with open(self.captions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Expect list of dicts [{"image": "path.jpg", "caption": "..."}] or dict {"path.jpg": "caption"}
                    if isinstance(data, list):
                        for entry in data:
                            img_path = self._resolve_image_path(entry["image"])
                            if img_path.exists():
                                self.samples.append({"image_path": img_path, "caption": entry.get("caption", "")})
                    elif isinstance(data, dict):
                        for img_name, cap in data.items():
                            img_path = self._resolve_image_path(img_name)
                            if img_path.exists():
                                self.samples.append({"image_path": img_path, "caption": cap})
            elif ext == ".jsonl":
                with open(self.captions_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            img_path = self._resolve_image_path(entry["image"])
                            if img_path.exists():
                                self.samples.append({"image_path": img_path, "caption": entry.get("caption", "")})
            elif ext == ".csv":
                with open(self.captions_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img_key = next((k for k in ("image", "image_path", "filename", "file") if k in row), None)
                        cap_key = next((k for k in ("caption", "text", "prompt") if k in row), None)
                        if img_key and cap_key:
                            img_path = self._resolve_image_path(row[img_key])
                            if img_path.exists():
                                self.samples.append({"image_path": img_path, "caption": row[cap_key]})
            elif ext == ".txt":
                # Line-by-line format: image_filename\tcaption
                with open(self.captions_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t", 1)
                        if len(parts) == 2:
                            img_path = self._resolve_image_path(parts[0])
                            if img_path.exists():
                                self.samples.append({"image_path": img_path, "caption": parts[1]})

        elif self.image_dir and self.image_dir.exists():
            # Scan directory for images and optional paired .txt files
            valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            for p in sorted(self.image_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in valid_exts:
                    txt_path = p.with_suffix(".txt")
                    caption = ""
                    if txt_path.exists():
                        try:
                            caption = txt_path.read_text(encoding="utf-8").strip()
                        except Exception:
                            caption = ""
                    self.samples.append({"image_path": p, "caption": caption})

    def _resolve_image_path(self, path_str: str) -> Path:
        """Resolve image path relative to image_dir or captions_file parent."""
        p = Path(path_str)
        if p.is_absolute() and p.exists():
            return p
        if self.image_dir and (self.image_dir / p).exists():
            return self.image_dir / p
        if self.captions_file and (self.captions_file.parent / p).exists():
            return self.captions_file.parent / p
        return self.image_dir / p if self.image_dir else p

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch item by index.
        
        Returns:
            Dictionary with keys:
            - 'image': Tensor of shape (3, H, W) normalized to [-1, 1].
            - 'caption': String caption.
            - (optional) 'input_ids': LongTensor of token IDs.
            - (optional) 'attention_mask': LongTensor attention mask.
            - (optional) 'pyramid': Dict of multi-scale tensors.
        """
        item = self.samples[idx]

        if item.get("synthetic", False):
            pil_img = self._generate_synthetic_image(item)
            caption = item["caption"]
        else:
            pil_img = Image.open(item["image_path"]).convert("RGB")
            caption = item.get("caption", "")

        image_tensor = self.transform(pil_img)

        result: Dict[str, Any] = {
            "image": image_tensor,
            "caption": caption,
        }

        if "input_ids" in item:
            result["input_ids"] = item["input_ids"]
        if "attention_mask" in item:
            result["attention_mask"] = item["attention_mask"]

        # Tokenize caption if tokenizer is provided
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "__call__"):
                enc = self.tokenizer(
                    [caption],
                    padding="max_length",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                result["input_ids"] = enc["input_ids"].squeeze(0)
                result["attention_mask"] = enc["attention_mask"].squeeze(0)
            elif hasattr(self.tokenizer, "encode"):
                ids = self.tokenizer.encode(caption, add_special_tokens=True)
                if len(ids) > self.max_seq_len:
                    ids = ids[:self.max_seq_len]
                mask = [1] * len(ids)
                # Pad to max_seq_len
                pad_id = getattr(self.tokenizer, "pad_token_id", 0)
                pad_len = self.max_seq_len - len(ids)
                if pad_len > 0:
                    ids = ids + [pad_id] * pad_len
                    mask = mask + [0] * pad_len
                result["input_ids"] = torch.tensor(ids, dtype=torch.long)
                result["attention_mask"] = torch.tensor(mask, dtype=torch.long)

        # Build multi-resolution pyramid if requested
        if self.return_pyramid:
            # Add batch dimension temporarily for pyramid generation
            pyramid = build_image_pyramid(
                image_tensor.unsqueeze(0),
                resolutions=self.scale_resolutions,
            )
            # Remove batch dimension for item
            result["pyramid"] = {s: t.squeeze(0) for s, t in pyramid.items()}

        return result


class SyntheticImageTextDataset(ScaleFlowDataset):
    """Convenience subclass for creating purely synthetic text-image datasets."""

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 256,
        tokenizer: Optional[Any] = None,
        max_seq_len: int = 64,
        return_pyramid: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(
            synthetic=True,
            num_synthetic_samples=num_samples,
            image_size=image_size,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            return_pyramid=return_pyramid,
            seed=seed,
        )


def collate_multiscale(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for ScaleFlow batches.
    
    Stacks images and token tensors, collects captions into a list of strings,
    and batches image pyramids if present.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    captions = [item["caption"] for item in batch]

    out: Dict[str, Any] = {
        "images": images,
        "captions": captions,
    }

    if "input_ids" in batch[0]:
        max_len = max(item["input_ids"].shape[-1] for item in batch)
        if all(item["input_ids"].shape[-1] == max_len for item in batch):
            out["input_ids"] = torch.stack([item["input_ids"] for item in batch], dim=0)
        else:
            pad_id = 0
            padded_list = []
            for item in batch:
                ids = item["input_ids"]
                if ids.shape[-1] < max_len:
                    pad = torch.full((max_len - ids.shape[-1],), pad_id, dtype=ids.dtype)
                    ids = torch.cat([ids, pad], dim=-1)
                padded_list.append(ids)
            out["input_ids"] = torch.stack(padded_list, dim=0)

    if "attention_mask" in batch[0]:
        max_len = max(item["attention_mask"].shape[-1] for item in batch)
        if all(item["attention_mask"].shape[-1] == max_len for item in batch):
            out["attention_mask"] = torch.stack([item["attention_mask"] for item in batch], dim=0)
        else:
            padded_list = []
            for item in batch:
                mask = item["attention_mask"]
                if mask.shape[-1] < max_len:
                    pad = torch.zeros(max_len - mask.shape[-1], dtype=mask.dtype)
                    mask = torch.cat([mask, pad], dim=-1)
                padded_list.append(mask)
            out["attention_mask"] = torch.stack(padded_list, dim=0)

    if "pyramid" in batch[0]:
        scales = list(batch[0]["pyramid"].keys())
        pyramids: Dict[int, torch.Tensor] = {}
        for s in scales:
            pyramids[s] = torch.stack([item["pyramid"][s] for item in batch], dim=0)
        out["pyramids"] = pyramids

    return out


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """Create a configured PyTorch DataLoader for ScaleFlow training or evaluation.
    
    Args:
        dataset: ScaleFlowDataset or other PyTorch Dataset.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle items.
        num_workers: Number of background worker processes.
        pin_memory: Whether to pin host memory.
        drop_last: Whether to drop the last incomplete batch.
        collate_fn: Collate function, defaults to collate_multiscale.
        
    Returns:
        Configured DataLoader instance.
    """
    if collate_fn is None:
        collate_fn = collate_multiscale

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

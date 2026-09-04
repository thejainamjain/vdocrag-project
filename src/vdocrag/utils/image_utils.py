"""
Image preprocessing helpers.

VDocRAG uses "dynamic high-resolution encoding" -- splitting a page image into
patches to preserve detail for varying aspect ratios rather than a single
naive resize (which distorts wide slides / long infographics). Qwen2-VL
handles dynamic resolution natively inside its processor, so most of the time
you won't need these helpers -- they're here for (a) Phi-3-vision baseline
compatibility and (b) any custom backbone you might plug in later.
"""

from typing import List, Tuple

from PIL import Image


def resize_keep_aspect(image: Image.Image, max_side: int = 1344) -> Image.Image:
    """Resize so the longer side == max_side, preserving aspect ratio."""
    w, h = image.size
    scale = max_side / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.BICUBIC)


def dynamic_crop(image: Image.Image, patch_size: int = 336, max_patches: int = 12) -> List[Image.Image]:
    """
    Simple dynamic cropping: splits a resized image into a grid of
    `patch_size` x `patch_size` tiles (with the last row/col possibly
    overlapping the edge), capped at `max_patches` tiles total.

    This is a simplified stand-in for the paper's dynamic high-resolution
    encoding -- good enough to demonstrate the idea; swap for the backbone's
    native image processor (Qwen2-VL does this internally) for production use.
    """
    w, h = image.size
    cols = max(1, w // patch_size)
    rows = max(1, h // patch_size)

    # cap total tiles
    while rows * cols > max_patches and (rows > 1 or cols > 1):
        if rows >= cols:
            rows -= 1
        else:
            cols -= 1

    tiles = []
    tile_w, tile_h = w // cols, h // rows
    for r in range(rows):
        for c in range(cols):
            box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
            tiles.append(image.crop(box))
    return tiles


def page_image_size(image_path: str) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size

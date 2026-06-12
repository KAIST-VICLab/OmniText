"""Character-width tables and character-aligned mask generation.

Extracted verbatim from the original OmniText notebooks. Provides the per-character
width weighting used to distribute target text across the editing region, plus the
mask-generation and polygon-shrinking helpers.
"""
import numpy as np
import torch


letter_number_width_weights = {
    "narrow": 1.0,
    "medium_narrow": 1.2,
    "standard_medium": 1.5,
    "slightly_wider": 1.8,
    "wide": 2.0,
    "extra_wide": 2.5
}

# Character-to-width mapping
char_width_dict = {}
for category, weight in letter_number_width_weights.items():
    for char in "IJijl1":
        char_width_dict[char] = letter_number_width_weights["narrow"]
    for char in "ftrFTR37":
        char_width_dict[char] = letter_number_width_weights["medium_narrow"]
    for char in "acegosuvxyzACEGLOSUVXYZ25":
        char_width_dict[char] = letter_number_width_weights["standard_medium"]
    for char in "bdhknpqBDHKNPQ469":
        char_width_dict[char] = letter_number_width_weights["slightly_wider"]
    for char in "mwMW08":
        char_width_dict[char] = letter_number_width_weights["wide"]
    for char in "MW08":
        char_width_dict[char] = letter_number_width_weights["extra_wide"]


def generate_char_masks_weighted(
    binary_mask: torch.Tensor,
    orig_len: int,
    target_text: str,
    char_width_dict: dict,
    default_width: float = 1.5
) -> list[torch.Tensor]:
    """
    Generate weighted character-aligned binary masks for the target text
    based on character width proportions.
    """
    H, W = binary_mask.shape
    target_len = len(target_text)

    # Step 1: Get bounding box of the original mask
    cols = binary_mask.any(dim=0)
    rows = binary_mask.any(dim=1)

    col_indices = torch.where(cols)[0]
    row_indices = torch.where(rows)[0]

    if len(col_indices) == 0 or len(row_indices) == 0:
        return [torch.zeros_like(binary_mask) for _ in range(target_len)]

    left, right = col_indices[0].item(), col_indices[-1].item() + 1
    top, bottom = row_indices[0].item(), row_indices[-1].item() + 1

    full_width = right - left

    # Step 2: Shrink bounding box if target is shorter
    if target_len < orig_len:
        scale = target_len / orig_len
        new_width = max(1, int(round(full_width * scale)))
        center = (left + right) // 2
        new_left = max(0, center - new_width // 2)
        new_right = min(W, new_left + new_width)
    else:
        new_left = left
        new_right = right

    box_width = new_right - new_left

    # Step 3: Convert target text into relative width weights
    weights = [char_width_dict.get(c, default_width) for c in target_text]
    total_weight = sum(weights)
    relative_widths = [w / total_weight for w in weights]
    pixel_widths = [w * box_width for w in relative_widths]

    # Step 4: Compute integer start/end column indices
    char_masks = []
    current_x = new_left

    for i, px_width in enumerate(pixel_widths):
        width = max(1, int(round(px_width)))
        col_start = int(current_x)
        col_end = int(min(W, col_start + width))

        # Safety clamp
        if col_end <= col_start:
            col_end = min(W, col_start + 1)

        char_mask = torch.zeros_like(binary_mask)
        char_mask[top:bottom, col_start:col_end] = binary_mask[top:bottom, col_start:col_end]

        # 🔥 Fix: if the resulting char_mask is all zero, steal from previous or next
        if char_mask.sum() == 0:
            if i > 0 and char_masks[i - 1].sum() > 0:
                char_mask = char_masks[i - 1].clone()
            elif i < target_len - 1:
                fallback_start = max(0, col_start - 1)
                fallback_end = min(W, col_end + 1)
                char_mask[top:bottom, fallback_start:fallback_end] = binary_mask[top:bottom, fallback_start:fallback_end]

        char_masks.append(char_mask)
        current_x = col_end

    return char_masks

def generate_char_masks(binary_mask: torch.Tensor, orig_len: int, target_len: int) -> list[torch.Tensor]:
    """
    Generate character-aligned binary masks for the target text by slicing the original
    bounding box evenly — shrinking the total width if target text is shorter.
    Ensures all output masks are non-zero.
    """
    H, W = binary_mask.shape

    # Get bounding box of the original mask
    cols = binary_mask.any(dim=0)
    rows = binary_mask.any(dim=1)

    col_indices = torch.where(cols)[0]
    row_indices = torch.where(rows)[0]

    if len(col_indices) == 0 or len(row_indices) == 0:
        # Fallback: all-zero (shouldn’t happen in your setup)
        return [torch.zeros_like(binary_mask) for _ in range(target_len)]

    left, right = col_indices[0].item(), col_indices[-1].item() + 1
    top, bottom = row_indices[0].item(), row_indices[-1].item() + 1

    full_width = right - left
    full_height = bottom - top

    # Shrink bounding box if target is shorter
    if target_len < orig_len:
        scale = target_len / orig_len
        new_width = max(1, int(round(full_width * scale)))
        center = (left + right) // 2
        new_left = max(0, center - new_width // 2)
        new_right = min(W, new_left + new_width)
    else:
        new_left = left
        new_right = right

    box_width = new_right - new_left
    char_width = box_width / target_len

    char_masks = []
    for i in range(target_len):
        start = int(round(i * char_width))
        end = int(round((i + 1) * char_width))

        # Ensure min width = 1 column
        if end == start:
            end = start + 1
        if end > box_width:
            end = box_width

        char_mask = torch.zeros_like(binary_mask)
        col_start = new_left + start
        col_end = new_left + end

        col_start = max(0, min(col_start, W - 1))
        col_end = max(col_start + 1, min(col_end, W))

        char_mask[top:bottom, col_start:col_end] = binary_mask[top:bottom, col_start:col_end]

        # 🔥 Fix: if the resulting char_mask is all zero, steal from previous or next
        if char_mask.sum() == 0:
            if i > 0 and char_masks[i - 1].sum() > 0:
                char_mask = char_masks[i - 1].clone()
            elif i < target_len - 1:
                # Look ahead at original binary mask area
                fallback_start = max(0, col_start - 1)
                fallback_end = min(W, col_end + 1)
                char_mask[top:bottom, fallback_start:fallback_end] = binary_mask[top:bottom, fallback_start:fallback_end]

        char_masks.append(char_mask)

    return char_masks



def calculate_text_weight(text, char_width_dict, default_weight=1.5):
    return sum(char_width_dict.get(c, default_weight) for c in text)

def shrink_polygon_width_only(polygon, source_text, target_text, char_width_dict):
    """
    Shrink the polygon horizontally based on text width ratio.
    polygon: list of 8 values [x0, y0, x1, y1, x2, y2, x3, y3]
    """
    # Step 1: Text weight calculation
    source_weight = calculate_text_weight(source_text, char_width_dict)
    target_weight = calculate_text_weight(target_text, char_width_dict)
    ratio = target_weight / source_weight  # < 1 if target shorter

    # Step 2: Reshape points
    points = np.array(polygon, dtype=np.float32).reshape(4, 2)  # [4, 2]

    # Step 3: Compute width
    x_left = (points[0, 0] + points[3, 0]) / 2  # average x of left side
    x_right = (points[1, 0] + points[2, 0]) / 2  # average x of right side
    width = x_right - x_left

    # Step 4: Compute new width
    new_width = width * ratio
    width_diff = (width - new_width) / 2  # equally shrink both sides

    # Step 5: Move points inward horizontally
    new_points = points.copy()
    for i in [0, 3]:  # left side points
        new_points[i, 0] += width_diff
    for i in [1, 2]:  # right side points
        new_points[i, 0] -= width_diff

    return new_points.flatten().astype(int).tolist()


def gaussian_attention_target(indices, size, sigma=1.0, device=None):
    grid = torch.arange(size, device=device)
    target = []
    for idx in indices:
        target.append(torch.exp(-0.5 * ((grid - idx).float() / sigma) ** 2))
    target = torch.stack(target, dim=0)  # [N, size]
    target = target / (target.sum(-1, keepdim=True) + 1e-8)  # normalize
    return target  # [N, size]

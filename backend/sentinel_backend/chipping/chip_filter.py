"""Per-chip quality filtering: cloud coverage and variance thresholds."""
import numpy as np


def compute_chip_stats(
    chip_array: np.ndarray,
    cloud_mask: np.ndarray | None,
    col_off: int,
    row_off: int,
) -> dict:
    """Compute cloud_pct and variance for one chip.

    Args:
        chip_array (np.ndarray): Shape (H, W, 3), dtype uint8.
        cloud_mask (np.ndarray or None): Full-image bool (H, W) mask; True = cloud.
            When None, cloud_pct = 0.0.
        col_off (int): Top-left pixel column offset of the chip in the source image.
        row_off (int): Top-left pixel row offset of the chip in the source image.

    Returns:
        dict: Dictionary with keys cloud_pct (float 0-1) and variance (float).
    """
    # Use the shape of the array passed in as the denominator.
    # When called from apply_chip_filters, chip_array always has the padded
    # nominal size (get_chip zero-pads edge chips), so the black padding is
    # included in the variance calculation.  cloud_pct is computed from the
    # mask slice, whose size equals the actual pixel region regardless of padding.
    h, w = chip_array.shape[:2]
    if cloud_mask is not None:
        patch = cloud_mask[row_off:row_off + h, col_off:col_off + w]
        cloud_pct = float(patch.sum()) / patch.size if patch.size > 0 else 0.0
    else:
        cloud_pct = 0.0
    variance = float(chip_array.astype(np.float32).var())
    return {"cloud_pct": cloud_pct, "variance": variance}


def apply_chip_filters(
    grid,
    cloud_mask: np.ndarray | None,
    cloud_thresh: float = 0.3,
    variance_thresh: float = 100.0,
    enable_cloud_filter: bool = True,
    enable_variance_filter: bool = True,
) -> tuple[list[int], list[int], list[dict]]:
    """Evaluate quality filters across all chips.

    Args:
        grid (ChipGrid): The chip grid to filter.
        cloud_mask (np.ndarray or None): Full-image bool cloud mask.
        cloud_thresh (float): Maximum allowed cloud fraction (0-1).
        variance_thresh (float): Minimum required pixel variance.
        enable_cloud_filter (bool): Whether to apply the cloud fraction filter.
        enable_variance_filter (bool): Whether to apply the variance filter.

    Returns:
        accepted_indices (list[int]): Indices of chips that passed all enabled filters.
        rejected_indices (list[int]): Indices of chips that failed at least one filter.
        chip_stats (list[dict]): One dict per chip with keys chip_index, cloud_pct,
            variance, rejected.
    """
    from sentinel_backend.chipping.gdal_chipper import get_chip
    accepted, rejected, stats = [], [], []
    for idx in range(grid.total):
        chip_arr, _ = get_chip(grid, idx)
        col_off, row_off, _, _ = grid.windows[idx]
        s = compute_chip_stats(chip_arr, cloud_mask, col_off, row_off)
        reject = False
        if enable_cloud_filter and s["cloud_pct"] > cloud_thresh:
            reject = True
        if enable_variance_filter and s["variance"] < variance_thresh:
            reject = True
        s["chip_index"] = idx
        s["rejected"] = reject
        stats.append(s)
        (rejected if reject else accepted).append(idx)
    return accepted, rejected, stats

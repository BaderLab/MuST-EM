"""
MorphDescriptor: 3D Morphological Feature Extraction for Organelle Instances.

This module provides tools for extracting quantitative morphological features
from 3D instance-segmented electron microscopy (EM) volumes. It supports:
  - Loading and normalising 3D EM volumes
  - Instance segmentation via watershed
  - Cross-slice instance tracking to build coherent 3D labels
  - Z-span filtering of small / spurious instances
  - PyRadiomics-based shape feature extraction with parallel processing

Designed for large-scale subcellular morphometry (e.g. mitochondria) in
serial-section EM datasets.

Usage
-----
    python morphology_features.py segment --input-dir <masks> --output-dir <out> [options]
    python morphology_features.py track   --input-dir <labelled> --output-dir <out> [options]
    python morphology_features.py extract --instance-dir <tracked> --em-dir <em> --output-csv <csv> [options]

Run ``python morphology_features.py <subcommand> --help`` for the full CLI of
each stage.

References
----------
van Griethuysen, J.J.M. et al. Computational Radiomics System to Decode the
Radiographic Phenotype. *Cancer Research* 77(21), e104–e107 (2017).
"""

import os
import csv
import gc
import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tifffile
from joblib import Parallel, delayed
from numpy.polynomial.polynomial import Polynomial
from PIL import Image
from radiomics.shape import RadiomicsShape
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    zoom,
)
from scipy.spatial.distance import cdist
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_opening,
    disk,
    h_maxima,
    remove_small_objects,
)
from skimage.segmentation import watershed
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MorphDescriptor:
    """Extract radiomics-based morphological features from a 3D labelled volume.

    Parameters
    ----------
    labelled_volume : np.ndarray
        3D integer array where each unique non-zero value is an instance label.
    em_folder : str
        Path to a directory of 2D EM slices (PNG) that correspond to the
        labelled volume.
    z_interpolation_factor : float, optional
        Factor by which to interpolate the EM stack along the z-axis so that
        voxel dimensions become approximately isotropic (default 1, i.e. no
        interpolation).
    """

    def __init__(
        self,
        labelled_volume: np.ndarray,
        em_folder: str,
        z_interpolation_factor: float = 1,
    ):
        self.labelled_volume = labelled_volume
        self.regions = regionprops(self.labelled_volume)
        logger.info("Number of labelled regions: %d", len(self.regions))

        self.em = self._load_and_normalise_em(em_folder, z_interpolation_factor)

    # -- EM loading & normalisation -----------------------------------------

    @staticmethod
    def _load_and_normalise_em(
        folder: str, z_interpolation_factor: float
    ) -> np.ndarray:
        """Load a stack of 2D EM slices, interpolate along z, and normalise.

        Intensity normalisation removes slow axial intensity drift by fitting
        a quadratic polynomial to per-slice mean intensity and subtracting the
        trend.

        Parameters
        ----------
        folder : str
            Directory containing PNG slices in alphanumerical order.
        z_interpolation_factor : float
            Zoom factor along the z-axis (1 = no change).

        Returns
        -------
        np.ndarray
            Normalised 3D EM volume.
        """
        png_files = sorted(
            (f for f in os.listdir(folder) if f.endswith(".png")),
            key=_extract_index,
        )
        if not png_files:
            raise FileNotFoundError(f"No PNG files found in {folder}")

        em_3d = np.stack(
            [imread(os.path.join(folder, f)) for f in png_files], axis=0
        )
        logger.info("Raw EM stack shape: %s", em_3d.shape)

        # Z-axis interpolation
        if z_interpolation_factor != 1:
            em_3d = zoom(em_3d, (z_interpolation_factor, 1, 1), order=1)
            logger.info(
                "Interpolated EM stack shape: %s", em_3d.shape
            )

        # Polynomial intensity normalisation along z
        z_indices = np.arange(em_3d.shape[0])
        mean_intensities = np.array(
            [em_3d[z].mean() for z in z_indices]
        )
        trend = Polynomial.fit(z_indices, mean_intensities, deg=2)(z_indices)
        normalised = np.empty_like(em_3d, dtype=np.float32)
        for z in range(em_3d.shape[0]):
            normalised[z] = em_3d[z].astype(np.float32) - trend[z]

        logger.info("Normalised EM stack shape: %s", normalised.shape)
        return normalised

    # -- Per-region feature extraction --------------------------------------

    @staticmethod
    def _extract_region_features(
        coords: np.ndarray,
        centroid: Tuple[float, float, float],
        region_idx: int,
        em: np.ndarray,
    ) -> Tuple[Dict, int]:
        """Compute PyRadiomics shape features for a single labelled region.

        Parameters
        ----------
        coords : np.ndarray
            Voxel coordinates of the region (N × 3; columns = z, y, x).
        centroid : tuple of float
            Region centroid (z, y, x) from ``regionprops``.
        region_idx : int
            Index used to identify this region in the output.
        em : np.ndarray
            Full 3D EM volume (used as the image input for PyRadiomics).

        Returns
        -------
        features : dict
            Shape feature values plus centroid and bounding-box metadata.
        region_idx : int
            Passthrough of the input index.
        """
        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
        z_min, z_max = int(z.min()), int(z.max())
        y_min, y_max = int(y.min()), int(y.max())
        x_min, x_max = int(x.min()), int(x.max())

        # Crop the EM sub-volume and build a binary mask
        em_crop = em[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
        mask_crop = np.zeros(em_crop.shape, dtype=np.uint8)
        mask_crop[z - z_min, y - y_min, x - x_min] = 1

        # PyRadiomics shape extraction
        em_sitk = sitk.GetImageFromArray(em_crop.astype(np.float32))
        mask_sitk = sitk.GetImageFromArray(mask_crop)
        shape_extractor = RadiomicsShape(em_sitk, mask_sitk)
        shape_extractor.enableAllFeatures()
        shape_features = shape_extractor.execute()

        features: Dict = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in shape_features.items()
        }

        # Centroid in the labelled-volume coordinate space
        features["centroid_z"] = float(centroid[0])
        features["centroid_y"] = float(centroid[1])
        features["centroid_x"] = float(centroid[2])

        # Bounding box in the labelled-volume coordinate space
        features["bbox_z_min"] = z_min
        features["bbox_z_max"] = z_max
        features["bbox_y_min"] = y_min
        features["bbox_y_max"] = y_max
        features["bbox_x_min"] = x_min
        features["bbox_x_max"] = x_max

        # Voxel count
        features["voxel_count"] = len(coords)

        logger.debug(
            "Region %d: centroid=(%.1f, %.1f, %.1f), "
            "bbox=(%dx%dx%d), n_features=%d",
            region_idx,
            features["centroid_x"],
            features["centroid_y"],
            features["centroid_z"],
            x_max - x_min + 1,
            y_max - y_min + 1,
            z_max - z_min + 1,
            len(shape_features),
        )
        return features, region_idx

    # -- Batch feature extraction -------------------------------------------

    def extract_features(
        self,
        xy_scale_factor: float = None,
        n_jobs: int = 16,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract shape features for every region in the labelled volume.

        Features are computed in parallel.  After extraction, centroid and
        bounding-box columns are additionally scaled to full-resolution
        coordinates using *xy_scale_factor* (the z-axis is not scaled).

        Parameters
        ----------
        xy_scale_factor : float, optional
            Ratio of full-resolution to working-resolution pixel size in XY
        n_jobs : int, optional
            Number of parallel workers (default 16).

        Returns
        -------
        features_df : pd.DataFrame
            One row per region.  Contains all PyRadiomics shape features plus
            centroid / bounding-box columns in both coordinate spaces.
        index_df : pd.DataFrame
            Region indices (useful for joining with other tables).
        """
        regions_data = [
            (r.coords, r.centroid, idx) for idx, r in enumerate(self.regions)
        ]

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._extract_region_features)(coords, centroid, idx, self.em)
            for coords, centroid, idx in regions_data
        )

        feature_dicts, indices = zip(*results)
        features_df = pd.DataFrame(feature_dicts)
        index_df = pd.DataFrame(indices, columns=["region_idx"])

        # Scale to full-resolution coordinates
        s = xy_scale_factor
        features_df["centroid_x_full"] = features_df["centroid_x"] * s
        features_df["centroid_y_full"] = features_df["centroid_y"] * s
        features_df["centroid_z_full"] = features_df["centroid_z"]
        for ax in ("x", "y"):
            for bound in ("min", "max"):
                col = f"bbox_{ax}_{bound}"
                if col in features_df.columns:
                    features_df[f"{col}_full"] = features_df[col] * s
        for bound in ("min", "max"):
            col = f"bbox_z_{bound}"
            if col in features_df.columns:
                features_df[f"{col}_full"] = features_df[col]

        logger.info(
            "Extracted %d features for %d regions.",
            features_df.shape[1],
            features_df.shape[0],
        )
        return features_df, index_df


# ---------------------------------------------------------------------------
# Instance segmentation helpers (slice-level)
# ---------------------------------------------------------------------------

def segment_slice(
    file_path: str,
    output_dir: str,
    opening_radius: int = 0,
    min_size: int = 0,
    h_max_threshold: float = 0.0,
    gaussian_sigma: float = 0.0,
    distance_sigma: float = 0.0,
) -> None:
    """Watershed-based instance segmentation of a single 2D binary mask.

    The pipeline: binarise → remove small objects → fill holes → (optional)
    morphological opening → Gaussian smoothing → distance transform →
    h-maxima suppression → watershed.

    All keyword arguments default to a no-op value (0), i.e. no opening, no
    size filtering, no smoothing, and no h-maxima suppression (every local
    maximum becomes a marker). In practice, some suppression/smoothing is
    usually needed to avoid oversegmentation — tune these for your dataset.

    Parameters
    ----------
    file_path : str
        Path to the input PNG mask (non-zero pixels = foreground).
    output_dir : str
        Directory where the labelled TIFF will be written.
    opening_radius : int, optional
        Radius for morphological opening (0 = skip, default).
    min_size : int, optional
        Minimum connected-component area in pixels to retain (0 = no
        filtering, default).
    h_max_threshold : float, optional
        Height parameter for ``h_maxima`` peak suppression (0 = no
        suppression, default; typically needs a positive value to avoid
        oversegmentation).
    gaussian_sigma : float, optional
        Sigma for Gaussian smoothing of the binary mask.
    distance_sigma : float, optional
        Sigma for Gaussian smoothing of the distance transform.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir, os.path.basename(file_path).replace(".png", ".tif")
    )
    if os.path.exists(out_path):
        return

    binary = imread(file_path) > 0
    binary = remove_small_objects(binary, min_size=min_size)
    filled = binary_fill_holes(binary)
    del binary

    if opening_radius > 0:
        filled = binary_opening(filled, disk(opening_radius))

    smoothed = gaussian_filter(filled.astype(np.float32), sigma=gaussian_sigma)
    del filled

    dist = distance_transform_edt(smoothed > 0.5)
    dist = gaussian_filter(dist, sigma=distance_sigma)
    markers = label(h_maxima(dist, h=h_max_threshold))
    labelled = label(
        watershed(-dist, markers, mask=(smoothed > 0.5)), connectivity=2
    )

    tifffile.imwrite(out_path, labelled.astype(np.uint16))
    del smoothed, dist, markers, labelled
    gc.collect()


# ---------------------------------------------------------------------------
# Cross-slice tracking
# ---------------------------------------------------------------------------

def _match_regions(
    prev_props: list,
    curr_props: list,
    dist_thresh: float,
    iou_thresh: float = None,
) -> Dict[int, int]:
    """Match instances between two consecutive slices.

    Matching is based on a combined score of centroid distance and
    intersection-over-union (IoU) of pixel coordinates.

    Parameters
    ----------
    prev_props : list
        ``regionprops`` of the previous slice.
    curr_props : list
        ``regionprops`` of the current slice.
    dist_thresh : float
        Maximum centroid distance (pixels) to consider a match.
    iou_thresh : float, optional
        Minimum IoU to accept a match (default 0.1).

    Returns
    -------
    dict
        Mapping from *current-slice* label → *previous-slice* label.
    """
    if not prev_props or not curr_props:
        return {}

    prev_centroids = np.array([r.centroid for r in prev_props])
    curr_centroids = np.array([r.centroid for r in curr_props])
    dists = cdist(curr_centroids, prev_centroids)

    matches: Dict[int, int] = {}
    used_prev: set = set()

    for i, curr in enumerate(curr_props):
        best_score, best_j = 0.0, None
        for j, prev in enumerate(prev_props):
            if dists[i, j] > dist_thresh:
                continue

            # Quick bounding-box overlap check
            bb_min = np.maximum(curr.bbox[:2], prev.bbox[:2])
            bb_max = np.minimum(curr.bbox[2:], prev.bbox[2:])
            if np.any(bb_min >= bb_max):
                continue

            curr_set = set(map(tuple, curr.coords))
            prev_set = set(map(tuple, prev.coords))
            intersection = len(curr_set & prev_set)
            union = len(curr_set | prev_set)
            iou = intersection / union if union > 0 else 0.0

            score = iou + 1.0 / (1.0 + dists[i, j])
            if iou > iou_thresh and score > best_score and j not in used_prev:
                best_score = score
                best_j = j

        if best_j is not None:
            matches[curr.label] = prev_props[best_j].label
            used_prev.add(best_j)

    return matches


def track_instances_across_slices(
    labelled_slices_dir: str,
    output_dir: str,
    downsampled_dir: Optional[str] = None,
    full_shape: Tuple[int, int] = (None, None),
    downsample_shape: Tuple[int, int] = (None, None),
    dist_thresh: float = None,
    iou_thresh: float = None,
) -> int:
    """Assign consistent instance IDs across a stack of 2D labelled slices.

    Each slice is loaded sequentially.  Instances in the current slice are
    matched to the previous slice using centroid distance + IoU.  Unmatched
    instances receive a new unique ID.

    Parameters
    ----------
    labelled_slices_dir : str
        Directory of per-slice instance-labelled TIFFs.
    output_dir : str
        Directory for full-resolution tracked label TIFFs.
    downsampled_dir : str, optional
        If given, also write downsampled label maps here.
    full_shape : tuple of int
        (H, W) of the full-resolution label maps.
    downsample_shape : tuple of int
        (H, W) for the downsampled copies.
    dist_thresh : float
        Maximum centroid distance for matching.
    iou_thresh : float
        Minimum IoU for matching.

    Returns
    -------
    int
        Total number of unique 3D instance IDs assigned.
    """
    os.makedirs(output_dir, exist_ok=True)
    if downsampled_dir is not None:
        os.makedirs(downsampled_dir, exist_ok=True)

    paths = sorted(
        [
            os.path.join(labelled_slices_dir, f)
            for f in os.listdir(labelled_slices_dir)
            if f.endswith(".tif")
        ],
        key=lambda p: _extract_index(p),
    )
    if not paths:
        raise FileNotFoundError(
            f"No TIF files found in {labelled_slices_dir}"
        )

    H, W = full_shape
    next_id = 1
    id_map: Dict[int, int] = {}

    # Initialise with the first slice
    first_labels = tifffile.imread(paths[0]).astype(np.int32)
    first_out = np.zeros((H, W), dtype=np.int32)
    prev_props = regionprops(first_labels)
    for region in prev_props:
        first_out[tuple(region.coords.T)] = next_id
        id_map[region.label] = next_id
        next_id += 1
    _save_label_slice(first_out, paths[0], output_dir, downsampled_dir, downsample_shape)

    # Sequential tracking
    for z in tqdm(range(1, len(paths)), desc="Tracking across slices"):
        curr_labels = tifffile.imread(paths[z]).astype(np.int32)
        curr_props = regionprops(curr_labels)
        matches = _match_regions(prev_props, curr_props, dist_thresh, iou_thresh)

        out_slice = np.zeros((H, W), dtype=np.int32)
        new_id_map: Dict[int, int] = {}
        for region in curr_props:
            if region.label in matches:
                assigned = id_map[matches[region.label]]
            else:
                assigned = next_id
                next_id += 1
            out_slice[tuple(region.coords.T)] = assigned
            new_id_map[region.label] = assigned

        _save_label_slice(out_slice, paths[z], output_dir, downsampled_dir, downsample_shape)
        prev_props = curr_props
        id_map = new_id_map

    total_ids = next_id - 1
    logger.info("Total tracked 3D instances: %d", total_ids)
    return total_ids


# ---------------------------------------------------------------------------
# Z-span filtering
# ---------------------------------------------------------------------------

def filter_by_z_span(
    volume: np.ndarray,
    z_threshold: int = None,
    id_map_csv: Optional[str] = None,
) -> np.ndarray:
    """Remove instances that span fewer than *z_threshold* z-slices.

    Parameters
    ----------
    volume : np.ndarray
        3D instance-labelled volume.
    z_threshold : int, optional
        Minimum number of distinct z-slices an instance must occupy to be
        retained (default 6).
    id_map_csv : str, optional
        If provided, write a CSV mapping original IDs to new filtered IDs.

    Returns
    -------
    np.ndarray
        Filtered volume with re-numbered instance labels.
    """
    filtered = np.zeros_like(volume)
    regions = regionprops(volume)
    new_id = 1
    id_mapping: Dict[int, int] = {}

    for region in regions:
        z_span = len(np.unique(region.coords[:, 0]))
        if z_span >= z_threshold:
            filtered[tuple(region.coords.T)] = new_id
            id_mapping[region.label] = new_id
            new_id += 1

    logger.info(
        "Retained %d / %d instances (z_threshold=%d).",
        new_id - 1,
        len(regions),
        z_threshold,
    )

    if id_map_csv is not None:
        os.makedirs(os.path.dirname(id_map_csv), exist_ok=True)
        with open(id_map_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["original_id", "filtered_id"])
            for old, new in id_mapping.items():
                writer.writerow([old, new])
        logger.info("ID mapping saved to %s", id_map_csv)

    return filtered


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_instance_volume(folder: str) -> np.ndarray:
    """Load a stack of 2D instance-labelled TIFFs into a 3D array.

    Parameters
    ----------
    folder : str
        Directory of TIFF slices in alphanumerical order.

    Returns
    -------
    np.ndarray
        3D integer array (Z × H × W).
    """
    tif_files = sorted(
        (f for f in os.listdir(folder) if f.endswith(".tif")),
        key=_extract_index,
    )
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {folder}")

    slices = [tifffile.imread(os.path.join(folder, f)) for f in tif_files]
    volume = np.stack(slices, axis=0)
    logger.info("Loaded instance volume: shape=%s", volume.shape)
    return volume


def _save_label_slice(
    label_slice: np.ndarray,
    source_path: str,
    output_dir: str,
    downsampled_dir: Optional[str],
    downsample_shape: Tuple[int, int],
) -> None:
    """Write a label slice (and optionally a downsampled copy) to disk."""
    fname = os.path.basename(source_path)
    tifffile.imwrite(
        os.path.join(output_dir, fname), label_slice.astype(np.uint32)
    )
    if downsampled_dir is not None:
        ds = cv2.resize(
            label_slice, downsample_shape[::-1], interpolation=cv2.INTER_NEAREST
        )
        tifffile.imwrite(
            os.path.join(downsampled_dir, fname), ds.astype(np.uint32)
        )


def _extract_index(file_path: str) -> int:
    """Parse a trailing integer index from a filename (e.g. 'mask_20.tif' → 20)."""
    base = os.path.basename(file_path)
    return int(base.rsplit("_", 1)[1].split(".")[0])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def segment_directory(
    input_dir: str,
    output_dir: str,
    opening_radius: int,
    min_size: int,
    h_max_threshold: float,
    gaussian_sigma: float,
    distance_sigma: float,
) -> None:
    """Run :func:`segment_slice` (watershed instance segmentation) on every
    PNG mask in *input_dir*, writing one labelled TIFF per slice to
    *output_dir*.
    """
    png_files = sorted(
        (f for f in os.listdir(input_dir) if f.endswith(".png")),
        key=_extract_index,
    )
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    for f in tqdm(png_files, desc="Segmenting slices"):
        segment_slice(
            os.path.join(input_dir, f),
            output_dir,
            opening_radius=opening_radius,
            min_size=min_size,
            h_max_threshold=h_max_threshold,
            gaussian_sigma=gaussian_sigma,
            distance_sigma=distance_sigma,
        )


def main() -> None:
    """CLI for the full pipeline: per-slice watershed segmentation, cross-slice
    instance tracking, z-span filtering, and morphological feature extraction.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="3D morphological feature extraction from EM volumes: "
                     "watershed segmentation, cross-slice tracking, and "
                     "PyRadiomics-based feature extraction."
    )
    parser.add_argument(
        "--log-file", default=None, help="Optional log file path."
    )
    sub = parser.add_subparsers(dest="command", required=True,
                                 help="Pipeline stage to run.")

    # --- segment: per-slice watershed instance segmentation ---
    sg = sub.add_parser(
        "segment",
        help="Watershed-based instance segmentation of 2D binary masks.",
    )
    sg.add_argument("--input-dir", required=True,
                     help="Directory of binary mask PNGs (one per slice).")
    sg.add_argument("--output-dir", required=True,
                     help="Directory for labelled TIFF output.")
    sg.add_argument("--opening-radius", type=int, default=0,
                     help="Morphological opening radius in pixels "
                          "(default: 0, i.e. skip). Dataset-specific.")
    sg.add_argument("--min-size", type=int, default=0,
                     help="Minimum connected-component area (pixels) to "
                          "retain (default: 0, i.e. no filtering). "
                          "Dataset-specific.")
    sg.add_argument("--h-max-threshold", type=float, default=0.0,
                     help="Height parameter for h-maxima peak suppression "
                          "(default: 0, i.e. no suppression). Usually needs "
                          "a positive value to avoid oversegmentation — "
                          "dataset-specific.")
    sg.add_argument("--gaussian-sigma", type=float, default=0.0,
                     help="Sigma for Gaussian smoothing of the binary mask "
                          "(default: 0, i.e. no smoothing). Dataset-specific.")
    sg.add_argument("--distance-sigma", type=float, default=0.0,
                     help="Sigma for Gaussian smoothing of the distance "
                          "transform (default: 0, i.e. no smoothing). "
                          "Dataset-specific.")

    # --- track: cross-slice instance tracking ---
    tk = sub.add_parser(
        "track", help="Assign consistent 3D instance IDs across slices.",
    )
    tk.add_argument("--input-dir", required=True,
                     help="Directory of per-slice labelled TIFFs (output of "
                          "'segment').")
    tk.add_argument("--output-dir", required=True,
                     help="Directory for full-resolution tracked TIFFs.")
    tk.add_argument("--downsampled-dir", default=None,
                     help="Optional directory for downsampled tracked TIFFs.")
    tk.add_argument("--full-height", type=int, required=True,
                     help="Height (pixels) of the full-resolution label maps.")
    tk.add_argument("--full-width", type=int, required=True,
                     help="Width (pixels) of the full-resolution label maps.")
    tk.add_argument("--downsample-height", type=int, default=None,
                     help="Height for downsampled copies (required if "
                          "--downsampled-dir is set).")
    tk.add_argument("--downsample-width", type=int, default=None,
                     help="Width for downsampled copies (required if "
                          "--downsampled-dir is set).")
    tk.add_argument("--dist-thresh", type=float, required=True,
                     help="Maximum centroid distance (pixels) for matching "
                          "instances between slices. Dataset-specific.")
    tk.add_argument("--iou-thresh", type=float, required=True,
                     help="Minimum IoU for matching instances between "
                          "slices. Dataset-specific.")

    # --- extract: z-filter + feature extraction ---
    ex = sub.add_parser(
        "extract",
        help="Z-span filter tracked instances and extract shape features.",
    )
    ex.add_argument(
        "--instance-dir",
        required=True,
        help="Directory of instance-labelled TIFF slices (output of 'track').",
    )
    ex.add_argument(
        "--em-dir",
        required=True,
        help="Directory of EM image slices (PNG).",
    )
    ex.add_argument(
        "--output-csv",
        required=True,
        help="Path for the output feature CSV.",
    )
    ex.add_argument(
        "--z-threshold",
        type=int,
        required=True,
        help="Minimum z-span (slices) to retain an instance. Dataset-specific"
             " — depends on section thickness and structure size.",
    )
    ex.add_argument(
        "--xy-scale",
        type=float,
        required=True,
        help="XY scale factor from working to full resolution. "
             "Dataset-specific.",
    )
    ex.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (default: -1, i.e. all cores).",
    )
    ex.add_argument(
        "--id-map-csv",
        default=None,
        help="Optional path to save the original→filtered ID mapping.",
    )
    args = parser.parse_args()

    # Configure logging
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="a"))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=handlers,
    )

    if args.command == "segment":
        segment_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            opening_radius=args.opening_radius,
            min_size=args.min_size,
            h_max_threshold=args.h_max_threshold,
            gaussian_sigma=args.gaussian_sigma,
            distance_sigma=args.distance_sigma,
        )

    elif args.command == "track":
        if args.downsampled_dir is not None and (
            args.downsample_height is None or args.downsample_width is None
        ):
            parser.error(
                "--downsample-height and --downsample-width are required "
                "when --downsampled-dir is set."
            )
        track_instances_across_slices(
            labelled_slices_dir=args.input_dir,
            output_dir=args.output_dir,
            downsampled_dir=args.downsampled_dir,
            full_shape=(args.full_height, args.full_width),
            downsample_shape=(args.downsample_height, args.downsample_width),
            dist_thresh=args.dist_thresh,
            iou_thresh=args.iou_thresh,
        )

    elif args.command == "extract":
        volume = load_instance_volume(args.instance_dir)
        filtered = filter_by_z_span(
            volume, z_threshold=args.z_threshold, id_map_csv=args.id_map_csv
        )
        del volume

        morph = MorphDescriptor(filtered, args.em_dir)
        features_df, _ = morph.extract_features(
            xy_scale_factor=args.xy_scale, n_jobs=args.n_jobs
        )

        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        features_df.to_csv(args.output_csv, index=False)
        logger.info("Features saved to %s", args.output_csv)


if __name__ == "__main__":
    main()
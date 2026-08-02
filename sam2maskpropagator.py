"""
Prompt-guided instance segmentation via SAM2 video propagation.

This module provides a pipeline for propagating instance segmentation masks
across serial-section image stacks (treated as video frames) using Meta's
Segment Anything Model 2 (SAM2) video predictor. The workflow supports:

  - **Prompt-based initialisation**: bounding-box or point-coordinate prompts
    on user-specified anchor frames.
  - **Bidirectional propagation**: masks are propagated both forward and
    backward from the anchor frame within a user-defined slice range.
  - **SWC-guided prompt generation**: automatic extraction of anchor-frame
    prompts from SWC morphological reconstructions, with optional
    skeleton-based masking for densely packed cells (e.g., hepatocytes).
  - **Instance-ID mask output**: each propagated object retains a unique
    integer label in the output masks (uint16 PNG).

The video predictor (``build_sam2_video_predictor``) maintains a memory bank
of spatial features across frames, enabling temporally coherent segmentation
that is substantially more efficient than per-frame image prediction for
whole-volume analysis.

Dependencies
------------
- segment-anything-2 (https://github.com/facebookresearch/segment-anything-2)
- PyTorch >= 2.0
- scikit-image, OpenCV, NumPy, Pillow, matplotlib

References
----------
Ravi, N. et al. SAM 2: Segment Anything in Images and Videos.
*arXiv preprint arXiv:2408.00714* (2024).
"""

import os
import gc
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, erosion, disk

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

Image.MAX_IMAGE_PIXELS = 500_000_000

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def overlay_mask(
    mask: np.ndarray,
    ax: plt.Axes,
    obj_id: Optional[int] = None,
    alpha: float = 0.5,
) -> None:
    """Overlay a coloured semi-transparent mask on a matplotlib Axes.

    Parameters
    ----------
    mask : np.ndarray
        2-D or 3-D mask array (values in [0, 255]).
    ax : matplotlib.axes.Axes
        Target axes for the overlay.
    obj_id : int, optional
        Object ID used to select a colour from the ``tab10`` colour-map.
    alpha : float
        Overlay opacity (purely cosmetic; default 0.5).
    """
    mask = mask.astype(np.float32) / 255.0
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], alpha])
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class SAM2VideoMaskPropagator:
    """Propagate instance segmentation masks across an image stack using SAM2.

    Parameters
    ----------
    model_cfg : str
        SAM2 model configuration name (e.g., ``"sam2_hiera_l.yaml"``).
    sam2_checkpoint : str
        Path to the SAM2 pre-trained checkpoint file.
    video_dir : str
        Directory containing the frame images (``.jpg`` or ``.png``),
        treated as a video sequence by the SAM2 video predictor.
    output_dir : str
        Directory where output instance-mask PNGs will be saved.
    label_type : str, optional
        Cell/structure type being segmented. Controls type-specific logic
        (e.g., ``"hepatocyte"``, ``"cholangiocyte"``, ``"arteriole"``,
        ``"endothelial"``, ``"portalvein"``, ``"bileduct"``).
    text_file : str, optional
        Path to a tab-separated file recording per-object anchor-frame
        indices and bounding-box coordinates.
    input_swc_dir : str, optional
        Directory of SWC files used for automatic prompt generation.
    mask_dir : str, optional
        Directory of pre-computed semantic masks (e.g., skeleton masks for
        hepatocyte boundary propagation).
    output_visual_dir : str, optional
        Directory for saving visualisation overlays.
    ann_obj_id : list of int, optional
        Object IDs for ``run_video_only`` mode.
    ann_frame_idx : list of int, optional
        Anchor frame indices for ``run_video_only`` mode.
    box : list of list, optional
        Bounding-box prompts ``[x1, y1, x2, y2]`` for ``run_video_only``.
    device : torch.device, optional
        Compute device. Defaults to CUDA if available.
    frame_sort_key : callable, optional
        Function to extract a sortable key from frame filenames.
        Defaults to extracting the integer after the first underscore
        (e.g., ``"frame_042.png"`` → ``42``).
    """

    # ---- construction ----

    def __init__(
        self,
        model_cfg: str,
        sam2_checkpoint: str,
        video_dir: str,
        output_dir: str,
        label_type: Optional[str] = None,
        text_file: Optional[str] = None,
        input_swc_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        output_visual_dir: Optional[str] = None,
        ann_obj_id: Optional[List[int]] = None,
        ann_frame_idx: Optional[List[int]] = None,
        box: Optional[List[List[float]]] = None,
        device: Optional[torch.device] = None,
        frame_sort_key=None,
    ):
        self.model_cfg = model_cfg
        self.sam2_checkpoint = sam2_checkpoint
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.label_type = label_type
        self.text_file = text_file
        self.input_swc_dir = input_swc_dir
        self.mask_dir = mask_dir
        self.output_visual_dir = output_visual_dir
        self.ann_obj_id = ann_obj_id
        self.ann_frame_idx = ann_frame_idx
        self.box = box
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._frame_sort_key = frame_sort_key or (
            lambda p: int(p.split("_")[1].split(".")[0])
        )
        self.frame_names = self._get_sorted_frame_names()
        self.video_segments: Dict[int, np.ndarray] = {}

    # ---- internal helpers ----

    def _get_sorted_frame_names(self) -> List[str]:
        """Return image filenames in *video_dir* sorted by frame index."""
        names = [
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1].lower() in {".jpg", ".png"}
        ]
        names.sort(key=self._frame_sort_key)
        return names

    # ---- prompt-file I/O ----

    def read_object_prompts(self) -> List[dict]:
        """Parse the tab-separated prompt file into a list of object dicts.

        Each dict contains:
            ``start_slice_id``, ``middle_slice_id``, ``end_slice_id``,
            ``x_top``, ``y_top``, ``x_bottom``, ``y_bottom``.

        The file format differs slightly for hepatocytes (no leading SWC-file
        column) versus other cell types.

        Returns
        -------
        list of dict
        """
        objects = []
        with open(self.text_file, "r") as fh:
            for line in fh:
                fields = line.strip().split("\t")
                if self.label_type == "hepatocyte":
                    offset = 0  # no SWC-file-ID column
                else:
                    offset = 1  # first column is SWC file ID
                objects.append({
                    "start_slice_id": int(fields[0 + offset]),
                    "middle_slice_id": int(fields[1 + offset]),
                    "end_slice_id": int(fields[2 + offset]),
                    "x_top": float(fields[3 + offset]),
                    "y_top": float(fields[4 + offset]),
                    "x_bottom": float(fields[5 + offset]),
                    "y_bottom": float(fields[6 + offset]),
                })
        return objects

    # ---- single-object propagation ----

    def _propagate_object(self, obj: dict, obj_id: int) -> None:
        """Propagate a single object's mask bidirectionally from its anchor frame.

        The predictor state is reset before each object so that prompts from
        previous objects do not interfere.  Masks are propagated forward from
        ``middle_slice_id`` to ``end_slice_id``, then backward to
        ``start_slice_id``.

        Parameters
        ----------
        obj : dict
            Object prompt dictionary (see :meth:`read_object_prompts`).
        obj_id : int
            Unique instance ID assigned to this object in the output masks.
        """
        self.predictor.reset_state(self.inference_state)

        box = np.array(
            [obj["x_top"], obj["y_top"], obj["x_bottom"], obj["y_bottom"]],
            dtype=np.float32,
        )
        self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=obj["middle_slice_id"],
            obj_id=obj_id,
            box=box,
        )

        anchor = obj["middle_slice_id"]

        # Forward propagation: anchor → end
        with torch.amp.autocast("cuda"):
            for frame_idx, _, logits in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=anchor
            ):
                if frame_idx > obj["end_slice_id"]:
                    break
                self._assign_mask(frame_idx, logits, obj_id)
                del logits
                torch.cuda.empty_cache()

        # Backward propagation: anchor → start
        with torch.amp.autocast("cuda"):
            for frame_idx, _, logits in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=anchor, reverse=True
            ):
                if frame_idx < obj["start_slice_id"]:
                    break
                self._assign_mask(frame_idx, logits, obj_id)
                del logits
                torch.cuda.empty_cache()

    def _assign_mask(
        self,
        frame_idx: int,
        logits: torch.Tensor,
        obj_id: int,
    ) -> None:
        """Write thresholded mask logits into the instance-label volume.

        Parameters
        ----------
        frame_idx : int
            Index of the current frame.
        logits : torch.Tensor
            Raw mask logits from the SAM2 video predictor.
        obj_id : int
            Instance ID to assign where the mask is positive.
        """
        mask = (logits[0] > 0.0).cpu().numpy()
        if frame_idx not in self.video_segments:
            self.video_segments[frame_idx] = np.zeros_like(mask, dtype=np.uint16)
        self.video_segments[frame_idx][mask > 0] = obj_id

    # ---- mask I/O ----

    def _save_instance_mask(self, frame_idx: int) -> None:
        """Save the instance-label mask for a single frame as a uint16 PNG.

        Parameters
        ----------
        frame_idx : int
            Frame index to save.
        """
        if frame_idx not in self.video_segments:
            return

        mask = self.video_segments[frame_idx]
        if mask.dtype != np.uint16:
            mask = mask.astype(np.uint16)

        # Collapse singleton dimensions (e.g. [1, H, W] → [H, W])
        if mask.ndim == 3:
            mask = mask.squeeze()
        if mask.ndim != 2:
            logger.warning(
                "Skipping frame %d: unexpected mask shape %s", frame_idx, mask.shape
            )
            return

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f"mask_{frame_idx:04d}.png")
        Image.fromarray(mask).save(save_path, format="PNG")
        logger.info("Saved mask: %s", save_path)

    # ---- SWC prompt generation ----

    def _find_frame_index(self, slice_id: int, image_dir: str) -> Optional[int]:
        """Return the frame index whose filename contains *slice_id*.

        Uses the same ``self._frame_sort_key`` ordering as ``self.frame_names``
        so indices returned here line up with frames indexed elsewhere.

        Parameters
        ----------
        slice_id : int
            Integer slice identifier embedded in the filename.
        image_dir : str
            Directory of frame images.

        Returns
        -------
        int or None
        """
        names = sorted(os.listdir(image_dir), key=self._frame_sort_key)
        for i, fname in enumerate(names):
            if f"{slice_id:03d}" in fname:
                return i
        return None

    def parse_swc_files(self) -> List[dict]:
        """Extract anchor-frame prompts from SWC morphological reconstructions.

        Each SWC file yields the start, middle, and end slice indices plus a
        single representative coordinate on the middle slice.  Coordinates are
        scaled from SWC conventions (µm) to pixel space assuming a
        ``1000 / 8 / 2`` scale factor.

        Returns
        -------
        list of dict
            One entry per SWC file with keys ``file_name``,
            ``start_slice_id``, ``middle_slice_id``, ``end_slice_id``,
            ``coordinate``.
        """
        if self.input_swc_dir is None:
            raise ValueError("input_swc_dir must be set to use SWC prompts.")

        swc_entries = []
        for swc_filename in os.listdir(self.input_swc_dir):
            filepath = os.path.join(self.input_swc_dir, swc_filename)
            min_z, max_z = 99999, 0
            coordinates = []

            with open(filepath, "r") as fh:
                rows = [
                    line.strip().split()
                    for line in fh
                    if not line.startswith("#")
                ]

            for row in rows:
                z = int(float(row[4]) * 20)
                min_z, max_z = min(min_z, z), max(max_z, z)

            if min_z == 99999:
                continue
            mid_z = (min_z + max_z) // 2

            # Retrieve the first coordinate on the middle slice
            scale = 1000.0 / 8.0 / 2.0
            for row in rows:
                if int(float(row[4]) * 20) == mid_z:
                    coordinates = [float(row[2]) * scale, float(row[3]) * scale]
                    break

            if not coordinates:
                continue

            swc_entries.append({
                "file_name": swc_filename,
                "start_slice_id": self._find_frame_index(min_z, self.video_dir),
                "middle_slice_id": self._find_frame_index(mid_z, self.video_dir),
                "end_slice_id": self._find_frame_index(max_z, self.video_dir),
                "coordinate": coordinates,
            })
        return swc_entries

    def _place_mask_in_full_frame(
        self,
        cropped_mask: np.ndarray,
        top_x: int,
        top_y: int,
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Map a cropped mask back into a full-frame-sized array.

        Parameters
        ----------
        cropped_mask : np.ndarray
            Binary mask produced on the cropped sub-image.
        top_x, top_y : int
            Top-left corner of the crop in full-frame coordinates.
        original_shape : tuple of int
            ``(H, W)`` of the full frame.

        Returns
        -------
        np.ndarray
            Full-frame mask with the cropped region filled in.
        """
        full_mask = np.zeros(original_shape, dtype=np.uint8)
        h, w = cropped_mask.shape
        full_mask[top_x : top_x + h, top_y : top_y + w] = cropped_mask
        return full_mask

    def generate_prompts_from_swc(self) -> None:
        """Generate bounding-box prompts from SWC files using the SAM2 image predictor.

        For each SWC entry the method:
        1. Crops a region around the SWC coordinate on the anchor slice.
        2. Runs SAM2 image prediction with the coordinate as a point prompt.
        3. Retains the largest connected component as the object mask.
        4. Derives a bounding box and appends it to ``self.text_file``.

        For hepatocytes the crop is taken from pre-computed skeleton masks
        (``self.mask_dir``) to improve boundary-based propagation.
        """
        swc_data = self.parse_swc_files()
        image_predictor = SAM2ImagePredictor(
            build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        )

        crop_size = 6000 if self.label_type == "hepatocyte" else 2000
        half = crop_size // 2
        erosion_radius = 10

        for entry in swc_data:
            mid_idx = entry["middle_slice_id"]
            coords = entry["coordinate"]
            if mid_idx is None:
                continue

            # Load the appropriate image. `mid_idx` is a position into
            # `self.frame_names` (video_dir); for the hepatocyte branch this
            # assumes mask_dir contains the same number of files, in the same
            # frame order, as video_dir.
            if self.label_type == "hepatocyte":
                mask_names = sorted(
                    os.listdir(self.mask_dir), key=self._frame_sort_key
                )
                img_path = os.path.join(self.mask_dir, mask_names[mid_idx])
                image = np.array(Image.open(img_path).convert("RGB"))
                binary = (image > 0).astype(bool)
                image = (skeletonize(binary) * 255).astype(np.uint8)
            else:
                img_path = os.path.join(self.video_dir, self.frame_names[mid_idx])
                image = np.array(Image.open(img_path).convert("RGB"))
                img_range = image.max() - image.min()
                if (image.max() > 255 or image.min() < 0) and img_range > 0:
                    image = np.uint8(255 * (image - image.min()) / img_range)

            # Crop around the SWC coordinate
            top_y = max(int(coords[0] - half), 0)
            top_x = max(int(coords[1] - half), 0)
            bot_y = min(int(coords[0] + half), image.shape[0])
            bot_x = min(int(coords[1] + half), image.shape[1])
            crop = image[top_x:bot_x, top_y:bot_y, :]
            adj_point = np.array([coords[0] - top_y, coords[1] - top_x])

            # SAM2 image prediction on the crop
            image_predictor.set_image(crop)
            mask, _, _ = image_predictor.predict(
                point_coords=np.array([adj_point]),
                point_labels=np.array([1]),
                multimask_output=False,
            )

            # Place mask in full frame and keep largest component
            full_mask = self._place_mask_in_full_frame(
                mask[0], top_x, top_y, image.shape[:2]
            )
            labelled = label(full_mask > 0, connectivity=2)
            largest = max(regionprops(labelled), key=lambda r: r.area, default=None)
            if largest is None:
                continue
            full_mask = ((labelled == largest.label).astype(np.uint8) * 255)

            # Derive bounding box with a small margin
            margin = 10
            rows_nz, cols_nz = np.where(full_mask > 0)
            x_top = max(cols_nz.min() - margin, 0)
            y_top = max(rows_nz.min() - margin, 0)
            x_bot = min(cols_nz.max() + margin, image.shape[1])
            y_bot = min(rows_nz.max() + margin, image.shape[0])

            # Save visualisation (optional)
            if self.output_visual_dir:
                eroded = erosion((full_mask > 0), disk(erosion_radius))
                eroded_vis = (eroded * 255).astype(np.uint8)

                os.makedirs(self.output_visual_dir, exist_ok=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.set_title(f"Anchor slice {mid_idx}")
                ax.imshow(image, cmap="gray")
                ax.scatter(coords[0], coords[1], c="red", s=20, zorder=5)
                overlay_mask(eroded_vis, ax)
                rect = Rectangle(
                    (x_top, y_top), x_bot - x_top, y_bot - y_top,
                    linewidth=2, edgecolor="blue", facecolor="none",
                )
                ax.add_patch(rect)
                fig.savefig(
                    os.path.join(self.output_visual_dir, f"prompt_{mid_idx:04d}.png"),
                    dpi=300,
                )
                plt.close(fig)

            # Append prompt to text file
            with open(self.text_file, "a") as fh:
                fh.write(
                    f"{entry['file_name']}\t{entry['start_slice_id']}\t"
                    f"{mid_idx}\t{entry['end_slice_id']}\t"
                    f"{x_top}\t{y_top}\t{x_bot}\t{y_bot}\n"
                )

    # ---- main entry points ----

    def run(self) -> None:
        """Run prompt-guided mask propagation for all objects in the prompt file.

        Workflow
        --------
        1. Build the SAM2 video predictor and initialise the inference state.
        2. Read object prompts from ``self.text_file``.
        3. For each object, propagate bidirectionally from its anchor frame.
        4. Save instance-label masks for all frames that received predictions.
        """
        self.predictor = build_sam2_video_predictor(
            self.model_cfg, self.sam2_checkpoint, device=self.device
        )

        # For hepatocytes, propagate on skeleton masks rather than raw EM
        source_dir = (
            self.mask_dir
            if self.label_type == "hepatocyte" and self.mask_dir
            else self.video_dir
        )
        self.inference_state = self.predictor.init_state(video_path=source_dir)

        objects = self.read_object_prompts()
        logger.info("Loaded %d object prompts.", len(objects))

        for obj_id, obj in enumerate(objects, start=1):
            logger.info("Propagating object %d / %d", obj_id, len(objects))
            self._propagate_object(obj, obj_id)
            torch.cuda.empty_cache()
            gc.collect()

        # Save all frames that contain at least one prediction
        for frame_idx in sorted(self.video_segments):
            self._save_instance_mask(frame_idx)

        logger.info("Done. Masks saved to %s", self.output_dir)

    def run_video_only(self) -> None:
        """Run propagation for structures with direct box prompts (no SWC / text file).

        This mode is intended for large structures such as blood vessels
        (arterioles, portal veins, bile ducts) where a small number of
        bounding-box prompts on specific frames is sufficient.

        Propagation direction(s) are selected based on ``self.label_type``:
        - **arteriole**: bidirectional from each anchor frame.
        - **portalvein**: bidirectional from a single anchor frame.
        - **bileduct**: forward-only from the first frame.
        """
        predictor = build_sam2_video_predictor(
            self.model_cfg, self.sam2_checkpoint, device=self.device
        )
        inference_state = predictor.init_state(video_path=self.video_dir)
        predictor.reset_state(inference_state)

        # Register all box prompts
        for i in range(len(self.ann_obj_id)):
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=self.ann_frame_idx[i],
                obj_id=self.ann_obj_id[i],
                box=np.array(self.box[i], dtype=np.float32),
            )

        video_segments: Dict[int, Dict[int, np.ndarray]] = {}

        # Select propagation strategy based on structure type
        if self.label_type == "arteriole":
            # Bidirectional from each unique anchor frame
            anchor_frames = sorted(set(self.ann_frame_idx))
            for anchor in anchor_frames:
                for reverse in (False, True):
                    with torch.amp.autocast("cuda"):
                        for fidx, obj_ids, logits in predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=anchor,
                            reverse=reverse,
                        ):
                            video_segments.setdefault(fidx, {})
                            for j, oid in enumerate(obj_ids):
                                video_segments[fidx][oid] = (
                                    (logits[j] > 0.0).cpu().numpy()
                                )

        elif self.label_type == "portalvein":
            anchor = self.ann_frame_idx[0]
            for reverse in (False, True):
                for fidx, obj_ids, logits in predictor.propagate_in_video(
                    inference_state, start_frame_idx=anchor, reverse=reverse
                ):
                    video_segments.setdefault(fidx, {})
                    for j, oid in enumerate(obj_ids):
                        video_segments[fidx][oid] = (logits[j] > 0.0).cpu().numpy()

        elif self.label_type == "bileduct":
            for fidx, obj_ids, logits in predictor.propagate_in_video(
                inference_state
            ):
                video_segments.setdefault(fidx, {})
                for j, oid in enumerate(obj_ids):
                    video_segments[fidx][oid] = (logits[j] > 0.0).cpu().numpy()

        else:
            raise ValueError(f"Unsupported label_type for video-only mode: {self.label_type}")

        del inference_state
        gc.collect()

        # Save combined binary masks
        self._save_combined_masks(video_segments)

    def _save_combined_masks(
        self,
        video_segments: Dict[int, Dict[int, np.ndarray]],
    ) -> None:
        """Merge per-object binary masks and save one PNG per frame.

        Parameters
        ----------
        video_segments : dict
            ``{frame_idx: {obj_id: mask_array, ...}, ...}``
        """
        os.makedirs(self.output_dir, exist_ok=True)
        for fidx in sorted(video_segments):
            combined = None
            for mask in video_segments[fidx].values():
                m = np.squeeze(mask)
                if m.dtype != np.uint8:
                    m = (m * 255).astype(np.uint8)
                combined = m if combined is None else np.maximum(combined, m)
            if combined is not None:
                Image.fromarray(combined).save(
                    os.path.join(self.output_dir, f"mask_{fidx:04d}.png")
                )
        logger.info("Saved combined masks to %s", self.output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for SAM2 video mask propagation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SAM2 video mask propagation for instance segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--label_type", type=str, required=True,
        choices=[
            "cholangiocyte", "arteriole", "endothelial",
            "hepatocyte", "portalvein", "bileduct",
        ],
        help="Structure type to segment.",
    )
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml",
                        help="SAM2 model configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SAM2 checkpoint.")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory of input frame images.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output instance masks.")
    parser.add_argument("--text_file", type=str, default=None,
                        help="Tab-separated prompt file.")
    parser.add_argument("--swc_dir", type=str, default=None,
                        help="Directory of SWC files for prompt generation.")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Directory of pre-computed semantic masks.")
    parser.add_argument("--visual_dir", type=str, default=None,
                        help="Directory for visualisation output.")
    parser.add_argument("--mode", type=str, default="propagate",
                        choices=["generate_prompts", "propagate", "video_only"],
                        help="Pipeline stage to execute.")
    parser.add_argument(
        "--box_prompts_json", type=str, default=None,
        help="Required for --mode video_only. Path to a JSON file containing "
             "a list of box prompts, e.g. "
             '[{"obj_id": 1, "frame_idx": 0, "box": [x0, y0, x1, y1]}, ...]. '
             "Frame indices, object IDs, and box coordinates are dataset- "
             "and structure-specific and must be supplied by the user.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    ann_obj_id = ann_frame_idx = box = None
    if args.mode == "video_only":
        if not args.box_prompts_json:
            parser.error("--mode video_only requires --box_prompts_json.")
        with open(args.box_prompts_json, "r") as fh:
            prompts = json.load(fh)
        ann_obj_id = [p["obj_id"] for p in prompts]
        ann_frame_idx = [p["frame_idx"] for p in prompts]
        box = [p["box"] for p in prompts]

    propagator = SAM2VideoMaskPropagator(
        model_cfg=args.model_cfg,
        sam2_checkpoint=args.checkpoint,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        label_type=args.label_type,
        text_file=args.text_file,
        input_swc_dir=args.swc_dir,
        mask_dir=args.mask_dir,
        output_visual_dir=args.visual_dir,
        ann_obj_id=ann_obj_id,
        ann_frame_idx=ann_frame_idx,
        box=box,
    )

    if args.mode == "generate_prompts":
        propagator.generate_prompts_from_swc()
    elif args.mode == "propagate":
        propagator.run()
    elif args.mode == "video_only":
        propagator.run_video_only()


if __name__ == "__main__":
    main()
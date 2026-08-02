# MuST-EM: Multiscale Tissue Electron Microscopy

This repository contains the deep-learning segmentation and downstream quantitative
analysis code for **MuST-EM**, an end-to-end volume electron microscopy (vEM) workflow
combining large-volume SBF-SEM acquisition, expert pathologist annotation,
deep-learning-based segmentation across scales, quantitative morphometric analysis, and
inter-structure contact analysis.

## Introduction

We applied MuST-EM to an intact periportal region of human liver tissue, imaging a
contiguous volume of 152 × 140 × 33 µm³ at 8 nm pixel size that captures tissue,
vascular, cellular, and organellar architecture within a single dataset. Automated
segmentation enabled comprehensive annotation of the full volume, from which we
quantified structural relationships between bile duct lumens and cholangiocytes and
characterised sinusoidal capillary branching organisation. At the organelle scale,
morphometric analysis of 35,790 mitochondria revealed pronounced shape heterogeneity,
and quantitative contact analysis demonstrated that elongated mitochondria with
narrowing sites exhibit preferential short-range ER contact at those sites, a spatial
signature consistent with models of ER-mediated mitochondrial remodelling.

<p align="center">
  <img src="figures/automated_segmentation.png" alt="Automated segmentation overview" width="800">
</p>

## Repository Contents

| Path | Description |
| --- | --- |
| `sam2maskpropagator.py` | Prompt-guided instance segmentation (vascular/cellular level) via SAM2 video mask propagation. |
| `morphology_features.py` | 3D morphological feature extraction from instance-segmented organelle volumes (PyRadiomics-based). |
| `mito_er_analysis.py` | Mitochondrial morphology clustering, per-hepatocyte distribution analysis, and ER–mitochondria narrowing-site analysis. |
| `nnUNet/` | Modified copy of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) (organelle-level segmentation), Apache License 2.0 — see `nnUNet/NOTICE.md`. |
| `SAM2/` | Modified copy of [SAM2](https://github.com/facebookresearch/sam2) (vascular/cellular-level segmentation backbone), Apache License 2.0 — see `SAM2/NOTICE.md`. |
| `create_run_example.slurm` | Example Slurm submission script (Compute Canada environment) for running `mito_er_analysis.py` on an HPC cluster. |

## System Requirements

- **OS:** Linux (tested on a Compute Canada Slurm cluster; other Linux distributions with a compatible CUDA driver should work).
- **Python:** 3.10
- **Hardware:** an NVIDIA H100 GPU (80 GB VRAM) with CUDA 12.x was used for all GPU
  stages (SAM2 `sam2maskpropagator.py` and nnU-Net training/inference); a smaller GPU
  may work for inference on smaller inputs but has not been tested.
  `morphology_features.py` and `mito_er_analysis.py` are CPU-only and were run with 32
  CPU cores / 190 GB RAM (see `create_run_example.slurm`); both are configurable via the
  `--n-jobs` / multiprocessing options in each script for smaller machines.
- **Non-standard hardware:** none required.
- **Expected run time:** varies by dataset size and stage; SAM2/nnU-Net inference on a
  full tile takes on the order of minutes to hours per volume, and `mito_er_analysis.py`
  stages take minutes on the settings above for a single tile.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BaderLab/Multiscale_human_liver_vEM.git
   cd Multiscale_human_liver_vEM
   ```

2. **Create and activate a virtual environment** (Python 3.10), e.g.:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
   An example HPC (Slurm) setup is provided in `create_run_example.slurm`.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This top-level `requirements.txt` merges the dependencies of the vendored
   [nnUNet](nnUNet/requirements.txt) and [SAM2](SAM2/requirements.txt) copies with
   those of the analysis scripts. Typical install time is a few minutes on a
   standard workstation with a working internet connection (longer if PyTorch
   needs to build/download CUDA wheels).

4. **Download SAM2 checkpoints** (required for `sam2maskpropagator.py`):
   ```bash
   cd SAM2 && bash download_ckpts.sh && cd ..
   ```

5. **Download trained nnU-Net checkpoints** for all segmented organelles from
   [Zenodo](https://zenodo.org/records/17360859) (required for organelle segmentation).

## Getting Started

### 1. Vascular and Cellular Level Segmentation

`sam2maskpropagator.py` uses SAM2 to propagate instance masks across a serial-section
image stack from user-supplied prompts:

```bash
python sam2maskpropagator.py \
  --label_type portalvein \
  --checkpoint SAM2/sam2_hiera_large.pt \
  --model_cfg sam2_hiera_l.yaml \
  --video_dir /path/to/em_slices \
  --output_dir /path/to/output_masks \
  --mode propagate
```

Run `python sam2maskpropagator.py --help` for the full set of options, including
SWC-guided prompt generation (`--mode generate_prompts`) and video-only propagation
(`--mode video_only`).

### 2. Organelle Segmentation

Organelle segmentation was performed using [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
(vendored in `nnUNet/`, see `nnUNet/NOTICE.md`) with pretraining and fine-tuning, using
the standard `nnUNetv2_predict` inference entry point and the checkpoints from
[Zenodo](https://zenodo.org/records/17360859).

### 3. Mitochondrial Morphology Feature Extraction

`morphology_features.py` exposes three subcommands covering the full path from
binary masks to a feature table: per-slice watershed instance segmentation
(`segment`), cross-slice instance tracking into coherent 3D labels (`track`), and
[PyRadiomics](https://pyradiomics.readthedocs.io/)-based shape feature extraction
(`extract`):

```bash
python morphology_features.py segment \
  --input-dir /path/to/binary_masks \
  --output-dir /path/to/labelled_slices \
  [--opening-radius <radius>] [--min-size <min_size>] \
  [--h-max-threshold <threshold>] [--gaussian-sigma <sigma>] \
  [--distance-sigma <sigma>]

python morphology_features.py track \
  --input-dir /path/to/labelled_slices \
  --output-dir /path/to/tracked_instances \
  --full-height <height> --full-width <width> \
  --dist-thresh <max_centroid_distance> --iou-thresh <min_iou>

python morphology_features.py extract \
  --instance-dir /path/to/tracked_instances \
  --em-dir /path/to/em_slices \
  --output-csv /path/to/output/mito_features.csv \
  --z-threshold <min_z_span> \
  --xy-scale <working_to_full_res_ratio>
```

`segment`'s five parameters are all optional and default to a no-op (0 = skip
opening / no size filtering / no smoothing / no h-maxima suppression); in
practice some suppression and smoothing is usually needed to avoid
oversegmentation. Every other placeholder above (tile/volume dimensions,
matching thresholds, z-span, XY scale factor) depends on your acquisition
parameters (pixel size, tile layout, section thickness, etc.) — run
`python morphology_features.py <subcommand> --help` for a full description of
each option.

### 4. Mitochondria–ER Interaction Analysis

`mito_er_analysis.py` exposes four subcommands for the clustering, per-hepatocyte, and
ER–mitochondria narrowing-site analyses described in the manuscript:

```bash
python mito_er_analysis.py cluster \
  --feature_csv /path/to/mito_features.csv \
  --output_root /path/to/output \
  --ks_threshold 0.1

python mito_er_analysis.py hepatocyte \
  --pca_csv /path/to/output/pca_features.csv \
  --hepatocyte_mask_folder /path/to/hepatocyte_masks \
  --mito_instance_folder /path/to/mito_instance_masks \
  --outlier_mask /path/to/outlier_mask \
  --output_dir /path/to/output \
  --z_threshold 6

python mito_er_analysis.py er-mito \
  --mito_folder /path/to/mito_instance_masks \
  --er_folder /path/to/er_masks \
  --cluster_csv /path/to/output/pca_features.csv \
  --output_dir /path/to/output \
  --z_end 596 \
  --tile_size 2000 --grid_cols 10 \
  --exclude_end_fraction 0.1 --min_ratio_threshold 0.7 \
  --n_slices 20 --near_fraction 0.5

python mito_er_analysis.py enrichment \
  --result_dir /path/to/output \
  --output_dir /path/to/output/enrichment
```

All thresholds above (KS cutoff, z-span, tile geometry, narrowing-site detection
parameters, etc.) depend on your acquisition and tiling setup — run
`python mito_er_analysis.py <subcommand> --help` for the full CLI of each stage.

## License

The original code in this repository (`mito_er_analysis.py`, `morphology_features.py`,
`sam2maskpropagator.py`, and `create_run_example.slurm`) is released under the MIT
License — see [LICENSE.txt](LICENSE.txt). The vendored, modified copies of nnU-Net and
SAM2 in `nnUNet/` and `SAM2/` retain their original Apache License 2.0 — see
`nnUNet/LICENSE`/`nnUNet/NOTICE.md` and `SAM2/LICENSE`/`SAM2/NOTICE.md`.

## Data Availability

Input human liver volume EM data and derived segmentation/feature datasets are
described in the Data Availability statement of the accompanying manuscript. We
gratefully acknowledge [OpenOrganelle](https://openorganelle.janelia.org/) and
[Parlakgül et al. (2022)](https://www.nature.com/articles/s41586-022-04488-5) for
making the mouse liver volume electron microscopy data used during method development
publicly available. Trained nnU-Net model checkpoints are available on
[Zenodo](https://zenodo.org/records/17360859).

## Acknowledgements

We thank the [SAM2](https://arxiv.org/abs/2408.00714) and [nnUNet](https://www.nature.com/articles/s41592-020-01008-z) teams for making their source code publicly available. We also thank the [PyRadiomics](https://doi.org/10.1158/0008-5472.CAN-17-0339) team for their open-source morphological feature extraction package. We gratefully acknowledge [OpenOrganelle](https://openorganelle.janelia.org/) and [Parlakgül et al. (2022)](https://www.nature.com/articles/s41586-022-04488-5) for making the mouse liver volume electron microscopy data publicly available.

## Citation

<!-- Update this once the manuscript is published -->
```bibtex
@article{MuST-EM,
  title   = {MuST-EM: an end-to-end workflow for multiscale volume electron microscopy of intact tissue},
  author  = {},
  journal = {},
  volume  = {},
  pages   = {},
  year    = {2026}
}
```

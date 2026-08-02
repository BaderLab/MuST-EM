"""
Mitochondrial morphology feature analysis + mitochondria-ER interaction pipeline.

This module performs dimensionality reduction, morphological analysis,
per-hepatocyte distribution analysis, and ER–mitochondria interaction quantification
from 3D electron microscopy data. Key analysis steps include:

1. **Feature clustering** — PCA on radiomics shape descriptors
2. **Per-hepatocyte analysis** — morphological distribution in each hepatocyte.
3. **ER–mitochondria narrowing site analysis** — identification of narrowing
   sites along the major axis, quantification of ER surface contact enrichment
   near vs. far from narrowing site region, and multi-tile parallelised processing.

Usage
-----
    python mito_er_analysis.py cluster --feature_csv <path> --output_root <path> [options]

Subcommands: ``cluster``, ``hepatocyte``, ``er-mito``, ``enrichment``.
Run ``python mito_er_analysis.py <subcommand> --help`` for the full CLI of each stage.
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import argparse
import gc
import glob
import logging
import multiprocessing as mp
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import binary_fill_holes, gaussian_filter, zoom
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import (
    binomtest,
    f_oneway,
    ks_2samp,
    kruskal,
    kstest,
    mannwhitneyu,
    median_abs_deviation,
    rankdata,
    skewnorm,
    t as t_dist,
    ttest_ind,
    wilcoxon,
)
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import ball, binary_dilation, binary_erosion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper / Utility Functions
# =============================================================================

def _skewnorm_mode(shape: float, loc: float, scale: float) -> float:
    """Compute the mode of a skew-normal distribution via dense-grid search.

    A closed-form mode does not exist for the skew-normal; this numerical
    approach is reliable for visualisation and reporting.
    """
    xs = np.linspace(loc - 6 * scale, loc + 6 * scale, 20_001)
    pdf = skewnorm.pdf(xs, shape, loc=loc, scale=scale)
    return float(xs[np.argmax(pdf)])


def _empirical_percentile(values: np.ndarray) -> np.ndarray:
    """Convert values to empirical percentiles on a 0–1 scale."""
    return (rankdata(values) - 1) / (len(values) - 1)


def _fit_skewnorm(
    values: np.ndarray,
    name: str = "PC",
    ks_threshold: float = None,
) -> Tuple[np.ndarray, dict]:
    """Fit a skew-normal distribution and return CDF-based indices.

    Falls back to empirical percentiles when the Kolmogorov–Smirnov D
    statistic exceeds *ks_threshold*.

    Returns
    -------
    index : np.ndarray
        CDF values (0–1).
    fit_info : dict
        Parameters and goodness-of-fit diagnostics.
    """
    try:
        a, loc, scale = skewnorm.fit(values)
        D, pval = kstest(
            values, lambda x: skewnorm.cdf(x, a, loc=loc, scale=scale)
        )
        logger.info(
            "[%s SkewNorm] shape=%.3f, loc=%.3f, scale=%.3f | KS D=%.3f, p=%.3f",
            name, a, loc, scale, D, pval,
        )
        if D > ks_threshold:
            logger.warning(
                "[%s] Poor skew-normal fit (D > %.2f); using empirical percentile.",
                name, ks_threshold,
            )
            return (
                _empirical_percentile(values),
                {"model": "empirical_percentile", "median": float(np.median(values))},
            )

        index = skewnorm.cdf(values, a, loc=loc, scale=scale)
        fit_info = {
            "model": "skewnorm",
            "shape": a,
            "loc": loc,
            "scale": scale,
            "ks_D": D,
            "ks_p": pval,
            "mode": _skewnorm_mode(a, loc, scale),
            "median": float(np.median(values)),
        }
        return index, fit_info

    except (RuntimeError, ValueError, FloatingPointError) as exc:
        # Only catch errors expected from a failed distribution fit (e.g.
        # scipy optimizer non-convergence); anything else (bad arguments,
        # programming errors) should propagate instead of being masked.
        logger.warning("[%s] Skew-normal fit failed (%s); using empirical percentile.", name, exc)
        return (
            _empirical_percentile(values),
            {"model": "empirical_percentile", "median": float(np.median(values))},
        )


def _assign_soft_labels(
    index_values: np.ndarray,
    high_conf: float = None,
    low_conf: float = None,
) -> np.ndarray:
    """Assign soft morphology labels based on index thresholds."""
    return np.where(
        index_values >= high_conf,
        "elongated-like",
        np.where(index_values <= low_conf, "constricted-like", "intermediate"),
    )


def _fit_skewnorm_group(x: np.ndarray) -> Tuple[float, float, float]:
    """Fit a skew-normal to a 1-D array; returns ``(shape, loc, scale)``."""
    x = np.asarray(x, dtype=float)
    if x.size < 3 or np.nanstd(x) == 0:
        return (0.0, float(np.nanmean(x)), max(float(np.nanstd(x)), 1e-8))
    try:
        a, loc, scale = skewnorm.fit(x)
        if not (np.isfinite(a) and np.isfinite(loc) and np.isfinite(scale) and scale > 0):
            raise RuntimeError("invalid skewnorm fit")
        return (float(a), float(loc), float(scale))
    except Exception:
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x)) if np.nanstd(x) > 0 else 1e-8
        return (0.0, mu, sd)


def _classify_by_skewnorm_quantiles(
    df: pd.DataFrame,
    hep_col: str,
    val_col: str,
    low_q: float,
    high_q: float,
) -> pd.DataFrame:
    """Classify mitochondria per hepatocyte using skew-normal quantiles.

    For each hepatocyte, fits a skew-normal to *val_col* and labels rows
    below *low_q* as ``constricted-like``, above *high_q* as
    ``elongated-like``, and the remainder as ``intermediate``.
    """
    out = df.copy()
    out[["sn_alpha", "sn_loc", "sn_scale", "q_low", "q_high", "conf_tail"]] = np.nan
    out["class_skewnorm"] = pd.Categorical(
        values=["intermediate"] * len(out),
        categories=["constricted-like", "intermediate", "elongated-like"],
    )

    for hep, idx in out.groupby(hep_col).indices.items():
        vals = out.loc[idx, val_col].to_numpy(float)
        a, loc, scale = _fit_skewnorm_group(vals)
        ql = skewnorm.ppf(low_q, a, loc=loc, scale=scale)
        qh = skewnorm.ppf(high_q, a, loc=loc, scale=scale)

        x = out.loc[idx, val_col].to_numpy(float)
        cdf = skewnorm.cdf(x, a, loc=loc, scale=scale)
        conf = np.maximum(cdf, 1 - cdf)

        lab = np.where(
            x <= ql, "constricted-like",
            np.where(x >= qh, "elongated-like", "intermediate"),
        )

        out.loc[idx, ["sn_alpha", "sn_loc", "sn_scale"]] = (a, loc, scale)
        out.loc[idx, ["q_low", "q_high"]] = (ql, qh)
        out.loc[idx, "class_skewnorm"] = lab
        out.loc[idx, "conf_tail"] = conf

    return out


def _icc_approx(y: np.ndarray, groups: np.ndarray) -> float:
    """Approximate ICC(1) for an unbalanced one-way random-effects design.

    ICC ≈ (MS_between − MS_within) / (MS_between + (n̄ − 1) × MS_within)
    """
    df_tmp = pd.DataFrame({"y": y, "g": groups})
    gstats = df_tmp.groupby("g").agg(
        n=("y", "size"),
        mean=("y", "mean"),
        var=("y", lambda s: np.var(s, ddof=1) if len(s) > 1 else 0.0),
    )
    k = len(gstats)
    N = int(gstats["n"].sum())
    grand_mean = float(np.average(gstats["mean"], weights=gstats["n"]))

    ss_between = float(((gstats["mean"] - grand_mean) ** 2 * gstats["n"]).sum())
    df_between = k - 1
    ms_between = ss_between / max(df_between, 1)

    ss_within = float(((gstats["n"] - 1) * gstats["var"]).sum())
    df_within = N - k
    ms_within = ss_within / max(df_within, 1)

    n_bar = float(gstats["n"].mean())
    denom = ms_between + (n_bar - 1.0) * ms_within
    if denom <= 0:
        return np.nan
    return float((ms_between - ms_within) / denom)


def pad_to_shape(vol: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Zero-pad *vol* symmetrically to *target_shape*."""
    pad_width = []
    for i in range(len(target_shape)):
        diff = target_shape[i] - vol.shape[i]
        pad_before = diff // 2
        pad_width.append((pad_before, diff - pad_before))
    return np.pad(vol, pad_width, mode="constant", constant_values=0)


def get_tile_offset(
    tile_id: int, tile_size: int, grid_cols: int
) -> Tuple[int, int]:
    """Return (x_offset, y_offset) for a grid tile."""
    row = tile_id // grid_cols
    col = tile_id % grid_cols
    return col * tile_size, row * tile_size


# =============================================================================
# Volume I/O
# =============================================================================

def _extract_index_from_filename(file_name: str) -> int:
    """Extract numeric index from filenames like ``mask_20.png``."""
    return int(file_name.rsplit("_", 1)[1].split(".")[0])


def get_instance_mask(
    input_folder: str,
    zrange: Optional[Tuple[int, int]] = None,
    tile_number: Optional[int] = None,
    tile_size: int = None,
    grid_size: int = None,
) -> np.ndarray:
    """Load and stack 2-D mask slices into a 3-D volume.

    Supports optional z-cropping and spatial tiling.
    """
    files = sorted(
        (f for f in os.listdir(input_folder)
         if f.lower().endswith((".tif", ".tiff", ".png"))),
        key=_extract_index_from_filename,
    )
    if not files:
        raise ValueError(f"No image files found in {input_folder}")

    if zrange:
        files = files[zrange[0]: zrange[1]]

    if tile_number is not None:
        row = tile_number // grid_size
        col = tile_number % grid_size
        y_start, y_end = row * tile_size, (row + 1) * tile_size
        x_start, x_end = col * tile_size, (col + 1) * tile_size
    else:
        y_start = x_start = 0
        y_end = x_end = None

    slices = []
    for f in files:
        path = os.path.join(input_folder, f)
        if f.lower().endswith((".tif", ".tiff")):
            with tifffile.TiffFile(path) as tif:
                img = tif.pages[0].asarray()[y_start:y_end, x_start:x_end]
        else:
            img = imread(path)[y_start:y_end, x_start:x_end]
        slices.append(img.copy())
        del img
        gc.collect()

    return np.stack(slices, axis=0)


def get_mask_hepa(input_folder: str) -> np.ndarray:
    """Load hepatocyte instance-labelled mask slices into a 3-D volume."""
    file_names = sorted(
        (f for f in os.listdir(input_folder) if f.endswith(".png")),
        key=_extract_index_from_filename,
    )
    volume = [imread(os.path.join(input_folder, f)) for f in file_names]
    return np.stack(volume, axis=0)


def zfiltering(volume: np.ndarray, z_threshold: int) -> np.ndarray:
    """Remove objects spanning fewer than *z_threshold* unique z-slices.

    Returns a re-labelled volume with consecutively numbered IDs.
    """
    filtered = np.zeros_like(volume)
    new_id = 1
    for region in regionprops(volume):
        z_span = len(np.unique(region.coords[:, 0]))
        if z_span >= z_threshold:
            filtered[tuple(region.coords.T)] = new_id
            new_id += 1
    logger.info("Retained %d objects after z-filtering (threshold=%d).", new_id - 1, z_threshold)
    return filtered


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_scree(pca: PCA, save_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Generate and save a PCA scree plot."""
    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(1, len(explained_var) + 1)
    ax1.bar(x, explained_var, alpha=0.7, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_ylim(0, max(explained_var) * 1.1)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_var, "r-o", markersize=4, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_ylim(0, 105)

    plt.title("PCA Scree Plot")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return explained_var, cumulative_var


def plot_pc_distribution(
    pc_values: np.ndarray,
    fit_info: dict,
    pc_name: str,
    save_path: str,
    low_conf: float,
    high_conf: float,
) -> None:
    """Plot PC histogram with skew-normal fit and threshold bands."""
    plt.figure(figsize=(9, 6))
    plt.hist(pc_values, bins=60, density=True, alpha=0.35, edgecolor="none",
             label=f"{pc_name} density")

    xs = np.linspace(pc_values.min(), pc_values.max(), 2048)

    if fit_info["model"] == "skewnorm":
        pdf = skewnorm.pdf(xs, fit_info["shape"], loc=fit_info["loc"],
                           scale=fit_info["scale"])
        plt.plot(xs, pdf, lw=2, label="Skew-normal fit")
        try:
            def _inv_cdf(p):
                return brentq(
                    lambda t: skewnorm.cdf(t, fit_info["shape"],
                                           fit_info["loc"], fit_info["scale"]) - p,
                    xs[0], xs[-1],
                )
            thr_low = _inv_cdf(low_conf)
            thr_high = _inv_cdf(high_conf)
            plt.axvspan(xs[0], thr_low, alpha=0.08, color="C3",
                        label=f"Index ≤ {low_conf:g}")
            plt.axvspan(thr_high, xs[-1], alpha=0.08, color="C2",
                        label=f"Index ≥ {high_conf:g}")
        except Exception:
            pass
        title_extra = f"(skewnorm; KS D={fit_info['ks_D']:.2f}, p={fit_info['ks_p']:.2f})"
    else:
        title_extra = "(empirical percentile)"

    plt.title(f"{pc_name} Distribution {title_extra}")
    plt.xlabel(pc_name)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pca_continuous(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    evr: np.ndarray,
    save_path: str,
    cmap: str = "cividis_r",
) -> None:
    """Scatter plot of PCA coordinates coloured by a continuous variable."""
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        df[x_col], df[y_col], c=df[color_col],
        s=25, alpha=0.85, edgecolors="white", linewidth=0.3, cmap=cmap,
    )
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label(color_col, fontsize=12, weight="bold")
    cbar.ax.tick_params(labelsize=10)

    plt.title(
        f"PCA of cell shape features\nPC1: {evr[0]*100:.1f}% | PC2: {evr[1]*100:.1f}%",
        fontsize=13, weight="bold", pad=15,
    )
    plt.xlabel("PC1", fontsize=12, weight="bold")
    plt.ylabel("PC2", fontsize=12, weight="bold")
    plt.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca_categorical(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    evr: np.ndarray,
    save_path: str,
) -> None:
    """Scatter plot of PCA coordinates coloured by categorical labels."""
    # Colorblind-safe palette (Paul Tol's muted scheme)
    colors = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
              "#DDCC77", "#CC6677", "#882255", "#AA4499", "#661100"]

    label_order = ["constricted-like", "intermediate", "elongated-like"]
    label_to_int = {lab: i for i, lab in enumerate(label_order)}
    n_labels = len(label_order)
    cmap = mcolors.ListedColormap(colors[:n_labels])
    colors_idx = np.array([label_to_int[l] for l in df[label_col]])

    plt.figure(figsize=(10, 8))
    plt.scatter(
        df[x_col], df[y_col], c=colors_idx, cmap=cmap,
        s=25, alpha=0.85, edgecolors="white", linewidth=0.3,
        vmin=-0.5, vmax=n_labels - 0.5,
    )

    legend_elements = [
        Patch(facecolor=colors[i], edgecolor="black", linewidth=0.8, label=lab)
        for i, lab in enumerate(label_order)
    ]
    plt.legend(
        handles=legend_elements, title="Cell morphology",
        title_fontsize=11, fontsize=10, loc="best",
        frameon=True, fancybox=False, edgecolor="black", framealpha=0.95,
    )

    plt.title(
        f"PCA of cell shape features\nPC1: {evr[0]*100:.1f}% | PC2: {evr[1]*100:.1f}%",
        fontsize=13, weight="bold", pad=15,
    )
    plt.xlabel("PC1", fontsize=12, weight="bold")
    plt.ylabel("PC2", fontsize=12, weight="bold")
    plt.tick_params(labelsize=10)
    plt.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_on_pca(
    df: pd.DataFrame,
    feature_cols: list,
    output_dir: str,
    percentile_clip: Tuple[float, float] = (0, 100),
) -> None:
    """Generate per-feature PCA overlay plots."""
    os.makedirs(output_dir, exist_ok=True)
    lo_pct, hi_pct = percentile_clip

    for feat in feature_cols:
        vals = df[feat].to_numpy().copy()
        lo, hi = np.percentile(vals, lo_pct), np.percentile(vals, hi_pct)
        vals = np.clip(vals, lo, hi)

        plt.figure(figsize=(10, 8))
        sc = plt.scatter(df["PC1"], df["PC2"], c=vals, cmap="coolwarm_r",
                         s=30, alpha=0.8, edgecolor="k")
        cbar = plt.colorbar(sc)
        cbar.set_label(f"{feat} (clipped {lo_pct}–{hi_pct}%)", rotation=270, labelpad=15)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA with {feat}")
        plt.savefig(os.path.join(output_dir, f"{feat}_on_PCA.png"), dpi=300)
        plt.close()


# =============================================================================
# Heatmap Generation
# =============================================================================

def heat_map_generate(
    df: pd.DataFrame,
    heatmap_path: str,
    index_col: str = "PC1",
    zscore_clip: float = None,
    figsize: Tuple[int, int] = [None, None]
) -> dict:
    """Generate a heatmap of morphological features sorted by *index_col*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and the sorting column.
    heatmap_path : str
        Output path for the saved figure.
    index_col : str
        Column used to order samples along the x-axis.
    zscore_clip : float
        Symmetric z-score clipping threshold.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    dict
        Metadata including sample/feature counts and ordering info.
    """
    df = df.copy()

    if index_col not in df.columns:
        raise ValueError(f"Index column '{index_col}' not found in DataFrame")
    if not pd.api.types.is_numeric_dtype(df[index_col]):
        raise ValueError(f"Index column '{index_col}' must be numeric")

    order_vals = df[index_col].astype(float).to_numpy()

    # Remove rows with non-finite sorting values
    mask = np.isfinite(order_vals)
    if (~mask).sum() > 0:
        logger.warning("Removed %d rows with invalid %s values.", (~mask).sum(), index_col)
        df = df.loc[mask].reset_index(drop=True)
        order_vals = order_vals[mask]
    if len(df) == 0:
        raise ValueError("No valid samples remaining after removing NaN values")

    col_order = np.argsort(order_vals)

    # Identify feature columns (exclude annotations and derived columns)
    drop_cols = {"PC1", "PC2", "elongation_index", "UMAP1", "UMAP2"}
    non_feature_patterns = ["id", "label", "class", "group", "sample"]
    for col in df.columns:
        if any(pat in col.lower() for pat in non_feature_patterns):
            drop_cols.add(col)

    feature_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in drop_cols
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    logger.info("Heatmap: %d features × %d samples, sorted by '%s'.",
                len(feature_cols), len(df), index_col)

    feat_mat = df[feature_cols].to_numpy(dtype=float)

    # Column-wise z-score normalisation
    feat_mean = feat_mat.mean(axis=0, keepdims=True)
    feat_std = feat_mat.std(axis=0, keepdims=True) + 1e-8
    feat_mat_z = (feat_mat - feat_mean) / feat_std
    if zscore_clip is not None:
        feat_mat_z = np.clip(feat_mat_z, -zscore_clip, zscore_clip)

    # Hierarchical clustering of features
    try:
        Z_feat = linkage(feat_mat_z.T, method="ward")
        dend = dendrogram(Z_feat, no_plot=True)
        row_order = dend["leaves"]
    except Exception as exc:
        logger.warning("Feature clustering failed (%s); using original order.", exc)
        row_order = list(range(len(feature_cols)))

    H = feat_mat_z.T[row_order, :][:, col_order]
    ordered_feature_names = [feature_cols[i] for i in row_order]

    ord_vals_sorted = order_vals[col_order].astype(float)
    top_bar = ord_vals_sorted.reshape(1, -1)
    idx_median = float(np.nanmedian(top_bar))
    max_dev = max(float(np.nanmax(top_bar)) - idx_median,
                  idx_median - float(np.nanmin(top_bar)))
    vmin_idx = idx_median - max_dev
    vmax_idx = idx_median + max_dev

    # --- Plot ---
    sns.set(context="notebook", style="white")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 19], hspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])
    im_top = ax_top.imshow(top_bar, aspect="auto", interpolation="nearest",
                           cmap="cividis_r", vmin=vmin_idx, vmax=vmax_idx)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title(f"{index_col} (z-scored: low → high)", fontsize=10, pad=2)
    cbar_top = fig.colorbar(im_top, ax=ax_top, fraction=0.15, pad=0.02)
    cbar_top.ax.tick_params(labelsize=8)
    cbar_top.set_label("z-score", rotation=270, labelpad=12, fontsize=8)

    ax_main = fig.add_subplot(gs[1, 0])
    hm = ax_main.imshow(H, aspect="auto", interpolation="nearest", cmap="vlag_r",
                         vmin=-zscore_clip, vmax=zscore_clip)
    ax_main.set_yticks(np.arange(len(ordered_feature_names)))
    ax_main.set_yticklabels(ordered_feature_names, fontsize=9)
    ax_main.set_xticks([])
    ax_main.set_xlabel(f"Mitochondria (sorted by {index_col})", fontsize=10)
    ax_main.set_ylabel("Features (hierarchically clustered)", fontsize=10)
    ax_main.set_title(
        f"Morphological Feature Heatmap (sorted by {index_col})", fontsize=12, pad=10,
    )
    cbar = fig.colorbar(hm, ax=ax_main, fraction=0.046, pad=0.02)
    cbar.set_label("Feature z-score", rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved → %s", heatmap_path)

    return {
        "n_samples": len(col_order),
        "n_features": len(feature_cols),
        "sorting_col": index_col,
        "feature_names": ordered_feature_names,
    }


# =============================================================================
# Core Analysis: PCA + Morphological Indexing
# =============================================================================

# Columns generated by PyRadiomics that are non-numeric diagnostics
_PYRADIOMICS_DIAGNOSTIC_COLS = [
    "diagnostics_Versions_PyRadiomics", "diagnostics_Versions_Numpy",
    "diagnostics_Versions_SimpleITK", "diagnostics_Versions_PyWavelet",
    "diagnostics_Versions_Python", "diagnostics_Configuration_Settings",
    "diagnostics_Configuration_EnabledImageTypes", "diagnostics_Image-original_Hash",
    "diagnostics_Image-original_Dimensionality", "diagnostics_Image-original_Spacing",
    "diagnostics_Image-original_Size", "diagnostics_Mask-original_Hash",
    "diagnostics_Mask-original_Spacing", "diagnostics_Mask-original_Size",
    "diagnostics_Mask-original_BoundingBox",
    "diagnostics_Mask-original_CenterOfMassIndex",
    "diagnostics_Mask-original_CenterOfMass",
]

# Coordinate / bounding-box columns to exclude from feature PCA
_COORD_COLS = [
    "centroid_x_2k", "centroid_y_2k", "centroid_z_2k",
    "centroid_x_10k", "centroid_y_10k", "centroid_z_10k",
    "bbox_x_min_2k", "bbox_x_max_2k", "bbox_y_min_2k", "bbox_y_max_2k",
    "bbox_z_min_2k", "bbox_z_max_2k",
    "bbox_x_min_10k", "bbox_x_max_10k", "bbox_y_min_10k", "bbox_y_max_10k",
    "bbox_z_min_10k", "bbox_z_max_10k", "voxel_count",
]


def cluster_feature(
    feature_file: str,
    output_root: str,
    high_conf: float,
    low_conf: float,
    outlier_threshold: float,
    ks_threshold: float,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """Run PCA on radiomics features and compute morphology indices.

    Performs full PCA, outlier removal, skew-normal fitting of PC1/PC2,
    and generates diagnostic plots (scree, distributions, scatter).

    Parameters
    ----------
    feature_file : str
        Path to CSV with radiomic shape features.
    output_root : str
        Root directory for all outputs.
    high_conf / low_conf : float
        CDF thresholds for soft label assignment.
    outlier_threshold : float
        Z-score beyond which samples are removed.
    ks_threshold : float
        Kolmogorov-Smirnov D-statistic above which the skew-normal fit is
        rejected in favour of an empirical-percentile fallback. Dataset
        dependent — inspect the KS D values logged for your PC1/PC2 fits
        and choose accordingly (there is no generic default).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    df_out : pd.DataFrame
        Annotated DataFrame with PC scores, indices, and labels.
    results : dict
        Fit information, PCA loadings, and variance explained.
    """
    csv_dir = os.path.join(output_root, "csv_feature_files")
    fig_dir = os.path.join(output_root, "result_figures")
    feature_pca_dir = os.path.join(fig_dir, "featuremaponPCA")
    for d in [csv_dir, fig_dir, feature_pca_dir]:
        os.makedirs(d, exist_ok=True)

    # --- Load & preprocess ---
    df = pd.read_csv(feature_file)
    logger.info("Loaded %d samples from %s", len(df), feature_file)

    df = df.drop(columns=_PYRADIOMICS_DIAGNOSTIC_COLS, errors="ignore")

    # Invert elongation/flatness so higher → more constricted (consistent axis)
    for col in ["Elongation", "Flatness"]:
        if col in df.columns:
            logger.info("Inverting %s (higher → more constricted).", col)
            df[col] = -df[col]

    df = df.dropna()
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in all_numeric if c not in _COORD_COLS]
    logger.info("Using %d numeric features (excluding coordinates).", len(feature_cols))

    # --- Standardise ---
    scaler = StandardScaler()
    Xz = scaler.fit_transform(df[feature_cols].values)

    # --- Full PCA (scree) ---
    n_full = min(len(feature_cols), len(df) - 1)
    pca_full = PCA(n_components=n_full, random_state=random_state)
    pca_full.fit(Xz)

    explained_var, cumulative_var = plot_scree(
        pca_full, os.path.join(fig_dir, "pca_scree_plot.png")
    )
    n80 = int(np.argmax(cumulative_var >= 80) + 1)
    n90 = int(np.argmax(cumulative_var >= 90) + 1)
    n95 = int(np.argmax(cumulative_var >= 95) + 1)
    logger.info("PC1: %.1f%%, PC2: %.1f%% | 80%%/90%%/95%%: %d/%d/%d components.",
                explained_var[0], explained_var[1], n80, n90, n95)

    for pc_idx, pc_name in enumerate(["PC1", "PC2"]):
        loadings = pd.Series(pca_full.components_[pc_idx], index=feature_cols)
        logger.info("[%s] Top-5 positive: %s", pc_name,
                    dict(loadings.nlargest(5)))
        logger.info("[%s] Top-5 negative: %s", pc_name,
                    dict(loadings.nsmallest(5)))

    pca_info = {
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative_variance": cumulative_var,
        "n_components_80pct": n80,
        "n_components_90pct": n90,
        "n_components_95pct": n95,
        "pc1_loadings": dict(zip(feature_cols, pca_full.components_[0])),
        "pc2_loadings": dict(zip(feature_cols, pca_full.components_[1])),
    }

    # --- Outlier removal ---
    outlier_mask = np.any(np.abs(Xz) > outlier_threshold, axis=1)
    n_outliers = int(outlier_mask.sum())
    logger.info("Removing %d outliers (%.1f%%) with |z| > %.1f.",
                n_outliers, 100 * n_outliers / len(df), outlier_threshold)
    np.save(os.path.join(csv_dir, "outlier_mask.npy"), outlier_mask)
    np.save(os.path.join(csv_dir, "kept_indices.npy"), np.where(~outlier_mask)[0])

    df = df[~outlier_mask].reset_index(drop=True)
    Xz = Xz[~outlier_mask]

    # --- Final 2-component PCA ---
    pca_final = PCA(n_components=2, random_state=random_state)
    pcs = pca_final.fit_transform(Xz)
    pc1, pc2 = pcs[:, 0], pcs[:, 1]
    evr = pca_final.explained_variance_ratio_

    # Skew-normal indices
    ei_pc1, fit_info_pc1 = _fit_skewnorm(pc1, "PC1", ks_threshold=ks_threshold)
    ei_pc2, fit_info_pc2 = _fit_skewnorm(pc2, "PC2", ks_threshold=ks_threshold)
    labels = _assign_soft_labels(ei_pc1, high_conf, low_conf)

    unique, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        logger.info("  %s: %d (%.1f%%)", lbl, cnt, 100 * cnt / len(labels))

    # --- Assemble output ---
    df_out = df.copy()
    df_out["PC1"] = pc1
    df_out["PC2"] = pc2
    df_out["elongation_index_PC1"] = ei_pc1
    df_out["elongation_index_PC2"] = ei_pc2
    df_out["label_soft"] = labels

    output_csv = os.path.join(csv_dir, "cluster_df_PCA_Skew.csv")
    df_out.to_csv(output_csv, index=False)
    logger.info("Saved annotated CSV → %s", output_csv)

    # --- Plots ---
    plot_pc_distribution(pc1, fit_info_pc1, "PC1",
                         os.path.join(fig_dir, "pc1_skew_distribution.png"),
                         low_conf, high_conf)
    plot_pc_distribution(pc2, fit_info_pc2, "PC2",
                         os.path.join(fig_dir, "pc2_skew_distribution.png"),
                         low_conf, high_conf)
    plot_pca_continuous(df_out, "PC1", "PC2", "elongation_index_PC1", evr,
                        os.path.join(fig_dir, "pca_prob_EI_PC1.png"))
    plot_pca_continuous(df_out, "PC1", "PC2", "elongation_index_PC2", evr,
                        os.path.join(fig_dir, "pca_prob_EI_PC2.png"))
    plot_pca_categorical(df_out, "PC1", "PC2", "label_soft", evr,
                         os.path.join(fig_dir, "pca_soft_labels.png"))
    plot_feature_on_pca(df_out, feature_cols, feature_pca_dir)

    results = {
        "fit_info_pc1": fit_info_pc1,
        "fit_info_pc2": fit_info_pc2,
        "pca_info": pca_info,
        "explained_variance_ratio": evr,
    }
    return df_out, results


# =============================================================================
# Per-Hepatocyte Distribution Analysis
# =============================================================================
def _compute_summary_and_stats(
    data: pd.DataFrame,
    high_conf: float,
    low_conf: float,
) -> Tuple[pd.DataFrame, dict, dict]:
    """Per-hepatocyte summary statistics and cross-cell homeostasis tests.

    Fits a skew-normal per hepatocyte, classifies each mitochondrion,
    and computes ANOVA, Kruskal–Wallis, ICC, and CV diagnostics.
    """
    d = _classify_by_skewnorm_quantiles(data, "hep_id", "PC1", low_conf, high_conf)

    g = d.groupby("hep_id")
    summary = g.agg(
        n=("PC1", "size"),
        mean_PC1=("PC1", "mean"),
        sd_PC1=("PC1", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
        alpha=("sn_alpha", "first"),
        loc=("sn_loc", "first"),
        scale=("sn_scale", "first"),
        q_low=("q_low", "first"),
        q_high=("q_high", "first"),
        frac_constricted=("class_skewnorm", lambda s: np.mean(s == "constricted-like")),
        frac_intermediate=("class_skewnorm", lambda s: np.mean(s == "intermediate")),
        frac_elongated=("class_skewnorm", lambda s: np.mean(s == "elongated-like")),
    ).reset_index()

    # Cross-hepatocyte tests
    y_groups = [grp["PC1"].to_numpy(float) for _, grp in d.groupby("hep_id")]
    anova_p = kruskal_p = np.nan
    try:
        if len(y_groups) >= 2:
            anova_p = float(f_oneway(*y_groups).pvalue)
            kruskal_p = float(kruskal(*y_groups).pvalue)
    except Exception:
        pass

    icc = _icc_approx(d["PC1"].to_numpy(float), d["hep_id"].to_numpy(int))

    cv_means = (
        float(np.std(summary["mean_PC1"], ddof=1) / (np.mean(summary["mean_PC1"]) + 1e-12))
        if len(summary) > 1 else np.nan
    )

    stats_out = {
        "anova_p": anova_p,
        "kruskal_p": kruskal_p,
        "icc_approx": icc,
        "cv_mean_PC1": cv_means,
        "n_hepatocytes": int(summary.shape[0]),
        "low_quantile": float(low_conf),
        "high_quantile": float(high_conf),
    }
    notes = {"classification": "per-hepatocyte skew-normal quantiles"}

    logger.info(
        "Per-hepatocyte stats — ANOVA p=%.3g, Kruskal p=%.3g, ICC≈%.3f, CV≈%.3f",
        anova_p, kruskal_p, icc, cv_means,
    )
    return summary, notes, stats_out

def _generate_violin(
    data: pd.DataFrame,
    out_dir: str,
    hep_col: str = "hep_id",
    value_col: str = "PC1",
    top_k: int = None,
    min_mito_per_hep: int = None,
    sort_by: str = "mean",
    title: Optional[str] = None,
    filename: str = "violin_pc1_topHep.png",
) -> pd.DataFrame:
    """Violin + boxplot of per-hepatocyte PC1 distributions.

    Parameters
    ----------
    sort_by : str
        ``"mean"`` (default) to order by mean PC1, or ``"hep_id"`` for label order.
    """
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    os.makedirs(out_dir, exist_ok=True)

    counts = data.groupby(hep_col).size()
    counts = counts[counts >= min_mito_per_hep].sort_values(ascending=False)
    if len(counts) == 0:
        raise ValueError(f"No hepatocytes with ≥ {min_mito_per_hep} mitochondria.")
    hep_list = counts.head(top_k).index.tolist()
    dplot = data[data[hep_col].isin(hep_list)].copy()

    g = dplot.groupby(hep_col)[value_col]
    summary = pd.DataFrame({
        hep_col: g.count().index,
        "n": g.count().values,
        "mean": g.mean().values,
        "sd": g.std(ddof=1).fillna(0.0).values,
    })
    dfree = np.maximum(summary["n"] - 1, 1)
    tcrit = t_dist.ppf(0.975, dfree)
    summary["ci_half"] = tcrit * (summary["sd"] / np.sqrt(summary["n"]))

    if sort_by == "mean":
        summary = summary.sort_values("mean").reset_index(drop=True)
        sort_label = "sorted by mean PC1"
    else:
        summary = summary.sort_values(hep_col).reset_index(drop=True)
        sort_label = ""

    order = summary[hep_col].tolist()
    dplot["_hep_str"] = dplot[hep_col].astype(str)

    # Desaturated colour palette
    base_cmap = plt.colormaps.get_cmap("hsv")
    colors = []
    for i in range(len(order)):
        rgba = base_cmap(i / len(order))
        hsv = rgb_to_hsv(np.array(rgba[:3]))
        hsv[1] *= 0.55
        hsv[2] = min(hsv[2] * 0.8, 0.75)
        colors.append(hsv_to_rgb(hsv))
    palette = {str(h): colors[i] for i, h in enumerate(order)}

    sns.set(context="notebook", style="white")
    fig_w = max(12, 0.45 * len(order) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    sns.violinplot(
        data=dplot, x="_hep_str", y=value_col,
        order=[str(h) for h in order], palette=palette,
        inner=None, cut=1, scale="width", linewidth=0.8, alpha=0.4, ax=ax,
    )
    sns.boxplot(
        data=dplot, x="_hep_str", y=value_col,
        order=[str(h) for h in order],
        whis=(5, 95), width=0.22, showcaps=True, showfliers=False,
        meanline=True, showmeans=True,
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=0),
        meanprops=dict(color="k", linewidth=2.0),
        zorder=3, ax=ax,
    )
    for i, patch in enumerate(ax.patches):
        if i < len(order):
            c = colors[i]
            patch.set_facecolor((*c, 0.7))
            patch.set_edgecolor("k")

    ax.set_xlabel(
        f"Hepatocyte ({sort_label})" if sort_label else "Hepatocyte",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("PC1 morphology index", fontsize=12, fontweight="bold")
    ttl = title or f"PC1 distribution per hepatocyte (top {len(order)})"
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(range(len(order)), ha="right", fontsize=8)
    ax.set_title(ttl, fontsize=13, fontweight="bold", pad=10)
    ax.grid(False)
    sns.despine(ax=ax, left=False, bottom=False)

    y_max = ax.get_ylim()[1]
    for i, row in summary.iterrows():
        ax.text(i, y_max, f"n={row['n']}", ha="center", va="bottom",
                fontsize=6, color="gray", rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=400, bbox_inches="tight")
    plt.close()
    logger.info("Saved violin plot: %s", filename)
    return summary


def per_hepatocyte_mito_distribution(
    pca_csv: str,
    hepatocyte_mask_folder: str,
    mito_instance_folder: str,
    outlier_mask_file: str,
    output_dir: str,
    high_conf: float,
    low_conf: float,
    z_threshold: int,
    cache_file: Optional[str] = None,
    force_recompute: bool = False,
) -> Tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """Assign mitochondria to hepatocytes and analyse per-cell distributions.

    Parameters
    ----------
    pca_csv : str
        CSV with at least ``VoxelVolume`` and ``PC1_morphology_index`` columns.
    hepatocyte_mask_folder : str
        Folder with hepatocyte instance-labelled mask slices.
    mito_instance_folder : str
        Folder with mitochondria instance-labelled mask slices.
    outlier_mask_file : str
        Path to ``outlier_mask.npy`` from ``cluster_feature``.
    output_dir : str
        Directory for figures and cache.
    z_threshold : int
        Minimum z-span (in slices) for a mitochondrion instance to be kept.
        Dataset-specific — use the same value as `morphology_features.py`.
    cache_file : str, optional
        Path for pickle cache (speeds up re-runs).

    Returns
    -------
    summary, notes, stats, data
    """
    if cache_file is None:
        cache_file = os.path.join(output_dir, "cached_hepatocyte_mito_data.pkl")

    if not force_recompute and os.path.exists(cache_file):
        try:
            logger.info("Loading cached data from %s", cache_file)
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            data = cached["data"]
            if "p" not in data.columns:
                data["p"] = np.nan

            summary, notes, stats = _compute_summary_and_stats(data, high_conf, low_conf)
            _generate_violin(data, out_dir=output_dir, top_k=30,
                             sort_by="mean",
                             filename="violin_pc1_per_hepatocyte.png")
            return summary, notes, stats, data

        except Exception as exc:
            logger.warning("Cache load failed (%s); recomputing.", exc)

    # --- Load masks ---
    logger.info("Loading hepatocyte and mitochondria masks...")
    hep_lab = get_mask_hepa(hepatocyte_mask_folder)
    inst = zfiltering(get_instance_mask(mito_instance_folder), z_threshold)
    mito_regs = regionprops(inst)
    logger.info("Mitochondria regions: %d", len(mito_regs))

    # --- Load CSV ---
    df = pd.read_csv(pca_csv)
    for c in ["VoxelVolume", "PC1_morphology_index"]:
        if c not in df.columns:
            raise ValueError(f"'{c}' missing in {pca_csv}")

    # --- Outlier removal ---
    outlier_mask = np.load(outlier_mask_file)
    logger.info("Outlier mask: %d outliers.", outlier_mask.sum())
    mito_regs = [r for i, r in enumerate(mito_regs) if not outlier_mask[i]]

    if len(df) != len(mito_regs):
        raise ValueError(
            f"CSV rows ({len(df)}) ≠ mito regions ({len(mito_regs)})."
        )

    vol_all = df["VoxelVolume"].to_numpy(float)
    pc1_all = df["PC1_morphology_index"].to_numpy(float)

    # --- Centroid → hepatocyte mapping ---
    cents = np.array([np.round(r.centroid).astype(int) for r in mito_regs])
    Z, Y, X = hep_lab.shape
    inb = (
        (cents[:, 0] >= 0) & (cents[:, 0] < Z)
        & (cents[:, 1] >= 0) & (cents[:, 1] < Y)
        & (cents[:, 2] >= 0) & (cents[:, 2] < X)
    )
    n_oob = int((~inb).sum())
    if n_oob:
        logger.warning("%d mito centroids out of bounds; dropping.", n_oob)

    cents, vol_all, pc1_all = cents[inb], vol_all[inb], pc1_all[inb]
    hep_ids = hep_lab[cents[:, 0], cents[:, 1], cents[:, 2]]
    in_hep = hep_ids > 0
    n_nohep = int((~in_hep).sum())
    if n_nohep:
        logger.info("%d mito centroids outside any hepatocyte.", n_nohep)

    data = pd.DataFrame({
        "hep_id": hep_ids[in_hep].astype(int),
        "PC1": pc1_all[in_hep],
        "vol": vol_all[in_hep],
    })

    # --- Cache ---
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"data": data, "n_oob": n_oob, "n_nohep": n_nohep}, f)
    except Exception as exc:
        logger.warning("Failed to save cache: %s", exc)

    summary, notes, stats = _compute_summary_and_stats(data, high_conf, low_conf)
    notes.update({"n_regions": len(mito_regs), "n_out_of_bounds": n_oob,
                  "n_no_hepatocyte": n_nohep})

    _generate_violin(data, out_dir=output_dir, top_k=30,
                     sort_by="mean",
                     filename="violin_pc1_per_hepatocyte.png")

    return summary, notes, stats, data
# =============================================================================
# ER–Mitochondria narrowing Analysis
# =============================================================================

def classify_mito_by_pc2(
    pc2_value: float,
    pc2_narrowing_low: float = None,
    pc2_narrowing_high: float = None,
) -> str:
    """Classify a mitochondrion as ``"narrowing"``, ``"normal"``, or ``"artifact"``."""
    if pc2_value >= pc2_narrowing_high:
        return "artifact"
    elif pc2_value > pc2_narrowing_low:
        return "narrowing"
    return "normal"


def find_narrowing_site(
    mito_mask: np.ndarray,
    surface_coords: np.ndarray,
    n_slices: int = None,
    exclude_end_fraction: float = None,
    min_ratio_threshold: float = None,
) -> Tuple:
    """Identify a narrowing site along the major axis of a mitochondrion.

    Returns
    -------
    constrict_pos : float or None
        Normalised position (0–1) along the major axis.
    constrict_pt : ndarray or None
        3-D coordinate of the narrowing point.
    min_median_ratio : float or None
        Cross-section area ratio at narrowing vs. median.
    near_surface_coords, far_surface_coords : ndarray
        Surface voxels near/far from the narrowing.
    is_valid : bool
        Whether a valid narrowing was found.
    """
    coords = np.argwhere(mito_mask)
    if len(coords) < 10:
        return None, None, None, surface_coords, np.empty((0, 3)), False

    centered = coords - coords.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    major_axis = eigenvectors[:, -1]

    projections = centered @ major_axis
    proj_min, proj_max = projections.min(), projections.max()
    proj_range = proj_max - proj_min
    if proj_range < 1e-6:
        return None, None, None, surface_coords, np.empty((0, 3)), False

    valid_start = proj_min + exclude_end_fraction * proj_range
    valid_end = proj_max - exclude_end_fraction * proj_range

    slice_edges = np.linspace(proj_min, proj_max, n_slices + 1)
    areas = np.array([
        ((projections >= slice_edges[i]) & (projections < slice_edges[i + 1])).sum()
        for i in range(n_slices)
    ])
    centers = (slice_edges[:-1] + slice_edges[1:]) / 2
    valid_mask = (centers >= valid_start) & (centers <= valid_end)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0 or areas[valid_idx].max() == 0:
        return None, None, None, surface_coords, np.empty((0, 3)), False

    min_valid = valid_idx[np.argmin(areas[valid_idx])]
    median_area = np.median(areas[areas > 0])
    min_area = areas[min_valid]
    ratio = min_area / median_area if median_area > 0 else 1.0

    if ratio > min_ratio_threshold:
        return None, None, ratio, surface_coords, np.empty((0, 3)), False

    constrict_proj = centers[min_valid]
    constrict_pos = (constrict_proj - proj_min) / proj_range
    constrict_pt = coords[np.argmin(np.abs(projections - constrict_proj))]

    # Split surface into near/far (±25% of range around narrowing)
    surface_centered = surface_coords - coords.mean(axis=0)
    surface_proj = surface_centered @ major_axis
    threshold = 0.25 * proj_range
    near_mask = np.abs(surface_proj - constrict_proj) < threshold

    return (constrict_pos, constrict_pt, ratio,
            surface_coords[near_mask], surface_coords[~near_mask], True)


def find_middle_region(
    mito_mask: np.ndarray,
    surface_coords: np.ndarray,
    near_fraction: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """For non-narrowing mitochondria, define the middle 50 % as "nearby"."""
    coords = np.argwhere(mito_mask)
    if len(coords) < 10:
        return surface_coords, np.empty((0, 3))

    centered = coords - coords.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
    major_axis = eigenvectors[:, -1]

    projections = centered @ major_axis
    proj_min, proj_max = projections.min(), projections.max()
    proj_range = proj_max - proj_min
    if proj_range < 1e-6:
        return surface_coords, np.empty((0, 3))

    margin = (1 - near_fraction) / 2
    near_start = proj_min + margin * proj_range
    near_end = proj_max - margin * proj_range

    surface_proj = (surface_coords - coords.mean(axis=0)) @ major_axis
    near_mask = (surface_proj >= near_start) & (surface_proj <= near_end)
    return surface_coords[near_mask], surface_coords[~near_mask]


def _analyze_single_mito(args: tuple) -> Optional[dict]:
    """Worker function for parallel narrowing + ER contact analysis."""
    (mito_id, mito_mask, er_crop, zmin, mito_info,
     pc2_low, pc2_high, exclude_end_frac, min_ratio_thr, is_narrowing,
     n_slices, near_fraction) = args

    try:
        pc2 = mito_info.get("PC2", 0)
        pc1 = mito_info.get("PC1", 0)
        ftype = classify_mito_by_pc2(pc2, pc2_low, pc2_high)
        if ftype in ("artifact", "normal"):
            return None

        er_dilated = binary_dilation(er_crop, ball(2))
        surface = binary_dilation(mito_mask, ball(1)) & ~mito_mask
        surface_coords = np.argwhere(surface)

        if is_narrowing:
            (constrict_pos, constrict_pt, ratio,
             near_surf, far_surf, valid) = find_narrowing_site(
                mito_mask, surface_coords,
                n_slices=n_slices,
                exclude_end_fraction=exclude_end_frac,
                min_ratio_threshold=min_ratio_thr,
            )
            # Fallback: if no valid constriction found, use middle 50%
            if not valid:
                near_surf, far_surf = find_middle_region(
                    mito_mask, surface_coords, near_fraction=near_fraction
                )
                valid = True
        else:
            near_surf, far_surf = find_middle_region(
                mito_mask, surface_coords, near_fraction=near_fraction
            )
            constrict_pos = constrict_pt = ratio = None
            valid = True

        contact_total = set(map(tuple, np.argwhere(surface & er_dilated)))
        near_set = set(map(tuple, near_surf))
        far_set = set(map(tuple, far_surf)) if len(far_surf) > 0 else set()

        contact_near = len(near_set & contact_total)
        contact_far = len(far_set & contact_total)
        area_nearby = len(near_surf)
        area_total = len(surface_coords)

        if area_total == 0:
            return None

        return {
            "mito_id": mito_id,
            "csv_idx": mito_info.get("_csv_idx", -1),
            "zmin": zmin,
            "fission_type": ftype,
            "PC1": pc1,
            "PC2": pc2,
            "centroid_x_10k": mito_info.get("centroid_x_10k", 0),
            "centroid_y_10k": mito_info.get("centroid_y_10k", 0),
            "centroid_z_10k": mito_info.get("centroid_z_10k", 0),
            "has_valid_narrowing": valid,
            "narrowing_pos": constrict_pos,
            "narrowing_coords": tuple(constrict_pt) if constrict_pt is not None else None,
            "min_median_ratio": ratio,
            "surface_area": area_total,
            "nearby_surface_area": area_nearby,
            "nearby_contact_count": contact_near,
            "far_contact_count": contact_far,
            "nearby_contact_ratio": contact_near / area_nearby if area_nearby > 0 else 0,
            "total_contact_ratio": (contact_near + contact_far) / area_total,
        }
    except Exception as exc:
        logger.error("Mito %s failed: %s", mito_id, exc, exc_info=True)
        return None


def match_by_bbox_with_tile_offset(
    props: list,
    analysis_df: pd.DataFrame,
    z_range: Tuple[int, int],
    tile_id: int,
    coord_space: str = None,
    tile_size: int = None,
    grid_cols: int = None,
    bbox_tolerance: int = None,
    downsample_tolerance: int = None,
) -> Tuple[list, list, list]:
    """Match volume regions to CSV rows using bounding-box IoU with tile offset.

    Returns ``(matches, unmatched_props, unmatched_csv_indices)``.
    """
    x_off, y_off = get_tile_offset(tile_id, tile_size, grid_cols)
    z_off = z_range[0]
    total_tol = bbox_tolerance + downsample_tolerance

    sfx = "_10k" if coord_space == "10k" else "_2k"
    cols = {ax: (f"bbox_{ax}_min{sfx}", f"bbox_{ax}_max{sfx}") for ax in ("x", "y", "z")}
    required = [c for pair in cols.values() for c in pair]
    missing = [c for c in required if c not in analysis_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Pre-filter to tile region
    xmc, xMc = cols["x"]
    ymc, yMc = cols["y"]
    zmc, zMc = cols["z"]
    in_tile = (
        (analysis_df[xmc] <= x_off + tile_size + total_tol)
        & (analysis_df[xMc] >= x_off - total_tol)
        & (analysis_df[ymc] <= y_off + tile_size + total_tol)
        & (analysis_df[yMc] >= y_off - total_tol)
        & (analysis_df[zmc] <= z_off + (z_range[1] - z_range[0]) + total_tol)
        & (analysis_df[zMc] >= z_off - total_tol)
    )
    df_tile = analysis_df[in_tile].copy()
    if len(df_tile) == 0:
        return [], list(props), list(analysis_df.index)

    csv_bboxes = df_tile[[xmc, xMc, ymc, yMc, zmc, zMc]].values
    csv_indices = df_tile.index.values

    matches, unmatched_props = [], []
    matched_csv = set()

    for prop in props:
        bb = prop.bbox
        vx0, vx1 = bb[2] + x_off, bb[5] + x_off
        vy0, vy1 = bb[1] + y_off, bb[4] + y_off
        vz0, vz1 = bb[0] + z_off, bb[3] + z_off
        vol_vol = (vx1 - vx0) * (vy1 - vy0) * (vz1 - vz0)

        best_idx, best_iou = None, 0
        for i, (xm, xM, ym, yM, zm, zM) in enumerate(csv_bboxes):
            if csv_indices[i] in matched_csv:
                continue
            ix0 = max(vx0, xm - total_tol)
            ix1 = min(vx1, xM + total_tol)
            iy0 = max(vy0, ym - total_tol)
            iy1 = min(vy1, yM + total_tol)
            iz0 = max(vz0, zm - total_tol)
            iz1 = min(vz1, zM + total_tol)
            if ix1 > ix0 and iy1 > iy0 and iz1 > iz0:
                inter = (ix1 - ix0) * (iy1 - iy0) * (iz1 - iz0)
                csv_vol = (xM - xm) * (yM - ym) * (zM - zm)
                union = vol_vol + csv_vol - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_idx = iou, i

        if best_idx is not None and best_iou > 0:
            ci = csv_indices[best_idx]
            matched_csv.add(ci)
            row = df_tile.loc[ci].to_dict()
            row["_csv_idx"] = ci
            row["_match_iou"] = best_iou
            matches.append((prop, row, best_iou))
        else:
            unmatched_props.append(prop)

    unmatched_csv = [i for i in csv_indices if i not in matched_csv]
    return matches, unmatched_props, unmatched_csv


def analyze_narrowing_mito_er_interaction(
    input_folder_mito: str,
    input_folder_er: str,
    cluster_csv: str,
    output_dir_base: str,
    pc2_narrowing_low: float = None,
    pc2_narrowing_high: float = None,
    pc1_low: float = None,
    pc1_high: float = None,
    exclude_end_fraction: float = None,
    min_ratio_threshold: float = None,
    n_slices: int = None,
    near_fraction: float = None,
    coord_space: str = None,
    tile_size: int = None,
    grid_cols: int = None,
    bbox_tolerance: int = 0,
    downsample_tolerance: int = 0,
    z_range: Optional[Tuple[int, int]] = None,
    tile_id: Optional[int] = None,
    save_visualization: bool = True,
    is_narrowing: bool = True,
) -> list:
    """Analyse ER–mitochondria contact at narrowing sites (per tile).

    Loads mito/ER volumes for a single tile, matches mitochondria to CSV
    entries via bounding-box IoU, quantifies surface contact near vs. far
    from narrowing sites, and exports per-mito results to CSV.

    Parameters
    ----------
    tile_id : int, optional
        Defaults to ``SLURM_ARRAY_TASK_ID`` environment variable or 0.
    n_slices : int
        Number of cross-sections sampled along the mitochondrion's major
        axis when searching for a narrowing site. Dataset/resolution
        dependent; must be supplied by the caller.
    near_fraction : float
        Fraction (0-1) of the major-axis length defined as "near" the
        narrowing site (or, for non-narrowing mitochondria, the middle
        region). Dataset dependent; must be supplied by the caller.
    is_narrowing : bool
        If True, analyse narrowing sites; if False, use the middle 50 %.
    """
    if tile_id is None:
        tile_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if z_range is None:
        raise ValueError(
            "z_range must be specified (start, end) — it depends on the "
            "z-extent of your volume and has no generic default."
        )

    os.makedirs(output_dir_base, exist_ok=True)
    output_csv = os.path.join(
        output_dir_base,
        f"er_mito_narrowing_pc2_{pc2_narrowing_low}_{pc2_narrowing_high}_tile{tile_id}.csv",
    )

    cluster_df = pd.read_csv(cluster_csv)
    logger.info("Loaded %d rows from %s", len(cluster_df), cluster_csv)

    # PC filtering
    pc1_mask = (cluster_df["PC1"] >= pc1_low) & (cluster_df["PC1"] <= pc1_high)
    non_artifact = (cluster_df["PC2"] < pc2_narrowing_high) & pc1_mask
    analysis_df = cluster_df[non_artifact].copy()

    if len(analysis_df) == 0:
        logger.warning("No mitochondria for analysis after filtering.")
        return []
    analysis_df = analysis_df.reset_index(drop=True)

    # Load volumes
    mito_vol = get_instance_mask(
        input_folder_mito, zrange=z_range, tile_number=tile_id,
        tile_size=tile_size, grid_size=grid_cols,
    )
    er_vol = get_instance_mask(
        input_folder_er, zrange=z_range, tile_number=tile_id,
        tile_size=tile_size, grid_size=grid_cols,
    )
    props = regionprops(mito_vol)
    logger.info("Tile %d: %d mito regions, volume shape %s",
                tile_id, len(props), mito_vol.shape)

    matches, unmatched_p, unmatched_c = match_by_bbox_with_tile_offset(
        props, analysis_df, z_range, tile_id,
        coord_space=coord_space, tile_size=tile_size, grid_cols=grid_cols,
        bbox_tolerance=bbox_tolerance, downsample_tolerance=downsample_tolerance,
    )
    logger.info("Matched: %d | Unmatched volume: %d | Unmatched CSV: %d",
                len(matches), len(unmatched_p), len(unmatched_c))
    if not matches:
        return []

    # Prepare parallel args
    pad = 10
    args_list = []
    for prop, info, iou in matches:
        zmin, ymin, xmin, zmax, ymax, xmax = map(int, prop.bbox)
        sl = (
            slice(max(zmin - pad, 0), min(zmax + pad, mito_vol.shape[0])),
            slice(max(ymin - pad, 0), min(ymax + pad, mito_vol.shape[1])),
            slice(max(xmin - pad, 0), min(xmax + pad, mito_vol.shape[2])),
        )
        args_list.append((
            prop.label, (mito_vol[sl] == prop.label), er_vol[sl],
            sl[0].start + z_range[0], info,
            pc2_narrowing_low, pc2_narrowing_high,
            exclude_end_fraction, min_ratio_threshold, is_narrowing,
            n_slices, near_fraction,
        ))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = [r for r in pool.imap_unordered(_analyze_single_mito, args_list)
                   if r is not None]

    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        logger.info("Saved %d results → %s", len(results), output_csv)

    return results


# =============================================================================
# ER Enrichment Post-Processing
# =============================================================================

def load_all_tiles(
    result_dir: str,
    file_pattern: str = "er_mito_narrowing_pc2_*_tile*.csv",
) -> pd.DataFrame:
    """Load and concatenate per-tile CSV results."""
    files = sorted(glob.glob(os.path.join(result_dir, file_pattern)))
    if not files:
        raise ValueError(f"No files matching '{file_pattern}' in {result_dir}")

    dfs = []
    for f in files:
        tile_df = pd.read_csv(f)
        tile_df["tile_id"] = int(f.split("tile")[-1].replace(".csv", ""))
        dfs.append(tile_df)
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d mitochondria from %d tiles.", len(df), len(files))
    return df


def compute_enrichment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ER contact enrichment at narrowing sites.

    Adds ``enrichment_nearby_total``, ``enrichment_nearby_far``, and
    ``log2_enrichment_nearby_far`` columns.
    """
    df = df.copy()
    df["far_surface_area"] = df["surface_area"] - df["nearby_surface_area"]
    df["far_contact_ratio"] = df["far_contact_count"] / (df["far_surface_area"] + 1e-6)
    df["enrichment_nearby_total"] = df["nearby_contact_ratio"] / (df["total_contact_ratio"] + 1e-6)
    df["enrichment_nearby_far"] = df["nearby_contact_ratio"] / (df["far_contact_ratio"] + 1e-6)
    df["log2_enrichment_nearby_far"] = np.log2(df["enrichment_nearby_far"] + 1e-6)

    for col in ("enrichment_nearby_total", "enrichment_nearby_far"):
        df[col] = df[col].clip(0, 100)
    df["log2_enrichment_nearby_far"] = df["log2_enrichment_nearby_far"].clip(-10, 10)
    return df


def statistical_tests(
    df: pd.DataFrame,
    enrichment_col: str = "enrichment_nearby_far",
    n_perm: int = 10_000,
    random_state: Optional[int] = None,
) -> dict:
    """Statistical tests for median enrichment against H₀: median = 1.

    Parameters
    ----------
    random_state : int, optional
        Seed for the sign-flip permutation test's RNG. Leave unset for a
        non-reproducible run; set explicitly to reproduce a reported p-value.

    Returns a dictionary with permutation p-value, binomial test, effect size.
    """
    enrichment = df[enrichment_col].dropna()
    enrichment = enrichment[np.isfinite(enrichment)].values
    n = len(enrichment)

    obs_median = np.median(enrichment)
    obs_mad = float(median_abs_deviation(enrichment, scale="normal"))
    median_diff = obs_median - 1.0
    robust_d = median_diff / obs_mad if obs_mad != 0 else 0

    # One-sample sign-flip permutation test
    rng = np.random.default_rng(random_state)
    diffs = enrichment - 1.0
    perm_stats = np.zeros(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_stats[i] = np.median(diffs * signs)
    p_perm = float((np.abs(perm_stats) >= np.abs(median_diff)).mean())

    n_enriched = int((enrichment > 1).sum())
    binom_p = float(binomtest(n_enriched, n, p=0.5, alternative="greater").pvalue)

    results = {
        "n": n,
        "median": obs_median,
        "mad": obs_mad,
        "robust_d": robust_d,
        "perm_pvalue": p_perm,
        "binom_pvalue": binom_p,
        "prop_enriched": n_enriched / n,
    }
    logger.info(
        "Enrichment: median=%.3f, d=%.3f, perm p=%.2e, binom p=%.2e, prop=%.1f%%",
        obs_median, robust_d, p_perm, binom_p, 100 * n_enriched / n,
    )
    return results


def plot_enrichment_analysis(
    df: pd.DataFrame,
    output_dir: str,
    prefix: str = "er_mito_enrichment",
) -> None:
    """Generate multi-panel enrichment visualisation figures."""
    os.makedirs(output_dir, exist_ok=True)
    df_valid = df[df["has_valid_narrowing"] == True].copy()
    if len(df_valid) == 0:
        logger.warning("No valid narrowings to plot.")
        return

    plt.rcParams["font.size"] = 12
    from scipy import stats as sp_stats

    # --- Panel 1: 2×2 overview ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1a. Scatter: nearby vs far
    ax = axes[0, 0]
    ax.scatter(df_valid["far_contact_ratio"], df_valid["nearby_contact_ratio"],
               alpha=0.5, s=20, c=(242 / 255, 128 / 255, 12 / 255))
    ax.grid(False)
    max_val = max(df_valid["far_contact_ratio"].max(),
                  df_valid["nearby_contact_ratio"].max())
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.7, label="1:1 line")
    slope, intercept, r, p, _ = sp_stats.linregress(
        df_valid["far_contact_ratio"], df_valid["nearby_contact_ratio"])
    x_reg = np.linspace(0, max_val, 100)
    ax.plot(x_reg, slope * x_reg + intercept, "r-",
            label=f"slope={slope:.2f}, p={p:.2e}")
    ax.set_xlabel("Far contact ratio")
    ax.set_ylabel("Nearby contact ratio")
    ax.set_title("ER Contact: Nearby vs Far")

    # 1b. Enrichment histogram
    ax = axes[0, 1]
    enrich = df_valid["enrichment_nearby_far"].clip(0, 10)
    ax.hist(enrich, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(1, color="red", ls="--", lw=2, label="No enrichment")
    ax.axvline(enrich.median(), color="green", ls="-", lw=2,
               label=f"Median ({enrich.median():.2f})")
    ax.set_xlabel("Enrichment (nearby / far)")
    ax.set_ylabel("Count")
    ax.set_title("ER Contact Enrichment Distribution")

    # 1c. Log2 enrichment
    ax = axes[1, 0]
    log2 = df_valid["log2_enrichment_nearby_far"].clip(-5, 5)
    ax.hist(log2, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", ls="--", lw=2)
    ax.axvline(log2.median(), color="green", ls="-", lw=2,
               label=f"Median ({log2.median():.2f})")
    n_pos = int((log2 > 0).sum())
    n_neg = int((log2 < 0).sum())
    ax.text(0.95, 0.95,
            f"Enriched: {n_pos} ({n_pos / len(log2):.1%})\n"
            f"Depleted: {n_neg} ({n_neg / len(log2):.1%})",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Log₂ Enrichment")
    ax.set_ylabel("Count")
    ax.set_title("Log₂ Enrichment (>0 = enriched)")

    # 1d. Enrichment vs PC2
    ax = axes[1, 1]
    sc = ax.scatter(df_valid["PC2"],
                    df_valid["enrichment_nearby_far"].clip(0, 10),
                    c=df_valid["min_median_ratio"], cmap="viridis",
                    alpha=0.6, s=20)
    ax.axhline(1, color="red", ls="--", lw=2)
    ax.set_xlabel("PC2 (morphology)")
    ax.set_ylabel("Enrichment (nearby / far)")
    ax.set_title("Enrichment vs Morphology")
    plt.colorbar(sc, ax=ax, label="narrowing severity")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- Panel 2: Boxplot comparison ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = pd.DataFrame({
        "Contact Ratio": np.concatenate([
            df_valid["nearby_contact_ratio"].values,
            df_valid["far_contact_ratio"].values,
        ]),
        "Region": (["narrowing\n(nearby)"] * len(df_valid)
                   + ["Rest\n(far)"] * len(df_valid)),
    })
    sns.boxplot(data=plot_data, x="Region", y="Contact Ratio", ax=ax,
                palette=["steelblue", "coral"])
    stat_u, pval = mannwhitneyu(
        df_valid["nearby_contact_ratio"].values,
        df_valid["far_contact_ratio"].values,
        alternative="greater",
    )
    ax.set_ylabel("ER Contact Ratio")
    ax.set_title(f"ER Contact: narrowing vs Rest\n(Mann–Whitney U p={pval:.2e})")
    y_max = plot_data["Contact Ratio"].max()
    ax.plot([0, 0, 1, 1], [y_max * 1.05, y_max * 1.1, y_max * 1.1, y_max * 1.05],
            "k-", lw=1.5)
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
    ax.text(0.5, y_max * 1.12, sig, ha="center", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_boxplot.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Enrichment plots saved to %s", output_dir)


def analyze_er_mito_enrichment(
    result_dir: str,
    output_dir: Optional[str] = None,
    file_pattern: str = "er_mito_narrowing_pc2_*_tile*.csv",
    prefix: str = "er_mito_enrichment",
    min_median_ratio_threshold: float = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """End-to-end ER–mito enrichment analysis across all tiles.

    Loads per-tile CSVs, computes enrichment metrics, runs statistical
    tests, and generates publication-ready figures.
    """
    if output_dir is None:
        output_dir = result_dir

    df = load_all_tiles(result_dir, file_pattern)
    df_valid = df[df["has_valid_narrowing"] == True].copy()

    if "min_median_ratio" in df_valid.columns and min_median_ratio_threshold is not None:
        has_val = df_valid["min_median_ratio"].notna()
        meets = df_valid["min_median_ratio"] <= min_median_ratio_threshold
        df_valid = df_valid[meets | ~has_val].copy()
        logger.info("After severity filter: %d mitochondria.", len(df_valid))

    if len(df_valid) == 0:
        logger.error("No mitochondria with valid narrowings!")
        return df, {}

    df_valid = compute_enrichment_metrics(df_valid)
    stats_results = statistical_tests(
        df_valid, "enrichment_nearby_far", random_state=random_state
    )
    plot_enrichment_analysis(df_valid, output_dir, prefix)

    output_csv = os.path.join(output_dir, f"{prefix}_combined.csv")
    df_valid.to_csv(output_csv, index=False)
    logger.info("Combined results → %s", output_csv)

    return df_valid, stats_results


# =============================================================================
# CLI Entry Point
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mitochondrial morphology feature analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", help="Analysis stage to run.")

    # --- cluster ---
    cl = sub.add_parser("cluster", help="PCA + morphological indexing.")
    cl.add_argument("--feature_csv", required=True, help="Radiomic features CSV.")
    cl.add_argument("--output_root", required=True, help="Root output directory.")
    cl.add_argument("--high_conf", type=float, default=0.6)
    cl.add_argument("--low_conf", type=float, default=0.4)
    cl.add_argument("--outlier_threshold", type=float, default=4.0)
    cl.add_argument("--ks_threshold", type=float, required=True,
                     help="KS D-statistic above which the skew-normal fit "
                          "is rejected for an empirical-percentile "
                          "fallback. Dataset-specific; inspect the logged "
                          "KS D values for your PC1/PC2 fits to choose it.")
    cl.add_argument("--random_state", type=int, default=42)

    # --- hepatocyte ---
    hp = sub.add_parser("hepatocyte", help="Per-hepatocyte distribution analysis.")
    hp.add_argument("--pca_csv", required=True)
    hp.add_argument("--hepatocyte_mask_folder", required=True)
    hp.add_argument("--mito_instance_folder", required=True)
    hp.add_argument("--outlier_mask", required=True)
    hp.add_argument("--output_dir", required=True)
    hp.add_argument("--high_conf", type=float, default=0.8)
    hp.add_argument("--low_conf", type=float, default=0.2)
    hp.add_argument("--z_threshold", type=int, required=True,
                     help="Minimum z-span (slices) for a mitochondrion "
                          "instance to be kept; use the same value as "
                          "morphology_features.py --z-threshold.")

    # --- er-mito ---
    em = sub.add_parser("er-mito", help="ER–mitochondria narrowing analysis.")
    em.add_argument("--mito_folder", required=True)
    em.add_argument("--er_folder", required=True)
    em.add_argument("--cluster_csv", required=True)
    em.add_argument("--output_dir", required=True)
    em.add_argument("--pc2_low", type=float, default=0.6)
    em.add_argument("--pc2_high", type=float, default=0.8)
    em.add_argument("--pc1_low", type=float, default=0)
    em.add_argument("--pc1_high", type=float, default=10)
    em.add_argument("--tile_id", type=int, default=None,
                     help="Tile index. Defaults to $SLURM_ARRAY_TASK_ID or 0.")
    em.add_argument("--z_start", type=int, default=0,
                     help="First z-slice (inclusive) of the volume to load.")
    em.add_argument("--z_end", type=int, required=True,
                     help="Last z-slice (exclusive) of the volume to load. "
                          "Dataset-specific — set to your volume's z-extent.")
    em.add_argument("--coord_space", choices=["2k", "10k"], default=None,
                     help="Which bbox coordinate columns to match against "
                          "in the cluster CSV (bbox_*_2k vs bbox_*_10k).")
    em.add_argument("--tile_size", type=int, required=True,
                     help="Tile edge length in pixels (dataset-specific).")
    em.add_argument("--grid_cols", type=int, required=True,
                     help="Number of tile columns in the acquisition grid "
                          "(dataset-specific).")
    em.add_argument("--bbox_tolerance", type=int, default=0,
                     help="Extra pixel tolerance when matching volume "
                          "regions to CSV bounding boxes.")
    em.add_argument("--downsample_tolerance", type=int, default=0,
                     help="Additional pixel tolerance to account for "
                          "coordinate downsampling error.")
    em.add_argument("--exclude_end_fraction", type=float, required=True,
                     help="Fraction of the major axis excluded at each end "
                          "when searching for a narrowing site (0-0.5).")
    em.add_argument("--min_ratio_threshold", type=float, required=True,
                     help="Max cross-section-area ratio (narrowest/median) "
                          "to qualify as a narrowing site.")
    em.add_argument("--n_slices", type=int, required=True,
                     help="Number of cross-sections sampled along the major "
                          "axis when searching for a narrowing site.")
    em.add_argument("--near_fraction", type=float, required=True,
                     help="Fraction (0-1) of the major-axis length defined "
                          "as 'near' the narrowing site / middle region.")
    em.add_argument("--is_narrowing", action="store_true", default=True)
    em.add_argument("--no_narrowing", dest="is_narrowing", action="store_false")

    # --- enrichment ---
    en = sub.add_parser("enrichment", help="Post-hoc enrichment analysis across tiles.")
    en.add_argument("--result_dir", required=True)
    en.add_argument("--output_dir", default=None)
    en.add_argument("--file_pattern", default="er_mito_narrowing_pc2_*_tile*.csv")
    en.add_argument("--min_median_ratio", type=float, default=0.8)
    en.add_argument("--random_state", type=int, default=None,
                     help="Seed for the sign-flip permutation test RNG. "
                          "Set explicitly for a reproducible p-value.")

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "cluster":
        cluster_feature(
            feature_file=args.feature_csv,
            output_root=args.output_root,
            high_conf=args.high_conf,
            low_conf=args.low_conf,
            outlier_threshold=args.outlier_threshold,
            ks_threshold=args.ks_threshold,
            random_state=args.random_state,
        )

    elif args.command == "hepatocyte":
        per_hepatocyte_mito_distribution(
            pca_csv=args.pca_csv,
            hepatocyte_mask_folder=args.hepatocyte_mask_folder,
            mito_instance_folder=args.mito_instance_folder,
            outlier_mask_file=args.outlier_mask,
            output_dir=args.output_dir,
            high_conf=args.high_conf,
            low_conf=args.low_conf,
            z_threshold=args.z_threshold,
        )

    elif args.command == "er-mito":
        analyze_narrowing_mito_er_interaction(
            input_folder_mito=args.mito_folder,
            input_folder_er=args.er_folder,
            cluster_csv=args.cluster_csv,
            output_dir_base=args.output_dir,
            pc2_narrowing_low=args.pc2_low,
            pc2_narrowing_high=args.pc2_high,
            pc1_low=args.pc1_low,
            pc1_high=args.pc1_high,
            exclude_end_fraction=args.exclude_end_fraction,
            min_ratio_threshold=args.min_ratio_threshold,
            n_slices=args.n_slices,
            near_fraction=args.near_fraction,
            coord_space=args.coord_space,
            tile_size=args.tile_size,
            grid_cols=args.grid_cols,
            bbox_tolerance=args.bbox_tolerance,
            downsample_tolerance=args.downsample_tolerance,
            z_range=(args.z_start, args.z_end),
            tile_id=args.tile_id,
            is_narrowing=args.is_narrowing,
        )

    elif args.command == "enrichment":
        analyze_er_mito_enrichment(
            result_dir=args.result_dir,
            output_dir=args.output_dir,
            file_pattern=args.file_pattern,
            min_median_ratio_threshold=args.min_median_ratio,
            random_state=args.random_state,
        )

    else:
        parser.print_help()

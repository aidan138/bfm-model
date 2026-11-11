"""
Copyright 2025 (C) TNO. Licensed under the MIT license.

Importable helpers to compare model predictions against ground truth for a single batch.

Extracted from the Streamlit viewers (``prediction_viewer.py`` and
``documentation/rollout_visualisation.py``) so they can be used directly in notebooks/scripts.

Expected layout (as produced by ``test_lighting.py`` / ``BFM.predict_step``):
    pred[group][var] -> [B, H, W]       or [B, C, H, W]       (C = pressure levels)
    gt[group][var]   -> [B, T, H, W]    or [B, T, C, H, W]    (T index 0 is the target)
``pred`` / ``gt`` may be dicts or ``Batch`` namedtuples.

Example:
    from bfm_model.bfm.eval_plots import plot_maps, batch_metrics, get_lat_lon, select_field

    w = torch.load("pre-train_test_exports/window_00000.pt", weights_only=False)
    pred, gt = w["pred"], w["gt"]
    lats, lons = get_lat_lon(gt)
    p, g = select_field(pred, gt, "surface_variables", "t2m")
    plot_maps(p, g, lats, lons, "t2m", show_diff=True)
    batch_metrics(pred, gt)  # DataFrame of RMSE/MAE/Bias/Corr/Skill per variable
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ArrayLike = Any  # torch.Tensor | np.ndarray


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a tensor/array to a float numpy array (handles bf16 and GPU tensors)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=float)


def as_dict(batch: Any) -> Dict[str, Any]:
    """Return a ``Batch`` namedtuple or dict as a plain dict of variable groups."""
    if isinstance(batch, dict):
        return batch
    if hasattr(batch, "_asdict"):
        return batch._asdict()
    raise TypeError(f"Cannot interpret batch of type {type(batch)}")


def _get_metadata(gt: Any) -> Any:
    meta = as_dict(gt)["batch_metadata"]
    return meta._asdict() if hasattr(meta, "_asdict") else meta


def get_lat_lon(gt: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 1D latitude/longitude arrays from a ground-truth batch's metadata."""
    meta = _get_metadata(gt)
    lats, lons = to_numpy(meta["latitudes"]), to_numpy(meta["longitudes"])
    # Metadata is collated with a leading batch dim.
    return (lats[0] if lats.ndim > 1 else lats), (lons[0] if lons.ndim > 1 else lons)


def select_field(
    pred: Any,
    gt: Any,
    group: str,
    var: str,
    sample: int = 0,
    gt_time: int = 0,
    level: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return matching 2D (H, W) prediction and ground-truth fields for one variable.

    Args:
        group: variable group, e.g. ``"surface_variables"``.
        var: variable name within the group.
        sample: index in the batch dimension.
        gt_time: time index in the ground truth (0 = the predicted target).
        level: channel index for atmospheric variables (pressure level). Defaults to 0.
    """
    p = to_numpy(as_dict(pred)[group][var])[sample]
    g = to_numpy(as_dict(gt)[group][var])[sample][gt_time]
    if p.ndim == 3:
        ci = 0 if level is None else level
        p, g = p[ci], g[ci]
    return p, g


def _flatten(pred: ArrayLike, gt: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    pr = to_numpy(pred).ravel()
    ob = to_numpy(gt).ravel()
    mask = np.isfinite(pr) & np.isfinite(ob)
    return pr[mask], ob[mask]


def compute_metrics(pred: ArrayLike, gt: ArrayLike) -> Dict[str, float]:
    """RMSE, MAE, bias, Pearson correlation and skill (vs. spatial-mean climatology), ignoring NaNs."""
    p, o = _flatten(pred, gt)
    if p.size == 0:
        return {k: float("nan") for k in ("RMSE", "MAE", "Bias", "Corr", "Skill")}
    err = p - o
    denom = np.sum((o - o.mean()) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = float(np.corrcoef(o, p)[0, 1]) if p.size > 1 else float("nan")
        skill = float(1.0 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "Bias": float(np.mean(err)),
        "Corr": corr,
        "Skill": skill,
    }


def batch_metrics(
    pred: Any,
    gt: Any,
    groups: Optional[Sequence[str]] = None,
    sample: int = 0,
    gt_time: int = 0,
    per_level: bool = False,
) -> pd.DataFrame:
    """
    Compute :func:`compute_metrics` for every variable in a batch.

    Args:
        groups: variable groups to include (default: all ``*_variables`` groups in ``pred``).
        per_level: for atmospheric variables, one row per pressure level instead of pooling all levels.

    Returns:
        DataFrame indexed by (group, variable, level) with metric columns.
    """
    pred_d, gt_d = as_dict(pred), as_dict(gt)
    groups = groups or [k for k in pred_d if k.endswith("_variables")]
    rows = []
    for group in groups:
        if pred_d.get(group) is None or gt_d.get(group) is None:
            continue
        for var, p_t in pred_d[group].items():
            if var not in gt_d[group]:
                continue
            p = to_numpy(p_t)[sample]
            g = to_numpy(gt_d[group][var])[sample][gt_time]
            if p.ndim == 3 and per_level:
                for ci in range(p.shape[0]):
                    rows.append({"group": group, "variable": var, "level": ci, **compute_metrics(p[ci], g[ci])})
            else:
                rows.append({"group": group, "variable": var, "level": None, **compute_metrics(p, g)})
    return pd.DataFrame(rows).set_index(["group", "variable", "level"])


def plot_maps(
    pred: ArrayLike,
    gt: ArrayLike,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str = "",
    show_diff: bool = False,
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Quick-look map panels: Prediction | Ground Truth (| Prediction - Ground Truth).

    Prediction and ground truth share one colour scale; the difference uses a symmetric diverging scale.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    p, g = to_numpy(pred).squeeze(), to_numpy(gt).squeeze()
    proj = ccrs.PlateCarree()
    ncols = 3 if show_diff else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5 * ncols, 4), subplot_kw=dict(projection=proj))
    vmin, vmax = np.nanmin([np.nanmin(p), np.nanmin(g)]), np.nanmax([np.nanmax(p), np.nanmax(g)])
    for a, dat, lbl in zip(ax[:2], (p, g), ("Prediction", "Ground Truth")):
        mesh = a.pcolormesh(lons, lats, dat, cmap=cmap, vmin=vmin, vmax=vmax, transform=proj, shading="auto")
        a.add_feature(cfeature.COASTLINE, lw=0.4)
        a.set_title(f"{title}\n{lbl}" if title else lbl)
    fig.colorbar(mesh, ax=ax[:2].tolist(), shrink=0.75)
    if show_diff:
        d = p - g
        lim = np.nanmax(np.abs(d)) or 1.0
        dm = ax[2].pcolormesh(lons, lats, d, cmap="RdBu_r", vmin=-lim, vmax=lim, transform=proj, shading="auto")
        ax[2].add_feature(cfeature.COASTLINE, lw=0.4)
        ax[2].set_title(f"{title}\nPrediction - Ground Truth" if title else "Prediction - Ground Truth")
        fig.colorbar(dm, ax=ax[2], shrink=0.75)
    return fig


def plot_maps_informative(
    pred: ArrayLike,
    gt: ArrayLike,
    lats: np.ndarray,
    lons: np.ndarray,
    var_group: str,
    var_name: str,
    timestamp: str,
    cbar_label: str = "Distribution density",
) -> plt.Figure:
    """Publication-style two-panel map with gridlines, labels and a shared right-hand colourbar."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    p, g = to_numpy(pred).squeeze(), to_numpy(gt).squeeze()
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), subplot_kw=dict(projection=proj))
    vmin, vmax = np.nanmin([np.nanmin(p), np.nanmin(g)]), np.nanmax([np.nanmax(p), np.nanmax(g)])
    mesh = None
    for i, (a, dat, lbl) in enumerate(zip(ax, (p, g), ("Prediction", "Ground Truth"))):
        mesh = a.pcolormesh(lons, lats, dat, cmap="viridis", vmin=vmin, vmax=vmax, transform=proj, shading="auto")
        a.add_feature(cfeature.COASTLINE, lw=0.4)
        gl = a.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--")
        gl.top_labels = gl.right_labels = False
        if i > 0:
            gl.left_labels = False
        a.set_title(lbl)

    cax = inset_axes(
        ax[-1],
        width="2.5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1, 1),
        bbox_transform=ax[-1].transAxes,
        borderpad=0,
    )
    fig.colorbar(mesh, cax=cax, label=cbar_label)
    fig.suptitle(f"Variable Group: {var_group} | Variable: {var_name} — Timestamp: {timestamp}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_raw_triple(var_name: str, pred_tensor: ArrayLike, gt_tensor: ArrayLike, timestamps: Any = None):
    """
    Interactive Plotly figure: Ground Truth | Prediction | Difference (GT - Pred), no map projection.

    Accepts full tensors (first sample / first timestep / first channel are taken) or 2D fields.
    """
    import plotly.express as px
    from plotly.subplots import make_subplots

    def _to_2d(x: ArrayLike) -> np.ndarray:
        a = to_numpy(x)
        while a.ndim > 2:
            a = a[0]
        return a

    p, g = _to_2d(pred_tensor), _to_2d(gt_tensor)
    diff = g - p
    p, g, diff = np.flipud(p), np.flipud(g), np.flipud(diff)
    t_label = timestamps[-1] if isinstance(timestamps, (list, tuple)) and timestamps else (timestamps or "")
    at = f" @ {t_label}" if t_label else ""
    fig = make_subplots(rows=1, cols=3, subplot_titles=(f"Ground Truth{at}", f"Prediction{at}", "Difference"))
    for data, colorscale, col in [(g, "Viridis", 1), (p, "Viridis", 2), (diff, "thermal", 3)]:
        for trace in px.imshow(data, color_continuous_scale=colorscale).data:
            fig.add_trace(trace, row=1, col=col)
    fig.update_layout(title_text=f"{var_name} Comparison", showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=400)
    for c in (1, 2, 3):
        fig.update_xaxes(constrain="domain", row=1, col=c)
        fig.update_yaxes(scaleanchor="x", row=1, col=c)
    return fig

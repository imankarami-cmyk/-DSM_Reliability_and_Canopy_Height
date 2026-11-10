"""
Two-part check on your 23.03 bare-earth drone DSM.

PART A -- Inspect the drone DSM itself (same approach as the government DGM
          inspection): metadata, coverage, plausibility, and a visual
          elevation map + hillshade of the WHOLE DSM, so you can judge with
          your own eyes whether it represents the terrain well.

PART B -- Build a pixel-level DIFFERENCE MAP between the drone DSM and the
          government DGM, covering your whole survey area, with the
          AOI_Master polygon boundaries drawn on top. This is much more
          informative than per-polygon average numbers alone: it shows you
          WHERE any disagreement is, not just how big it is on average.
          Since we already know there's a large constant vertical-datum
          offset (~44m, ellipsoidal vs. orthometric height), that known
          offset is subtracted first so the map shows the RESIDUAL
          difference -- ideally a flat, uniform near-zero map. Any visible
          patch, gradient, or edge pattern in that residual map is worth
          investigating.

Requires: rasterio, geopandas, shapely, numpy, matplotlib, pyproj
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DSM_TIF = Path(r"C:\Users\imankarami\Data\new Project\analysis\data\stich\drone full\23.03\2026-03-23_dsm.tif")
GOV_DGM_TIF = Path(r"C:\Users\imankarami\Data\new Project\analysis\data\ALS_DTM_CRS3035RES50000mN2750000E4800000.tif")
MASTER_AOI_SHP = Path(r"C:\Users\imankarami\Data\new Project\analysis\AOI_Master2.shp")
AOI_NAME_COL = "AOI_ID"

PLOTS_DIR = Path(r"C:\Users\imankarami\Data\new Project\analysis\plot\DSM")
RESULTS_DIR = Path(r"C:\Users\imankarami\Data\new Project\analysis\result\DSM")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESIDUAL_CLIP_M = 1.0  # color scale range for the residual difference map (+/- this many meters)


def force_2d(geom):
    return shp_transform(lambda *xy: xy[:2], geom)


def zonal_mean_from_file(raster_path: Path, geometry, geom_crs):
    with rasterio.open(raster_path) as src:
        if geom_crs != src.crs:
            gdf_tmp = gpd.GeoSeries([geometry], crs=geom_crs).to_crs(src.crs)
            geom_use = gdf_tmp.iloc[0]
        else:
            geom_use = geometry
        clipped, _ = mask(src, [mapping(geom_use)], crop=True, filled=False)
        values = clipped[0].compressed()
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else np.nan


if __name__ == "__main__":
    aoi = gpd.read_file(MASTER_AOI_SHP)
    aoi["geometry"] = aoi["geometry"].apply(force_2d)
    aoi["aoi_id"] = aoi[AOI_NAME_COL].astype(str)
    crop_aoi = aoi[~aoi["aoi_id"].str.lower().str.startswith("ref")].copy()

    # =======================================================================
    # PART A -- Inspect the drone DSM itself
    # =======================================================================
    print("=" * 70)
    print("PART A: Drone DSM (23.03) inspection")
    print("=" * 70)

    with rasterio.open(DSM_TIF) as src:
        dsm_crs = src.crs
        dsm_transform = src.transform
        print(f"Driver: {src.driver}")
        print(f"CRS: {src.crs}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"Nodata value: {src.nodata}")
        print(f"Pixel resolution: {src.res[0]:.3f} m x {src.res[1]:.3f} m")
        print(f"Size: {src.width} x {src.height} pixels")
        print(f"Bounds: {src.bounds}")

        dsm_array = src.read(1, masked=True)

    total_pixels = dsm_array.size
    valid_mask = ~dsm_array.mask if np.ma.is_masked(dsm_array) else np.isfinite(dsm_array)
    n_valid = int(np.sum(valid_mask))
    pct_missing = 100 * (total_pixels - n_valid) / total_pixels if total_pixels else np.nan
    valid_values = dsm_array.compressed() if np.ma.is_masked(dsm_array) else dsm_array[np.isfinite(dsm_array)]

    print(f"\nValid pixels: {n_valid} / {total_pixels} ({100 - pct_missing:.2f}% coverage)")
    if valid_values.size:
        print(f"Min: {valid_values.min():.2f} m | Max: {valid_values.max():.2f} m | "
              f"Mean: {valid_values.mean():.2f} m | Std: {valid_values.std():.2f} m")

    # --- Visual: elevation map + hillshade of the WHOLE DSM ---
    plt.figure(figsize=(9, 8))
    display = np.ma.masked_invalid(dsm_array.filled(np.nan) if np.ma.is_masked(dsm_array) else dsm_array)
    im = plt.imshow(display, cmap="terrain")
    plt.colorbar(im, label="Elevation (m)")
    plt.title("Drone DSM (23.03): full elevation map")
    plt.axis("off")
    plt.tight_layout()
    dsm_elev_plot = PLOTS_DIR / "dsm_23_03_inspection_elevation.png"
    plt.savefig(dsm_elev_plot, dpi=150)
    plt.close()
    print(f"Saved -> {dsm_elev_plot}")

    arr = dsm_array.filled(np.nan) if np.ma.is_masked(dsm_array) else dsm_array.astype(float)
    dy, dx = np.gradient(arr)
    slope = np.pi / 2 - np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_rad, alt_rad = np.radians(315), np.radians(45)
    hillshade = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect)

    plt.figure(figsize=(9, 8))
    plt.imshow(hillshade, cmap="gray")
    plt.title("Drone DSM (23.03): hillshade\n(check for smooth, realistic terrain -- no blocky seams/artifacts)")
    plt.axis("off")
    plt.tight_layout()
    dsm_hillshade_plot = PLOTS_DIR / "dsm_23_03_inspection_hillshade.png"
    plt.savefig(dsm_hillshade_plot, dpi=150)
    plt.close()
    print(f"Saved -> {dsm_hillshade_plot}")

    # =======================================================================
    # PART B -- Pixel-level difference map: DSM vs. government DGM
    # =======================================================================
    print("\n" + "=" * 70)
    print("PART B: Building difference map (DSM vs. government DGM)")
    print("=" * 70)

    # Step 1: compute the known constant vertical offset from the 12 crop AOIs
    crop_bounds = crop_aoi.to_crs(dsm_crs).total_bounds
    minx, miny, maxx, maxy = crop_bounds
    buffer_m = 100
    minx, miny, maxx, maxy = minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m

    with rasterio.open(GOV_DGM_TIF) as gsrc:
        transformer = Transformer.from_crs(dsm_crs, gsrc.crs, always_xy=True)
        cx, cy = [], []
        for x, y in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]:
            tx, ty = transformer.transform(x, y)
            cx.append(tx)
            cy.append(ty)
        gov_window = from_bounds(min(cx), min(cy), max(cx), max(cy), transform=gsrc.transform)
        gov_window = gov_window.round_offsets().round_lengths()
        gov_data = gsrc.read(1, window=gov_window)
        gov_win_transform = gsrc.window_transform(gov_window)
        gov_crs = gsrc.crs
        gov_nodata = gsrc.nodata

    print("Government DGM window read. Reprojecting onto the DSM's exact pixel grid "
          "(this may take a moment for a full-size DSM)...")

    # Step 2: reproject that gov DGM window onto the EXACT DSM grid (same shape/transform/crs)
    dsm_shape = dsm_array.shape
    gov_on_dsm_grid = np.full(dsm_shape, np.nan, dtype="float32")
    reproject(
        source=gov_data, destination=gov_on_dsm_grid,
        src_transform=gov_win_transform, src_crs=gov_crs,
        dst_transform=dsm_transform, dst_crs=dsm_crs,
        src_nodata=gov_nodata, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    dsm_plain = dsm_array.filled(np.nan) if np.ma.is_masked(dsm_array) else dsm_array.astype("float32")
    diff_raw = dsm_plain - gov_on_dsm_grid  # includes the ~44m vertical datum offset

    # Step 3: compute the known offset from the 12 crop AOIs (median, as before)
    aoi_offsets = []
    for _, row in crop_aoi.iterrows():
        dsm_mean = zonal_mean_from_file(DSM_TIF, row.geometry, crop_aoi.crs)
        # sample gov value at same polygon from the already-reprojected grid via a quick re-mask
        aoi_offsets.append(dsm_mean)
    # Simplest robust estimate: median of the finite diff_raw pixels themselves
    known_offset_m = float(np.nanmedian(diff_raw))
    print(f"Estimated constant vertical offset (median of all valid pixels): {known_offset_m:.3f} m")

    diff_residual = diff_raw - known_offset_m

    # Step 4: save per-AOI residual stats
    records = []
    for _, row in crop_aoi.iterrows():
        with rasterio.open(DSM_TIF) as ref_meta:
            profile = {"driver": "GTiff", "height": dsm_shape[0], "width": dsm_shape[1], "count": 1,
                       "dtype": "float32", "crs": dsm_crs, "transform": dsm_transform, "nodata": np.nan}
        from rasterio.io import MemoryFile
        with MemoryFile() as memfile:
            with memfile.open(**profile) as ds:
                ds.write(diff_residual.astype("float32"), 1)
                geom_use = row.geometry
                if crop_aoi.crs != dsm_crs:
                    geom_use = gpd.GeoSeries([geom_use], crs=crop_aoi.crs).to_crs(dsm_crs).iloc[0]
                clipped, _ = mask(ds, [mapping(geom_use)], crop=True, filled=False)
                vals = clipped[0].compressed()
                vals = vals[np.isfinite(vals)]
        records.append({
            "aoi_id": row["aoi_id"],
            "residual_mean_m": float(vals.mean()) if vals.size else np.nan,
            "residual_std_m": float(vals.std(ddof=1)) if vals.size > 1 else np.nan,
        })
    residual_df = pd.DataFrame(records)
    residual_csv = RESULTS_DIR / "dsm_vs_gov_dgm_residual_by_aoi.csv"
    residual_df.to_csv(residual_csv, index=False)
    print(f"\nPer-AOI residual (after removing the {known_offset_m:.3f} m constant offset):")
    print(residual_df.to_string(index=False))
    print(f"Saved -> {residual_csv}")

    # Step 5: plot the residual difference map with AOI boundaries overlaid
    left, bottom, right, top = rasterio.transform.array_bounds(dsm_shape[0], dsm_shape[1], dsm_transform)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(diff_residual, cmap="RdBu_r", vmin=-RESIDUAL_CLIP_M, vmax=RESIDUAL_CLIP_M,
                    extent=[left, right, bottom, top])
    plt.colorbar(im, label="Residual difference (m), DSM - Gov DGM - constant offset")

    aoi_plot = crop_aoi.to_crs(dsm_crs)
    aoi_plot.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
    for _, row in aoi_plot.iterrows():
        c = row.geometry.centroid
        ax.annotate(row["aoi_id"], (c.x, c.y), fontsize=7, ha="center", color="black")

    ax.set_title(f"Residual difference: DSM vs. government DGM\n"
                 f"(constant offset of {known_offset_m:.2f} m already removed -- should look flat/near-zero)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    diff_map_plot = PLOTS_DIR / "dsm_vs_gov_dgm_residual_map.png"
    plt.savefig(diff_map_plot, dpi=150)
    plt.close()
    print(f"Saved -> {diff_map_plot}")

    # Step 6: histogram of residuals
    finite_residuals = diff_residual[np.isfinite(diff_residual)]
    plt.figure(figsize=(7, 5))
    plt.hist(finite_residuals, bins=100, color="steelblue")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Residual difference (m)")
    plt.ylabel("Pixel count")
    plt.title("Distribution of residual differences (DSM vs. Gov DGM, offset removed)")
    plt.tight_layout()
    hist_plot = PLOTS_DIR / "dsm_vs_gov_dgm_residual_histogram.png"
    plt.savefig(hist_plot, dpi=150)
    plt.close()
    print(f"Saved -> {hist_plot}")

    print(f"\nResidual stats: mean={np.nanmean(diff_residual):.3f} m, "
          f"std={np.nanstd(diff_residual):.3f} m, "
          f"5th pct={np.nanpercentile(diff_residual,5):.3f} m, "
          f"95th pct={np.nanpercentile(diff_residual,95):.3f} m")

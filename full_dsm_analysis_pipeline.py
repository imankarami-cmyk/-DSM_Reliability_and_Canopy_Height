"""
Complete DSM validation and canopy height analysis pipeline.

STEP 1 -- Validate the 23.03 bare-earth DSM against the government DGM
          (per crop AOI: 6m/9m/12m/control), to catch any gross
          georeferencing error before trusting it as ground truth.

STEP 2 -- Check elevation drift across ALL flight dates using stable
          reference surfaces (Ref_1-Ref_6, on roads/paths that don't
          change height over the season). This tells you whether your
          flights are well co-registered with each other.

STEP 3 -- Canopy Height Model: for every flight date, compute
          CHM = DSM_date - DSM_23.03, per crop AOI, giving you actual
          plant height over the season.

All CSVs   -> C:\\Users\\imankarami\\Data\\new Project\\analysis\\result\\DSM
All plots  -> C:\\Users\\imankarami\\Data\\new Project\\analysis\\plot\\DSM

Requires: rasterio, geopandas, shapely, numpy, pandas, matplotlib, seaborn, pyproj
"""

from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

# ===========================================================================
# CONFIG -- edit these paths for your setup
# ===========================================================================

MASTER_AOI_SHP = Path(r"C:\Users\imankarami\Data\new Project\analysis\AOI_Master2_ref.shp")
AOI_NAME_COL = "AOI_ID"

GOV_DGM_TIF = Path(r"C:\Users\imankarami\Data\new Project\analysis\data\ALS_DTM_CRS3035RES50000mN2750000E4800000.tif")
STICH_ROOT = Path(r"C:\Users\imankarami\Data\new Project\analysis\data\stich\drone full")
FLIGHT_YEAR = 2026
GROUND_REFERENCE_FOLDER_NAME = "23.03"  # bare-earth flight, used as ground for CHM

VARIANT_ORDER = ["6m", "9m", "12m", "control"]

PLOTS_DIR = Path(r"C:\Users\imankarami\Data\new Project\analysis\plot\DSM")
RESULTS_DIR = Path(r"C:\Users\imankarami\Data\new Project\analysis\result\DSM")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OFFSET_FLAG_THRESHOLD_M = 1.0     # step 1: flag AOI whose gov-DGM offset deviates from median by more than this
DRIFT_FLAG_THRESHOLD_M = 0.03     # step 2: flag a date whose reference-surface drift exceeds this
PLAUSIBLE_MIN_M = -0.10           # step 3: plausibility bounds for mean canopy height
PLAUSIBLE_MAX_M = 1.30


# ===========================================================================
# SHARED HELPERS
# ===========================================================================

def force_2d(geom):
    return shp_transform(lambda *xy: xy[:2], geom)


def parse_aoi_name(name: str):
    m = re.match(r"^(.*)_(\d+)$", str(name).strip())
    if not m:
        return str(name).lower(), "1"
    return m.group(1).lower(), m.group(2)


def discover_date_folders(root: Path, year: int):
    entries = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = re.match(r"^(\d{2})\.(\d{2})$", p.name.strip())
        if not m:
            continue
        day, month = m.group(1), m.group(2)
        date_label = f"{year}-{month}-{day}"
        entries.append((date_label, p, p.name))
    entries.sort(key=lambda x: x[0])
    return entries


def find_dsm_file(folder: Path, date_label: str) -> Path | None:
    exact = folder / f"{date_label}_dsm.tif"
    if exact.exists():
        return exact
    matches = [f for f in folder.glob("*.tif") if "dsm" in f.stem.lower()]
    return matches[0] if matches else None


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


def zonal_mean_from_array(data_array, transform, crs, geometry, geom_crs):
    if geom_crs != crs:
        gdf_tmp = gpd.GeoSeries([geometry], crs=geom_crs).to_crs(crs)
        geometry = gdf_tmp.iloc[0]
    height, width = data_array.shape
    profile = {"driver": "GTiff", "height": height, "width": width, "count": 1,
               "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan}
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data_array, 1)
            clipped, _ = mask(dataset, [mapping(geometry)], crop=True, filled=False)
            values = clipped[0].compressed()
            values = values[np.isfinite(values)]
            return float(values.mean()) if values.size else np.nan


def zonal_stats_from_array(data_array, transform, crs, geometry, geom_crs, prefix: str) -> dict:
    if geom_crs != crs:
        gdf_tmp = gpd.GeoSeries([geometry], crs=geom_crs).to_crs(crs)
        geometry = gdf_tmp.iloc[0]
    height, width = data_array.shape
    profile = {"driver": "GTiff", "height": height, "width": width, "count": 1,
               "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan}
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data_array, 1)
            clipped, _ = mask(dataset, [mapping(geometry)], crop=True, filled=False)
            values = clipped[0].compressed()
            values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_n_pixels": 0, f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan,
                f"{prefix}_std": np.nan, f"{prefix}_min": np.nan, f"{prefix}_max": np.nan}
    return {
        f"{prefix}_n_pixels": int(values.size), f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
        f"{prefix}_min": float(values.min()), f"{prefix}_max": float(values.max()),
    }


def get_reprojected_dgm_window(gov_dgm_path: Path, target_crs, bounds_in_target_crs, buffer_m=100):
    minx, miny, maxx, maxy = bounds_in_target_crs
    minx, miny, maxx, maxy = minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m

    with rasterio.open(gov_dgm_path) as src:
        transformer = Transformer.from_crs(target_crs, src.crs, always_xy=True)
        corners_x, corners_y = [], []
        for x, y in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]:
            tx, ty = transformer.transform(x, y)
            corners_x.append(tx)
            corners_y.append(ty)

        window = from_bounds(min(corners_x), min(corners_y), max(corners_x), max(corners_y), transform=src.transform)
        window = window.round_offsets().round_lengths()

        data = src.read(1, window=window)
        win_transform = src.window_transform(window)
        src_crs = src.crs
        nodata = src.nodata

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, target_crs, data.shape[1], data.shape[0],
        *rasterio.transform.array_bounds(data.shape[0], data.shape[1], win_transform)
    )
    dst_array = np.full((dst_height, dst_width), np.nan, dtype="float32")
    reproject(
        source=data, destination=dst_array, src_transform=win_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=target_crs, src_nodata=nodata, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst_array, dst_transform, target_crs


def reproject_to_reference(src_path: Path, ref_transform, ref_crs, ref_shape):
    with rasterio.open(src_path) as src:
        dst_array = np.full(ref_shape, np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1), destination=dst_array,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_transform, dst_crs=ref_crs,
            src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear,
        )
    return dst_array


# ===========================================================================
# LOAD AOI (crop zones + reference polygons, from the one master shapefile)
# ===========================================================================

print("=" * 70)
print("Loading master AOI shapefile...")
print("=" * 70)

aoi_all = gpd.read_file(MASTER_AOI_SHP)
aoi_all["geometry"] = aoi_all["geometry"].apply(force_2d)
aoi_all["name_str"] = aoi_all[AOI_NAME_COL].astype(str)
parsed = aoi_all["name_str"].apply(parse_aoi_name)
aoi_all["variant"] = [p[0] for p in parsed]
aoi_all["repetition"] = [p[1] for p in parsed]
aoi_all["aoi_id"] = aoi_all["name_str"]

crop_aoi = aoi_all[~aoi_all["variant"].str.startswith("ref")].copy()
ref_aoi = aoi_all[aoi_all["variant"].str.startswith("ref")].copy()

print(f"Crop AOIs ({len(crop_aoi)}): {sorted(crop_aoi['aoi_id'].tolist())}")
print(f"Reference AOIs ({len(ref_aoi)}): {sorted(ref_aoi['aoi_id'].tolist())}")

date_folders = discover_date_folders(STICH_ROOT, FLIGHT_YEAR)
print(f"\nFound {len(date_folders)} flight date folders: {[d for d, _, _ in date_folders]}")

ground_entry = next((e for e in date_folders if e[2] == GROUND_REFERENCE_FOLDER_NAME), None)
if ground_entry is None:
    raise FileNotFoundError(f"Ground reference folder '{GROUND_REFERENCE_FOLDER_NAME}' not found.")
ground_date_label, ground_folder, _ = ground_entry
ground_dsm_path = find_dsm_file(ground_folder, ground_date_label)
if ground_dsm_path is None:
    raise FileNotFoundError(f"No DSM file found in {ground_folder}")


# ===========================================================================
# STEP 1 -- Validate 23.03 DSM against the government DGM (crop AOIs)
# ===========================================================================

print("\n" + "=" * 70)
print("STEP 1: Validating 23.03 bare-earth DSM against government DGM")
print("=" * 70)

with rasterio.open(ground_dsm_path) as dsm_src:
    dsm_crs = dsm_src.crs

crop_bounds = crop_aoi.to_crs(dsm_crs).total_bounds
dgm_array, dgm_transform, dgm_crs = get_reprojected_dgm_window(GOV_DGM_TIF, dsm_crs, tuple(crop_bounds), buffer_m=100)

step1_records = []
for _, row in crop_aoi.iterrows():
    dsm_mean = zonal_mean_from_file(ground_dsm_path, row.geometry, crop_aoi.crs)
    dgm_mean = zonal_mean_from_array(dgm_array, dgm_transform, dgm_crs, row.geometry, crop_aoi.crs)
    offset = dsm_mean - dgm_mean if np.isfinite(dsm_mean) and np.isfinite(dgm_mean) else np.nan
    step1_records.append({
        "aoi_id": row["aoi_id"], "dsm_mean_elevation_m": dsm_mean,
        "government_dgm_mean_elevation_m": dgm_mean, "offset_m": offset,
    })

step1_df = pd.DataFrame(step1_records)
median_offset = step1_df["offset_m"].median()
step1_df["deviation_from_median_m"] = step1_df["offset_m"] - median_offset
step1_df["flagged"] = step1_df["deviation_from_median_m"].abs() > OFFSET_FLAG_THRESHOLD_M

step1_csv = RESULTS_DIR / "step1_dsm_vs_government_dgm_validation.csv"
step1_df.to_csv(step1_csv, index=False)
print(step1_df.to_string(index=False))
std_offset = step1_df["offset_m"].std()
print(f"\nMedian offset: {median_offset:.3f} m | Std across AOIs: {std_offset:.3f} m")
print(f"Saved -> {step1_csv}")

plt.figure(figsize=(9, 5))
colors = ["crimson" if f else "steelblue" for f in step1_df["flagged"]]
plt.bar(step1_df["aoi_id"], step1_df["offset_m"], color=colors)
plt.axhline(median_offset, color="black", linestyle="--", linewidth=1, label=f"Median ({median_offset:.2f} m)")
plt.ylabel("DSM - Government DGM (m)")
plt.title("Step 1: Elevation offset per AOI, 23.03 DSM vs. government DGM")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
step1_plot = PLOTS_DIR / "step1_dsm_vs_government_dgm_offsets.png"
plt.savefig(step1_plot, dpi=150)
plt.close()
print(f"Saved -> {step1_plot}")


# ===========================================================================
# STEP 2 -- Reference-surface (road) drift check across all dates
# ===========================================================================

print("\n" + "=" * 70)
print("STEP 2: Checking elevation drift on stable reference surfaces (roads)")
print("=" * 70)

step2_records = []
for date_label, folder, folder_name in date_folders:
    dsm_path = find_dsm_file(folder, date_label)
    if dsm_path is None:
        print(f"WARNING: no DSM found for {date_label}, skipping.")
        continue
    print(f"Reading reference elevations for {date_label} ...")
    for _, row in ref_aoi.iterrows():
        elev = zonal_mean_from_file(dsm_path, row.geometry, ref_aoi.crs)
        step2_records.append({"date": date_label, "ref_id": row["aoi_id"], "elevation_m": elev})

step2_df = pd.DataFrame(step2_records)
step2_df["date"] = pd.to_datetime(step2_df["date"])

ref_date_ts = pd.Timestamp(ground_date_label)
baseline = step2_df[step2_df["date"] == ref_date_ts].set_index("ref_id")["elevation_m"]
step2_df["baseline_elevation_m"] = step2_df["ref_id"].map(baseline)
step2_df["drift_m"] = step2_df["elevation_m"] - step2_df["baseline_elevation_m"]

step2_csv = RESULTS_DIR / "step2_reference_surface_drift.csv"
step2_df.to_csv(step2_csv, index=False)

step2_summary = (
    step2_df.groupby("date", as_index=False)
    .agg(mean_drift_m=("drift_m", "mean"), std_drift_m=("drift_m", "std"), n_refs=("ref_id", "nunique"))
)
step2_summary["flagged"] = step2_summary["mean_drift_m"].abs() > DRIFT_FLAG_THRESHOLD_M
step2_summary_csv = RESULTS_DIR / "step2_reference_surface_drift_summary.csv"
step2_summary.to_csv(step2_summary_csv, index=False)

print(step2_summary.to_string(index=False))
print(f"Saved -> {step2_csv}")
print(f"Saved -> {step2_summary_csv}")

plt.figure(figsize=(9, 5.5))
sns.lineplot(data=step2_df, x="date", y="elevation_m", hue="ref_id", marker="o")
plt.ylabel("Elevation (m)")
plt.title("Step 2: Reference surface (road/path) elevation across flight dates")
plt.xticks(rotation=45)
plt.tight_layout()
step2_plot = PLOTS_DIR / "step2_reference_surface_drift.png"
plt.savefig(step2_plot, dpi=150)
plt.close()
print(f"Saved -> {step2_plot}")


# ===========================================================================
# STEP 3 -- Canopy Height Model (CHM) across the season, per crop AOI
# ===========================================================================

print("\n" + "=" * 70)
print("STEP 3: Computing canopy height model (CHM) across the season")
print("=" * 70)

with rasterio.open(ground_dsm_path) as ref_src:
    ground_array = ref_src.read(1, masked=True).astype("float32").filled(np.nan)
    ref_transform = ref_src.transform
    ref_crs = ref_src.crs
    ref_shape = ground_array.shape

crop_aoi_ref = crop_aoi.to_crs(ref_crs)

step3_records = []
for date_label, folder, folder_name in date_folders:
    dsm_path = find_dsm_file(folder, date_label)
    if dsm_path is None:
        print(f"WARNING: no DSM found for {date_label}, skipping.")
        continue
    print(f"Processing CHM for {date_label} ...")

    if folder_name == GROUND_REFERENCE_FOLDER_NAME:
        date_array = ground_array.copy()
    else:
        date_array = reproject_to_reference(dsm_path, ref_transform, ref_crs, ref_shape)

    chm_array = date_array - ground_array

    for _, row in crop_aoi_ref.iterrows():
        stats = zonal_stats_from_array(chm_array, ref_transform, ref_crs, row.geometry, crop_aoi_ref.crs, "chm")
        rec = {"aoi_id": row["aoi_id"], "variant": row["variant"], "repetition": row["repetition"],
               "date": date_label, **stats}
        mean_val = rec["chm_mean"]
        rec["implausible"] = np.isfinite(mean_val) and not (PLAUSIBLE_MIN_M <= mean_val <= PLAUSIBLE_MAX_M)
        step3_records.append(rec)

step3_df = pd.DataFrame(step3_records)
step3_df["date"] = pd.to_datetime(step3_df["date"])
step3_long_csv = RESULTS_DIR / "step3_canopy_height_by_polygon_long.csv"
step3_df.to_csv(step3_long_csv, index=False)
print(f"Saved -> {step3_long_csv}")

self_check = step3_df[step3_df["date"] == ref_date_ts]
max_self_check_dev = self_check["chm_mean"].abs().max()
print(f"\nSelf-check ({ground_date_label} vs itself): max deviation = {max_self_check_dev:.3f} m "
      f"({'OK' if max_self_check_dev < 0.05 else 'WARNING -- check pipeline'})")

flagged = step3_df[step3_df["implausible"]]
if not flagged.empty:
    print(f"\nWARNING: {len(flagged)} polygon-date combos flagged as implausible:")
    print(flagged[["date", "aoi_id", "chm_mean"]].to_string(index=False))
else:
    print(f"\nNo implausible CHM values detected (within {PLAUSIBLE_MIN_M}m to {PLAUSIBLE_MAX_M}m).")

step3_summary = (
    step3_df.groupby(["variant", "date"], as_index=False)
    .agg(n_polygons=("aoi_id", "nunique"), chm_mean=("chm_mean", "mean"), chm_std=("chm_mean", "std"))
    .sort_values(["variant", "date"])
)
step3_summary_csv = RESULTS_DIR / "step3_canopy_height_summary.csv"
step3_summary.to_csv(step3_summary_csv, index=False)
print(f"Saved -> {step3_summary_csv}")
print(step3_summary.to_string(index=False))

variant_order_present = [v for v in VARIANT_ORDER if v in step3_df["variant"].unique()]
dates_sorted = sorted(step3_df["date"].unique())
n_cols = min(4, len(dates_sorted))
n_rows = int(np.ceil(len(dates_sorted) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4 * n_rows), squeeze=False)
for i, d in enumerate(dates_sorted):
    ax = axes[i // n_cols][i % n_cols]
    sns.boxplot(data=step3_df[step3_df["date"] == d], x="variant", y="chm_mean",
                order=variant_order_present, ax=ax)
    ax.set_title(pd.Timestamp(d).strftime("%Y-%m-%d"))
    ax.set_xlabel("")
    ax.set_ylabel("Canopy height (m)" if i % n_cols == 0 else "")
for i in range(len(dates_sorted), n_rows * n_cols):
    axes[i // n_cols][i % n_cols].axis("off")
fig.suptitle("Step 3: Canopy height by zone, per date (DSM_date - DSM_23.03)", y=1.02)
plt.tight_layout()
step3_plot1 = PLOTS_DIR / "step3_canopy_height_by_variant_per_date.png"
plt.savefig(step3_plot1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {step3_plot1}")

plt.figure(figsize=(9, 5.5))
sns.lineplot(data=step3_df, x="date", y="chm_mean", hue="variant", hue_order=variant_order_present,
             marker="o", estimator="mean", errorbar="sd")
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.ylabel("Canopy height (m)")
plt.xlabel("Date")
plt.title("Step 3: Canopy height over the season, by zone")
plt.xticks(rotation=45)
plt.tight_layout()
step3_plot2 = PLOTS_DIR / "step3_canopy_height_timeline.png"
plt.savefig(step3_plot2, dpi=150)
plt.close()
print(f"Saved -> {step3_plot2}")

print("\n" + "=" * 70)
print("ALL STEPS COMPLETE")
print(f"CSVs  -> {RESULTS_DIR}")
print(f"Plots -> {PLOTS_DIR}")
print("=" * 70)

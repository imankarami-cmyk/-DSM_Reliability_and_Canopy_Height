"""
Inspect the 23.03 bare-earth DSM across its FULL extent to judge how well
it represents real terrain before trusting it as the ground reference for
canopy height calculations.

Handles large orthomosaic files safely:
  - Statistics (coverage, min/max/mean/std) are computed by streaming
    through the file block-by-block, so it never loads the whole raster
    into memory at once.
  - The visual renders (elevation map, hillshade, noise/roughness map) use
    a downsampled overview so they stay fast and readable regardless of
    the file's native resolution.

WHAT THIS CHECKS:
  1. File-level metadata.
  2. Data coverage: how much of the DSM is real data vs. nodata/holes
     (common near orthomosaic edges or over reflective/problem surfaces).
  3. Elevation plausibility across the whole field.
  4. Visual terrain check (elevation map + hillshade) -- smooth, natural
     terrain is good; blocky seams, holes, or spikes are red flags.
  5. A local roughness/noise map -- flat farmland should look "quiet"
     almost everywhere; scattered bright speckle indicates point-cloud
     noise/blunders in the photogrammetry reconstruction.

Requires: rasterio, numpy, matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import generic_filter

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DSM_TIF = Path(r"C:\Users\imankarami\Data\new Project\analysis\data\stich\drone full\23.03\2026-03-23_dsm.tif")

PLOTS_DIR = Path(r"C:\Users\imankarami\Data\new Project\analysis\plot\DSM")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_OVERVIEW_DIM = 1500  # max width/height (pixels) for the visual renders


if __name__ == "__main__":
    print("=" * 70)
    print("PART 1: File-level metadata")
    print("=" * 70)

    with rasterio.open(DSM_TIF) as src:
        print(f"Driver: {src.driver}")
        print(f"CRS: {src.crs}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"Nodata value: {src.nodata}")
        print(f"Pixel resolution: {src.res[0]:.3f} m x {src.res[1]:.3f} m")
        print(f"Full DSM size: {src.width} x {src.height} pixels")
        print(f"Extent: {(src.width * src.res[0]):.1f} m x {(src.height * src.res[1]):.1f} m")
        print(f"Bounds: {src.bounds}")

        width, height = src.width, src.height
        nodata = src.nodata

        print("\n" + "=" * 70)
        print("PART 2: Streaming statistics (memory-safe, full resolution)")
        print("=" * 70)

        n_valid = 0
        n_total = 0
        vmin, vmax = np.inf, -np.inf
        running_sum = 0.0
        running_sumsq = 0.0

        for _, window in src.block_windows(1):
            block = src.read(1, window=window)
            n_total += block.size
            if nodata is not None:
                valid = block[block != nodata]
            else:
                valid = block[np.isfinite(block)]
            if valid.size:
                n_valid += valid.size
                vmin = min(vmin, valid.min())
                vmax = max(vmax, valid.max())
                running_sum += valid.sum()
                running_sumsq += np.square(valid, dtype=np.float64).sum()

        n_missing = n_total - n_valid
        pct_missing = 100 * n_missing / n_total if n_total else np.nan

        if n_valid > 0:
            vmean = running_sum / n_valid
            vstd = np.sqrt(max(running_sumsq / n_valid - vmean**2, 0))
        else:
            vmean = vstd = np.nan

        print(f"Total pixels: {n_total:,}")
        print(f"Valid pixels: {n_valid:,} ({100 - pct_missing:.2f}%)")
        print(f"Missing/nodata pixels: {n_missing:,} ({pct_missing:.2f}%)")
        print(f"Min: {vmin:.2f} m | Max: {vmax:.2f} m | Mean: {vmean:.2f} m | Std: {vstd:.2f} m")
        print(f"Elevation range: {vmax - vmin:.2f} m")

        implausible = (vmax - vmin) > 50  # bare flat field shouldn't span more than this
        if implausible:
            print("WARNING: elevation range is large for what should be relatively flat "
                  "farmland -- check for spikes/blunders (see noise map below).")
        else:
            print("Elevation range looks reasonable for flat-to-gently-rolling farmland.")

        print("\n" + "=" * 70)
        print("PART 3: Visual inspection (downsampled overview, full extent)")
        print("=" * 70)

        scale = min(1.0, MAX_OVERVIEW_DIM / max(width, height))
        out_width = max(1, int(width * scale))
        out_height = max(1, int(height * scale))
        print(f"Rendering overview at {out_width} x {out_height} pixels "
              f"(downsampled from {width} x {height})")

        overview = src.read(
            1, out_shape=(1, out_height, out_width), resampling=Resampling.average
        )
        if nodata is not None:
            overview = np.ma.masked_equal(overview, nodata)
        else:
            overview = np.ma.masked_invalid(overview)

    # --- Elevation map ---
    plt.figure(figsize=(9, 8))
    im = plt.imshow(overview, cmap="terrain")
    plt.colorbar(im, label="Elevation (m)")
    plt.title("23.03 bare-earth DSM: full-extent elevation\n(look for smooth, natural-looking terrain)")
    plt.axis("off")
    elev_path = PLOTS_DIR / "dsm_2303_inspection_elevation.png"
    plt.tight_layout()
    plt.savefig(elev_path, dpi=150)
    plt.close()
    print(f"Saved -> {elev_path}")

    # --- Hillshade ---
    arr = overview.filled(np.nan).astype(float)
    dy, dx = np.gradient(arr)
    slope = np.pi / 2 - np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_rad, alt_rad = np.radians(315), np.radians(45)
    hillshade = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect)

    plt.figure(figsize=(9, 8))
    plt.imshow(hillshade, cmap="gray")
    plt.title("23.03 bare-earth DSM: hillshade\n(check for seams, holes, blocky artifacts)")
    plt.axis("off")
    hillshade_path = PLOTS_DIR / "dsm_2303_inspection_hillshade.png"
    plt.tight_layout()
    plt.savefig(hillshade_path, dpi=150)
    plt.close()
    print(f"Saved -> {hillshade_path}")

    # --- Local roughness/noise map ---
    print("Computing local roughness map (this may take a moment)...")

    def local_std(values):
        v = values[np.isfinite(values)]
        return v.std() if v.size > 1 else 0.0

    roughness = generic_filter(arr, local_std, size=3, mode="nearest")

    plt.figure(figsize=(9, 8))
    im2 = plt.imshow(roughness, cmap="inferno", vmax=np.nanpercentile(roughness, 99))
    plt.colorbar(im2, label="Local elevation std (m), 3x3 window")
    plt.title("23.03 bare-earth DSM: local roughness/noise map\n(should be mostly dark/uniform over bare soil; "
              "bright speckle = point-cloud noise)")
    plt.axis("off")
    roughness_path = PLOTS_DIR / "dsm_2303_inspection_roughness.png"
    plt.tight_layout()
    plt.savefig(roughness_path, dpi=150)
    plt.close()
    print(f"Saved -> {roughness_path}")

    # --- Missing data map, if relevant ---
    if n_missing > 0:
        plt.figure(figsize=(9, 8))
        plt.imshow(~overview.mask if hasattr(overview, "mask") else np.ones_like(arr, dtype=bool),
                   cmap="RdYlGn", vmin=0, vmax=1)
        plt.title(f"Data coverage map ({pct_missing:.1f}% missing overall, shown in red)")
        plt.axis("off")
        coverage_path = PLOTS_DIR / "dsm_2303_inspection_coverage.png"
        plt.tight_layout()
        plt.savefig(coverage_path, dpi=150)
        plt.close()
        print(f"Saved -> {coverage_path}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    high_roughness_frac = np.mean(roughness > 0.1)  # fraction of pixels with >10cm local std
    print(f"Fraction of area with high local roughness (>0.10m std in 3x3 window): "
          f"{high_roughness_frac*100:.2f}%")

    if pct_missing > 5:
        print(f"USE WITH CAUTION: {pct_missing:.1f}% missing data across the DSM.")
    elif implausible:
        print("USE WITH CAUTION: elevation range looks too large for flat farmland -- inspect "
              "the elevation map for isolated spikes.")
    elif high_roughness_frac > 0.10:
        print("USE WITH CAUTION: more than 10% of the area shows high local roughness -- "
              "inspect the roughness map for where the noise is concentrated (e.g. field "
              "edges, specific patches) before trusting those areas as ground reference.")
    else:
        print("LOOKS RELIABLE: good data coverage, plausible elevation range, low background "
              "noise. Safe to use as the ground reference for canopy height.")

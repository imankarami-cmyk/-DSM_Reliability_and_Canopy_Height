"""
Compute the official geoid undulation (N) at your farm's coordinates,
using the BEV Austrian Geoid 2008 (GRS80) model, and compare it to your
empirically measured DSM-vs-government-DGM offset (44.188 m).

Run this on your own machine with normal internet access (this needs to
reach cdn.proj.org to fetch the official geoid grid the first time).

Requires: pyproj (pip install pyproj)
"""

from pyproj import Transformer, network

# ---------------------------------------------------------------------------
# CONFIG -- replace with your farm's actual coordinates
# ---------------------------------------------------------------------------

# Farm coordinates (converted from your UTM33N AOI centroid: 635465.96, 5321941.25)
FARM_LON = 16.817317
FARM_LAT = 48.036378

EMPIRICAL_OFFSET_M = 44.188  # your measured offset from the AOI validation


if __name__ == "__main__":
    network.set_network_enabled(True)  # allow pyproj to fetch the BEV geoid grid if not already downloaded

    # EPSG:4937 = ETRS89 3D geographic (lon, lat, ellipsoidal height)
    # EPSG:9274 = EVRF2000 Austria height (orthometric height, government DGM's system)
    transformer = Transformer.from_crs("EPSG:4937", "EPSG:9274", always_xy=True)

    # Feed ellipsoidal height = 0 so the returned orthometric height directly
    # gives us -N (i.e. N = 0 - H)
    lon_out, lat_out, H = transformer.transform(FARM_LON, FARM_LAT, 0.0)
    N = 0.0 - H

    print(f"Farm coordinates: lon={FARM_LON}, lat={FARM_LAT}")
    print(f"Official geoid undulation N at this location: {N:.3f} m")
    print(f"Your empirically measured offset: {EMPIRICAL_OFFSET_M:.3f} m")
    print(f"Difference: {abs(N - EMPIRICAL_OFFSET_M):.3f} m")

    if abs(N - EMPIRICAL_OFFSET_M) < 0.5:
        print("\n=> MATCH: your empirical offset is consistent with the official geoid "
              "undulation for this location. This confirms the 44.19m offset is the "
              "expected ellipsoidal-to-orthometric height difference, not a data error.")
    else:
        print("\n=> Larger difference than expected -- double check the farm coordinates "
              "used here, and confirm your drone DSM's height really is raw ellipsoidal "
              "(unconverted) height.")

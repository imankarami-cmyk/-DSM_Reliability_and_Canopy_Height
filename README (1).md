================================================================================
README — Scripts needed to reproduce DSM_Reliability_and_Canopy_Height_Report.docx
================================================================================

Run these 4 scripts, in order. Every other script in this folder is not
needed for the report.

Before running anything: make sure MASTER_AOI_SHP and AOI_NAME_COL are set
the same way in every script below (they currently differ between files --
check each script's CONFIG section near the top).


--------------------------------------------------------------------------------
1) inspect_own_dsm_2303.py                              -> Report Figures 1, 2, 3
--------------------------------------------------------------------------------
Produces:
    dsm_2303_inspection_elevation.png   -> Figure 1
    dsm_2303_inspection_hillshade.png   -> Figure 2
    dsm_2303_inspection_roughness.png   -> Figure 3


--------------------------------------------------------------------------------
2) compute_geoid_undulation.py                          -> Report Section 2.2 table
--------------------------------------------------------------------------------
Produces:
    printed geoid undulation (N) value, compared against the empirical offset
Requires internet access (fetches the BEV geoid grid on first run).
Update FARM_LON / FARM_LAT / EMPIRICAL_OFFSET_M at the top before running.


--------------------------------------------------------------------------------
3) inspect_dsm_and_compare_to_government.py             -> Report Figure 4 + residual table
--------------------------------------------------------------------------------
Run PART B only (skip PART A -- already covered by script #1 above).
Produces:
    dsm_vs_gov_dgm_residual_map.png            -> Figure 4
    dsm_vs_gov_dgm_residual_by_aoi.csv          -> Section 2.3 residual table


--------------------------------------------------------------------------------
4) full_dsm_analysis_pipeline.py                        -> Report Sections 3 and 4
--------------------------------------------------------------------------------
Produces:
    Step 2 output -> Section 3 drift table
    Step 3 output -> Section 4 canopy height table
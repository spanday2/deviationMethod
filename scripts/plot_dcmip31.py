# #!/usr/bin/env python3
# """
# Plot DCMIP 3-1 potential-temperature perturbation using an analytically
# computed base-state potential temperature.

# No command-line arguments are required. Change the paths and options in
# the USER SETTINGS section below, then run:

#     python plot_dcmip31.py
# """

# from pathlib import Path
# import warnings

# import matplotlib

# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import netCDF4 as nc
# import numpy as np

# warnings.filterwarnings("ignore")


# # ============================================================
# # USER SETTINGS
# # ============================================================

# # Path to your model result NetCDF file.
# INPUT_FILE = (
#     "/home/shp000/site8/U2_data/raid/ppp6/"
#     "deviationMethod/results/dcmip_31.nc"
# )

# # Directory where PNG figures will be saved.
# OUTPUT_DIR = "./plots/theta_sections"

# # Number of output times to plot.
# # Use None to plot every available output time.
# MAX_TIMES = None

# # PNG resolution.
# DPI = 300

# # Save the analytically computed base-state potential temperature.
# SAVE_BASE_STATE = True

# # Output path for the computed base state.
# BASE_STATE_FILE = "theta_base_computed.npy"

# # Number of contour intervals.
# NUMBER_OF_CONTOUR_LEVELS = 17

# # Use the middle horizontal row, as in the original plotting script.
# # Set to an integer to select a specific row.
# YROW = None


# # ============================================================
# # DCMIP 3-1 CONSTANTS
# # ============================================================

# GRAVITY = 9.80616        # m s^-2
# RD = 287.05              # J kg^-1 K^-1
# CPD = 1005.46            # J kg^-1 K^-1
# P0 = 100000.0            # Pa
# TEQ = 300.0              # K
# U0 = 20.0                # m s^-1
# BRUNT_VAISALA = 0.01     # s^-1


# def analytic_base_theta(latitude_radians, height_m):
#     """
#     Compute the analytic DCMIP 3-1 base-state potential temperature.

#     Parameters
#     ----------
#     latitude_radians : numpy.ndarray
#         Latitude in radians. Expected to be broadcast-compatible with
#         height_m. In this script, its shape is (panel, 1, j, i).

#     height_m : numpy.ndarray
#         Model elevation in metres. Expected shape:
#         (panel, vertical_level, j, i).

#     Returns
#     -------
#     numpy.ndarray
#         Base-state potential temperature in kelvin, with the same shape
#         as height_m.
#     """
#     kappa = RD / CPD

#     big_g = GRAVITY**2 / (BRUNT_VAISALA**2 * CPD)

#     surface_temperature = big_g + (TEQ - big_g) * np.exp(
#         -(
#             U0**2
#             * BRUNT_VAISALA**2
#             / (4.0 * GRAVITY**2)
#         )
#         * (np.cos(2.0 * latitude_radians) - 1.0)
#     )

#     surface_pressure = (
#         P0
#         * np.exp(
#             U0**2
#             / (4.0 * big_g * RD)
#             * (np.cos(2.0 * latitude_radians) - 1.0)
#         )
#         * (surface_temperature / TEQ) ** (1.0 / kappa)
#     )

#     theta_surface = (
#         surface_temperature
#         * (P0 / surface_pressure) ** kappa
#     )

#     theta_base = theta_surface * np.exp(
#         BRUNT_VAISALA**2 * height_m / GRAVITY
#     )

#     return theta_base


# def split_longitude_segments(longitudes):
#     """
#     Split a longitude row at the periodic 0/360-degree seam.

#     Parameters
#     ----------
#     longitudes : numpy.ndarray
#         One-dimensional longitude array.

#     Returns
#     -------
#     tuple
#         Longitude values converted to [0, 360), and a list of boolean
#         masks defining one or two continuous longitude segments.
#     """
#     longitude_360 = np.mod(np.asarray(longitudes), 360.0)

#     if longitude_360.size < 2:
#         return longitude_360, [
#             np.ones(longitude_360.shape, dtype=bool)
#         ]

#     longitude_range = (
#         np.nanmax(longitude_360)
#         - np.nanmin(longitude_360)
#     )

#     if longitude_range < 180.0:
#         return longitude_360, [
#             np.ones(longitude_360.shape, dtype=bool)
#         ]

#     longitude_maximum = np.nanmax(longitude_360)

#     high_longitude_mask = (
#         longitude_360 > longitude_maximum - 180.0
#     )

#     low_longitude_mask = ~high_longitude_mask

#     return longitude_360, [
#         high_longitude_mask,
#         low_longitude_mask,
#     ]


# def validate_shapes(theta, elevation, latitudes, longitudes):
#     """
#     Check that the NetCDF variables have the expected dimensions.
#     """
#     if theta.ndim != 5:
#         raise ValueError(
#             "Expected theta dimensions "
#             "(time, panel, vertical_level, j, i), "
#             f"but found shape {theta.shape}"
#         )

#     number_of_times, number_of_panels, nk, nj, ni = theta.shape

#     expected_elevation_shape = (
#         number_of_panels,
#         nk,
#         nj,
#         ni,
#     )

#     if elevation.shape != expected_elevation_shape:
#         raise ValueError(
#             "Unexpected elev shape.\n"
#             f"Expected: {expected_elevation_shape}\n"
#             f"Found:    {elevation.shape}"
#         )

#     expected_horizontal_shape = (
#         number_of_panels,
#         nj,
#         ni,
#     )

#     if latitudes.shape != expected_horizontal_shape:
#         raise ValueError(
#             "Unexpected lats shape.\n"
#             f"Expected: {expected_horizontal_shape}\n"
#             f"Found:    {latitudes.shape}"
#         )

#     if longitudes.shape != expected_horizontal_shape:
#         raise ValueError(
#             "Unexpected lons shape.\n"
#             f"Expected: {expected_horizontal_shape}\n"
#             f"Found:    {longitudes.shape}"
#         )

#     return (
#         number_of_times,
#         number_of_panels,
#         nk,
#         nj,
#         ni,
#     )


# def load_data(input_file):
#     """
#     Load all required variables from the NetCDF file.
#     """
#     required_variables = (
#         "theta",
#         "elev",
#         "lats",
#         "lons",
#         "time",
#     )

#     with nc.Dataset(input_file, "r") as dataset:
#         missing_variables = [
#             variable_name
#             for variable_name in required_variables
#             if variable_name not in dataset.variables
#         ]

#         if missing_variables:
#             raise KeyError(
#                 "The input file is missing required variables: "
#                 + ", ".join(missing_variables)
#             )

#         theta = np.asarray(
#             dataset.variables["theta"][:],
#             dtype=np.float64,
#         )

#         elevation = np.asarray(
#             dataset.variables["elev"][:],
#             dtype=np.float64,
#         )

#         latitudes = np.asarray(
#             dataset.variables["lats"][:],
#             dtype=np.float64,
#         )

#         longitudes = np.asarray(
#             dataset.variables["lons"][:],
#             dtype=np.float64,
#         )

#         times = np.asarray(
#             dataset.variables["time"][:],
#             dtype=np.float64,
#         )

#     return theta, elevation, latitudes, longitudes, times


# def plot_theta_section(
#     time_index,
#     time_value,
#     theta_prime,
#     elevation,
#     longitudes,
#     yrow,
#     panel_indices,
#     output_directory,
# ):
#     """
#     Save one longitude-height section of theta perturbation.
#     """
#     section = theta_prime[:, :, yrow, :]

#     absolute_maximum = float(
#         np.nanmax(np.abs(section))
#     )

#     if (
#         not np.isfinite(absolute_maximum)
#         or absolute_maximum == 0.0
#     ):
#         absolute_maximum = 1.0e-12

#     contour_levels = np.linspace(
#         -absolute_maximum,
#         absolute_maximum,
#         NUMBER_OF_CONTOUR_LEVELS,
#     )

#     fig, axis = plt.subplots(
#         figsize=(11.0, 3.8)
#     )

#     contour_handle = None

#     for panel in panel_indices:
#         longitude_360, segment_masks = split_longitude_segments(
#             longitudes[panel, yrow, :]
#         )

#         panel_elevation = elevation[panel, :, yrow, :]
#         panel_theta_prime = theta_prime[panel, :, yrow, :]

#         for segment_mask in segment_masks:
#             if np.count_nonzero(segment_mask) < 2:
#                 continue

#             segment_longitudes = longitude_360[segment_mask]
#             sorting_indices = np.argsort(segment_longitudes)
#             segment_longitudes = segment_longitudes[sorting_indices]

#             segment_elevation = (
#                 panel_elevation[:, segment_mask][:, sorting_indices]
#                 / 1000.0
#             )

#             segment_theta_prime = (
#                 panel_theta_prime[:, segment_mask][:, sorting_indices]
#             )

#             longitude_mesh = np.broadcast_to(
#                 segment_longitudes[None, :],
#                 segment_elevation.shape,
#             )

#             contour_handle = axis.contourf(
#                 longitude_mesh,
#                 segment_elevation,
#                 segment_theta_prime,
#                 levels=contour_levels,
#                 cmap="RdBu_r",
#                 extend="both",
#             )

#             axis.plot(
#                 segment_longitudes,
#                 segment_elevation[0, :],
#                 color="black",
#                 linewidth=0.8,
#             )

#     if contour_handle is None:
#         plt.close(fig)

#         raise RuntimeError(
#             "No valid longitude segments were found "
#             f"for time index {time_index}."
#         )

#     colorbar = fig.colorbar(
#         contour_handle,
#         ax=axis,
#         pad=0.02,
#     )

#     colorbar.set_label(
#         r"$\Delta\theta"
#         r"=\theta-\overline{\theta}$ (K)"
#     )

#     axis.set_xlabel(
#         "Longitude (degrees east)"
#     )

#     axis.set_ylabel(
#         "Height (km)"
#     )

#     axis.set_xlim(
#         0.0,
#         360.0,
#     )

#     axis.set_title(
#         "DCMIP 3-1 potential-temperature perturbation, "
#         f"t = {time_value:g} s"
#     )

#     axis.grid(
#         alpha=0.2,
#         linewidth=0.5,
#     )

#     fig.tight_layout()

#     output_file = (
#         output_directory
#         / f"theta_prime_{time_index:04d}.png"
#     )

#     fig.savefig(
#         output_file,
#         dpi=DPI,
#         bbox_inches="tight",
#     )

#     plt.close(fig)

#     print(
#         f"Saved: {output_file}\n"
#         f"  time = {time_value:g} s\n"
#         f"  theta-prime minimum = "
#         f"{np.nanmin(section):.6e} K\n"
#         f"  theta-prime maximum = "
#         f"{np.nanmax(section):.6e} K"
#     )


# def main():
#     input_file = Path(INPUT_FILE)
#     output_directory = Path(OUTPUT_DIR)

#     if not input_file.exists():
#         raise FileNotFoundError(
#             f"Input NetCDF file not found:\n{input_file}"
#         )

#     output_directory.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     (
#         theta,
#         elevation,
#         latitudes,
#         longitudes,
#         times,
#     ) = load_data(input_file)

#     (
#         number_of_times,
#         number_of_panels,
#         nk,
#         nj,
#         ni,
#     ) = validate_shapes(
#         theta,
#         elevation,
#         latitudes,
#         longitudes,
#     )

#     if times.size != number_of_times:
#         raise ValueError(
#             "The time coordinate does not match the theta "
#             "time dimension.\n"
#             f"theta time dimension: {number_of_times}\n"
#             f"time coordinate size: {times.size}"
#         )

#     if YROW is None:
#         yrow = nj // 2
#     else:
#         yrow = int(YROW)

#     if not 0 <= yrow < nj:
#         raise ValueError(
#             f"YROW={yrow} is outside the valid range "
#             f"0 to {nj - 1}."
#         )

#     # Convert latitude from degrees to radians and add a
#     # singleton vertical dimension:
#     #
#     #     (panel, j, i) -> (panel, 1, j, i)
#     #
#     # It then broadcasts against elevation:
#     #
#     #     elevation shape:
#     #     (panel, vertical_level, j, i)
#     latitude_radians = (
#         np.deg2rad(latitudes)[:, None, :, :]
#     )

#     theta_base = analytic_base_theta(
#         latitude_radians,
#         elevation,
#     )

#     if theta_base.shape != elevation.shape:
#         raise ValueError(
#             "Computed base-state shape does not match "
#             "the elevation shape.\n"
#             f"theta_base shape: {theta_base.shape}\n"
#             f"elevation shape:  {elevation.shape}"
#         )

#     if SAVE_BASE_STATE:
#         base_state_file = Path(BASE_STATE_FILE)

#         base_state_file.parent.mkdir(
#             parents=True,
#             exist_ok=True,
#         )

#         np.save(
#             base_state_file,
#             theta_base,
#         )

#         print(
#             f"Saved computed base state: "
#             f"{base_state_file}"
#         )

#     if MAX_TIMES is None:
#         number_to_plot = number_of_times
#     else:
#         number_to_plot = min(
#             number_of_times,
#             int(MAX_TIMES),
#         )

#     # The original plotting code uses panels 0, 1, 2, and 3.
#     # If fewer than four panels are available, use all of them.
#     panel_indices = list(
#         range(min(4, number_of_panels))
#     )

#     print()
#     print("Input summary")
#     print("-------------")
#     print(f"Input file:       {input_file}")
#     print(f"Output directory: {output_directory}")
#     print(f"theta shape:      {theta.shape}")
#     print(f"theta-base shape: {theta_base.shape}")
#     print(f"elevation shape:  {elevation.shape}")
#     print(f"latitude shape:   {latitudes.shape}")
#     print(f"longitude shape:  {longitudes.shape}")
#     print(f"selected y row:   {yrow}")
#     print(f"panels plotted:   {panel_indices}")
#     print(f"times plotted:    {number_to_plot}")
#     print()

#     for time_index in range(number_to_plot):
#         theta_prime = (
#             theta[time_index, :, :, :, :]
#             - theta_base
#         )

#         plot_theta_section(
#             time_index=time_index,
#             time_value=float(times[time_index]),
#             theta_prime=theta_prime,
#             elevation=elevation,
#             longitudes=longitudes,
#             yrow=yrow,
#             panel_indices=panel_indices,
#             output_directory=output_directory,
#         )

#     print()
#     print(
#         f"Finished. Saved {number_to_plot} PNG files "
#         f"in {output_directory}"
#     )


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3

"""
Plot DCMIP 3-1 potential-temperature perturbation along an
equatorial longitude-height section on a cube-sphere grid.
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

warnings.filterwarnings("ignore")


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE = (
    "/home/shp000/site8/U2_data/raid/ppp6/"
    "deviationMethod/results/dcmip_31.nc"
)

OUTPUT_DIR = "./plots/theta_sections_journal"


# ------------------------------------------------------------
# Times to plot, in seconds
#
# Examples:
#
# PLOT_TIMES = [1800.0]
# PLOT_TIMES = [3600.0]
# PLOT_TIMES = [1800.0, 3600.0]
# ------------------------------------------------------------

PLOT_TIMES = [1800.0]


# ------------------------------------------------------------
# PNG resolution
# ------------------------------------------------------------

DPI = 300


# ------------------------------------------------------------
# Save analytically computed base state
# ------------------------------------------------------------

SAVE_BASE_STATE = True
BASE_STATE_FILE = "theta_base_computed.npy"


# ------------------------------------------------------------
# Colormap
# ------------------------------------------------------------

COLORMAP = "RdYlBu"


# ------------------------------------------------------------
# Filled contour settings
#
# These control ONLY contourf.
#
# The levels are automatically generated with:
#
# np.linspace(COLOR_MIN, COLOR_MAX, NUMBER_OF_FILLED_LEVELS)
# ------------------------------------------------------------

COLOR_MIN = -0.09
COLOR_MAX = 0.09

NUMBER_OF_FILLED_LEVELS = 10


# ------------------------------------------------------------
# Contour line settings
#
# These control ONLY the black contour lines.
#
# Put exactly the contour-line values you want here.
#
# Examples:
#
# CONTOUR_LINE_LEVELS = [-0.09, 0.09]
#
# CONTOUR_LINE_LEVELS = [
#     -0.09, -0.06, -0.03,
#      0.00,
#      0.03,  0.06,  0.09
# ]
# ------------------------------------------------------------

SHOW_CONTOUR_LINES = True

CONTOUR_LINE_LEVELS = [
    -0.01,
     0.01,
]

CONTOUR_LINE_WIDTH = 0.4


# ------------------------------------------------------------
# Manual colorbar ticks and labels
#
# Each entry is:
#
#     (tick_position, "label")
#
# This avoids any possibility of different numbers of tick
# positions and labels.
#
# Example:
#
# (-0.09, "-0.09")
#
# means:
#     put a colorbar tick at -0.09
#     and label it "-0.09"
#
# You can put ANY label you want.
# ------------------------------------------------------------

COLORBAR_TICKS_AND_LABELS = [
    (-0.09, "-0.09"),
    (-0.07, "-0.07"),
    (-0.05, "-0.05"),
    (-0.03, "-0.03"),
    (-0.01, "-0.01"),
    (0.01, "0.01"),
    ( 0.03, "0.03"),
    ( 0.05, "0.05"),
    ( 0.07, "0.07"),
    ( 0.09, "0.09"),
]


# ------------------------------------------------------------
# Colorbar label
# ------------------------------------------------------------

COLORBAR_LABEL = r"$\Delta\theta$ (K)"


# ------------------------------------------------------------
# Which side of equator to use?
#
# "south" = nearest row with mean latitude <= 0
# "north" = nearest row with mean latitude >= 0
# ------------------------------------------------------------

EQUATOR_SIDE = "south"


# ------------------------------------------------------------
# Maximum distance of selected row from equator
# ------------------------------------------------------------

MAX_EQUATOR_DISTANCE_DEG = 2.0


# ------------------------------------------------------------
# Maximum neighbouring-panel gap to stitch
# ------------------------------------------------------------

MAX_STITCH_GAP_FACTOR = 3.0


# ============================================================
# DCMIP 3-1 CONSTANTS
# ============================================================

GRAVITY = 9.80616
RD = 287.05
CPD = 1005.46
P0 = 100000.0
TEQ = 300.0
U0 = 20.0
BRUNT_VAISALA = 0.01


# ============================================================
# ANALYTIC BASE-STATE POTENTIAL TEMPERATURE
# ============================================================

def analytic_base_theta(latitude_radians, height_m):

    kappa = RD / CPD

    big_g = GRAVITY**2 / (BRUNT_VAISALA**2 * CPD)

    surface_temperature = big_g + (TEQ - big_g) * np.exp(
        -(U0**2 * BRUNT_VAISALA**2 / (4.0 * GRAVITY**2))
        * (np.cos(2.0 * latitude_radians) - 1.0)
    )

    surface_pressure = (
        P0
        * np.exp(
            U0**2 / (4.0 * big_g * RD)
            * (np.cos(2.0 * latitude_radians) - 1.0)
        )
        * (surface_temperature / TEQ) ** (1.0 / kappa)
    )

    theta_surface = surface_temperature * (P0 / surface_pressure) ** kappa

    theta_base = theta_surface * np.exp(
        BRUNT_VAISALA**2 * height_m / GRAVITY
    )

    return theta_base


# ============================================================
# LOAD DATA
# ============================================================

def load_data(input_file):

    required_variables = (
        "theta",
        "elev",
        "lats",
        "lons",
        "time",
    )

    with nc.Dataset(input_file, "r") as dataset:

        missing_variables = [
            name
            for name in required_variables
            if name not in dataset.variables
        ]

        if missing_variables:
            raise KeyError(
                "Input file is missing variables: "
                + ", ".join(missing_variables)
            )

        theta = np.asarray(
            dataset.variables["theta"][:],
            dtype=np.float64,
        )

        elevation = np.asarray(
            dataset.variables["elev"][:],
            dtype=np.float64,
        )

        latitudes = np.asarray(
            dataset.variables["lats"][:],
            dtype=np.float64,
        )

        longitudes = np.asarray(
            dataset.variables["lons"][:],
            dtype=np.float64,
        )

        times = np.asarray(
            dataset.variables["time"][:],
            dtype=np.float64,
        )

    return theta, elevation, latitudes, longitudes, times


# ============================================================
# VALIDATE SHAPES
# ============================================================

def validate_shapes(
    theta,
    elevation,
    latitudes,
    longitudes,
):

    if theta.ndim != 5:
        raise ValueError(
            "Expected theta dimensions "
            "(time, panel, level, j, i), "
            f"but found {theta.shape}"
        )

    (
        number_of_times,
        number_of_panels,
        nk,
        nj,
        ni,
    ) = theta.shape

    if elevation.shape != (
        number_of_panels,
        nk,
        nj,
        ni,
    ):
        raise ValueError(
            f"Unexpected elevation shape: "
            f"{elevation.shape}"
        )

    if latitudes.shape != (
        number_of_panels,
        nj,
        ni,
    ):
        raise ValueError(
            f"Unexpected latitude shape: "
            f"{latitudes.shape}"
        )

    if longitudes.shape != (
        number_of_panels,
        nj,
        ni,
    ):
        raise ValueError(
            f"Unexpected longitude shape: "
            f"{longitudes.shape}"
        )

    if number_of_panels % 6 != 0:
        raise ValueError(
            f"Panel count {number_of_panels} "
            "is not divisible by 6."
        )

    panels_per_face = number_of_panels // 6

    return (
        number_of_times,
        number_of_panels,
        nk,
        nj,
        ni,
        panels_per_face,
    )


# ============================================================
# FIND EQUATORIAL ROW
# ============================================================

def choose_equatorial_row(
    latitudes,
    panel,
):

    panel_lats = np.asarray(
        latitudes[panel],
        dtype=np.float64,
    )

    row_mean_lat = np.nanmean(
        panel_lats,
        axis=1,
    )

    side = EQUATOR_SIDE.lower()

    if side == "south":

        valid_rows = np.where(
            row_mean_lat <= 0.0
        )[0]

        if valid_rows.size == 0:
            return None

        j = valid_rows[
            np.argmax(
                row_mean_lat[valid_rows]
            )
        ]

    elif side == "north":

        valid_rows = np.where(
            row_mean_lat >= 0.0
        )[0]

        if valid_rows.size == 0:
            return None

        j = valid_rows[
            np.argmin(
                row_mean_lat[valid_rows]
            )
        ]

    else:

        raise ValueError(
            "EQUATOR_SIDE must be "
            "'south' or 'north'."
        )

    j = int(j)

    mean_lat = float(
        row_mean_lat[j]
    )

    if abs(mean_lat) > MAX_EQUATOR_DISTANCE_DEG:
        return None

    return j, mean_lat


# ============================================================
# SPLIT PERIODIC LONGITUDE SEGMENT
# ============================================================

def split_periodic_longitude_segment(
    longitude,
    elevation,
    field,
):

    longitude = np.mod(
        np.asarray(
            longitude,
            dtype=np.float64,
        ),
        360.0,
    )

    elevation = np.asarray(
        elevation,
        dtype=np.float64,
    )

    field = np.asarray(
        field,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(longitude)
        &
        np.any(
            np.isfinite(elevation),
            axis=0,
        )
        &
        np.any(
            np.isfinite(field),
            axis=0,
        )
    )

    longitude = longitude[valid]
    elevation = elevation[:, valid]
    field = field[:, valid]

    if longitude.size < 2:
        return []

    order = np.argsort(
        longitude
    )

    longitude = longitude[order]
    elevation = elevation[:, order]
    field = field[:, order]

    gaps = np.diff(
        longitude
    )

    if gaps.size == 0:

        return [
            {
                "longitude": longitude,
                "elevation": elevation,
                "field": field,
                "lon_start": float(longitude[0]),
                "lon_end": float(longitude[-1]),
            }
        ]

    positive_gaps = gaps[
        gaps > 0.0
    ]

    if positive_gaps.size > 0:

        typical_gap = float(
            np.median(
                positive_gaps
            )
        )

    else:

        typical_gap = 0.0

    largest_internal_index = int(
        np.argmax(
            gaps
        )
    )

    largest_internal_gap = float(
        gaps[
            largest_internal_index
        ]
    )

    if typical_gap > 0.0:

        crosses_seam = (
            largest_internal_gap
            >
            3.0 * typical_gap
        )

    else:

        crosses_seam = (
            largest_internal_gap
            >
            180.0
        )

    if not crosses_seam:

        return [
            {
                "longitude": longitude,
                "elevation": elevation,
                "field": field,
                "lon_start": float(longitude[0]),
                "lon_end": float(longitude[-1]),
            }
        ]

    split_index = largest_internal_index + 1

    pieces = []

    lon_low = longitude[:split_index]
    elev_low = elevation[:, :split_index]
    field_low = field[:, :split_index]

    if lon_low.size >= 2:

        pieces.append(
            {
                "longitude": lon_low,
                "elevation": elev_low,
                "field": field_low,
                "lon_start": float(lon_low[0]),
                "lon_end": float(lon_low[-1]),
            }
        )

    lon_high = longitude[split_index:]
    elev_high = elevation[:, split_index:]
    field_high = field[:, split_index:]

    if lon_high.size >= 2:

        pieces.append(
            {
                "longitude": lon_high,
                "elevation": elev_high,
                "field": field_high,
                "lon_start": float(lon_high[0]),
                "lon_end": float(lon_high[-1]),
            }
        )

    return pieces


# ============================================================
# COLLECT GLOBAL EQUATORIAL SEGMENTS
# ============================================================

def collect_equatorial_segments(
    theta_prime,
    elevation,
    latitudes,
    longitudes,
    panels_per_face,
):

    segments = []

    number_of_side_panels = (
        4 * panels_per_face
    )

    for panel in range(
        number_of_side_panels
    ):

        selected = choose_equatorial_row(
            latitudes,
            panel,
        )

        if selected is None:
            continue

        j, mean_lat = selected

        longitude = longitudes[
            panel,
            j,
            :
        ]

        panel_elevation = (
            elevation[
                panel,
                :,
                j,
                :
            ]
            / 1000.0
        )

        panel_field = theta_prime[
            panel,
            :,
            j,
            :
        ]

        pieces = split_periodic_longitude_segment(
            longitude=longitude,
            elevation=panel_elevation,
            field=panel_field,
        )

        for piece in pieces:

            piece["panel"] = panel
            piece["j"] = j
            piece["mean_lat"] = mean_lat

            piece["field_min"] = float(
                np.nanmin(
                    piece["field"]
                )
            )

            piece["field_max"] = float(
                np.nanmax(
                    piece["field"]
                )
            )

            segments.append(
                piece
            )

    segments.sort(
        key=lambda item:
        item["lon_start"]
    )

    return segments


# ============================================================
# STITCH NEIGHBOURING PANELS
# ============================================================

def stitch_equatorial_segments(
    segments,
):

    longitude_parts = []
    elevation_parts = []
    field_parts = []

    for segment_index, segment in enumerate(
        segments
    ):

        longitude = np.asarray(
            segment["longitude"],
            dtype=np.float64,
        )

        segment_elevation = np.asarray(
            segment["elevation"],
            dtype=np.float64,
        )

        field = np.asarray(
            segment["field"],
            dtype=np.float64,
        )

        if segment_index > 0:

            previous_segment = segments[
                segment_index - 1
            ]

            previous_longitude = np.asarray(
                previous_segment["longitude"],
                dtype=np.float64,
            )

            previous_elevation = np.asarray(
                previous_segment["elevation"],
                dtype=np.float64,
            )

            previous_field = np.asarray(
                previous_segment["field"],
                dtype=np.float64,
            )

            previous_lon_end = float(
                previous_longitude[-1]
            )

            current_lon_start = float(
                longitude[0]
            )

            gap = (
                current_lon_start
                -
                previous_lon_end
            )

            previous_dlon = np.diff(
                previous_longitude
            )

            current_dlon = np.diff(
                longitude
            )

            previous_positive_dlon = previous_dlon[
                previous_dlon > 0.0
            ]

            current_positive_dlon = current_dlon[
                current_dlon > 0.0
            ]

            spacing_values = []

            if previous_positive_dlon.size > 0:

                spacing_values.append(
                    float(
                        np.median(
                            previous_positive_dlon
                        )
                    )
                )

            if current_positive_dlon.size > 0:

                spacing_values.append(
                    float(
                        np.median(
                            current_positive_dlon
                        )
                    )
                )

            if spacing_values:

                typical_spacing = float(
                    np.mean(
                        spacing_values
                    )
                )

            else:

                typical_spacing = np.nan

            should_stitch = (
                np.isfinite(gap)
                and
                np.isfinite(typical_spacing)
                and
                gap > 0.0
                and
                gap <= MAX_STITCH_GAP_FACTOR * typical_spacing
            )

            if should_stitch:

                bridge_longitude = np.array(
                    [
                        0.5
                        *
                        (
                            previous_lon_end
                            +
                            current_lon_start
                        )
                    ],
                    dtype=np.float64,
                )

                bridge_elevation = (
                    0.5
                    *
                    (
                        previous_elevation[:, -1:]
                        +
                        segment_elevation[:, :1]
                    )
                )

                bridge_field = (
                    0.5
                    *
                    (
                        previous_field[:, -1:]
                        +
                        field[:, :1]
                    )
                )

                longitude_parts.append(
                    bridge_longitude
                )

                elevation_parts.append(
                    bridge_elevation
                )

                field_parts.append(
                    bridge_field
                )

        longitude_parts.append(
            longitude
        )

        elevation_parts.append(
            segment_elevation
        )

        field_parts.append(
            field
        )

    longitude_global = np.concatenate(
        longitude_parts
    )

    elevation_global = np.concatenate(
        elevation_parts,
        axis=1,
    )

    field_global = np.concatenate(
        field_parts,
        axis=1,
    )

    order = np.argsort(
        longitude_global
    )

    longitude_global = longitude_global[
        order
    ]

    elevation_global = elevation_global[
        :,
        order
    ]

    field_global = field_global[
        :,
        order
    ]

    if longitude_global.size > 1:

        keep = np.ones(
            longitude_global.size,
            dtype=bool,
        )

        keep[1:] = (
            np.abs(
                np.diff(
                    longitude_global
                )
            )
            >
            1.0e-10
        )

        longitude_global = longitude_global[
            keep
        ]

        elevation_global = elevation_global[
            :,
            keep
        ]

        field_global = field_global[
            :,
            keep
        ]

    return (
        longitude_global,
        elevation_global,
        field_global,
    )


# ============================================================
# JOURNAL-STYLE PLOT
# ============================================================

def plot_theta_section(
    time_index,
    time_value,
    theta_prime,
    elevation,
    latitudes,
    longitudes,
    panels_per_face,
    output_directory,
):

    segments = collect_equatorial_segments(
        theta_prime=theta_prime,
        elevation=elevation,
        latitudes=latitudes,
        longitudes=longitudes,
        panels_per_face=panels_per_face,
    )

    if not segments:

        raise RuntimeError(
            f"No equatorial segments found "
            f"for time index {time_index}."
        )

    (
        longitude_global,
        elevation_global,
        field_global,
    ) = stitch_equatorial_segments(
        segments
    )

    # ========================================================
    # Filled contour levels
    #
    # Controlled ONLY by:
    #
    # COLOR_MIN
    # COLOR_MAX
    # NUMBER_OF_FILLED_LEVELS
    # ========================================================

    if NUMBER_OF_FILLED_LEVELS < 2:

        raise ValueError(
            "NUMBER_OF_FILLED_LEVELS must be at least 2."
        )

    filled_contour_levels = np.linspace(
        COLOR_MIN,
        COLOR_MAX,
        NUMBER_OF_FILLED_LEVELS,
    )

    # ========================================================
    # Contour line levels
    #
    # Completely independent from contourf.
    # ========================================================

    contour_line_levels = np.asarray(
        CONTOUR_LINE_LEVELS,
        dtype=np.float64,
    )

    if SHOW_CONTOUR_LINES:

        if contour_line_levels.size < 1:

            raise ValueError(
                "CONTOUR_LINE_LEVELS must contain "
                "at least one value."
            )

        if contour_line_levels.size > 1:

            if np.any(
                np.diff(
                    contour_line_levels
                )
                <= 0.0
            ):

                raise ValueError(
                    "CONTOUR_LINE_LEVELS must be "
                    "strictly increasing."
                )

    # ========================================================
    # Colorbar ticks and manual labels
    #
    # Position and label are stored together, so there cannot
    # be a tick/label length mismatch.
    # ========================================================

    colorbar_ticks = np.asarray(
        [
            item[0]
            for item in COLORBAR_TICKS_AND_LABELS
        ],
        dtype=np.float64,
    )

    colorbar_tick_labels = [
        item[1]
        for item in COLORBAR_TICKS_AND_LABELS
    ]

    # ========================================================
    # Journal plotting style
    # ========================================================

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
        }
    )

    fig, axis = plt.subplots(
        figsize=(
            7.6,
            3.45,
        )
    )

    longitude_mesh = np.broadcast_to(
        longitude_global[
            None,
            :
        ],
        elevation_global.shape,
    )

    # ========================================================
    # Filled contours
    #
    # ONLY filled_contour_levels are used here.
    # ========================================================

    contour_handle = axis.contourf(
        longitude_mesh,
        elevation_global,
        field_global,
        levels=filled_contour_levels,
        cmap=COLORMAP,
        extend="both",
    )

    # ========================================================
    # Optional contour lines
    #
    # ONLY CONTOUR_LINE_LEVELS are used here.
    # ========================================================

    if SHOW_CONTOUR_LINES:

        axis.contour(
            longitude_mesh,
            elevation_global,
            field_global,
            levels=contour_line_levels,
            colors="black",
            linewidths=CONTOUR_LINE_WIDTH,
        )

    # ========================================================
    # Surface
    # ========================================================

    axis.plot(
        longitude_global,
        elevation_global[
            0,
            :
        ],
        color="black",
        linewidth=0.8,
    )

    # ========================================================
    # Axes
    # ========================================================

    axis.set_xlabel(
        "Longitude",
        fontsize=17,
        labelpad=5,
    )

    axis.set_ylabel(
        "H (km)",
        fontsize=17,
        labelpad=5,
    )

    axis.set_xlim(
        0.0,
        360.0,
    )

    axis.set_xticks(
        [
            0,
            90,
            180,
            270,
            360,
        ]
    )

    # --------------------------------------------------------
    # Remove top and bottom whitespace
    # --------------------------------------------------------

    y_min = float(
        np.nanmin(
            elevation_global
        )
    )

    y_max = float(
        np.nanmax(
            elevation_global
        )
    )

    axis.set_ylim(
        y_min,
        y_max,
    )

    axis.margins(
        x=0.0,
        y=0.0,
    )

    axis.tick_params(
        axis="both",
        which="major",
        labelsize=14,
        direction="out",
        top=False,
        right=False,
    )

    # ========================================================
    # No title
    # ========================================================


    # ========================================================
    # Colorbar
    # ========================================================

    colorbar = fig.colorbar(
        contour_handle,
        ax=axis,
        ticks=colorbar_ticks,
        pad=0.025,
        fraction=0.045,
        aspect=25,
    )

    colorbar.set_label(
        COLORBAR_LABEL,
        fontsize=17,
        labelpad=8,
    )

    colorbar.ax.tick_params(
        labelsize=13,
        width=1.0,
        length=3.5,
    )

    colorbar.ax.set_yticklabels(
        colorbar_tick_labels
    )

    # ========================================================
    # Layout
    # ========================================================

    fig.tight_layout(
        pad=0.5
    )

    # ========================================================
    # Save
    # ========================================================

    output_file = (
        output_directory
        /
        f"theta_prime_t"
        f"{int(round(time_value)):05d}s.png"
    )

    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {output_file}\n"
        f"  time = {time_value:g} s\n"
        f"  minimum = "
        f"{np.nanmin(field_global):.6e} K\n"
        f"  maximum = "
        f"{np.nanmax(field_global):.6e} K"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    input_file = Path(
        INPUT_FILE
    )

    output_directory = Path(
        OUTPUT_DIR
    )

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input NetCDF file not found:\n"
            f"{input_file}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load
    # ========================================================

    (
        theta,
        elevation,
        latitudes,
        longitudes,
        times,
    ) = load_data(
        input_file
    )

    # ========================================================
    # Validate
    # ========================================================

    (
        number_of_times,
        number_of_panels,
        nk,
        nj,
        ni,
        panels_per_face,
    ) = validate_shapes(
        theta,
        elevation,
        latitudes,
        longitudes,
    )

    if times.size != number_of_times:

        raise ValueError(
            "Time coordinate mismatch.\n"
            f"theta times: {number_of_times}\n"
            f"time values: {times.size}"
        )

    # ========================================================
    # Compute analytic base state
    # ========================================================

    latitude_radians = np.deg2rad(
        latitudes
    )[
        :,
        None,
        :,
        :
    ]

    theta_base = analytic_base_theta(
        latitude_radians,
        elevation,
    )

    if theta_base.shape != elevation.shape:

        raise ValueError(
            "theta_base shape mismatch.\n"
            f"theta_base: {theta_base.shape}\n"
            f"elevation:  {elevation.shape}"
        )

    if SAVE_BASE_STATE:

        np.save(
            BASE_STATE_FILE,
            theta_base,
        )

        print(
            f"Saved computed base state: "
            f"{BASE_STATE_FILE}"
        )

    # ========================================================
    # Select requested physical times
    # ========================================================

    selected_time_indices = []

    for requested_time in PLOT_TIMES:

        time_index = int(
            np.argmin(
                np.abs(
                    times
                    -
                    requested_time
                )
            )
        )

        actual_time = float(
            times[
                time_index
            ]
        )

        if not np.isclose(
            actual_time,
            requested_time,
            rtol=0.0,
            atol=1.0e-8,
        ):

            raise ValueError(
                f"Requested time "
                f"{requested_time:g} s "
                f"does not exist in the file.\n"
                f"Nearest available time is "
                f"{actual_time:g} s."
            )

        selected_time_indices.append(
            time_index
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("Input summary")
    print("-------------")

    print(
        f"Input file:              "
        f"{input_file}"
    )

    print(
        f"Output directory:        "
        f"{output_directory}"
    )

    print(
        f"theta shape:             "
        f"{theta.shape}"
    )

    print(
        f"elevation shape:         "
        f"{elevation.shape}"
    )

    print(
        f"latitude shape:          "
        f"{latitudes.shape}"
    )

    print(
        f"longitude shape:         "
        f"{longitudes.shape}"
    )

    print(
        f"total panels/subpanels:  "
        f"{number_of_panels}"
    )

    print(
        f"panels per cube face:    "
        f"{panels_per_face}"
    )

    print(
        f"side-face panels tested: "
        f"{4 * panels_per_face}"
    )

    print(
        f"equator side:            "
        f"{EQUATOR_SIDE}"
    )

    print(
        f"filled color range:      "
        f"[{COLOR_MIN}, {COLOR_MAX}]"
    )

    print(
        f"number filled levels:    "
        f"{NUMBER_OF_FILLED_LEVELS}"
    )

    print(
        f"contour-line levels:     "
        f"{CONTOUR_LINE_LEVELS}"
    )

    print(
        f"contour lines:           "
        f"{SHOW_CONTOUR_LINES}"
    )

    print(
        f"colorbar ticks/labels:   "
        f"{COLORBAR_TICKS_AND_LABELS}"
    )

    print(
        f"requested times:         "
        f"{PLOT_TIMES}"
    )

    print(
        f"selected time indices:   "
        f"{selected_time_indices}"
    )

    print()

    # ========================================================
    # Plot requested times
    # ========================================================

    for time_index in selected_time_indices:

        theta_prime = (
            theta[
                time_index
            ]
            -
            theta_base
        )

        plot_theta_section(
            time_index=time_index,
            time_value=float(
                times[
                    time_index
                ]
            ),
            theta_prime=theta_prime,
            elevation=elevation,
            latitudes=latitudes,
            longitudes=longitudes,
            panels_per_face=panels_per_face,
            output_directory=output_directory,
        )

    print()

    print(
        f"Finished. Saved "
        f"{len(selected_time_indices)} "
        f"figure(s) in "
        f"{output_directory}"
    )


if __name__ == "__main__":

    main()
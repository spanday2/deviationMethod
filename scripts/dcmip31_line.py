#!/usr/bin/env python3

"""
Compare DCMIP 3-1 potential-temperature perturbation along the equator
at the ACTUAL DG vertical solution point nearest a requested height.

No vertical interpolation is performed.

The same vertical index k is used at every longitude.
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

EXPERIMENTS = [
    {
        "label": "Rusanov (No WB)",
        "file": "/home/shp000/site8/U2_data/raid/ppp6/deviationMethod/results/dcmip_31_rusanov.nc",
        "linestyle": "-",
    },
    {
        "label": "AUSM+UP (No WB)",
        "file": "/home/shp000/site8/U2_data/raid/ppp6/deviationMethod/results/dcmip_31_ausmplusup.nc",
        "linestyle": "--",
    },
    {
        "label": "Rusanov (WB)",
        "file": "/home/shp000/site8/U2_data/raid/ppp6/deviationMethod/results/dcmip_31_rusanov_wb.nc",
        "linestyle": "-.",
    },
    {
        "label": "AUSM+UP (WB)",
        "file": "/home/shp000/site8/U2_data/raid/ppp6/deviationMethod/results/dcmip_31_ausmplusup_wb.nc",
        "linestyle": ":",
    },
]


# Time to plot
PLOT_TIME = 1800.0


# Requested height.
# The script does NOT interpolate vertically.
# It selects the actual DG solution point nearest this value.
TARGET_HEIGHT_KM = 5.0


# Output
OUTPUT_FILE = "dcmip31_theta_profile_nearest_5km_t1800s.png"

DPI = 300


# Which side of equator to use
EQUATOR_SIDE = "south"

MAX_EQUATOR_DISTANCE_DEG = 2.0

MAX_STITCH_GAP_FACTOR = 3.0


# ============================================================
# PLOT SETTINGS
# ============================================================

# Width / height = 2.666
FIGURE_WIDTH = 8.0
FIGURE_HEIGHT = FIGURE_WIDTH / 2.666

LINE_WIDTH = 2.0

X_MIN = 0.0
X_MAX = 360.0

X_TICKS = [
    0,
    90,
    180,
    270,
    360,
]

SHOW_ZERO_LINE = True


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

    with nc.Dataset(input_file, "r") as dataset:

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
            "EQUATOR_SIDE must be 'south' or 'north'."
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

    order = np.argsort(longitude)

    longitude = longitude[order]
    elevation = elevation[:, order]
    field = field[:, order]

    gaps = np.diff(longitude)

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
        np.argmax(gaps)
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
# SELECT ACTUAL DG VERTICAL SOLUTION POINT
# ============================================================

def select_actual_vertical_level(
    elevation_global,
    target_height_km,
):

    if elevation_global.ndim != 2:

        raise ValueError(
            "Expected elevation_global dimensions "
            "(vertical_level, longitude)."
        )

    level_heights = np.nanmedian(
        elevation_global,
        axis=1,
    )

    if not np.any(
        np.isfinite(level_heights)
    ):

        raise RuntimeError(
            "No finite vertical heights were found."
        )

    vertical_index = int(
        np.nanargmin(
            np.abs(
                level_heights
                -
                target_height_km
            )
        )
    )

    actual_heights = np.asarray(
        elevation_global[
            vertical_index,
            :
        ],
        dtype=np.float64,
    )

    valid_heights = actual_heights[
        np.isfinite(
            actual_heights
        )
    ]

    if valid_heights.size == 0:

        raise RuntimeError(
            f"No valid heights at vertical index "
            f"k={vertical_index}."
        )

    mean_height = float(
        np.mean(
            valid_heights
        )
    )

    median_height = float(
        np.median(
            valid_heights
        )
    )

    minimum_height = float(
        np.min(
            valid_heights
        )
    )

    maximum_height = float(
        np.max(
            valid_heights
        )
    )

    return (
        vertical_index,
        actual_heights,
        mean_height,
        median_height,
        minimum_height,
        maximum_height,
        level_heights,
    )


# ============================================================
# EXTRACT ONE EXPERIMENT
# ============================================================

def extract_profile(
    input_file,
):

    theta, elevation, latitudes, longitudes, times = load_data(
        input_file
    )

    if theta.ndim != 5:

        raise ValueError(
            f"Unexpected theta shape: "
            f"{theta.shape}"
        )

    (
        number_of_times,
        number_of_panels,
        nk,
        nj,
        ni,
    ) = theta.shape

    if number_of_panels % 6 != 0:

        raise ValueError(
            f"Panel count {number_of_panels} "
            "is not divisible by 6."
        )

    panels_per_face = (
        number_of_panels
        //
        6
    )

    time_index = int(
        np.argmin(
            np.abs(
                times
                -
                PLOT_TIME
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
        PLOT_TIME,
        rtol=0.0,
        atol=1.0e-8,
    ):

        raise ValueError(
            f"Requested time "
            f"{PLOT_TIME:g} s "
            f"does not exist in "
            f"{input_file}.\n"
            f"Nearest available time is "
            f"{actual_time:g} s."
        )

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

    theta_prime = (
        theta[
            time_index
        ]
        -
        theta_base
    )

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
            f"for {input_file}"
        )

    (
        longitude_global,
        elevation_global,
        field_global,
    ) = stitch_equatorial_segments(
        segments
    )

    (
        vertical_index,
        actual_heights,
        mean_height,
        median_height,
        minimum_height,
        maximum_height,
        level_heights,
    ) = select_actual_vertical_level(
        elevation_global,
        TARGET_HEIGHT_KM,
    )

    # ========================================================
    # Actual DG solution point.
    # NO vertical interpolation.
    # ========================================================

    profile = np.asarray(
        field_global[
            vertical_index,
            :
        ],
        dtype=np.float64,
    )

    return {
        "longitude": longitude_global,
        "profile": profile,
        "actual_time": actual_time,
        "vertical_index": vertical_index,
        "actual_heights": actual_heights,
        "mean_height": mean_height,
        "median_height": median_height,
        "minimum_height": minimum_height,
        "maximum_height": maximum_height,
        "level_heights": level_heights,
        "nk": nk,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Journal plotting style
    # ========================================================

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 14,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "lines.linewidth": LINE_WIDTH,
            "legend.frameon": False,
        }
    )

    fig, axis = plt.subplots(
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT,
        )
    )

    reference_vertical_index = None
    reference_mean_height = None

    print()
    print("============================================================")
    print("DCMIP 3-1 ACTUAL DG GRID-POINT PROFILE")
    print("============================================================")
    print(f"Requested height = {TARGET_HEIGHT_KM:.6f} km")
    print("No vertical interpolation is performed.")
    print()

    for experiment_index, experiment in enumerate(
        EXPERIMENTS
    ):

        input_file = Path(
            experiment["file"]
        )

        if not input_file.exists():

            raise FileNotFoundError(
                f"File not found:\n"
                f"{input_file}"
            )

        result = extract_profile(
            input_file
        )

        longitude = result[
            "longitude"
        ]

        profile = result[
            "profile"
        ]

        actual_time = result[
            "actual_time"
        ]

        vertical_index = result[
            "vertical_index"
        ]

        mean_height = result[
            "mean_height"
        ]

        median_height = result[
            "median_height"
        ]

        minimum_height = result[
            "minimum_height"
        ]

        maximum_height = result[
            "maximum_height"
        ]

        if experiment_index == 0:

            reference_vertical_index = vertical_index
            reference_mean_height = mean_height

            print("Actual vertical DG solution points:")
            print("-----------------------------------")

            for k, height in enumerate(
                result["level_heights"]
            ):

                print(
                    f"k = {k:3d} : "
                    f"{height:.12f} km"
                )

            print()
            print("Selected vertical point")
            print("-----------------------")
            print(f"k                    = {vertical_index}")
            print(f"requested height     = {TARGET_HEIGHT_KM:.12f} km")
            print(f"mean actual height   = {mean_height:.12f} km")
            print(f"median actual height = {median_height:.12f} km")
            print(f"minimum height       = {minimum_height:.12f} km")
            print(f"maximum height       = {maximum_height:.12f} km")
            print(
                f"difference from 5 km = "
                f"{mean_height - TARGET_HEIGHT_KM:+.12f} km"
            )
            print(
                f"difference            = "
                f"{1000.0 * (mean_height - TARGET_HEIGHT_KM):+.6f} m"
            )
            print()

        else:

            if vertical_index != reference_vertical_index:

                raise RuntimeError(
                    f"Experiment "
                    f"{experiment['label']} selected "
                    f"k={vertical_index}, but the first "
                    f"experiment selected "
                    f"k={reference_vertical_index}."
                )

        valid = (
            np.isfinite(
                longitude
            )
            &
            np.isfinite(
                profile
            )
        )

        axis.plot(
            longitude[
                valid
            ],
            profile[
                valid
            ],
            linestyle=experiment[
                "linestyle"
            ],
            linewidth=LINE_WIDTH,
            label=experiment[
                "label"
            ],
        )

        print(
            f"{experiment['label']}"
        )

        print(
            f"  time                 = "
            f"{actual_time:.6f} s"
        )

        print(
            f"  vertical index k     = "
            f"{vertical_index}"
        )

        print(
            f"  mean actual height   = "
            f"{mean_height:.12f} km"
        )

        print(
            f"  minimum height       = "
            f"{minimum_height:.12f} km"
        )

        print(
            f"  maximum height       = "
            f"{maximum_height:.12f} km"
        )

        print(
            f"  minimum DeltaTheta   = "
            f"{np.nanmin(profile):.12e} K"
        )

        print(
            f"  maximum DeltaTheta   = "
            f"{np.nanmax(profile):.12e} K"
        )

        print()

    # ========================================================
    # Axes
    # ========================================================

    axis.set_xlabel(
        "Longitude",
        fontsize=18,
        labelpad=5,
    )

    axis.set_ylabel(
        r"$\Delta\theta$ (K)",
        fontsize=18,
        labelpad=6,
    )

    axis.set_xlim(
        X_MIN,
        X_MAX,
    )

    axis.set_xticks(
        X_TICKS
    )

    axis.margins(
        x=0.0
    )

    axis.tick_params(
        axis="both",
        which="major",
        direction="out",
        top=False,
        right=False,
        labelsize=15,
        width=1.2,
        length=5.0,
    )

    # ========================================================
    # Zero line
    # ========================================================

    if SHOW_ZERO_LINE:

        axis.axhline(
            0.0,
            linewidth=0.8,
            color="black",
            zorder=0,
        )

    # ========================================================
    # Legend
    # ========================================================

    axis.legend(
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            1.02,
        ),
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=2.5,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    # ========================================================
    # Spines
    # ========================================================

    for spine in axis.spines.values():

        spine.set_linewidth(
            1.2
        )

    # ========================================================
    # Layout
    # ========================================================

    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.20,
        top=0.80,
    )

    # ========================================================
    # Save
    # ========================================================

    fig.savefig(
        OUTPUT_FILE,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print("============================================================")
    print(f"Selected k            = {reference_vertical_index}")
    print(f"Selected mean height  = {reference_mean_height:.12f} km")
    print(f"Selected mean height  = {1000.0 * reference_mean_height:.6f} m")
    print(f"Saved                 = {OUTPUT_FILE}")
    print("============================================================")


if __name__ == "__main__":

    main()
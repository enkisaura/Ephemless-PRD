from typing import Literal
import pandas as pd
import numpy as np
from tqdm import tqdm

import bits

from parse import SENSOR_COL_LIST

DEFAULT_GNSS_TABLE = {
    "gal": 29.6e6,
    "gps": 26.56e6,
    "glo": 25.508e6,
    "bei": 27.878e6
}


def get_g_rover(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes the rovers geometry matrix alongside their variance.

    Model
    -----
    - db_x = cos(heading) * displacement
    - db_y = sin(heading) * displacement

    - var_db_x = (sin(heading) * displacement)**2 * Var(heading) + cos(heading)**2 * Var(displacement)
    - var_db_y = (cos(heading) * displacement)**2 * Var(heading) + sin(heading)**2 * Var(displacement)

    :param df: Dataframe containing sensor data
    :return: Dataframe containing g_rover and dataframe containing its corresponding variances
    """
    ## Compute G_rover
    # Get first sensor measurement at each epoch
    sensor_df = df.groupby("time")[["time", "dt"] + SENSOR_COL_LIST].first()

    # Compute 2D displacement from heading + displacement
    g_rover_df = pd.DataFrame()
    g_rover_df["db_x1"] = np.cos(sensor_df["heading_rad_rx1"]) * sensor_df["displacement_m_rx1"]
    g_rover_df["db_y1"] = np.sin(sensor_df["heading_rad_rx1"]) * sensor_df["displacement_m_rx1"]
    g_rover_df["db_x2"] = np.cos(sensor_df["heading_rad_rx2"]) * sensor_df["displacement_m_rx2"]
    g_rover_df["db_y2"] = np.sin(sensor_df["heading_rad_rx2"]) * sensor_df["displacement_m_rx2"]

    ## Compute variance of G_rover
    # Compute tachometer variance
    sensor_df["var_displacement_m_rx1"] = (
            sensor_df["displacement_quant_m_rx1"]**2
            + sensor_df["displacement_scale_rx1"]**2 * sensor_df["displacement_m_rx1"]**2)
    sensor_df["var_displacement_m_rx2"] = (
            sensor_df["displacement_quant_m_rx2"]**2
            + sensor_df["displacement_scale_rx2"]**2 * sensor_df["displacement_m_rx2"]**2)

    # Compute IMU variance
    sensor_df["var_heading_rad_rx1"] = (
            sensor_df["heading_quant_rad_rx1"]**2
            + sensor_df["heading_arw_rx1"]**2 * sensor_df["dt"]
            + sensor_df["heading_gbi_rx1"]**2 * sensor_df["dt"]**2)
    sensor_df["var_heading_rad_rx2"] = (
            sensor_df["heading_quant_rad_rx2"]**2
            + sensor_df["heading_arw_rx2"]**2 * sensor_df["dt"]
            + sensor_df["heading_gbi_rx2"]**2 * sensor_df["dt"]**2)

    var_g_rover_df = pd.DataFrame()
    var_g_rover_df["var_db_x1"] = (g_rover_df["db_y1"]**2 * sensor_df["var_heading_rad_rx1"]
                               + np.cos(sensor_df["heading_rad_rx1"]) ** 2 * sensor_df["var_displacement_m_rx1"])
    var_g_rover_df["var_db_y1"] = (g_rover_df["db_x1"]**2 * sensor_df["var_heading_rad_rx1"]
                               + np.sin(sensor_df["heading_rad_rx1"]) ** 2 * sensor_df["var_displacement_m_rx1"])
    var_g_rover_df["var_db_x2"] = (g_rover_df["db_y2"]**2 * sensor_df["var_heading_rad_rx2"]
                               + np.cos(sensor_df["heading_rad_rx2"]) ** 2 * sensor_df["var_displacement_m_rx2"])
    var_g_rover_df["var_db_y2"] = (g_rover_df["db_x2"]**2 * sensor_df["var_heading_rad_rx2"]
                               + np.sin(sensor_df["heading_rad_rx2"]) ** 2 * sensor_df["var_displacement_m_rx2"])

    return g_rover_df, var_g_rover_df

def get_between_epoch_prd(df: pd.DataFrame, meas_col: str, sigma_meas_col:str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes between epochs pseudorange differences alongside its variance.

    :param df: Dataframe with GNSS observations
    :param meas_col: Observation to difference
    :param sigma_meas_col: Standard deviation of the observation
    :return: between epoch differences, variance of between epoch differences. Dataframe have epochs as lines, SV ID as
    column.
    """
    # Compute between epoch prd
    prd_pivot = df.pivot_table(index="time", columns="sv_id", values=meas_col, aggfunc="first")
    prd_pivot = prd_pivot.diff()

    # Compute variance of between epoch prd
    var_prd_pivot = df.pivot_table(index="time", columns="sv_id", values=sigma_meas_col, aggfunc="first")
    var_prd_pivot = var_prd_pivot**2
    var_prd_pivot = var_prd_pivot.rolling(2).sum()

    return prd_pivot, var_prd_pivot


def compute(df: pd.DataFrame, dt: float, min_sv: int = 4, on_cp: bool = True, sv_model: bool = True,
            gnss_table: dict = DEFAULT_GNSS_TABLE) -> pd.DataFrame:
    ## Clean up
    df = df.sort_values(by=["time", "gnss_id_rx1"]).reset_index(drop=True)

    ## Compute G_rover
    g_rover_df, var_g_rover_df = get_g_rover(df)

    ## Compute between epoch differences
    if on_cp:
        meas_col = "sd_cp"
    else:
        meas_col = "sd"

    prd_df, var_prd_df = get_between_epoch_prd(df, meas_col=meas_col, var_meas_col="sigma_prd")

    ## Compute geometry matrix for each windows of dt seconds
    all_times = prd_df.index.values
    n_times_total = len(all_times)
    delta = np.timedelta64(dt, "s")
    end_index = 0
    for start_index in tqdm(range(n_times_total), desc="Computing geometry matrix", total=n_times_total - dt):
        t_start = all_times[start_index]
        t_end = t_start + delta

        # Find end of window of size dt seconds
        while end_index < n_times_total and all_times[end_index] < t_end:
            end_index += 1
        if end_index >= n_times_total:
            break

        # Extract window
        window_times = all_times[start_index:end_index]
        if len(window_times) == 0:
            continue

        # Extract GNSS observables
        at_epoch_prd_df = prd_df.iloc[start_index:end_index]
        at_epoch_var_prd_df = var_prd_df.iloc[start_index:end_index]

        # Extract G rover
        at_epoch_g_rover_np = g_rover_df.iloc[start_index:end_index].to_numpy()
        at_epoch_var_g_rover_np = var_g_rover_df.iloc[start_index:end_index].to_numpy()

        # Find SV that are valid along the entirety of the window
        mask_complete = at_epoch_prd_df.notna().all(axis=0)
        sv_list = at_epoch_prd_df.columns[mask_complete].tolist()
        constellation_table = []
        for sv_id in sv_list:
            constellation_table.append(sv_id[:3])

        constellation_table = np.array(constellation_table)

        # Check if window is solvable
        if len(sv_list) <= min_sv or len(sv_list) < at_epoch_g_rover_np.shape[1]:
            continue

        # Extract PRD matrix with only valid SV
        at_epoch_prd_np = at_epoch_prd_df[sv_list].to_numpy()
        at_epoch_var_prd_np = at_epoch_var_prd_df[sv_list].to_numpy()

        # Extract timestamps in elapsed time format
        elapsed_time_np = bits.convert.time.utc_to_seconds(window_times) / np.timedelta64(1, "s")

        # Compute the window
        result = compute_window(prd=at_epoch_prd_np, var_prd=at_epoch_var_prd_np,
                                g_rover=at_epoch_g_rover_np, var_g_rover=at_epoch_var_g_rover_np,
                                elapsed_time=elapsed_time_np,
                                constellation_table=constellation_table, gnss_table=gnss_table, sv_model=sv_model)



def compute_window(prd: np.ndarray, var_prd: np.ndarray, g_rover: np.ndarray, var_g_rover: np.ndarray,
                   elapsed_time: np.ndarray, constellation_table:np.ndarray|None=None,
                   gnss_table: dict = DEFAULT_GNSS_TABLE, sv_model: bool = True) -> pd.DataFrame:
    pass


def old_compute_geometry_matrix(df: pd.DataFrame, dt: float, min_sv: int = 4, mode: Literal["sd", "dd"] = "dd",
                            on_cp: bool = True,
                            dynamic_sv: bool = True, semi_major_axis_table: dict = {"gal": 29.6e6,
                                                                                    "gps": 26.56e6,
                                                                                    "glo": 25.508e6,
                                                                                    "bei": 27.878e6}) -> pd.DataFrame:
    # Clean up
    df = df.sort_values("time").reset_index(drop=True)
    delta = np.timedelta64(dt, "s")

    # Initialize results columns
    df["gamma_rad"] = np.nan
    df["e_x"] = np.nan
    df["e_y"] = np.nan

    # Extract measurements
    if dynamic_sv:
        if mode == "dd":
            df = df[df["gnss_id1"] == df["gnss_id2"]]
            gnss_id_column = "gnss_id1"
        else:
            gnss_id_column = "gnss_id_rx1"
        df = df.sort_values(by=["time", gnss_id_column]).reset_index(drop=True)

    if on_cp:
        meas_col = mode + "_cp"
    else:
        meas_col = mode

    pivot_full = df.pivot_table(index="time", columns="sv_id", values=meas_col, aggfunc="first")
    sigma_pivot_full = df.pivot_table(index="time", columns="sv_id", values="sigma_prd", aggfunc="first")
    sensors_full = df.groupby("time")[["displacement_m_rx1", "heading_rad_rx1",
                                       "displacement_m_rx2", "heading_rad_rx2"]].first()
    sigma_sensors_full = df.groupby("time")[["sigma_displacement_m_rx1", "sigma_heading_rad_rx1",
                                             "sigma_displacement_m_rx2", "sigma_heading_rad_rx2"]].first()

    # Compute between-epochs differences
    pivot_full = pivot_full.diff()
    sigma_sensors_full = np.sqrt(2) * sigma_sensors_full  # If prd independent wich is not true...

    # Compute 2D displacement from heading + displacement
    """
    d_x = cos(heading) * d
    d_y = sin(heading) * d

    var_dx = d² * sin(heading)² * var_heading + cos(heading)² * var_d
    var_dy = d² * cos(heading)² * var_heading + sin(heading)² * var_d
    """
    db_full = pd.DataFrame()
    db_full["db_x1"] = np.cos(sensors_full["heading_rad_rx1"]) * sensors_full["displacement_m_rx1"]
    db_full["db_y1"] = np.sin(sensors_full["heading_rad_rx1"]) * sensors_full["displacement_m_rx1"]
    db_full["db_x2"] = np.cos(sensors_full["heading_rad_rx2"]) * sensors_full["displacement_m_rx2"]
    db_full["db_y2"] = np.sin(sensors_full["heading_rad_rx2"]) * sensors_full["displacement_m_rx2"]

    var_db_full = pd.DataFrame()
    var_db_full["db_x1"] = (sensors_full["displacement_m_rx1"] ** 2 * np.sin(sensors_full["heading_rad_rx1"]) ** 2 *
                            sigma_sensors_full["sigma_heading_rad_rx1"] ** 2
                            + np.cos(sensors_full["heading_rad_rx1"]) ** 2 * sigma_sensors_full[
                                "sigma_displacement_m_rx1"] ** 2)
    var_db_full["db_y1"] = (sensors_full["displacement_m_rx1"] ** 2 * np.cos(sensors_full["heading_rad_rx1"]) ** 2 *
                            sigma_sensors_full["sigma_heading_rad_rx1"] ** 2
                            + np.sin(sensors_full["heading_rad_rx1"]) ** 2 * sigma_sensors_full[
                                "sigma_displacement_m_rx1"] ** 2)
    var_db_full["db_x2"] = (sensors_full["displacement_m_rx2"] ** 2 * np.sin(sensors_full["heading_rad_rx2"]) ** 2 *
                            sigma_sensors_full["sigma_heading_rad_rx2"] ** 2
                            + np.cos(sensors_full["heading_rad_rx2"]) ** 2 * sigma_sensors_full[
                                "sigma_displacement_m_rx2"] ** 2)
    var_db_full["db_y2"] = (sensors_full["displacement_m_rx2"] ** 2 * np.cos(sensors_full["heading_rad_rx2"]) ** 2 *
                            sigma_sensors_full["sigma_heading_rad_rx2"] ** 2
                            + np.sin(sensors_full["heading_rad_rx2"]) ** 2 * sigma_sensors_full[
                                "sigma_displacement_m_rx2"] ** 2)

    # Geat rid of first timestamp
    pivot_full = pivot_full.iloc[1:]
    db_full = db_full.iloc[1:]
    sigma_pivot_full = sigma_pivot_full.iloc[1:]
    var_db_full = var_db_full.iloc[1:]

    # Compute geometry matrix for each windows of dt seconds
    all_times = pivot_full.index.values
    n_times_total = len(all_times)
    end_index = 0
    for start_index in tqdm(range(n_times_total), desc="Computing geometry matrix", total=n_times_total - dt):
        t_start = all_times[start_index]
        t_end = t_start + delta

        # Find end of window of size dt seconds
        while end_index < n_times_total and all_times[end_index] < t_end:
            end_index += 1
        if end_index >= n_times_total:
            break

        # Extract window
        window_times = all_times[start_index:end_index]
        if len(window_times) == 0:
            continue

        prd_pivot = pivot_full.iloc[start_index:end_index]
        sigma_prd_pivot = sigma_pivot_full.iloc[start_index:end_index]

        G_rover = db_full.iloc[start_index:end_index].to_numpy()
        var_g = var_db_full.iloc[start_index:end_index].to_numpy()

        # Find SV that are valid along the entirety of the window
        mask_complete = prd_pivot.notna().all(axis=0)
        sv_list = prd_pivot.columns[mask_complete].tolist()
        constellation_table = []
        for sv_id in sv_list:
            constellation_table.append(sv_id[:3])

        constellation_table = np.array(constellation_table)

        # Check if window is solvable
        if len(sv_list) <= min_sv or len(sv_list) < G_rover.shape[1]:
            continue

        # Extract PRD matrix with only valid SV
        PRD = prd_pivot[sv_list].to_numpy()
        var_y = (sigma_prd_pivot[sv_list].to_numpy()) ** 2

        # Extract timestamps
        dt_np = bits.convert.time.utc_to_seconds(window_times) / np.timedelta64(1, "s")

        # Compute geometry matrix
        result = window_geometry_matrix(PRD, var_y, G_rover, var_g, constellation_table, dt_np, semi_major_axis_table,
                                        dynamic_sv=dynamic_sv, mode=mode)

        if result is None:
            txt = f"Could not compute geometry matrix between {all_times[start_index]} and {all_times[end_index]}"
            warn(txt)
            continue

        # Save results
        G, Q_G, gamma = result
        time_mask = df["time"] == window_times[-1]

        # Map used lines in the window dataframe
        window_map = time_mask & (df["sv_id"].isin(sv_list))

        # Map SV line index in the window dataframe with corresponding id from sv_list
        sv_to_col = {sv_name: i for i, sv_name in enumerate(sv_list)}  # map sv_name with sv_list id
        col_idx = df.loc[time_mask, "sv_id"].map(sv_to_col)  # Map dataframe lines with corresponding id from sv_list
        valid = col_idx.notna()  # Find valid SV
        sv_map_idx = col_idx[valid].astype(int)  # Keep only valid SV

        # Gamma
        df.loc[window_map, "gamma_rad"] = gamma

        # G
        df.loc[window_map, "e_x"] = G[0, sv_map_idx.to_numpy()]
        df.loc[window_map, "e_y"] = G[1, sv_map_idx.to_numpy()]

        # Q_G
        df.loc[window_map, "sigma_e_xx"] = Q_G[sv_map_idx.to_numpy(), 0, 0]
        df.loc[window_map, "sigma_e_yy"] = Q_G[sv_map_idx.to_numpy(), 1, 1]
        df.loc[window_map, "sigma_e_xy"] = Q_G[sv_map_idx.to_numpy(), 0, 1]

    return df
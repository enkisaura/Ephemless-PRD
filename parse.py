import os
import pandas as pd
import numpy as np

import bits
import bits_prd

SENSOR_COL_LIST = ["displacement_m_rx1", "displacement_m_rx2",
                   "displacement_quant_m_rx1", "displacement_quant_m_rx2",
                   "displacement_scale_rx1", "displacement_scale_rx2",
                   "heading_rad_rx1", "heading_rad_rx2",
                   "heading_quant_rad_rx1", "heading_quant_rad_rx2",
                   "heading_arw_rx1", "heading_arw_rx2",
                   "heading_gbi_rx1", "heading_gbi_rx2"]

def obs(rx1_folder_path:str, rx2_folder_path:str, obs_file_name:str="obs.rnx", sigma_prd:float=0.7) \
        -> pd.DataFrame:
    """
    Computes pseudorange differences from two rovers.

    Parse Rinex files, computes pseudorange and carrier phase single differences

    :param rx1_folder_path: Rover 1 folder path
    :param rx2_folder_path: Rover 2 folder path
    :param obs_file_name: Rinex file name
    :param sigma_prd: Standard deviation of SD (m)
    :return: parsed PRD
    """
    # 1. Parse Rinex
    rx1_filepath = os.path.join(rx1_folder_path, obs_file_name)
    rx2_filepath = os.path.join(rx2_folder_path, obs_file_name)

    rx1_obs_pd = bits.utils.fast_parse(rx1_filepath, bits.parse.raw.rinex)
    rx2_obs_pd = bits.utils.fast_parse(rx2_filepath, bits.parse.raw.rinex)

    # 2. Compute PRD
    prd_df = bits_prd.code_prd.get_prd(rx1_obs_pd, rx2_obs_pd, time_between_meas=1, compute_dd=False,
                                       meas_col_name="pr_m")
    cp_df = bits_prd.code_prd.get_prd(rx1_obs_pd, rx2_obs_pd, time_between_meas=1, compute_dd=False,
                                      meas_col_name="cp_m")

    # 3. Merge pr and cp
    prd_df = pd.merge_asof(prd_df, cp_df[["time", "sv_id", "sd"]], on="time", by=["sv_id"], suffixes=("", "_cp"))

    prd_df["sigma_prd"] = sigma_prd

    # Clean
    prd_df = prd_df.dropna(subset=["sd", "sd_cp"])

    return prd_df


def sensors_from_rmc(rx1_folder_path:str, rx2_folder_path:str, rmc_file_name:str="nmea_simulator.txt", dt=-18,
                   sigma_heading:float=1e-3, sigma_displacement:float=1e-2) -> pd.DataFrame:
    """
    Parse tachometers/imu from NMEA-RMC files

    :param rx1_folder_path: Rover 1 folder path
    :param rx2_folder_path: Rover 2 folder path
    :param rmc_file_name: Rmc file
    :param dt: Time to UTC (-18 for GPST)
    :param sigma_heading: Standard deviation of heading ()
    :param sigma_displacement: Standard deviation of tachometer ()
    :return: Parsed dataframe {
    "time": UTC time (np.datetime); "dt": Elapsed time since last epoch (seconds);
    "displacement_m_rx1"/"displacement_m_rx2": Displacement since last epoch (m);
    "displacement_quant_m_rx1"/"displacement_quant_m_rx2": std of tachometers' quantization noise (m);
    "displacement_scale_rx1"/"displacement_scale_rx2": std of tachometers' scale factor noise (m/m);
    "heading_rad_rx1"/"heading_rad_rx2": heading (rad);
    "heading_quant_rad_rx1"/"heading_quant_rad_rx2": std of imu's quantization noise (rad);
    "heading_arw_rx1"/"heading_arw_rx2": std of imu's Angle Random Walk (ARW) (rad/sqrt(s));
    "heading_gbi_rx1"/"heading_gbi_rx2": std of imu's Gyroscope Bias Instability (GBI) (rad/s);
    }
    """
    ## Parse NMEA-RMC
    rx1_filepath = os.path.join(rx1_folder_path, rmc_file_name)
    rx2_filepath = os.path.join(rx2_folder_path, rmc_file_name)

    rx1_sensors_pd = bits.utils.fast_parse(rx1_filepath, bits.parse.pvt.rmc)
    rx2_sensors_pd = bits.utils.fast_parse(rx2_filepath, bits.parse.pvt.rmc)

    ## Clean
    # For some reason, skydel provides time in GPST
    dt = np.timedelta64(dt, "s")
    rx1_sensors_pd["time"] += dt
    rx2_sensors_pd["time"] += dt

    # For some reason, skydel already provides track in radians
    rx1_sensors_pd["cog_rad"] = np.rad2deg(rx1_sensors_pd["cog_rad"])
    rx2_sensors_pd["cog_rad"] = np.rad2deg(rx2_sensors_pd["cog_rad"])

    # Sort
    rx1_sensors_pd = rx1_sensors_pd.sort_values("time").reset_index(drop=True)
    rx2_sensors_pd = rx2_sensors_pd.sort_values("time").reset_index(drop=True)

    ## Merge
    # RX1
    sensor_df = rx1_sensors_pd[["time", "speed_mps", "cog_rad"]]
    sensor_df = sensor_df.rename(columns={'speed_mps': 'speed_mps_rx1', 'cog_rad': 'heading_rad_rx1'})
    sensor_df["heading_quant_rad_rx1"] = sigma_heading
    sensor_df["heading_arw_rx1"] = 0
    sensor_df["heading_gbi_rx1"] = 0
    sensor_df["displacement_quant_m_rx1"] = sigma_displacement
    sensor_df["displacement_scale_rx1"] = 0

    # RX2
    rx2_sensors_pd = rx2_sensors_pd.rename(columns={'cog_rad': 'heading_rad_rx2', 'speed_mps': 'speed_mps_rx2'})
    sensor_df = merge(sensor_df, rx2_sensors_pd, on_cols=["heading_rad_rx2", "speed_mps_rx2"])
    sensor_df["heading_quant_rad_rx2"] = sigma_heading
    sensor_df["heading_arw_rx2"] = 0
    sensor_df["heading_gbi_rx2"] = 0
    sensor_df["displacement_quant_m_rx2"] = sigma_displacement
    sensor_df["displacement_scale_rx2"] = 0

    # Compute displacement
    sensor_df["displacement_m_rx1"] = sensor_df["dt"] * sensor_df["speed_mps_rx1"]
    sensor_df["displacement_m_rx2"] = sensor_df["dt"] * sensor_df["speed_mps_rx2"]

    return sensor_df


def sensors_from_md26():
    pass


def merge(main_df: pd.DataFrame, on_df: pd.DataFrame, on_cols: list=SENSOR_COL_LIST) -> pd.DataFrame:
    """
    Merges on_df into main_df by extrapolating columns from on_cols by "time". Adds "dt" column, ellapsed time since
    last epoch.

    :param main_df: Dataframe to be merged on
    :param on_df: Dataframe to merge on main_df
    :param on_cols: Columns from on_df to merge
    :return: Merged dataframe
    """
    assert "time" in main_df.columns, "No time column in dataframe"
    assert "time" in on_df.columns, "No time column in dataframe"

    for col in on_cols:
        assert col in on_df.columns, f"No {col} column in dataframe"

    # Clean
    main_df = main_df.sort_values("time").reset_index(drop=True)
    on_df = on_df.sort_values("time").reset_index(drop=True)

    # Merge
    for col in on_cols:
        main_df[col] = np.interp(main_df["time"], on_df["time"], on_df[col], left=np.nan, right=np.nan)

    # Clean
    main_df = main_df.dropna(subset=on_cols)

    # Add elapsed time since last epoch
    main_df["dt"] = get_dt(main_df)

    return main_df


def get_dt(df: pd.DataFrame) -> np.ndarray:
    assert "time" in df.columns, "No time column in dataframe"

    group_id = (df["time"] != df["time"].shift()).cumsum() # Group by epoch
    group_time = df.groupby(group_id)["time"].transform("first") # Epoch's time
    unique_group_times = df.groupby(group_id)["time"].first() # Time list (one per epoch)
    prev_group_time = unique_group_times.shift().reindex(group_id).values # Preceding epoch
    return (group_time.values - prev_group_time) / np.timedelta64(1, "s")


def sample(df:pd.DataFrame, sample_size:int, sample_count:int, dump:int=0) -> pd.DataFrame:
    """
    Samples "sample_count" of "sample_size" number of epochs from "df". Dumps the "dump" first epochs.

    :param df: Dataframe to be sampled
    :param sample_size: number of epochs in one sample
    :param sample_count: number of samples to keep
    :param dump: number of epochs to dump at the beginning of df
    :return: Sampled dataframe
    """
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True) # Clean

    # Get all unique epochs from df
    unique_time_list = df["time"].unique().tolist()

    # Computes the minimum required number of epochs
    required_lenght = dump + sample_size + sample_count

    if len(unique_time_list) < required_lenght:
        txt = f"Not enough timestamps, requires {required_lenght}, got {len(df)}"
        raise Exception(txt)

    # Find epochs to keep from df
    timestamp_to_keep = unique_time_list[dump:required_lenght]

    # Sample df
    df = df[df["time"].isin(timestamp_to_keep)]

    return df

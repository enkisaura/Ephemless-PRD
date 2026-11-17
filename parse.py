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
                   "sigma_arw_rx1", "sigma_arw_rx2",
                   "sigma_gbi_rx1", "sigma_gbi_rx2"]

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
    "sigma_arw_rx1"/"sigma_arw_rx2": std of imu's Angle Random Walk (ARW) (rad/sqrt(s));
    "sigma_gbi_rx1"/"sigma_gbi_rx2": std of imu's Gyroscope Bias Instability (GBI) (rad/s);
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
    sensor_df["sigma_arw_rx1"] = 0
    sensor_df["sigma_gbi_rx1"] = 0
    sensor_df["displacement_quant_m_rx1"] = sigma_displacement
    sensor_df["displacement_scale_rx1"] = 0

    # RX2
    rx2_sensors_pd = rx2_sensors_pd.rename(columns={'cog_rad': 'heading_rad_rx2', 'speed_mps': 'speed_mps_rx2'})
    sensor_df = merge(sensor_df, rx2_sensors_pd, on_cols=["heading_rad_rx2", "speed_mps_rx2"])
    sensor_df["heading_quant_rad_rx2"] = sigma_heading
    sensor_df["sigma_arw_rx2"] = 0
    sensor_df["sigma_gbi_rx2"] = 0
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



def old_get_sensors_sim(rx1_folder_path:str, rx2_folder_path:str, prd_df:pd.DataFrame,
                    gt_file_name:str="nmea_simulator.txt", dt=0,
                    sigma_heading:float=1e-3, sigma_displacement:float=1e-2, sigma_prd:float=0.7):
    sensors_df = prd_df.copy()
    sensors_df["sigma_prd"] = sigma_prd
    rx1_filepath = os.path.join(rx1_folder_path, gt_file_name)
    rx2_filepath = os.path.join(rx2_folder_path, gt_file_name)

    rx1_sensors_pd = bits.utils.fast_parse(rx1_filepath, bits.parse.pvt.rmc)
    rx2_sensors_pd = bits.utils.fast_parse(rx2_filepath, bits.parse.pvt.rmc)

    # Clean
    rx1_sensors_pd = rx1_sensors_pd.sort_values("time").reset_index(drop=True)
    rx2_sensors_pd = rx2_sensors_pd.sort_values("time").reset_index(drop=True)
    sensors_df = sensors_df.sort_values("time").reset_index(drop=True)
    # For some reason, skydel provides time in GPST
    dt = np.timedelta64(dt, "s")
    rx1_sensors_pd["time"] += dt
    rx2_sensors_pd["time"] += dt
    # For some reason, skydel already provides track in radians
    rx1_sensors_pd["cog_rad"] = np.rad2deg(rx1_sensors_pd["cog_rad"])
    rx2_sensors_pd["cog_rad"] = np.rad2deg(rx2_sensors_pd["cog_rad"])

    sensors_df["heading_rad_rx1"] = np.interp(sensors_df["time"], rx1_sensors_pd["time"], rx1_sensors_pd["cog_rad"])
    sensors_df["heading_rad_rx2"] = np.interp(sensors_df["time"], rx2_sensors_pd["time"], rx2_sensors_pd["cog_rad"])
    sensors_df["sigma_heading_rad_rx1"] = sigma_heading
    sensors_df["sigma_heading_rad_rx2"] = sigma_heading

    # Add angle bias on rx2
    if False:
        angle_bias = np.random.uniform(0, 2 * np.pi)
        print(f"Angle bias: {angle_bias}rad")
        sensors_df["heading_rad_rx2"] = sensors_df["heading_rad_rx2"] + angle_bias

        # rewrap dans [-pi, pi]
        sensors_df["heading_rad_rx2"] = np.mod(sensors_df["heading_rad_rx2"] + np.pi, 2 * np.pi) - np.pi

    sensors_df["speed_mps_rx1"] = np.interp(sensors_df["time"], rx1_sensors_pd["time"], rx1_sensors_pd["speed_mps"])
    sensors_df["speed_mps_rx2"] = np.interp(sensors_df["time"], rx2_sensors_pd["time"], rx2_sensors_pd["speed_mps"])

    group_id = (sensors_df["time"] != sensors_df["time"].shift()).cumsum()

    # Temps de début de chaque groupe
    group_time = sensors_df.groupby(group_id)["time"].transform("first")

    # Temps du groupe précédent (unique par groupe, puis décalé)
    unique_group_times = sensors_df.groupby(group_id)["time"].first()
    prev_group_time = unique_group_times.shift().reindex(group_id).values

    sensors_df["dt"] = (group_time.values - prev_group_time) / np.timedelta64(1, "s")

    sensors_df["displacement_m_rx1"] = sensors_df["dt"] * sensors_df["speed_mps_rx1"]
    sensors_df["displacement_m_rx2"] = sensors_df["dt"] * sensors_df["speed_mps_rx2"]
    sensors_df["sigma_displacement_m_rx1"] = sigma_displacement
    sensors_df["sigma_displacement_m_rx2"] = sigma_displacement

    return sensors_df
from typing import Literal
import pandas as pd
import os

import parse
import geometry_matrix
import utils

def get_obs(rx1_folder_path, rx2_folder_path, obs_file_name:str="obs.rnx", sigma_prd:float=0.7,
            rmc_file_name:str="nmea_simulator.txt", dt:float=0.0,
            tacho_quant:float=1e-2, tacho_scale:float=0.0,
            heading_quant:float=1e-3, heading_arw:float=0.0, heading_bi:float=0.0,
            sensor_filetype:Literal["rmc", "md26"]="rmc"):
    """

    :param rx1_folder_path:
    :param rx2_folder_path:
    :param obs_file_name:
    :param sigma_prd:
    :param rmc_file_name:
    :param dt:
    :param tacho_quant:
    :param tacho_scale:
    :param heading_quant:
    :param heading_arw:
    :param heading_bi:
    :param sensor_filetype:
    :return:
    """
    # Parse GNSS obs and compute single differences
    prd_df = parse.obs(rx1_folder_path, rx2_folder_path, obs_file_name=obs_file_name, sigma_prd=sigma_prd)

    # Parse IMU and tachometer
    if sensor_filetype == "rmc":
        sensors_df = parse.sensors_from_rmc(rx1_folder_path, rx2_folder_path, rmc_file_name=rmc_file_name,
                                            dt=dt, sigma_heading=heading_quant, sigma_displacement=tacho_quant)
    elif sensor_filetype == "md26":
        sensors_df = parse.sensors_from_md26()
    else:
        raise ValueError("unknown sensor_filetype")

    return parse.merge(prd_df, sensors_df)


if __name__ == "__main__":
    main_folder = "/home/enki/Documents/ephemless_data"
    pickle_directory = "output"
    runs = ["3ms", "10ms", "30ms"]
    run1 = runs[0]
    run2 = runs[1]
    skydel_dt = -18
    sample_size = 20*60 # seconds

    sigma_prd = 0.7 # m
    sigma_tacho = 1e-2 # m
    sigma_heading = 1e-3 # rad

    rx1_folder_path = f"{main_folder}/{run1}"
    rx2_folder_path = f"{main_folder}/{run2}"

    # Create pickle directory
    if not os.path.exists(pickle_directory):
        os.makedirs(pickle_directory)

    # Parse
    obs_df = get_obs(rx1_folder_path, rx2_folder_path, sigma_prd=sigma_prd, dt=skydel_dt,
                     tacho_quant=sigma_tacho, heading_quant=sigma_heading, sensor_filetype="rmc")

    obs_df = parse.sample(obs_df, sample_size=sample_size, sample_count=5, dump=5 * 60)

    # Compute G
    if True:
        g_df, report = geometry_matrix.compute(obs_df, dt=sample_size, min_sv=4, on_cp=True, sv_model=True, g_norm_tolerance=400)
        g_df.to_csv(f"{pickle_directory}/g_df.csv")
        utils.save_dict_to_csv(report, f"{pickle_directory}/g_df_report.csv")
        # exit()
    else:
        g_df = pd.read_pickle(f"{pickle_directory}/g_df.pkl")

    print("All done !")
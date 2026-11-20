from typing import Literal
import pandas as pd
import os

import parse
import geometry_matrix
import utils
from main import get_obs


def server_compute(compute_param:dict):
    # Create output directory
    if not os.path.exists(compute_param["output_folder"]):
        os.makedirs(compute_param["output_folder"])

    for i in range(len(compute_param["sample_size"])):
        print(f"Iteration {i}")
        try:
            result_filename = f"sv_{compute_param["sv_model"][i]}_{compute_param["sample_size"][i]}_{compute_param["rovers_folder_name"][i]}.csv"
            rx1_folder_path = f"{compute_param["input_folder"]}/{compute_param["rovers_folder_name"][i][0]}"
            rx2_folder_path = f"{compute_param["input_folder"]}/{compute_param["rovers_folder_name"][i][1]}"

            # Parse
            obs_df = get_obs(rx1_folder_path, rx2_folder_path, sigma_prd=compute_param["sigma_prd"], dt=compute_param["skydel_dt"],
                             tacho_quant=compute_param["sigma_tacho"], heading_quant=compute_param["sigma_heading"], sensor_filetype="rmc")

            # Compute G
            g_df, report = geometry_matrix.compute(obs_df, dt=compute_param["sample_size"][i], min_sv=4, on_cp=True,
                                                   sv_model=compute_param["sv_model"][i], g_norm_tolerance=400)
            g_df.to_csv(f"{compute_param["output_folder"]}/{result_filename}")
            utils.save_dict_to_csv(report, f"{compute_param["output_folder"]}/report_{result_filename}")
        except Exception as e:
            txt = f"\n\n######################\nDid not work....\n{e}\n######################\n\n"""
            print(txt)


    print("All done !")



if __name__ == "__main__":
    compute_param = {
        # Filepaths
        # TODO Change ca |
        # TODO           v
        "input_folder": "/home/enki/Documents/ephemless_data", # TODO <- Juste là
        # TODO                                   ^
        # TODO Fait un effort... c'est juste ici |
        "output_folder": "output",
        # Main params
        "skydel_dt": -18,
        # Standard deviation
        "sigma_prd": 0.7,  # m
        "sigma_tacho": 1e-2,  # m
        "sigma_heading": 1e-3,  # rad
        # Iterables
        "rovers_folder_name": [("3ms", "10ms"), ("3ms", "10ms"), ("3ms", "10ms"), ("3ms", "10ms")],
        "sample_size": [10*60, 20*60, 30*60, 40*60], # seconds
        "sv_model": [True, True, True, True],
    }

    server_compute(compute_param)


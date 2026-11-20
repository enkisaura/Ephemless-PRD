import numpy as np
import pandas as pd
from tqdm import tqdm
from warnings import warn

import utils

def compute(df:pd.DataFrame) -> pd.DataFrame:
    # Clean up
    df = df.copy()
    df = df.dropna(subset=["e_x", "e_y"], how="all")
    df = df.sort_values("time").reset_index(drop=True)

    baseline_pd_list = []
    for _, group in tqdm(df.groupby("time"), total=len(df["time"].unique()), desc="Computing baseline"):
        try:
            baseline_pd = compute_window(group)
        except Exception as e:
            txt = f"Could not compute baseline: {e}"
            warn(txt)
            continue
        baseline_pd_list.append(baseline_pd)

    return pd.concat(baseline_pd_list, ignore_index=True)


def compute_window(group:pd.DataFrame) -> pd.DataFrame:
    # SD measurements
    prd = group["sd"].to_numpy().reshape(-1, 1)
    Q_prd = np.diag(group["sigma_prd"].to_numpy() ** 2)

    # Geometry matrix
    ex = group["e_x"].to_numpy()
    ey = group["e_y"].to_numpy()
    G = np.vstack((ex, ey, np.ones_like(ex))).transpose()

    var_ex = group["var_e_xx"].to_numpy()
    var_ey = group["var_e_yy"].to_numpy()
    Q_g = np.vstack((var_ex, var_ey, np.zeros_like(var_ex))).transpose()

    baseline_estimate, cov_baseline_estimate = utils.wtls(prd, Q_prd, G, Q_g)

    # Save result
    baseline_serie = pd.Series({"time": group["time"].iloc[0]})
    baseline_serie["bx_rx_m"] = float(baseline_estimate[0][0])
    baseline_serie["by_rx_m"] = float(baseline_estimate[1][0])
    baseline_serie["bb_rx_m"] = float(baseline_estimate[-1][0])
    baseline_serie["baseline_m"] = float(np.linalg.norm(baseline_estimate[:2]))

    baseline_serie["cov_bxx_rx_m"] = float(cov_baseline_estimate[0][0])
    baseline_serie["cov_byy_rx_m"] = float(cov_baseline_estimate[1][1])
    baseline_serie["cov_bbb_rx_m"] = float(cov_baseline_estimate[2][2])
    baseline_serie["cov_bxy_rx_m"] = float(cov_baseline_estimate[0][1])
    baseline_serie["cov_bxb_rx_m"] = float(cov_baseline_estimate[0][2])
    baseline_serie["cov_byb_rx_m"] = float(cov_baseline_estimate[1][2])

    return baseline_serie.to_frame().T


if __name__ == "__main__":
    main_folder = "/home/enki/Documents/ephemless_data"
    pickle_directory = "output"
    df_name = "sv_30_60_g_df.pkl"

    df = pd.read_pickle(f"{pickle_directory}/{df_name}")

    df = df.dropna(subset=["e_x", "e_y"], how="all")

    baseline = compute(df)

    print("All done !")
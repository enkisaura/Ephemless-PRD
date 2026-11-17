import os
import numpy as np
from scipy.linalg import cho_factor, cho_solve, block_diag
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
from warnings import warn
from tqdm import tqdm

import bits
import bits_prd

def ols(G:np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Y = G @ X
    X = (G.T @ G)-1 @ G.T @ Y
    Returns: X
    """
    return np.linalg.inv(G.transpose() @ G) @ G.transpose() @ Y

def wls(G:np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    GtW = G.transpose() @ W
    return np.linalg.inv(GtW @ G) @ GtW @ Y

def vec(M:np.ndarray) -> tuple[np.ndarray, float, float]:
    return M.flatten(order="F").reshape(-1, 1), M.shape[0], M.shape[1]

def reshape(V:np.ndarray, n:float, m:float) -> np.ndarray:
    if len(V.shape) == 2:
        if V.shape[1] > 1:
            txt = f"Input vector must be 1-dimensional, got {V.shape}"
            raise ValueError(txt)
    if V.shape[0] != n*m:
        txt = f"Input vector lenght must be {n*m}, got {V.shape[0]}"
        raise ValueError(txt)
    return V.reshape((n, m), order="F")

def old_wtls(Y:np.ndarray, Q_y:np.ndarray, G:np.ndarray, Q_g:np.ndarray, threshold:float=1e-4, max_iterations:int=20) \
        -> tuple[np.ndarray, np.ndarray]:
    n = Y.shape[0]
    m = G.shape[1]
    X = wls(G, Y, np.linalg.inv(Q_y))
    for i in range(max_iterations):
        print(i)
        X_kron = np.kron(X, np.eye(n))
        QgXk = Q_g @ X_kron
        Q = Q_y + X_kron.transpose() @ QgXk
        W = np.linalg.inv(Q)
        vec_eg = -QgXk @ W @ (Y - G @ X)
        E_g = reshape(vec_eg, n, m)
        G_hat = G - E_g
        Y_hat = Y - E_g @ X
        X_new = wls(G_hat, Y_hat, W)

        print(np.linalg.norm(X - X_new))
        if np.linalg.norm(X - X_new) < threshold:
            break
        X = X_new

    X = X_new
    Q = Q_y + X_kron.transpose() @ Q_g @ np.kron(X, np.eye(n))
    W = np.linalg.inv(Q)
    residuals = Y - G @ X
    lamb = W @ residuals
    sigma_2 = ((lamb.transpose() @ residuals) / (n - m)).flatten()
    Q_x = sigma_2 * np.linalg.inv(G_hat.transpose() @ W @ G_hat)

    return X, Q_x


def wtls(Y: np.ndarray, Q_y: np.ndarray, G: np.ndarray, Q_g: np.ndarray,
         threshold: float = 1e-4, max_iterations: int = 20
         ) -> tuple[np.ndarray, np.ndarray]:
    """Q_g : (n, m) variances de chaque élément de G, ou (n*m, n*m) covariance pleine de vec(G)."""
    y = np.asarray(Y, float).ravel()
    G = np.asarray(G, float)
    Q_g = np.asarray(Q_g, float)
    n, m = G.shape
    diag = Q_g.shape == (n, m)
    if not diag:
        Qg4 = Q_g.reshape(m, n, m, n)          # Qg4[j, i, k, l] = Cov(G[i, j], G[l, k])

    def corrections(x):
        r = y - G @ x
        if diag:
            Q = Q_y + np.diag(Q_g @ x**2)
        else:
            T = np.tensordot(Qg4, x, axes=([2], [0]))   # blocs de Q_g X
            Q = Q_y + np.tensordot(x, T, axes=1)         # Xᵀ Q_g X
        cQ = cho_factor(Q)
        lam = cho_solve(cQ, r)
        E_g = -Q_g * np.outer(lam, x) if diag else -(T @ lam).T
        return r, cQ, lam, E_g

    WG = cho_solve(cho_factor(Q_y), G)
    x = np.linalg.solve(G.T @ WG, WG.T @ y)

    for _ in range(max_iterations):
        _, cQ, _, E_g = corrections(x)
        G_hat = G - E_g
        WG = cho_solve(cQ, G_hat)
        x_new = np.linalg.solve(G_hat.T @ WG, WG.T @ (y - E_g @ x))
        converged = np.linalg.norm(x_new - x) < threshold
        x = x_new
        if converged:
            break

    r, cQ, lam, E_g = corrections(x)
    G_hat = G - E_g
    N = G_hat.T @ cho_solve(cQ, G_hat)
    sigma_2 = (r @ lam) / (n - m)
    Q_x = sigma_2 * cho_solve(cho_factor(N), np.eye(m))

    return (x[:, None] if np.ndim(Y) == 2 else x), Q_x


def prd(rx1_folder_path:str, rx2_folder_path:str, obs_file_name:str="obs.rnx", compute_dd:bool=False) -> pd.DataFrame:
    rx1_filepath = os.path.join(rx1_folder_path, obs_file_name)
    rx2_filepath = os.path.join(rx2_folder_path, obs_file_name)

    rx1_obs_pd = bits.utils.fast_parse(rx1_filepath, bits.parse.raw.rinex)
    rx2_obs_pd = bits.utils.fast_parse(rx2_filepath, bits.parse.raw.rinex)

    prd_df = bits_prd.code_prd.get_prd(rx1_obs_pd, rx2_obs_pd, time_between_meas=1, compute_dd=compute_dd,
                                      pivot_sv_id=None, meas_col_name="pr_m")
    cp_df = bits_prd.code_prd.get_prd(rx1_obs_pd, rx2_obs_pd, time_between_meas=1, compute_dd=compute_dd,
                                      pivot_sv_id=None, meas_col_name="cp_m")

    if compute_dd:
        prd_df["sv_id"] = prd_df["sv_id1"] + prd_df["sv_id2"]
        cp_df["sv_id"] = cp_df["sv_id1"] + cp_df["sv_id2"]
        mode = "dd"
    else:
        mode = "sd"
    prd_df = pd.merge_asof(prd_df, cp_df[["time", "sv_id", mode]], on="time", by=["sv_id"], suffixes=("", "_cp"))

    return prd_df


def get_sensors_sim(rx1_folder_path:str, rx2_folder_path:str, prd_df:pd.DataFrame,
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


def compute_geometry_matrix(df: pd.DataFrame, dt:float, min_sv:int=4, mode:Literal["sd", "dd"]="dd", on_cp:bool=True,
                            dynamic_sv:bool=True, semi_major_axis_table:dict={"gal": 29.6e6,
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
        df = df.sort_values(by = ["time", gnss_id_column]).reset_index(drop=True)

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
    sigma_sensors_full = np.sqrt(2) * sigma_sensors_full # If prd independent wich is not true...

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
    var_db_full["db_x1"] = (sensors_full["displacement_m_rx1"] ** 2 * np.sin(sensors_full["heading_rad_rx1"]) ** 2 * sigma_sensors_full["sigma_heading_rad_rx1"] ** 2
                            + np.cos(sensors_full["heading_rad_rx1"]) ** 2 * sigma_sensors_full["sigma_displacement_m_rx1"] ** 2)
    var_db_full["db_y1"] = (sensors_full["displacement_m_rx1"] ** 2 * np.cos(sensors_full["heading_rad_rx1"]) ** 2 * sigma_sensors_full["sigma_heading_rad_rx1"] ** 2
                            + np.sin(sensors_full["heading_rad_rx1"]) ** 2 * sigma_sensors_full["sigma_displacement_m_rx1"] ** 2)
    var_db_full["db_x2"] = (sensors_full["displacement_m_rx2"] ** 2 * np.sin(sensors_full["heading_rad_rx2"]) ** 2 * sigma_sensors_full["sigma_heading_rad_rx2"] ** 2
                            + np.cos(sensors_full["heading_rad_rx2"]) ** 2 * sigma_sensors_full["sigma_displacement_m_rx2"] ** 2)
    var_db_full["db_y2"] = (sensors_full["displacement_m_rx2"] ** 2 * np.cos(sensors_full["heading_rad_rx2"]) ** 2 * sigma_sensors_full["sigma_heading_rad_rx2"] ** 2
                            + np.sin(sensors_full["heading_rad_rx2"]) ** 2 * sigma_sensors_full["sigma_displacement_m_rx2"] ** 2)

    # Geat rid of first timestamp
    pivot_full = pivot_full.iloc[1:]
    db_full = db_full.iloc[1:]
    sigma_pivot_full = sigma_pivot_full.iloc[1:]
    var_db_full = var_db_full.iloc[1:]

    # Compute geometry matrix for each windows of dt seconds
    all_times = pivot_full.index.values
    n_times_total = len(all_times)
    end_index = 0
    for start_index in tqdm(range(n_times_total), desc="Computing geometry matrix", total=n_times_total-dt):
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
        result = window_geometry_matrix(PRD, var_y, G_rover, var_g, constellation_table, dt_np, semi_major_axis_table, dynamic_sv=dynamic_sv, mode=mode)

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
        sv_to_col = {sv_name: i for i, sv_name in enumerate(sv_list)} # map sv_name with sv_list id
        col_idx = df.loc[time_mask, "sv_id"].map(sv_to_col) # Map dataframe lines with corresponding id from sv_list
        valid = col_idx.notna() # Find valid SV
        sv_map_idx = col_idx[valid].astype(int) # Keep only valid SV

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


def window_geometry_matrix(PRD:np.ndarray, var_y:np.ndarray, G_rover:np.ndarray, var_g: np.ndarray, constellation_table:np.ndarray, dt:np.ndarray,
                           semi_major_axis_table:None|dict, dynamic_sv:bool=True, mode:Literal["sd", "dd"]="dd") -> tuple[np.ndarray, np.ndarray, float]|None:
    constellation_table = np.array(constellation_table)
    # Check inputs
    if G_rover.shape[0] < G_rover.shape[1]:
        txt = f"Not enough measurements, need at least {G_rover.shape[1]}, got {G_rover.shape[0]}"
        warn(txt)
        return None

    if G_rover.shape[0] != PRD.shape[0]:
        txt = f"PRD measurements and rover geometry matrix are not the same length ({PRD.shape[0]}, {G_rover.shape[0]})"
        warn(txt)
        return None

    # 1. Reduce G_rover
    #try:
    if dynamic_sv:
        if mode == "sd":
            #corr_PRD, b0, b1, b2 = correc_clock_bias_dyn(PRD, var_y, G_rover, var_g, constellation_table, dt, semi_major_axis_table)
            corr_PRD, var_corr_prd, B, b_cov = correc_clock_bias_dyn(PRD, var_y, G_rover, var_g, constellation_table, dt, semi_major_axis_table)
        else:
            corr_PRD = PRD
        G_rover, var_G_rover, gamma = reduce_g_rover_dyn(corr_PRD, var_corr_prd, G_rover, var_g, constellation_table, dt, semi_major_axis_table)
        #G_rover, gamma = reduce_g_rover(PRD, G_rover, dt) # TODO gneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeh
    else:
        G_rover, gamma = reduce_g_rover(PRD, G_rover, dt)
    #except Exception as e:
    #    txt = f"Could not reduce rover geometry matrix: {e}"
    #    warn(txt)
    #    return None

    # 2. Compute geometry matrix
    try:
        if dynamic_sv:
            all_PRD = PRD
            all_var_y = var_y
            G = np.full((all_PRD.shape[1], 2), np.nan)
            Q_G = np.full((all_PRD.shape[1], 2, 2), np.nan)
            b_0_list = []
            b_1_list = []
            b_2_list = []
            var_b_list = []
            tho_k = (dt[0] - dt) / (dt[-1] - dt[0])
            # Correct clock bias
            if mode == "sd":
                for cle in np.unique(constellation_table):
                    mask = (constellation_table == cle)
                    # const_groups[cle] = all_PRD[:, mask]
                    PRD = all_PRD[:, mask]
                    var_y = all_var_y[:, mask]

                    n_sat = np.sqrt(bits.const.NU / (semi_major_axis_table[cle] ** 3))
                    n_app = n_sat - bits.const.OMEGA_E

                    c_tk = np.cos(n_app * dt)
                    s_tk = np.sin(n_app * dt)

                    G_rover_extended_0 = G_rover[:, 0] * c_tk
                    G_rover_extended_1 = G_rover[:, 1] * c_tk
                    G_rover_extended_2 = G_rover[:, 0] * s_tk
                    G_rover_extended_3 = G_rover[:, 1] * s_tk

                    G_rover_extended_b0 = np.ones_like(G_rover_extended_0)
                    G_rover_extended_b1 = tho_k
                    G_rover_extended_b2 = tho_k ** 2

                    G_rover_extended = np.vstack(
                        [G_rover_extended_0, G_rover_extended_1, G_rover_extended_2, G_rover_extended_3,
                         G_rover_extended_b0, G_rover_extended_b1, G_rover_extended_b2]).transpose()

                    var_g_extended_0 = var_G_rover[:, 0] * (c_tk) ** 2
                    var_g_extended_1 = var_G_rover[:, 1] * (c_tk) ** 2
                    var_g_extended_2 = var_G_rover[:, 0] * (s_tk) ** 2
                    var_g_extended_3 = var_G_rover[:, 1] * (s_tk) ** 2

                    var_g_extended_b0 = np.zeros_like(var_g_extended_0)
                    var_g_extended_b1 = np.zeros_like(var_g_extended_0)
                    var_g_extended_b2 = np.zeros_like(var_g_extended_0)

                    var_g_extended = np.vstack([var_g_extended_0, var_g_extended_1, var_g_extended_2, var_g_extended_3,
                                                var_g_extended_b0, var_g_extended_b1, var_g_extended_b2]).transpose()

                    # 1. rough estimate
                    if True:
                        for i in range(var_y.shape[1]):
                            Q_y = np.diag(var_y[:, i])
                            Y = PRD[:, i].reshape(-1, 1)
                            X, Q_x = wtls(Y=Y, Q_y=Q_y, G=G_rover_extended, Q_g=var_g_extended)

                            b_0_list += X[4].tolist()
                            b_1_list += X[5].tolist()
                            b_2_list += X[6].tolist()

                            var_b_list.append(Q_x[4:, 4:])
                    else:
                        cs = ols(G_rover_extended, PRD).transpose()

                        b_0_list += cs[:, 4].tolist()
                        b_1_list += cs[:, 5].tolist()
                        b_2_list += cs[:, 6].tolist()

                b_0 = np.median(b_0_list)
                b_1 = np.median(b_1_list)
                b_2 = np.median(b_2_list)

                b_cov = cov_mediane(var_b_list)

                # T = [[np.ones(1)], tho_k, tho_k**2]
                # B = [[b_0, b_1, b_2]]
                # corr_prd = all_PRD - (T @ B)
                # cov(corr_prd) = cov(all_prd) - T @ cov(B) @ T^T

                T = np.array([np.ones(tho_k.shape[0]), tho_k, tho_k ** 2]).transpose()
                B = np.array([[b_0], [b_1], [b_2]])

                all_PRD = all_PRD - (T @ B)

                var_all_prd = all_var_y + np.diag(T @ b_cov @ T.transpose()).reshape(-1, 1)

                #all_PRD = all_PRD - b_0 - (b_1 * tho_k).reshape(-1, 1) - (b_2 * tho_k ** 2).reshape(-1, 1)

            # Compute G
            for cle in np.unique(constellation_table):
                mask = (constellation_table == cle)
                # const_groups[cle] = all_PRD[:, mask]
                PRD = all_PRD[:, mask]
                var_PRD = var_all_prd[:, mask]

                n_sat = np.sqrt(bits.const.NU / (semi_major_axis_table[cle] ** 3))
                n_app = n_sat - bits.const.OMEGA_E

                c_tk = np.cos(n_app * dt)
                s_tk = np.sin(n_app * dt)

                G_rover_extended_0 = G_rover[:, 0] * c_tk
                G_rover_extended_1 = G_rover[:, 1] * c_tk
                G_rover_extended_2 = G_rover[:, 0] * s_tk
                G_rover_extended_3 = G_rover[:, 1] * s_tk

                G_rover_extended = np.vstack(
                    [G_rover_extended_0, G_rover_extended_1, G_rover_extended_2, G_rover_extended_3])

                G_rover_extended = G_rover_extended.transpose()

                var_g_extended_0 = var_G_rover[:, 0] * (c_tk) ** 2
                var_g_extended_1 = var_G_rover[:, 1] * (c_tk) ** 2
                var_g_extended_2 = var_G_rover[:, 0] * (s_tk) ** 2
                var_g_extended_3 = var_G_rover[:, 1] * (s_tk) ** 2

                var_g_extended = np.vstack([var_g_extended_0, var_g_extended_1, var_g_extended_2, var_g_extended_3]).transpose()

                # 1. rough estimate
                if True:
                    X_list = []
                    Q_x_list = []
                    for i in range(var_PRD.shape[1]):
                        Q_y = np.diag(var_PRD[:, i])
                        Y = PRD[:, i].reshape(-1, 1)
                        X, Q_x = wtls(Y=Y, Q_y=Q_y, G=G_rover_extended, Q_g=var_g_extended)

                        X_list.append(X)
                        Q_x_list.append(Q_x)
                    cs = np.hstack(X_list).transpose()
                else:
                    cs = ols(G_rover_extended, PRD).transpose()

                #G_x = c_tk[-1] * cs[:, 0] + s_tk[-1] * cs[:, 2]
                #G_y = c_tk[-1] * cs[:, 1] + s_tk[-1] * cs[:, 3]
                #G[mask] = np.vstack([G_x, G_y]).transpose()

                G_local = c_tk[-1] * cs[:, 0:2] + s_tk[-1] * cs[:, 2:4]
                G[mask] = G_local

                q_g_local_list = []
                for i in range(len(Q_x_list)):
                    q_g_local_list.append(c_tk[-1]**2 * Q_x_list[i][0:2, 0:2] + s_tk[-1]**2 * Q_x_list[i][2:4, 2:4])
                Q_G_local = np.stack(q_g_local_list)
                Q_G[mask] = Q_G_local

            G = G.transpose()
        else:
            G = ols(G_rover, PRD)
    except Exception as e:
        txt = f"Could not compute matrix: {e}"
        warn(txt)
        return None

    return G, Q_G, gamma

def correc_clock_bias_dyn(PRD:np.ndarray, var_y:np.ndarray, G_rover:np.ndarray, var_g:np.ndarray, constellation_table:np.ndarray, dt:np.ndarray,
                          semi_major_axis_table:None|dict) -> tuple[np.ndarray, float, float, float]:
    # 0. Organize by constellation
    b_0_list = []
    b_1_list = []
    b_2_list = []
    var_b_list = []
    all_PRD = PRD
    all_var_y = var_y
    constellation_table = np.array(constellation_table)
    tho_k = (dt[0] - dt) / (dt[-1] - dt[0])

    for cle in np.unique(constellation_table):
        mask = (constellation_table == cle)
        PRD = all_PRD[:, mask]
        var_y = all_var_y[:, mask]

        n_sat = np.sqrt(bits.const.NU / (semi_major_axis_table[cle] ** 3))
        n_app = n_sat - bits.const.OMEGA_E

        c_tk = np.cos(n_app * dt)
        s_tk = np.sin(n_app * dt)

        G_rover_extended_0 = G_rover[:, 0] * c_tk
        G_rover_extended_1 = G_rover[:, 1] * c_tk
        G_rover_extended_2 = G_rover[:, 0] * s_tk
        G_rover_extended_3 = G_rover[:, 1] * s_tk

        G_rover_extended_4 = G_rover[:, 2] * c_tk
        G_rover_extended_5 = G_rover[:, 3] * c_tk
        G_rover_extended_6 = G_rover[:, 2] * s_tk
        G_rover_extended_7 = G_rover[:, 3] * s_tk

        G_rover_extended_b0 = np.ones_like(G_rover_extended_0)
        G_rover_extended_b1 = tho_k
        G_rover_extended_b2 = tho_k ** 2

        G_rover_extended = np.vstack([G_rover_extended_0, G_rover_extended_1, G_rover_extended_2, G_rover_extended_3,
                                      G_rover_extended_4, G_rover_extended_5, G_rover_extended_6, G_rover_extended_7,
                                      G_rover_extended_b0, G_rover_extended_b1, G_rover_extended_b2]).transpose()

        var_g_extended_0 = var_g[:, 0] * (c_tk)**2
        var_g_extended_1 = var_g[:, 1] * (c_tk)**2
        var_g_extended_2 = var_g[:, 0] * (s_tk)**2
        var_g_extended_3 = var_g[:, 1] * (s_tk)**2

        var_g_extended_4 = var_g[:, 2] * (c_tk)**2
        var_g_extended_5 = var_g[:, 3] * (c_tk)**2
        var_g_extended_6 = var_g[:, 2] * (s_tk)**2
        var_g_extended_7 = var_g[:, 3] * (s_tk)**2

        var_g_extended_b0 = np.zeros_like(var_g_extended_0)
        var_g_extended_b1 = np.zeros_like(var_g_extended_0)
        var_g_extended_b2 = np.zeros_like(var_g_extended_0)

        var_g_extended = np.vstack([var_g_extended_0, var_g_extended_1, var_g_extended_2, var_g_extended_3,
                                          var_g_extended_4, var_g_extended_5, var_g_extended_6, var_g_extended_7,
                                          var_g_extended_b0, var_g_extended_b1, var_g_extended_b2]).transpose()

        #Q_G = np.diag(vec(var_g_extended)[0].flatten())
        Q_G = var_g_extended

        # 1. rough estimate
        if True:
            for i in range(var_y.shape[1]):
                Q_y = np.diag(var_y[:, i])
                Y = PRD[:, i].reshape(-1, 1)
                X, Q_x = wtls(Y=Y, Q_y=Q_y, G=G_rover_extended.copy(), Q_g=Q_G.copy())

                b_0_list.append(X[8, 0])
                b_1_list.append(X[9, 0])
                b_2_list.append(X[10, 0])

                var_b_list.append(Q_x[8:, 8:])
        else:
            G = ols(G_rover_extended, PRD)

            b_0_list += G[8].tolist()
            b_1_list += G[9].tolist()
            b_2_list += G[10].tolist()


    # 2 correct clock bias
    b_0 = np.median(b_0_list)
    b_1 = np.median(b_1_list)
    b_2 = np.median(b_2_list)

    b_cov = cov_mediane(var_b_list)

    # T = [[np.ones(1)], tho_k, tho_k**2]
    # B = [[b_0, b_1, b_2]]
    # corr_prd = all_PRD - (T @ B)
    # cov(corr_prd) = cov(all_prd) - T @ cov(B) @ T^T

    T = np.array([np.ones(tho_k.shape[0]), tho_k, tho_k**2]).transpose()
    B = np.array([[b_0], [b_1], [b_2]])

    corr_PRD = all_PRD - (T @ B)

    var_corr_prd = all_var_y + np.diag(T @ b_cov @ T.transpose()).reshape(-1, 1)

    return corr_PRD, var_corr_prd, B, b_cov

def cov_mediane(cov_list):
    """cov_list : liste de N matrices de covariance (p, p), une par vecteur."""
    cov_list = np.stack([np.asarray(S, dtype=float) for S in cov_list])  # (N, p, p)
    sig = np.sqrt(np.diagonal(cov_list, axis1=1, axis2=2))             # (N, p)
    rho = cov_list / (sig[:, :, None] * sig[:, None, :])                # (N, p, p)
    num = np.arcsin(np.clip(rho, -1, 1)).sum(axis=0)                  # (p, p)
    S = (1 / sig).sum(axis=0)                                         # (p,)
    return num / np.outer(S, S)


def reduce_g_rover_dyn(PRD:np.ndarray, var_PRD:np.ndarray, G_rover:np.ndarray, var_g:np.ndarray, constellation_table:np.ndarray, dt:np.ndarray,
                   semi_major_axis_table:None|dict) -> tuple[np.ndarray, np.ndarray, float]:
    # 0. Organize by constellation
    #gamma_rad = 0
    all_PRD = PRD
    all_var_PRD = var_PRD
    g_list = []
    var_g_list = []
    constellation_table = np.array(constellation_table)
    for cle in np.unique(constellation_table):
        mask = (constellation_table == cle)
        PRD = all_PRD[:, mask]
        var_PRD = all_var_PRD[:, mask]

        n_sat = np.sqrt(bits.const.NU / (semi_major_axis_table[cle] ** 3))
        n_app = n_sat - bits.const.OMEGA_E

        c_tk = np.cos(n_app * dt)
        s_tk = np.sin(n_app * dt)

        G_rover_extended_0 = G_rover[:, 0] * c_tk
        G_rover_extended_1 = G_rover[:, 1] * c_tk
        G_rover_extended_2 = G_rover[:, 0] * s_tk
        G_rover_extended_3 = G_rover[:, 1] * s_tk

        G_rover_extended_4 = G_rover[:, 2] * c_tk
        G_rover_extended_5 = G_rover[:, 3] * c_tk
        G_rover_extended_6 = G_rover[:, 2] * s_tk
        G_rover_extended_7 = G_rover[:, 3] * s_tk

        G_rover_extended = np.vstack([G_rover_extended_0, G_rover_extended_1, G_rover_extended_2, G_rover_extended_3,
                                      G_rover_extended_4, G_rover_extended_5, G_rover_extended_6, G_rover_extended_7]).transpose()

        var_G_rover_extended_0 = var_g[:, 0] * c_tk**2
        var_G_rover_extended_1 = var_g[:, 1] * c_tk**2
        var_G_rover_extended_2 = var_g[:, 0] * s_tk**2
        var_G_rover_extended_3 = var_g[:, 1] * s_tk**2

        var_G_rover_extended_4 = var_g[:, 2] * c_tk**2
        var_G_rover_extended_5 = var_g[:, 3] * c_tk**2
        var_G_rover_extended_6 = var_g[:, 2] * s_tk**2
        var_G_rover_extended_7 = var_g[:, 3] * s_tk**2

        var_G_rover_extended = np.vstack([var_G_rover_extended_0, var_G_rover_extended_1, var_G_rover_extended_2, var_G_rover_extended_3,
                                          var_G_rover_extended_4, var_G_rover_extended_5, var_G_rover_extended_6, var_G_rover_extended_7]).transpose()

        # 1. rough estimate
        for i in range(var_PRD.shape[1]):
            Q_y = np.diag(var_PRD[:, i])
            Y = PRD[:, i].reshape(-1, 1)
            g, Q_g = wtls(Y=Y, Q_y=Q_y, G=G_rover_extended.copy(), Q_g=var_G_rover_extended.copy())

            g_list.append(g)

            var_g_list.append(np.diag(Q_g).reshape(-1, 1))

    G = np.hstack(g_list)
    var_G = np.hstack(var_g_list)


    #Gneh = ols(G_rover_extended, PRD)

    # 2. Gamma estimate
    e_A1 = -G[0:2,:].transpose()
    e_A2 = -G[2:4, :].transpose()
    e_B1 = G[4:6, :].transpose()
    e_B2 = G[6:8, :].transpose()

    var_e_A1 = var_G[0:2, :]
    var_e_A2 = var_G[2:4, :]
    var_e_B1 = var_G[4:6, :]
    var_e_B2 = var_G[6:8, :]

    angle_list = []
    var_angle_list = []
    for i in range(e_B1.shape[1]):
        Q_b = np.diag(var_e_B1[i])
        b = e_B1[:, i].reshape(-1, 1)
        angle, Q_angle = wtls(Y=b, Q_y=Q_b, G=e_A1, Q_g=var_e_A1.transpose())

        angle_list.append(angle)

        var_angle_list.append(np.diag(Q_angle).reshape(-1, 1))

    angle_estimate1 = np.hstack(angle_list)
    var_angle_estimate1 = np.hstack(var_angle_list)

    angle_list = []
    var_angle_list = []
    for i in range(e_B2.shape[1]):
        Q_b = np.diag(var_e_B2[i])
        b = e_B2[:, i].reshape(-1, 1)
        angle, Q_angle = wtls(Y=b, Q_y=Q_b, G=e_A2, Q_g=var_e_A2.transpose())

        angle_list.append(angle)

        var_angle_list.append(np.diag(Q_angle).reshape(-1, 1))

    angle_estimate2 = np.hstack(angle_list)
    var_angle_estimate2 = np.hstack(var_angle_list)

    #angle_estimate1 = ols(e_A1, e_B1)
    #angle_estimate2 = ols(e_A2, e_B2)

    U, S, Vt = np.linalg.svd(angle_estimate1)
    R = U @ Vt
    # Correction pour garantir det(R) = +1 (pas une réflexion)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    cos_gamma = R[0, 0]
    sin_gamma = R[1, 0]
    gamma_rad1 = np.arctan2(sin_gamma, cos_gamma)

    U, S, Vt = np.linalg.svd(angle_estimate2)
    R = U @ Vt
    # Correction pour garantir det(R) = +1 (pas une réflexion)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    cos_gamma = R[0, 0]
    sin_gamma = R[1, 0]
    gamma_rad2 = np.arctan2(sin_gamma, cos_gamma)
    #gamma_rad = PRD.shape[1] * (gamma_rad1 + gamma_rad2)/2
    gamma_rad = (gamma_rad1 + gamma_rad2) / 2


    #gamma_rad /= all_PRD.shape[1]
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)

    var_cos_gamma = (np.max(np.diag(var_angle_estimate1)) + np.max(np.diag(var_angle_estimate2))) / 4
    var_sin_gamma = (np.max([var_angle_estimate1[0, 1], var_angle_estimate1[1, 0]])
                     + np.max([var_angle_estimate2[0, 1], var_angle_estimate2[1, 0]])) / 4

    # Reduce G_rover using gamma estimate
    reduced_G_rover_x = - G_rover[:, 0] + cos_gamma * G_rover[:, 2] - sin_gamma * G_rover[:, 3]
    reduced_G_rover_y = - G_rover[:, 1] + cos_gamma * G_rover[:, 3] + sin_gamma * G_rover[:, 2]
    reduced_G_rover = np.column_stack([reduced_G_rover_x, reduced_G_rover_y])

    var_reduced_G_rover_x = (var_g[:, 0]
                             + var_cos_gamma * (G_rover[:, 2]**2) + var_g[:, 2] * cos_gamma**2 + var_cos_gamma * var_g[:, 2]
                             + var_sin_gamma * (G_rover[:, 3]**2) + var_g[:, 3] * sin_gamma**2 + var_sin_gamma * var_g[:, 3])
    var_reduced_G_rover_y = (var_g[:, 1]
                             + var_cos_gamma * (G_rover[:, 3]**2) + var_g[:, 3] * cos_gamma**2 + var_cos_gamma * var_g[:, 3]
                             + var_sin_gamma * (G_rover[:, 2]**2) + var_g[:, 2] * sin_gamma**2 + var_sin_gamma * var_g[:, 2])
    var_reduced_G_rover = np.column_stack([var_reduced_G_rover_x, var_reduced_G_rover_y])

    return reduced_G_rover, var_reduced_G_rover, float(gamma_rad)


def reduce_g_rover(PRD:np.ndarray, G_rover:np.ndarray, dt:np.ndarray) -> tuple[np.ndarray, float]:
    # 1. rough estimate
    G = ols(G_rover, PRD)

    # 2. Gamma estimate
    e_A = -G[0:2,:].transpose()
    e_B = G[2:4, :].transpose()

    angle_estimate = ols(e_A, e_B)

    U, S, Vt = np.linalg.svd(angle_estimate)
    R = U @ Vt
    # Correction pour garantir det(R) = +1 (pas une réflexion)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    cos_gamma = R[0, 0]
    sin_gamma = R[1, 0]
    gamma_rad = np.arctan2(sin_gamma, cos_gamma)

    """
    N = G.shape[1]
    ex_A = G[0]
    ey_A = G[1]
    ex_B = G[2]
    ey_B = G[3]

    e_B = np.zeros(2 * N)
    e_A = np.zeros((2 * N, 2))

    for c in range(N):
        e_B[2 * c] = ex_B[c]
        e_B[2 * c + 1] = ey_B[c]
        e_A[2 * c, :] = [ex_A[c], -ey_A[c]]
        e_A[2 * c + 1, :] = [ey_A[c], ex_A[c]]

    angle_estimate = ols(e_A, e_B)
    norm = np.linalg.norm(angle_estimate)
    angle_estimate = angle_estimate / norm
    cos_gamma = angle_estimate[0]
    sin_gamma = angle_estimate[1]
    gamma_rad = np.arctan2(sin_gamma, cos_gamma)
    gamma_rad = 0.0 # TODO gneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeh
    cos_gamma = np.cos(gamma_rad) # TODO gneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeh
    sin_gamma = np.sin(gamma_rad) # TODO gneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeh
    """

    # Reduce G_rover using gamma estimate
    reduced_G_rover_x = - G_rover[:, 0] + cos_gamma * G_rover[:, 2] - sin_gamma * G_rover[:, 3]
    reduced_G_rover_y = - G_rover[:, 1] + cos_gamma * G_rover[:, 3] + sin_gamma * G_rover[:, 2]
    reduced_G_rover = np.column_stack([reduced_G_rover_x, reduced_G_rover_y])

    return reduced_G_rover, float(gamma_rad)


def compute_baseline(df:pd.DataFrame, mode="dd") -> pd.DataFrame:
    # Clean up
    df = df.copy()
    df = df.dropna(subset=["e_x", "e_y"], how="all")
    df = df.sort_values("time").reset_index(drop=True)

    tqdm_desc = f"Computing baseline with {mode}"
    baseline_pd_list = []
    for _, group in tqdm(df.groupby("time"), total=len(df["time"].unique()), desc=tqdm_desc):
        try:
            baseline_pd = window_compute_baseline(group, mode=mode)
        except Exception as e:
            txt = f"Could not compute baseline: {e}"
            warn(txt)
            continue
        baseline_pd_list.append(baseline_pd)

    return pd.concat(baseline_pd_list, ignore_index=True)


def window_compute_baseline(group, mode="dd") -> pd.DataFrame:
    PRD = group[mode].to_numpy().reshape(-1, 1)
    ex = group["e_x"].to_numpy()
    ey = group["e_y"].to_numpy()
    if mode == "sd":
        G = np.vstack((ex, ey, np.ones_like(ex))).transpose()
    else:
        G = np.vstack((ex, ey)).transpose()

    baseline_estimate = ols(G, PRD)

    # Save result
    baseline_serie = pd.Series({"time": group["time"].iloc[0],
                                "mode": mode,})
    baseline_serie["bx_rx_m"] = float(baseline_estimate[0][0])
    baseline_serie["by_rx_m"] = float(baseline_estimate[1][0])
    if mode == "sd":
        baseline_serie["bb_rx_m"] = float(baseline_estimate[-1][0])
    baseline_serie["baseline_m"] = float(np.linalg.norm(baseline_estimate[:2]))

    return baseline_serie.to_frame().T

def ground_truth(df:pd.DataFrame, rx1_folder_path:str, rx2_folder_path:str, gt_file_name:str="nmea_simulator.txt", dt=0) \
        -> pd.DataFrame:
    rx1_filepath = os.path.join(rx1_folder_path, gt_file_name)
    rx2_filepath = os.path.join(rx2_folder_path, gt_file_name)

    rx1_gt_df = bits.utils.fast_parse(rx1_filepath, bits.parse.pvt.rmc)
    rx2_gt_df = bits.utils.fast_parse(rx2_filepath, bits.parse.pvt.rmc)

    # Clean
    rx1_gt_df = rx1_gt_df.sort_values("time").reset_index(drop=True)
    rx2_gt_df = rx2_gt_df.sort_values("time").reset_index(drop=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = df["time"].astype("datetime64[ns]")

    # Compute baseline ground truth
    rx1_gt_df["x_rx_m_rx2"] = np.interp(rx1_gt_df["time"], rx2_gt_df["time"], rx2_gt_df["x_rx_m"])
    rx1_gt_df["y_rx_m_rx2"] = np.interp(rx1_gt_df["time"], rx2_gt_df["time"], rx2_gt_df["y_rx_m"])
    rx1_gt_df["z_rx_m_rx2"] = np.interp(rx1_gt_df["time"], rx2_gt_df["time"], rx2_gt_df["z_rx_m"])
    rx1_gt_df["baseline_m"] = np.sqrt((rx1_gt_df["x_rx_m_rx2"] - rx1_gt_df["x_rx_m"])**2 +
                                      (rx1_gt_df["y_rx_m_rx2"] - rx1_gt_df["y_rx_m"])**2 +
                                      (rx1_gt_df["z_rx_m_rx2"] - rx1_gt_df["z_rx_m"])**2)

    dt = np.timedelta64(dt, "s")
    rx1_gt_df["time"] += dt

    df["baseline_m_gt"] = np.interp(df["time"], rx1_gt_df["time"], rx1_gt_df["baseline_m"])
    df["baseline_error_m"] = df["baseline_m_gt"] - df["baseline_m"]

    return df


def sample_reduction(df:pd.DataFrame, sample_size:float, sample_count:int, dump:int=0) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    unique_time_list = df["time"].unique().tolist()

    required_lenght = dump + sample_size + sample_count

    if len(unique_time_list) < required_lenght:
        txt = f"Not enough timestamps, requires {required_lenght}, got {len(df)}"
        raise Exception(txt)

    timestamp_to_keep = unique_time_list[dump:required_lenght]

    df = df[df["time"].isin(timestamp_to_keep)]

    return df


if __name__ == "__main__":
    runs = ["3ms", "10ms", "30ms"]
    run1 = runs[0]
    run2 = runs[1]
    skydel_dt = -18
    sample_size = 30*60 # seconds

    rx1_folder_path = f"resources/{run1}"
    rx2_folder_path = f"resources/{run2}"

    if True:
        prd_df = prd(rx1_folder_path, rx2_folder_path, compute_dd=False)
        prd_df.to_pickle(f"prd.pkl")
        #exit()
    else:
        prd_df = pd.read_pickle("prd.pkl")

    sensors_df = get_sensors_sim(rx1_folder_path, rx2_folder_path, prd_df, dt=skydel_dt)

    #sensors_df = sample_reduction(sensors_df, sample_size=sample_size, sample_count=10, dump=40)#10*60)

    if True:
        g_df = compute_geometry_matrix(sensors_df, sample_size, min_sv=4, mode="sd", on_cp=True, dynamic_sv=True)
        g_df.to_pickle(f"g_df.pkl")
        #exit()
    else:
        g_df = pd.read_pickle("g_df.pkl")

    exit()

    baseline_df = compute_baseline(g_df, mode="sd")

    baseline_df = ground_truth(baseline_df, rx1_folder_path, rx2_folder_path, dt=skydel_dt)

    plt.figure("Baseline")
    plt.plot(baseline_df["time"], baseline_df["baseline_error_m"], label="Error", color="red")
    plt.plot(baseline_df["time"], baseline_df["baseline_m"], label="Estimate", color="blue")
    plt.plot(baseline_df["time"], baseline_df["baseline_m_gt"], linestyle="--", label="Ground truth", color="black")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Baseline (m)")
    plt.legend()

    plt.figure("Gamma")
    plt.scatter(g_df["time"], g_df["gamma_rad"], s=0.5, label="Gamma")
    #plt.scatter(g_df["time"], g_df["heading_rad_rx1"], s=0.5, label="heading rx1")
    #plt.scatter(g_df["time"], g_df["heading_rad_rx2"], s=0.5, label="heading rx2")
    #plt.scatter(g_df["time"], g_df["heading_rad_rx1"] - g_df["heading_rad_rx2"], s=0.5, label="heading diff")
    #plt.legend()
    plt.xlabel("Time (UTC)")
    plt.ylabel("Gamma (rad)")

    plt.show()

    print("gneh")
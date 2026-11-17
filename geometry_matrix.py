from typing import Literal
import pandas as pd
import numpy as np
from tqdm import tqdm
from warnings import warn

import bits

from parse import SENSOR_COL_LIST
import utils

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


def add_sv_model(g_rover: np.ndarray, var_g_rover: np.ndarray, elapsed_time: np.ndarray, semi_major_axis: float) \
        -> tuple[np.ndarray, np.ndarray]:
    g_rover = g_rover.copy()
    var_g_rover = var_g_rover.copy()

    # Compute constellation apparent period
    n_sat = np.sqrt(bits.const.NU / (semi_major_axis ** 3))
    n_app = n_sat - bits.const.OMEGA_E

    # Compute c/s coeff
    c_tk = np.cos(n_app * elapsed_time)
    s_tk = np.sin(n_app * elapsed_time)

    # Compute G_rover for rx1
    c_x = g_rover[:, 0] * c_tk
    c_y = g_rover[:, 1] * c_tk
    s_x = g_rover[:, 0] * s_tk
    s_y = g_rover[:, 1] * s_tk

    g_rover_sv_model = np.vstack([c_x, c_y, s_x, s_y])

    # Compute variance of G_rover for rx1
    var_c_x = var_g_rover[:, 0] * (c_tk) ** 2
    var_c_y = var_g_rover[:, 1] * (c_tk) ** 2
    var_s_x = var_g_rover[:, 0] * (s_tk) ** 2
    var_s_y = var_g_rover[:, 1] * (s_tk) ** 2

    var_g_rover_sv_model = np.vstack([var_c_x, var_c_y, var_s_x, var_s_y])

    if g_rover.shape[1] > 2:
        # Compute G_rover for rx2
        c_x_rx2 = g_rover[:, 2] * c_tk
        c_y_rx2 = g_rover[:, 3] * c_tk
        s_x_rx2 = g_rover[:, 2] * s_tk
        s_y_rx2 = g_rover[:, 3] * s_tk

        g_rover_sv_model = np.vstack([g_rover_sv_model, c_x_rx2, c_y_rx2, s_x_rx2, s_y_rx2])

        # Compute variance of G_rover for rx2
        var_c_x_rx2 = var_g_rover[:, 2] * (c_tk) ** 2
        var_c_y_rx2 = var_g_rover[:, 3] * (c_tk) ** 2
        var_s_x_rx2 = var_g_rover[:, 2] * (s_tk) ** 2
        var_s_y_rx2 = var_g_rover[:, 3] * (s_tk) ** 2

        var_g_rover_sv_model = np.vstack([var_g_rover_sv_model, var_c_x_rx2, var_c_y_rx2, var_s_x_rx2, var_s_y_rx2])

    return g_rover_sv_model.transpose(), var_g_rover_sv_model.transpose()


def add_rx_clock_model(g_rover: np.ndarray, var_g_rover: np.ndarray, elapsed_time: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray]:
    g_rover = g_rover.copy()
    var_g_rover = var_g_rover.copy()

    # Compute ellapsed time factor
    tho_k = (elapsed_time[0] - elapsed_time) / (elapsed_time[-1] - elapsed_time[0]).reshape(-1, 1)

    # Compute G_rover relative clock bias factors
    b0 = np.ones_like(g_rover).reshape(-1, 1)
    b1 = tho_k
    b2 = tho_k ** 2

    g_rover_clock_bias = np.hstack([g_rover, b0, b1, b2])

    # Compute variance of G_rover relative clock bias factors
    var_b0 = np.zeros_like(var_g_rover[:, 0]).reshape(-1, 1)
    var_b1 = np.zeros_like(var_g_rover[:, 0]).reshape(-1, 1)
    var_b2 = np.zeros_like(var_g_rover[:, 0]).reshape(-1, 1)

    var_g_rover_clock_bias = np.hstack([var_g_rover, var_b0, var_b1, var_b2])

    return g_rover_clock_bias, var_g_rover_clock_bias

def clock_bias_correction(prd: np.ndarray, var_prd: np.ndarray, g_rover: np.ndarray, var_g_rover: np.ndarray,
                          elapsed_time: np.ndarray, constellation_table: np.ndarray,
                          gnss_table: dict = DEFAULT_GNSS_TABLE, sv_model: bool = False) \
        -> tuple[np.ndarray, float, float, float]:


    for gnss_id in np.unique(constellation_table): # Iterate over each GNSS constellation
        ## Extract GNSS observables
        const_mask = (constellation_table == gnss_id)
        const_prd = prd[:, const_mask]
        const_var_prd = var_prd[:, const_mask]

        # Build G_rover_extended
        if sv_model:
            g_rover_extended, var_g_rover_extended = add_sv_model(g_rover, var_g_rover, elapsed_time, gnss_table[gnss_id])
        else:
            g_rover_extended = g_rover
            var_g_rover_extended = var_g_rover

        g_rover_extended, var_g_rover_extended = add_rx_clock_model(g_rover_extended, var_g_rover_extended, elapsed_time)


def angle_bias_correction():
    pass


def reduce_g_rover():
    pass


def old_correc_clock_bias(prd: np.ndarray, var_prd: np.ndarray, g_rover: np.ndarray, var_g_rover: np.ndarray,
                          elapsed_time: np.ndarray, constellation_table: np.ndarray,
                          gnss_table: dict = DEFAULT_GNSS_TABLE, sv_model: bool = False) \
        -> tuple[np.ndarray, float, float, float]:

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
                X, Q_x = utils.wtls(Y=Y, Q_y=Q_y, G=G_rover_extended.copy(), Q_g=Q_G.copy())

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

    b_cov = utils.cov_mediane(var_b_list)

    # T = [[np.ones(1)], tho_k, tho_k**2]
    # B = [[b_0, b_1, b_2]]
    # corr_prd = all_PRD - (T @ B)
    # cov(corr_prd) = cov(all_prd) - T @ cov(B) @ T^T

    T = np.array([np.ones(tho_k.shape[0]), tho_k, tho_k**2]).transpose()
    B = np.array([[b_0], [b_1], [b_2]])

    corr_PRD = all_PRD - (T @ B)

    var_corr_prd = all_var_y + np.diag(T @ b_cov @ T.transpose()).reshape(-1, 1)

    return corr_PRD, var_corr_prd, B, b_cov


def compute_window(prd: np.ndarray, var_prd: np.ndarray, g_rover: np.ndarray, var_g_rover: np.ndarray,
                   elapsed_time: np.ndarray, constellation_table:np.ndarray,
                   gnss_table: dict = DEFAULT_GNSS_TABLE, sv_model: bool = False) -> pd.DataFrame|None:
    ## Check inputs
    if sv_model and constellation_table is None:
        raise ValueError("Constellation table must be provided for SV model computation")

    if g_rover.shape[0] < g_rover.shape[1]:
        txt = f"Not enough measurements, need at least {g_rover.shape[1]}, got {g_rover.shape[0]}"
        warn(txt)
        return None

    if g_rover.shape[0] != prd.shape[0]:
        txt = f"PRD measurements and rover geometry matrix are not the same length ({prd.shape[0]}, {g_rover.shape[0]})"
        warn(txt)
        return None

    # 1. Reduce G_rover
    #try:
    if sv_model:
        corr_PRD, var_corr_prd, B, b_cov = correc_clock_bias(PRD, var_y, G_rover, var_g, constellation_table, dt, semi_major_axis_table)
        G_rover, var_G_rover, gamma = reduce_g_rover_dyn(corr_PRD, var_corr_prd, G_rover, var_g, constellation_table, dt, semi_major_axis_table)
    else:
        G_rover, gamma = reduce_g_rover(PRD, G_rover, dt)


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

        # Extract used GNSS id table for sv model
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


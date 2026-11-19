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

    g_rover_clock_bias = np.hstack([g_rover, clock_model(elapsed_time)])

    # Compute variance of G_rover relative clock bias factors
    var_b = np.zeros((var_g_rover.shape[0], 3)) # Variances for b0, b1, b2 are set to 0
    var_g_rover_clock_bias = np.hstack([var_g_rover, var_b])

    return g_rover_clock_bias, var_g_rover_clock_bias


def clock_model(elapsed_time: np.ndarray) -> np.ndarray:
    tho_k = ((elapsed_time[0] - elapsed_time) / (elapsed_time[-1] - elapsed_time[0]))
    clock_model = np.array([np.ones(tho_k.shape[0]), tho_k, tho_k ** 2])
    return clock_model.transpose()


def clock_bias_correction(prd: np.ndarray, var_prd: np.ndarray, g_rover: np.ndarray, var_g_rover: np.ndarray,
                          elapsed_time: np.ndarray, constellation_table: np.ndarray,
                          gnss_table: dict = DEFAULT_GNSS_TABLE, sv_model: bool = False) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    ## Estimate relative clock biases (b0, b1, b2)
    b_list = []
    var_b_list = []
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

        X, Q_x = utils.wtls_2D(Y=const_prd, Q_y=const_var_prd, G=g_rover_extended, Q_g=var_g_rover_extended)

        # Keep only the 3 last values that correspond to relative clock bias
        b_list.append(X[:, -3:])
        var_b_list.append(Q_x[:, -3:, -3:])

    ## Merge clock biases coeff estimates into a single clock estimate
    b_all_sv = np.vstack(b_list)
    b = np.median(b_all_sv, axis=0).reshape(-1, 1)

    var_b_all_sv = np.vstack(var_b_list)
    var_b = utils.cov_mediane(var_b_all_sv)

    ## Correct prd with estimated relative clock bias
    tho = clock_model(elapsed_time)
    clock_bias = (tho @ b)
    corr_prd = prd - clock_bias

    cov_clock_bias = tho @ var_b @ tho.transpose()
    Q_corr_prd_list = []
    for i in range(var_prd.shape[1]):
        local_Q_corr_prd = np.diag(var_prd[:, i]) + cov_clock_bias
        Q_corr_prd_list.append(local_Q_corr_prd)
    Q_corr_prd = np.stack(Q_corr_prd_list)

    return corr_prd, Q_corr_prd, clock_bias, cov_clock_bias


def angle_bias_correction(prd: np.ndarray, Q_prd: np.ndarray, g_rover: np.ndarray, var_g_rover: np.ndarray,
                          elapsed_time: np.ndarray, constellation_table: np.ndarray,
                          gnss_table: dict = DEFAULT_GNSS_TABLE, sv_model: bool = False) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ## G estimation
    G_list = []
    Q_g_list = []
    for gnss_id in np.unique(constellation_table): # Iterate over each GNSS constellation
        # Extract GNSS observables
        const_mask = (constellation_table == gnss_id)
        const_prd = prd[:, const_mask]
        const_Q_prd = Q_prd[const_mask]

        # Build G_rover_extended
        if sv_model:
            g_rover_extended, var_g_rover_extended = add_sv_model(g_rover, var_g_rover, elapsed_time,
                                                                  gnss_table[gnss_id])
        else:
            g_rover_extended = g_rover
            var_g_rover_extended = var_g_rover

        local_G, local_Q_g = utils.wtls_2D(Y=const_prd, Q_y=const_Q_prd, G=g_rover_extended, Q_g=var_g_rover_extended)

        G_list.append(local_G)
        Q_g_list.append(local_Q_g)

    G_all_sv = np.vstack(G_list)
    Q_g_all_sv = np.vstack(Q_g_list)

    ## Angle in G estimation
    if not sv_model:
        m1 = -G_all_sv[:, :2]
        m2 = G_all_sv[:, 2:]
        q_m1 = np.vstack([Q_g_all_sv[:, 0, 0], Q_g_all_sv[:, 1, 1]]).transpose()
        q_m2 = np.vstack([Q_g_all_sv[:, 2, 2], Q_g_all_sv[:, 3, 3]]).transpose()
        cos_a, sin_a, var_cos_a, var_sin_a = compute_angle_between_matrix(m1, m2, q_m1, q_m2)
    else:
        # First angle
        m1 = -G_all_sv[:, 0:2]
        m2 = G_all_sv[:, 4:6]
        q_m1 = np.vstack([Q_g_all_sv[:, 0, 0], Q_g_all_sv[:, 1, 1]]).transpose()
        q_m2 = np.vstack([Q_g_all_sv[:, 4, 4], Q_g_all_sv[:, 5, 5]]).transpose()
        cos_a1, sin_a1, var_cos_a1, var_sin_a1 = compute_angle_between_matrix(m1, m2, q_m1, q_m2)

        # Second angle
        m1 = -G_all_sv[:, 2:4]
        m2 = G_all_sv[:, 6:8]
        q_m1 = np.vstack([Q_g_all_sv[:, 2, 2], Q_g_all_sv[:, 3, 3]]).transpose()
        q_m2 = np.vstack([Q_g_all_sv[:, 6, 6], Q_g_all_sv[:, 7, 7]]).transpose()
        cos_a2, sin_a2, var_cos_a2, var_sin_a2 = compute_angle_between_matrix(m1, m2, q_m1, q_m2)

        # Merge
        cos_a = np.mean([cos_a1, cos_a2])
        sin_a = np.mean([sin_a1, sin_a2])
        var_cos_a = (var_cos_a1 + var_cos_a2)/4
        var_sin_a = (var_sin_a1 + var_sin_a2)/4

        a = np.arctan2(sin_a, cos_a)
        var_a = utils.cov_arctan2(sin_a, cos_a, var_sin_a, var_cos_a)

    ## Correct angle bias
    g_rover_x_corrected = - g_rover[:, 0] + cos_a * g_rover[:, 2] - sin_a * g_rover[:, 3]
    g_rover_y_corrected = - g_rover[:, 1] + cos_a * g_rover[:, 3] + sin_a * g_rover[:, 2]
    g_rover_corrected = np.column_stack([g_rover_x_corrected, g_rover_y_corrected])

    var_g_rover_x_corrected = (var_g_rover[:, 0]
                             + var_cos_a * (g_rover[:, 2]**2) + var_g_rover[:, 2] * cos_a**2 + var_cos_a * var_g_rover[:, 2]
                             + var_sin_a * (g_rover[:, 3]**2) + var_g_rover[:, 3] * sin_a**2 + var_sin_a * var_g_rover[:, 3])
    var_g_rover_y_corrected = (var_g_rover[:, 1]
                             + var_cos_a * (g_rover[:, 3]**2) + var_g_rover[:, 3] * cos_a**2 + var_cos_a * var_g_rover[:, 3]
                             + var_sin_a * (g_rover[:, 2]**2) + var_g_rover[:, 2] * sin_a**2 + var_sin_a * var_g_rover[:, 2])
    var_g_rover_corrected = np.column_stack([var_g_rover_x_corrected, var_g_rover_y_corrected])

    return g_rover_corrected, var_g_rover_corrected, a, var_a

def compute_angle_between_matrix(M1: np.ndarray, M2: np.ndarray, Q_m1:np.ndarray, Q_m2:np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mi = [[x_1i, y_1i],
          [x_2i, y_2i],
              ...
          [x_ni, y_ni]]

    M2 = M1 @ [[cos(alpha), -sin(alpha)],
               [sin(alpha), cos(alpha)]]

    :param M1:
    :param M2:
    :param Q_m1:
    :param Q_m2:
    :return:
    """
    angle_matrix, Q_angle_matrix = utils.wtls_2D(Y=M2, Q_y=Q_m2, G=M1, Q_g=Q_m1)

    U, S, Vt = np.linalg.svd(angle_matrix)
    R = U @ Vt
    # Correction pour garantir det(R) = +1 (pas une réflexion)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    cos_angle = R[0, 0]
    sin_angle = R[1, 0]

    var_cos = np.max([Q_angle_matrix[0, 0, 0], Q_angle_matrix[1, 1, 1]])
    var_sin = np.max([Q_angle_matrix[0, 1, 1], Q_angle_matrix[1, 0, 0]])

    return cos_angle, sin_angle, var_cos, var_sin

def reduce_g_rover(prd: np.ndarray, var_prd: np.ndarray, g_rover: np.ndarray, var_g_rover: np.ndarray,
                   elapsed_time: np.ndarray, constellation_table: np.ndarray,
                   gnss_table: dict = DEFAULT_GNSS_TABLE, sv_model: bool = False):
    corr_prd, Q_corr_prd, clock_bias, Q_clock_bias = (
        clock_bias_correction(prd=prd, var_prd=var_prd, g_rover=g_rover, var_g_rover=var_g_rover,
                              elapsed_time=elapsed_time, constellation_table=constellation_table, gnss_table=gnss_table,
                              sv_model=sv_model))

    reduced_g_rover, Q_reduced_g_rover, angle_bias, Q_angle_bias = (
        angle_bias_correction(prd=corr_prd, Q_prd=Q_corr_prd, g_rover=g_rover, var_g_rover=var_g_rover,
                              elapsed_time=elapsed_time, constellation_table=constellation_table, gnss_table=gnss_table,
                              sv_model=sv_model))

    return reduced_g_rover, Q_reduced_g_rover, corr_prd, Q_corr_prd, clock_bias, Q_clock_bias, angle_bias, Q_angle_bias


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
    reduced_g_rover, Q_reduced_g_rover, corr_prd, Q_corr_prd, clock_bias, Q_clock_bias, angle_bias, Q_angle_bias = (
        reduce_g_rover(prd=prd, var_prd=var_prd, g_rover=g_rover, var_g_rover=var_g_rover, elapsed_time=elapsed_time,
                       constellation_table=constellation_table, gnss_table=gnss_table, sv_model=sv_model))


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

    prd_df, var_prd_df = get_between_epoch_prd(df, meas_col=meas_col, sigma_meas_col="sigma_prd")

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


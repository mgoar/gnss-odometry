from functools import partial
from typing import List, Optional

import gtsam
import numpy as np
import pymap3d as pm
import pymap3d.vincenty as pmv
import scipy.optimize
from tqdm.auto import tqdm
import pandas as pd

import constants


def satellite_selection(df, column):
    """
    Args:
        df : DataFrame from device_gnss.csv
        column : Column name
    Returns:
        df: DataFrame with eliminated satellite signals
    """
    idx = df[column].notnull()
    idx &= df['CarrierErrorHz'] < 2.0e6  # carrier frequency error (Hz)
    idx &= df['SvElevationDegrees'] > 10.0  # elevation angle (deg)
    idx &= df['Cn0DbHz'] > 15.0  # C/N0 (dB-Hz)
    idx &= df['MultipathIndicator'] == 0  # Multipath flag

    return df[idx]


# Compute line-of-sight vector from user to satellite
def los_vector(xusr, xsat):
    """
    Args:
        xusr : user position in ECEF (m)
        xsat : satellite position in ECEF (m)
    Returns:
        u: unit line-of-sight vector in ECEF (m)
        rng: distance between user and satellite (m)
    """
    u = xsat - xusr
    rng = np.linalg.norm(u, axis=1).reshape(-1, 1)
    u /= rng

    return u, rng.reshape(-1)


# Compute Jacobian matrix
def jac_pr_residuals(x, xsat, pr, W):
    """
    Args:
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        pr : pseudorange (m)
        W : weight matrix
    Returns:
        W*J : Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(pr), 1])])  # J = [-ux -uy -uz 1]

    return W @ J


# Compute pseudorange residuals
def pr_residuals(x, xsat, pr, W):
    """
    Args:
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        pr : pseudorange (m)
        W : weight matrix
    Returns:
        residuals*W : pseudorange residuals
    """
    _, rng = los_vector(x[:3], xsat)

    # Approximate correction of the earth rotation (Sagnac effect) often used in GNSS positioning
    rng += constants.OmegaE * \
        (xsat[:, 0] * x[1] - xsat[:, 1] * x[0]) / constants.C

    # Add GPS L1 clock offset
    residuals = rng - (pr - x[3])

    return residuals @ W


# Compute Jacobian matrix
def jac_prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        W*J : Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(prr), 1])])

    return W @ J


# Compute pseudorange rate residuals
def prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        residuals*W : pseudorange rate residuals
    """
    u, rng = los_vector(x[:3], xsat)
    rate = np.sum((vsat-v[:3])*u, axis=1) \
        + constants.OmegaE / constants.C * (vsat[:, 1] * x[0] + xsat[:, 1] * v[0]
                                            - vsat[:, 0] * x[1] - xsat[:, 0] * v[1])

    residuals = rate - (prr - v[3])

    return residuals @ W


# Compute distance by Vincenty's formulae
def vincenty_distance(llh1, llh2):
    """
    Args:
        llh1 : [latitude,longitude] (deg)
        llh2 : [latitude,longitude] (deg)
    Returns:
        d : distance between llh1 and llh2 (m)
    """
    d, az = np.array(pmv.vdist(llh1[:, 0], llh1[:, 1], llh2[:, 0], llh2[:, 1]))

    return d


# Carrier smoothing of pseudarange
def carrier_smoothing(gnss_df):
    """
    Args:
        df : DataFrame from device_gnss.csv
    Returns:
        df: DataFrame with carrier-smoothing pseudorange 'pr_smooth'
    """
    carr_th = 1.5  # carrier phase jump threshold [m] ** 2.0 -> 1.5 **
    pr_th = 20.0  # pseudorange jump threshold [m]

    prsmooth = np.full_like(gnss_df['RawPseudorangeMeters'], np.nan)
    # Loop for each signal
    for (i, (svid_sigtype, df)) in enumerate((gnss_df.groupby(['Svid', 'SignalType']))):
        df = df.replace(
            {'AccumulatedDeltaRangeMeters': {0: np.nan}})  # 0 to NaN

        # Compare time difference between pseudorange/carrier with Doppler
        drng1 = df['AccumulatedDeltaRangeMeters'].diff(
        ) - df['PseudorangeRateMetersPerSecond']
        drng2 = df['RawPseudorangeMeters'].diff(
        ) - df['PseudorangeRateMetersPerSecond']

        # Check cycle-slip
        slip1 = (df['AccumulatedDeltaRangeState'].to_numpy()
                 & 2**1) != 0  # reset flag
        slip2 = (df['AccumulatedDeltaRangeState'].to_numpy()
                 & 2**2) != 0  # cycle-slip flag
        slip3 = np.fabs(drng1.to_numpy()) > carr_th  # Carrier phase jump
        slip4 = np.fabs(drng2.to_numpy()) > pr_th  # Pseudorange jump

        idx_slip = slip1 | slip2 | slip3 | slip4
        idx_slip[0] = True

        # groups with continuous carrier phase tracking
        df['group_slip'] = np.cumsum(idx_slip)

        # Psudorange - carrier phase
        df['dpc'] = df['RawPseudorangeMeters'] - \
            df['AccumulatedDeltaRangeMeters']

        # Absolute distance bias of carrier phase
        meandpc = df.groupby('group_slip')['dpc'].mean()
        df = df.merge(meandpc, on='group_slip', suffixes=('', '_Mean'))

        # Index of original gnss_df
        idx = (gnss_df['Svid'] == svid_sigtype[0]) & (
            gnss_df['SignalType'] == svid_sigtype[1])

        # Carrier phase + bias
        prsmooth[idx] = df['AccumulatedDeltaRangeMeters'] + df['dpc_Mean']

    # If carrier smoothing is not possible, use original pseudorange
    idx_nan = np.isnan(prsmooth)
    prsmooth[idx_nan] = gnss_df['RawPseudorangeMeters'][idx_nan]
    gnss_df['pr_smooth'] = prsmooth

    return gnss_df


# Compute score
def calc_score(llh, llh_gt):
    """
    Args:
        llh : [latitude,longitude] (deg)
        llh_gt : [latitude,longitude] (deg)
    Returns:
        score : (m)
    """
    d = vincenty_distance(llh, llh_gt)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])

    return score


# GNSS single point positioning using pseudorange
def point_positioning(gnss_df):
    # Add nominal frequency to each signal
    # Note: GLONASS is an FDMA signal, so each satellite has a different frequency
    CarrierFrequencyHzRef = gnss_df.groupby(['Svid', 'SignalType'])[
        'CarrierFrequencyHz'].median()
    gnss_df = gnss_df.merge(CarrierFrequencyHzRef, how='left', on=[
                            'Svid', 'SignalType'], suffixes=('', 'Ref'))
    gnss_df['CarrierErrorHz'] = np.abs(
        (gnss_df['CarrierFrequencyHz'] - gnss_df['CarrierFrequencyHzRef']))

    # Carrier smoothing
    gnss_df = carrier_smoothing(gnss_df)

    # GNSS single point positioning
    utcTimeMillis = gnss_df['utcTimeMillis'].unique()
    nepoch = len(utcTimeMillis)
    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    x_wls = np.full([nepoch, 3], np.nan)  # For saving position
    v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity
    cov_x = np.full([nepoch, 3, 3], np.nan)  # For saving position covariance
    cov_v = np.full([nepoch, 3, 3], np.nan)  # For saving velocity covariance

    # Loop for epochs
    for i, (t_utc, df) in enumerate(tqdm(gnss_df.groupby('utcTimeMillis'), total=nepoch)):
        # Valid satellite selection
        df_pr = satellite_selection(df, 'pr_smooth')
        df_prr = satellite_selection(df, 'PseudorangeRateMetersPerSecond')

        # Corrected pseudorange/pseudorange rate
        pr = (df_pr['pr_smooth'] + df_pr['SvClockBiasMeters'] - df_pr['IsrbMeters'] -
              df_pr['IonosphericDelayMeters'] - df_pr['TroposphericDelayMeters']).to_numpy()
        prr = (df_prr['PseudorangeRateMetersPerSecond'] +
               df_prr['SvClockDriftMetersPerSecond']).to_numpy()

        # Satellite position/velocity
        xsat_pr = df_pr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                         'SvPositionZEcefMeters']].to_numpy()
        xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                           'SvPositionZEcefMeters']].to_numpy()
        vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                       'SvVelocityZEcefMetersPerSecond']].to_numpy()

        # Weight matrix for peseudorange/pseudorange rate
        Wx = np.diag(1 / df_pr['RawPseudorangeUncertaintyMeters'].to_numpy())
        Wv = np.diag(
            1 / df_prr['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

        # Robust WLS requires accurate initial values for convergence,
        # so perform normal WLS for the first time
        if len(df_pr) >= 4:
            # Normal WLS
            if np.all(x0 == 0):
                opt = scipy.optimize.least_squares(
                    pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
                x0 = opt.x
            # Robust WLS for position estimation
            opt = scipy.optimize.least_squares(
                pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')
            if opt.status < 1 or opt.status == 2:
                print(f'i = {i} position lsq status = {opt.status}')
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wx @ opt.jac)
                cov_x[i, :, :] = cov[:3, :3]
                x_wls[i, :] = opt.x[:3]
                x0 = opt.x

        # Velocity estimation
        if len(df_prr) >= 4:
            if np.all(v0 == 0):  # Normal WLS
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x
            # Robust WLS for velocity estimation
            opt = scipy.optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
            if opt.status < 1:
                print(f'i = {i} velocity lsq status = {opt.status}')
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wv @ opt.jac)
                cov_v[i, :, :] = cov[:3, :3]
                v_wls[i, :] = opt.x[:3]
                v0 = opt.x

    return utcTimeMillis, x_wls, v_wls, cov_x, cov_v


def tdcp_factor_error(measurement: np.ndarray, this: gtsam.CustomFactor,
                      values: gtsam.Values,
                      jacobians: Optional[List[np.ndarray]]) -> float:
    """TDCP factor error function
    :param measurement: TDCP measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values_x: gtsam.Values (state nodes)
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    key1 = this.keys()[0]
    key2 = this.keys()[1]
    key3 = this.keys()[2]
    key4 = this.keys()[3]
    pos1, pos2, b1, b2 = values.atVector(key1), values.atVector(
        key2), values.atVector(key3), values.atVector(key4)

    u, _ = los_vector(np.array([measurement[2:5]]),
                      np.array([measurement[8:]]))

    Htk = np.hstack((np.squeeze(u), 1e-3))

    error = np.matmul(Htk, (pos2 - pos1)) - (
        (measurement[1]-measurement[0]) - constants.C / constants.L1 * (b2 - b1))
    if jacobians is not None:
        jacobians[0] = np.array([np.squeeze(u)[0]])
        jacobians[1] = np.array([np.squeeze(u)[1]])
        jacobians[2] = np.array([np.squeeze(u)[2]])
        jacobians[3] = np.array([1])

    return error


def relcycleslip_factor_error(measurement: np.ndarray, this: gtsam.CustomFactor,
                              values: gtsam.Values,
                              jacobians: Optional[List[np.ndarray]]) -> float:
    """Relative cycle slip factor error function
    :param measurement: TDCP measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    key1 = this.keys()[0]
    key2 = this.keys()[1]
    pos1, pos2 = values.atVector(key1), values.atVector(key2)

    error = (pos2 - pos1)
    if jacobians is not None:
        jacobians[0] = np.eye(1)
        jacobians[1] = -np.eye(1)

    return error


def process(z, userEcef, satEcef, svids, epochs, sigma_tdcp, sigma_relcycleslip, state_anchor, cycle_slip_anchor):

    unknown_x = [gtsam.symbol('x', t) for t in epochs]

    shape = (len(epochs), len(svids))

    unknown_B = np.array([gtsam.symbol('B', i)
                         for i in np.arange(len(epochs)*len(svids))])

    # We now can use nonlinear factor graphs
    factor_graph = gtsam.NonlinearFactorGraph()

    # Add factors for TDCP measurements
    tdcp_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma_tdcp)

    # Add the TDCP factors (deltat = 1)
    for i, (t1, t2) in enumerate(zip(epochs[0:-1], epochs[1:])):
        for j, k in enumerate(svids):
            factor = gtsam.CustomFactor(tdcp_model, [unknown_x[i], unknown_x[i + 1], unknown_B[np.ravel_multi_index(
                [[i], [j]], shape)], unknown_B[np.ravel_multi_index(
                    [[i+1], [j]], shape)]],
                partial(tdcp_factor_error, np.hstack((z.loc[(t1, k)]['tdcp'], z.loc[(t2, k)]['tdcp'], userEcef.loc[t2].values, satEcef.loc[(t1, k)].values, satEcef.loc[(t2, k)].values))))
            factor_graph.add(factor)

    # Add the TDCP factors (deltat > 1)
    for i, (t0, t1, t2) in enumerate(zip(epochs[0:], epochs[1:-1], epochs[2:])):
        for j, k in enumerate(svids):
            factor = gtsam.CustomFactor(tdcp_model, [unknown_x[i], unknown_x[i + 2], unknown_B[np.ravel_multi_index(
                [[i], [j]], shape)], unknown_B[np.ravel_multi_index(
                    [[i+2], [j]], shape)]],
                partial(tdcp_factor_error, np.hstack((z.loc[(t0, k)]['tdcp'], z.loc[(t2, k)]['tdcp'], userEcef.loc[t2].values, satEcef.loc[(t0, k)].values, satEcef.loc[(t2, k)].values))))
            factor_graph.add(factor)

    relcycleslip_model = gtsam.noiseModel.Isotropic.Sigma(
        1, sigma_relcycleslip)

    # Add the relative cycle slip factors
    for j, k in enumerate(svids):
        for i, (t1, ti) in enumerate(zip(epochs[0:-1], epochs[1:])):
            factor = gtsam.CustomFactor(relcycleslip_model, [unknown_B[np.ravel_multi_index(
                [[i], [j]], shape)], unknown_B[np.ravel_multi_index(
                    [[i+1], [j]], shape)]],
                partial(relcycleslip_factor_error, np.array([cycle_slip_anchor])))
            factor_graph.add(factor)

    # Add state anchor
    factor_graph.add(gtsam.PriorFactorVector(
        unknown_x[0], state_anchor, gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.5, 100]))))

    # Add cycle slip anchor
    for j, k in enumerate(svids):
        factor_graph.add(gtsam.PriorFactorVector(unknown_B[np.ravel_multi_index(
            [[0], [j]], shape)], np.array([cycle_slip_anchor]), gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_relcycleslip]))))

    # New Values container
    v = gtsam.Values()

    # Add initial state estimates
    for i, _ in enumerate(unknown_x):
        v.insert(unknown_x[i], np.hstack((userEcef.reset_index(
        ).iloc[i].values[1:], userEcef.reset_index().iloc[i].values[0])))

    # Add initial relative cycle slip estimates
    for i, _ in enumerate(unknown_B):
        v.insert(unknown_B[i], np.array([cycle_slip_anchor]))

    # Initialize optimizer
    params = gtsam.GaussNewtonParams()
    optimizer = gtsam.GaussNewtonOptimizer(factor_graph, v, params)

    # Optimize the factor graph
    result = optimizer.optimize()

    # Retrieve results
    x_fg = np.array([result.atVector(unknown_x[k])
                    for k in range(len(epochs))])

    B_fg = np.array([result.atVector(unknown_B[k])
                     for k in np.arange(len(epochs)*len(svids))])

    return x_fg, B_fg


if __name__ == "__main__":

    # Load data (GSDC 2022)
    path = 'data/smartphone-decimeter-2022/train/2021-03-16-US-MTV-1/GooglePixel4XL'

    drive, phone = path.split('/')[-2:]

    # Read GNSSdata
    gnss_df = pd.read_csv(f'{path}/device_gnss.csv')

    # Read ground truth (RTK)
    gt_df = pd.read_csv(f'{path}/ground_truth.csv')

    # Point positioning
    utc, x_wls, v_wls, cov_x, cov_v = point_positioning(gnss_df)

    # Convert to latitude and longitude
    llh_wls = np.array(pm.ecef2geodetic(
        x_wls[:, 0], x_wls[:, 1], x_wls[:, 2])).T

    # Baseline
    x_bl = gnss_df.groupby('TimeNanos')[
        ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].mean().to_numpy()
    llh_bl = np.array(pm.ecef2geodetic(
        x_bl[:, 0], x_bl[:, 1], x_bl[:, 2])).T

    # Ground truth
    llh_gt = gt_df[['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()

    # Distance from ground truth
    vd_bl = vincenty_distance(llh_bl, llh_gt)
    vd_wls = vincenty_distance(llh_wls, llh_gt)

    # Score
    score_bl = calc_score(llh_bl, llh_gt)
    score_wls = calc_score(llh_wls, llh_gt)

    print(f'Score Baseline   {score_bl:.4f} [m]')
    print(f'Score Robust WLS {score_wls:.4f} [m]')

    #########################################################################################

    # GPS L1 only
    gps_df = gnss_df.loc[gnss_df['SignalType'] == 'GPS_L1'].set_index(
        ['utcTimeMillis', 'Svid'], append=True)

    # Drop original index
    gps_df.reset_index(0, drop=True, inplace=True)

    # Valid ADR
    gps_df = gps_df.loc[gps_df['AccumulatedDeltaRangeState'] == 21, :]

    gps_df['tdcp'] = gps_df['AccumulatedDeltaRangeMeters'] + gps_df['TroposphericDelayMeters'] - \
        gps_df['IonosphericDelayMeters'] - gps_df['SvClockBiasMeters'] - \
        gps_df['BiasNanos'] * 1E-9 * constants.C

    userEcef = pd.DataFrame(x_wls, index=utc).rename_axis('utcTimeMillis')
    satEcef = gps_df[['SvPositionXEcefMeters',
                      'SvPositionYEcefMeters', 'SvPositionZEcefMeters']]

    svids = [12, 14, 19, 24]
    T = 30
    subset = gps_df.loc[gps_df.index.get_level_values(
        1).isin(svids)].dropna()
    epochs = subset.index.get_level_values(0).unique()[0:T]
    x_fg, B_fg = process(subset.loc[epochs, :], userEcef.loc[epochs], satEcef.loc[epochs], svids,
                         epochs, score_wls/4, 2., np.hstack((userEcef.iloc[0].values, utc[0])), 0)

    # Compute error using ground truth
    gt = gt_df.loc[gt_df["UnixTimeMillis"].isin(
        epochs), ["LatitudeDegrees", "LongitudeDegrees"]].to_numpy()
    llh_fg = np.array(pm.ecef2geodetic(
        x_fg[:, 0], x_fg[:, 1], x_fg[:, 2])).T

    score_fg = calc_score(llh_fg, gt)

    print(f"Score: {score_fg:.4f} [m]")

    import matplotlib.pyplot as plt
    # Cycle slip for each SV
    shape = (len(epochs), len(svids))
    fig, axs = plt.subplots(len(svids), sharex=True)
    for k in range(len(svids)):
        B_t_0 = np.squeeze(
            [B_fg[np.ravel_multi_index([[i], [k]], shape)] for i in range(len(epochs))])
        axs[k].stem(epochs, B_t_0.astype(int),
                    markerfmt=' ', label='SV: {}'.format(svids[k]))
        axs[k].set_yticks(
            np.arange(-30, 30, 10))
        axs[k].legend(loc='upper right')
        axs[k].grid()

    fig.text(0.5, 0.04, 'UTC Time [ms]', ha='center')
    fig.text(0.04, 0.5, '$B_t^k$', va='center', rotation='vertical')

    plt.show()

    # Point positioning
    utc, x_wls, v_wls, cov_x, cov_v = point_positioning(
        gnss_df.loc[gnss_df['utcTimeMillis'].isin(epochs)])

    # Convert to latitude and longitude
    llh_wls = np.array(pm.ecef2geodetic(
        x_wls[:, 0], x_wls[:, 1], x_wls[:, 2])).T

    # Baseline
    x_bl = gnss_df.loc[gnss_df['utcTimeMillis'].isin(epochs)].groupby('TimeNanos')[
        ['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].mean().to_numpy()
    llh_bl = np.array(pm.ecef2geodetic(
        x_bl[:, 0], x_bl[:, 1], x_bl[:, 2])).T

    # Ground truth
    llh_gt = gt_df.loc[gt_df['UnixTimeMillis'].isin(
        epochs)][['LatitudeDegrees', 'LongitudeDegrees']].to_numpy()

    # Distance from ground truth
    vd_bl = vincenty_distance(llh_bl, llh_gt)
    vd_wls = vincenty_distance(llh_wls, llh_gt)

    # Score
    score_bl = calc_score(llh_bl, llh_gt)
    score_wls = calc_score(llh_wls, llh_gt)

    print(f'Score Baseline   {score_bl:.4f} [m]')
    print(f'Score Robust WLS {score_wls:.4f} [m]')

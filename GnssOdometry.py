from functools import partial
from typing import List, Optional

import gtsam
import numpy as np
import pandas as pd
import pymap3d as pm

import constants
import utils


def tdcp_factor_error(measurement: np.ndarray, this: gtsam.CustomFactor,
                      values: gtsam.Values,
                      jacobians: Optional[List[np.ndarray]]) -> float:
    '''
    Computes TDCP factor error function

            Parameters:
                    measurement (np.ndarray): TDCP measurement, to be filled with `partial`
                    this (gtsam.CustomFactor): handle
                    values (gtsam.Values): state nodes values
                    jacobians (Optional[List[np.ndarray]]): Jacobians

            Returns:
                    error (np.ndarray): unwhitened error (m)
    '''

    key1 = this.keys()[0]
    key2 = this.keys()[1]
    key3 = this.keys()[2]
    key4 = this.keys()[3]
    pos1, pos2, b1, b2 = values.atVector(key1), values.atVector(
        key2), values.atVector(key3), values.atVector(key4)

    u, _ = utils.los_vector(np.array([measurement[2:5]]),
                            np.array([measurement[8:]]))

    Htk = np.hstack((np.squeeze(u), 1e-3))

    error = np.matmul(Htk, (pos2 - pos1)) - (
        (measurement[1]-measurement[0]) - constants.C / constants.L1 * (b2 - b1))
    if jacobians is not None:
        jacobians[0] = -np.array([np.squeeze(u)[0]])
        jacobians[1] = -np.array([np.squeeze(u)[1]])
        jacobians[2] = -np.array([np.squeeze(u)[2]])
        jacobians[3] = -np.array([1])

    return error


def relcycleslip_factor_error(measurement: np.ndarray, this: gtsam.CustomFactor,
                              values: gtsam.Values,
                              jacobians: Optional[List[np.ndarray]]) -> float:
    '''
    Computes relative cycle slip factor error function

            Parameters:
                    measurement (np.ndarray): relative cycle slip measurement, to be filled with `partial`
                    this (gtsam.CustomFactor): handle
                    values (gtsam.Values): relative cycle slip nodes values
                    jacobians (Optional[List[np.ndarray]]): Jacobians

            Returns:
                    error (np.ndarray): unwhitened error (m)
    '''

    key1 = this.keys()[0]
    key2 = this.keys()[1]
    b1, b2 = values.atVector(key1), values.atVector(key2)

    error = (b2 - b1)
    if jacobians is not None:
        jacobians[0] = -np.eye(1)
        jacobians[1] = np.eye(1)

    return error


def process(z, userEcef, satEcef, svids, epochs, sigma_tdcp, sigma_relcycleslip, state_anchor, cycle_slip_anchor):
    '''
    Processes data and performs factor graph optimization

            Parameters:
                    z (pandas.DataFrame): Pandas DataFrame containing GNSS measurements
                    userEcef (np.ndarray): user position in ECEF coordinates (m)
                    satEcef (np.ndarray): satellite position in ECEF coordinates (m)
                    svids (list): list of SV
                    epochs (list): list of epochs (ms)
                    sigma_tdcp (float): uncertainty in TDCP measurements (m)
                    sigma_relcycleslip (float): uncertainty in relative cycle slip measurements (cycles)
                    state_anchor (np.ndarray): state anchor (unary factor) (m)
                    cycle_slip_anchor (float): cycle slip anchor (unary factor) (cycles)

            Returns:
                    x_fg (np.ndarray): array of state variables
                    B_fg (np.ndarray): array of relative cycle slip variables
    '''
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
    utc, x_wls, v_wls, cov_x, cov_v = utils.point_positioning(gnss_df)

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
    vd_bl = utils.vincenty_distance(llh_bl, llh_gt)
    vd_wls = utils.vincenty_distance(llh_wls, llh_gt)

    # Score
    score_bl = utils.calc_score(llh_bl, llh_gt)
    score_wls = utils.calc_score(llh_wls, llh_gt)

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

    score_fg = utils.calc_score(llh_fg, gt)

    print(f"Score factor graph optimization: {score_fg:.4f} [m]")

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
    utc, x_wls, v_wls, cov_x, cov_v = utils.point_positioning(
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
    vd_bl = utils.vincenty_distance(llh_bl, llh_gt)
    vd_wls = utils.vincenty_distance(llh_wls, llh_gt)

    # Score
    score_bl = utils.calc_score(llh_bl, llh_gt)
    score_wls = utils.calc_score(llh_wls, llh_gt)

    print(f'Score Baseline   {score_bl:.4f} [m]')
    print(f'Score Robust WLS {score_wls:.4f} [m]')

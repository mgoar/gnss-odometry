import numpy as np
import pymap3d.vincenty as pmv
import scipy.optimize
from tqdm.auto import tqdm

import constants

""" utils.py module. Adapted from: https://www.kaggle.com/code/taroz1461/carrier-smoothing-robust-wls-kalman-smoother/notebook """


def satellite_selection(df, column):
    '''
    Returns a dataframe with a satellite selection

            Parameters:
                    df (pandas.DataFrame): Pandas dataframe
                    column (str): Pandas label (column) to filter

            Returns:
                    df (pandas.DataFrame): DataFrame with eliminated satellite signals
    '''

    idx = df[column].notnull()
    idx &= df['CarrierErrorHz'] < 2.0e6  # carrier frequency error (Hz)
    idx &= df['SvElevationDegrees'] > 10.0  # elevation angle (deg)
    idx &= df['Cn0DbHz'] > 15.0  # C/N0 (dB-Hz)
    idx &= df['MultipathIndicator'] == 0  # Multipath flag

    return df[idx]


def los_vector(x_u, x_S):
    '''
    Computes line-of-sight (LOS) vector from user to satellite

            Parameters:
                    x_u (np.ndarray): user position in ECEF coordinates (m)
                    x_S (np.ndarray): satellite position in ECEF coordinates (m)

            Returns:
                    u (np.ndarray): LOS unit vector in ECEF coordinates (m)
                    rng (float): distance between user and satellite (m)
    '''

    u = x_S - x_u

    # Compute norm
    rng = np.linalg.norm(u, axis=1).reshape(-1, 1)

    # Normalize
    u /= rng

    return u, rng.reshape(-1)


def jac_pr_residuals(x, x_S, pr, W):
    '''
    Computes Jacobian matrix of using pseudorange residuals

            Parameters:
                    x (np.ndarray): user position in ECEF coordinates (m)
                    x_S (np.ndarray): satellite position in ECEF coordinates (m)
                    pr (float): pseudorange (m)
                    W (np.ndarray): weight matrix

            Returns:
                    W*JG (np.ndarray): Jacobian matrix
    '''
    u, _ = los_vector(x[:3], x_S)
    J = np.hstack([-u, np.ones([len(pr), 1])])  # J = [-ux -uy -uz 1]

    return W @ J


def jac_prr_residuals(v, v_S, prr, x, x_S, W):
    '''
    Computes Jacobian matrix of using pseudorange rate residuals

            Parameters:
                    v (np.ndarray): user velocity in ECEF coordinates (m/s)
                    v_S (np.ndarray): satellite velocity in ECEF coordinates (m/s)
                    x (np.ndarray): user position in ECEF coordinates (m)
                    x_S (np.ndarray): satellite position in ECEF coordinates (m)
                    prr (float): pseudorange range (m/s)
                    W (np.ndarray): weight matrix

            Returns:
                    W*JG (np.ndarray): Jacobian matrix
    '''
    u, _ = los_vector(x[:3], x_S)
    J = np.hstack([-u, np.ones([len(prr), 1])])  # J = [-ux -uy -uz 1]

    return W @ J


def pr_residuals(x, x_S, pr, W):
    '''
    Computes pseudorange residuals

            Parameters:
                    x (np.ndarray): user position in ECEF coordinates (m)
                    x_S (np.ndarray): satellite position in ECEF coordinates (m)
                    pr (float): pseudorange (m)
                    W (np.ndarray): weight matrix

            Returns:
                    residuals*W (np.ndarray): pseudorange residuals
    '''
    _, rng = los_vector(x[:3], x_S)

    # Approximate correction of the earth rotation (Sagnac effect) often used in GNSS positioning
    rng += constants.OmegaE * \
        (x_S[:, 0] * x[1] - x_S[:, 1] * x[0]) / constants.C

    # Add GPS L1 clock offset
    residuals = rng - (pr - x[3])

    return residuals @ W


def prr_residuals(v, v_S, prr, x, xsat, W):
    '''
    Computes pseudorange rate residuals

            Parameters:
                    v (np.ndarray): user velocity in ECEF coordinates (m/s)
                    v_S (np.ndarray): satellite velocity in ECEF coordinates (m/s)
                    prr (float): pseudorange rate (m/s)
                    x (np.ndarray): user position in ECEF coordinates (m)
                    x_S (np.ndarray): satellite position in ECEF coordinates (m)
                    W (np.ndarray): weight matrix

            Returns:
                    residuals*W (np.ndarray): pseudorange residuals
    '''

    u, rng = los_vector(x[:3], xsat)
    rate = np.sum((v_S-v[:3])*u, axis=1) \
        + constants.OmegaE / constants.C * (v_S[:, 1] * x[0] + xsat[:, 1] * v[0]
                                            - v_S[:, 0] * x[1] - xsat[:, 0] * v[1])

    residuals = rate - (prr - v[3])

    return residuals @ W


def vincenty_distance(llh1, llh2):
    '''
    Computes Vincenty's distance between llh1 and llh2

            Parameters:
                    llh1 (np.ndarray): [latitude,longitude] (deg)[latitude,longitude] (deg)
                    llh2 (np.ndarray): [latitude,longitude] (deg)[latitude,longitude] (deg)

            Returns:
                    d (np.ndarray): distance between llh1 and llh2 (m)
    '''

    d, az = np.array(pmv.vdist(llh1[:, 0], llh1[:, 1], llh2[:, 0], llh2[:, 1]))

    return d


# Carrier smoothing of pseudarange
def carrier_smoothing(gnss_df):
    '''
    Computes pseudorange smoothing using carrier phase. See https://gssc.esa.int/navipedia/index.php/Carrier-smoothing_of_code_pseudoranges

            Parameters:
                    gnss_df (pandas.DataFrame): Pandas DataFrame containing pseudorange and carrier phase measurements

            Returns:
                    df (pandas.DataFrame): Pandas DataFrame with carrier-smoothing pseudorange 'pr_smooth'
    '''

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


def calc_score(llh, llh_gt):
    '''
    Computes distance score between llh and llh_gt (ground truth)

            Parameters:
                    llh (np.ndarray): [latitude,longitude] (deg)[latitude,longitude] (deg)
                    llh_gt (np.ndarray): [latitude,longitude] (deg)[latitude,longitude] (deg)

            Returns:
                    score (float): score (m)
    '''

    d = vincenty_distance(llh, llh_gt)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])

    return score


def point_positioning(gnss_df):
    '''
    Computes GNSS single point positioning using pseudorange

            Parameters:
                    gnss_df(pandas.DataFrame): Pandas DataFrame with pseudorange measurements

            Returns:
                    utcTimeMillis (np.ndarray): WLS UTC time (ms)
                    x_wls (np.ndarray): WLS position in ECEF coordinates (m)
                    v_wls (np.ndarray): WLS velocity in ECEF coordinates (m)
                    cov_x (np.ndarray): covariance associated to WLS position solution (m)
                    cov_v (np.ndarray): covariance associated to WLS velocity solution (m)
    '''
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

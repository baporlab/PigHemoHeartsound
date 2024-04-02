from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks


@dataclass
class Ecg_feat:
    idx_r_peak: np.ndarray
    hr: np.ndarray


@dataclass
class Hls_feat:
    time_s1: np.ndarray
    time_s2: np.ndarray
    val_s1: np.ndarray
    val_s2: np.ndarray
    intv: np.ndarray


@dataclass
class Abp_feat:
    idx_time_sbp: np.ndarray
    val_sbp: np.ndarray
    idx_time_dbp: np.ndarray
    val_dbp: np.ndarray
    val_pp: np.ndarray
    val_dpdtmax: np.ndarray


@dataclass
class Features:
    time_r_peaks: List[float] = field(default_factory=list)
    value_r_peaks: List[float] = field(default_factory=list)
    time_s1_peaks: List[float] = field(default_factory=list)
    value_s1_peaks: List[float] = field(default_factory=list)
    time_s2_peaks: List[float] = field(default_factory=list)
    value_s2_peaks: List[float] = field(default_factory=list)
    time_sbp: List[float] = field(default_factory=list)
    value_sbp: List[float] = field(default_factory=list)
    time_dbp: List[float] = field(default_factory=list)
    value_dbp: List[float] = field(default_factory=list)
    heart_rate: List[float] = field(default_factory=list)
    s1s2_ratio: List[float] = field(default_factory=list)
    pulse_pressure: List[float] = field(default_factory=list)
    dpdt_max: List[float] = field(default_factory=list)
    interval: List[float] = field(default_factory=list)
    interval_hr: List[float] = field(default_factory=list)


def butter_filter(data, lowcut, highcut, fs, order=2):
    """
    Apply a Butterworth filter to the input data.

    Parameters
    ----------
    data : array_like
        The input data to be filtered.
    lowcut : float
        The lower cutoff frequency of the filter.
    highcut : float
        The upper cutoff frequency of the filter.
    fs : float
        The sampling frequency of the input data.
    order : int, optional
        The order of the filter. Default is 2.

    Returns
    -------
    array_like
        The filtered output data.

    Notes
    -----
    This function applies a Butterworth filter to the input data using the specified cutoff frequencies and order.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import butter, filtfilt
    >>> data = np.random.randn(1000)
    >>> lowcut = 0.1
    >>> highcut = 0.5
    >>> fs = 10.0
    >>> filtered_data = butter_filter(data, lowcut, highcut, fs, order=4)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low == 0 and high == 1:
        return data
    elif low == 0 and high != 1:
        b, a = butter(order, highcut / nyq, btype="low")
    elif low != 0 and high == 1:
        b, a = butter(order, lowcut / nyq, btype="high")
    elif low != 0 and high != 1:
        b, a = butter(order, [low, high], btype="band")
    output = filtfilt(b, a, data)

    return output


def smooth(y, box_pts):
    """Smooth the input signal using a moving average filter.

    Parameters
    ----------
    y : array_like
        The input signal to be smoothed.
    box_pts : int
        The size of the moving average window.

    Returns
    -------
    array_like
        The smoothed signal.

    Notes
    -----
    This function applies a moving average filter to the input signal `y` using a window of size `box_pts`.
    The filter is applied using the `np.convolve` function with the `mode` parameter set to "same",
    which ensures that the output signal has the same length as the input signal.

    Examples
    --------
    >>> y = [1, 2, 3, 4, 5]
    >>> box_pts = 3
    >>> smooth(y, box_pts)
    array([1.        , 2.        , 3.        , 4.        , 3.66666667])
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def heart_filtering(x, sampling_rate, order=2, smtl=200):
    """Apply heart sound filtering to the input signal.

    Parameters
    ----------
    x : array_like
        The input signal to be filtered.
    sampling_rate : int or float
        The sampling rate of the input signal.
    order : int, optional
        The order of the bandpass filter. Default is 2.
    smtl : int, optional
        The smoothing window size. Default is 200.

    Returns
    -------
    array_like
        The filtered signal after heart sound filtering.
    """
    # bandpass filtering
    filtered_data = butter_filter(x, 25, 100, sampling_rate, order=order)

    # hilbert transform and make envelope
    hilbert_heart = hilbert(filtered_data)
    hilbert_heart_env = np.abs(hilbert_heart)
    if sampling_rate != 4000:
        smtl = int(sampling_rate / 20)
    y = smooth(hilbert_heart_env, smtl)

    return y


def calc_ecg_for_pig(
    time, signal, fs, height=None, threshold=None, distance=None
):
    """Calculate ECG for pig.

    This function calculates the ECG (Electrocardiogram) for a pig based on the given time and signal data.
    It uses the find_peaks function to detect the R-peaks in the signal and calculates the heart rate (HR)
    based on the time intervals between the R-peaks.

    Parameters
    ----------
    time : array_like
        The time values corresponding to the signal data.
    signal : array_like
        The ECG signal data.
    fs : float
        The sampling frequency of the signal.
    height : float, optional
        The minimum height of the peaks to be detected. If not specified, all peaks are considered.
    threshold : float, optional
        The minimum relative height of peaks to be detected. If not specified, all peaks are considered.
    distance : float, optional
        The minimum distance between peaks to be detected. If not specified, all peaks are considered.

    Returns
    -------
    rpeaks_idx : array_like
        The indices of the R-peaks in the signal.
    hr : array_like
        The heart rate (HR) values calculated based on the time intervals between the R-peaks.
    """
    rpeaks_idx, _ = find_peaks(
        signal, wlen=100, distance=distance, height=height, threshold=threshold
    )
    hr = 60 / np.diff(time[rpeaks_idx])
    hr = np.append(hr, hr[-1])  # To match the array lengths of time and ecg

    return rpeaks_idx, hr


def calc_heart_sound_with_ecg(
    time, signal, fs, rpeaks_idx, param_height=0.1, param_distance=0.2
):
    """Calculate heart sound features from heart sound signal.

    Parameters
    ----------
    time : array_like
        Array of time values corresponding to the heart sound signal.
    signal : array_like
        Array of heart sound signal values.
    fs : float
        Sampling frequency of the heart sound signal.
    rpeaks_idx : array_like
        Array of indices corresponding to the R-peaks in the ECG signal.
    param_height : float, optional
        The minimum height of peaks to be considered, as a fraction of the maximum peak height.
        Defaults to 0.1.
    param_distance : float, optional
        The minimum distance between peaks, as a fraction of the total length of the segment.
        Defaults to 0.2.

    Returns
    -------
    time_s1 : array_like
        Array of time values corresponding to the first heart sound (S1).
    time_s2 : array_like
        Array of time values corresponding to the second heart sound (S2).
    val_s1 : array_like
        Array of values corresponding to the first heart sound (S1).
    val_s2 : array_like
        Array of values corresponding to the second heart sound (S2).
    intv : array_like
        Array of intervals between S1 and S2.

    """
    win_length_find_peaks = int(fs * 0.5)
    time_s1 = []
    time_s2 = []
    val_s1 = []
    val_s2 = []
    intv = []

    for i in range(len(rpeaks_idx)):
        segment_start_idx = rpeaks_idx[i] - int(win_length_find_peaks / 10)
        segment_end_idx = rpeaks_idx[i] + win_length_find_peaks
        try:
            segment_time = time[segment_start_idx:segment_end_idx]
            segment_heart_sound = signal[segment_start_idx:segment_end_idx]
            filtered_heart_sound = heart_filtering(
                segment_heart_sound, fs, order=2
            )

            peaks, _ = find_peaks(
                filtered_heart_sound,
                height=np.max(filtered_heart_sound) * param_height,
                distance=len(segment_time) * param_distance,
            )
            if len(peaks) < 2:
                continue
            else:
                time_s1.append(segment_time[peaks[0]])
                time_s2.append(segment_time[peaks[1]])
                val_s1.append(filtered_heart_sound[peaks[0]])
                val_s2.append(filtered_heart_sound[peaks[1]])
                intv.append(segment_time[peaks[1]] - segment_time[peaks[0]])
        except ValueError:
            continue

    time_s1 = np.hstack(time_s1)
    time_s2 = np.hstack(time_s2)
    val_s1 = np.hstack(val_s1)
    val_s2 = np.hstack(val_s2)
    intv = np.hstack(intv)

    return time_s1, time_s2, val_s1, val_s2, intv


import numpy as np

def calculate_dP_dt_max(abp_waveform, onset_times, sample_rate):
    """Calculate the maximum dp/dt values for each beat in an arterial blood pressure (ABP) waveform.

    Parameters
    ----------
    abp_waveform : numpy.ndarray
        The ABP waveform as a 1-dimensional numpy array.
    onset_times : list
        A list of indices representing the onset times of each beat in the ABP waveform.
    sample_rate : float
        The sample rate (in Hz) of the ABP waveform.

    Returns
    -------
    numpy.ndarray
        An array containing the maximum dp/dt values for each beat in the ABP waveform.
    """
    # Calculate time between samples
    time_between_samples = 1 / sample_rate

    # Initialize an empty array to store the max dp/dt values for each beat
    dp_dt_max = np.zeros(len(onset_times))

    # Calculate the dp/dt for each beat
    for i, onset_index in enumerate(onset_times):
        # Find the indices of the ABP waveform corresponding to the current beat
        start_index = onset_index
        end_index = (
            onset_times[i + 1] - 1
            if i + 1 < len(onset_times)
            else len(abp_waveform)
        )

        # Calculate the first derivative of the ABP waveform for the current beat
        dP_dt = np.gradient(
            abp_waveform[start_index:end_index], time_between_samples
        )

        # Find the maximum dp/dt value for the current beat
        dp_dt_max[i] = np.max(dP_dt)

    return dp_dt_max


def run_wabp(abp, *args):
    """Detects the onset time of each beat in the ABP waveform.

    Parameters
    ----------
    abp : array_like
        The ABP waveform with units of mmHg.

    Returns
    -------
    ndarray
        An array containing the onset sample time of each beat in the ABP waveform.

    Notes
    -----
    This function detects the onset time (in samples) of each beat in the ABP waveform.
    The ABP waveform must have units of mmHg.

    Written by James Sun (xinsun@mit.edu) on Nov 19, 2005. This ABP onset
    detector is adapted from Dr. Wei Zong's wabp.c.

    LICENSE:
    This software is offered freely and without warranty under
    the GNU (v3 or later) public license. See license file for
    more information.
    """

    # Input checks
    varargin = args
    nargin = 1 + len(varargin)
    if nargin != 1:
        return "exactly 1 argment needed"
    try:
        if np.shape(abp)[1] != 1:
            return "Input must be a <nx1> vector"
    except IndexError:
        pass

    ##################################################

    # scale physiologic ABP
    offset = 1600
    scale = 20
    # Araw = abp * scale - offset
    A = abp
    # # LPF
    # A = (signal.lfilter(np.array([1,0,0,0,0,-2,0,0,0,0,1]),
    #             np.array([1,-2,1]), Araw) / 24) + 30
    # A = (A[3:] + offset)/scale # Takes care of 4 sample group delay

    # Slope-sum functionn
    dypos = np.diff(A, prepend=0)
    dypos[dypos <= 0] = 0
    ssf = np.convolve(np.ones(16), dypos)
    ssf = np.concatenate(([0, 0], ssf))

    # Decision rule
    avg0 = (
        np.sum(ssf[:1000]) / 1000
    )  # average of 1st 8 seconds (1000 samples) of SSF
    Threshold0 = 3 * avg0  # initial decision threshold

    # ignoring "learning period" for now
    lockout = 0  # lockout >0 means we are in refractory
    timer = 0
    z = np.zeros(100000)
    counter = 0

    for t in range(50, len(ssf) - 17):
        lockout = lockout - 1
        timer = (
            timer + 1
        )  # Timer used for counting time after previous ABP pulse

        if (lockout < 1) and (
            ssf[t] > (avg0 + 5)
        ):  # Not in refractory and SSF has exceeded threshold here
            timer = 0
            maxSSF = np.max(ssf[t : t + 17])  # Find local max of SSF
            minSSF = np.min(ssf[t - 17 : t])  # Find local min of SSF
            if maxSSF > (minSSF + 10):
                onset = (
                    0.01 * maxSSF
                )  # Onset is at the time in which local SSF just exceeds 0.01*maxSSF

                tt = np.arange(t - 17, t)
                dssf = ssf[tt] - ssf[tt - 1]
                counter = counter + 1
                try:
                    BeatTime = np.where(dssf < onset)[0][-1] + t - 17
                    z[counter] = BeatTime
                except:
                    counter = counter - 1

                Threshold0 = Threshold0 + 0.1 * (
                    maxSSF - Threshold0
                )  # adjust threshold
                avg0 = Threshold0 / 3  # adjust avg

                lockout = 32  # lock so prevent sensing right after detection (refractory period)

        if timer > 312:
            Threshold0 = Threshold0 - 1
            avg0 = Threshold0 / 3

    r = z[z.nonzero()] - 2
    r = r.astype(np.int64)
    return r


def abpfeature(abp, OnsetTimes, Fs):  # if 'Fs' was not set, put -1
    """Extracts features from ABP waveform.

    Parameters
    ----------
    abp : array_like
        ABP waveform signal.
    OnsetTimes : array_like
        Times of onset (in samples).
    Fs : int, optional
        Sampling frequency of the ABP waveform. Default is -1.

    Returns
    -------
    numpy.ndarray
        Beat-to-beat ABP features. The shape of the array is (n, 13), where n is the number of beats.
        Each row represents a beat and contains the following features:
        0 - Time of systole [samples]
        1 - Systolic BP [mmHg]
        2 - Time of diastole [samples]
        3 - Diastolic BP [mmHg]
        4 - Pulse pressure [mmHg]
        5 - Mean pressure [mmHg]
        6 - Beat Period [samples]
        7 - mean_dyneg
        8 - End of systole time (0.3*sqrt(RR) method)
        9 - Area under systole (0.3*sqrt(RR) method)
        10 - End of systole time (1st min-slope method)
        11 - Area under systole (1st min-slope method)
        12 - Pulse [samples]

    Notes
    -----
    - OnsetTimes must be obtained using wabp.m.

    Written by James Sun (xinsun@mit.edu) on Nov 19, 2005.
    Updated by Alistair Johnson, 2014.

    LICENSE:
    This software is offered freely and without warranty under
    the GNU (v3 or later) public license. See license file for
    more information.
    """

    if len(OnsetTimes) < 30:  # don't process anything if too little onsets
        r = []

    # P_sys, P_dias
    if Fs == -1:
        Fs = 125  # Default value of Fs

    Window = int(np.ceil(0.32 * Fs))
    OT = OnsetTimes[0:-1]
    BeatQty = len(OT)

    MinDomain = np.zeros((BeatQty, Window), dtype=int)
    MaxDomain = np.zeros((BeatQty, Window), dtype=int)
    for i in range(0, Window):  # Vectorized version
        MinDomain[:, i] = OT - i
        MaxDomain[:, i] = OT + i

    MinDomain[MinDomain < 1] = 1  # Error protection
    MaxDomain[MaxDomain < 1] = 1
    MinDomain[MinDomain > len(abp)] = len(abp)
    MaxDomain[MaxDomain > len(abp)] = len(abp)

    P_dias = np.min(abp[MinDomain], axis=1)
    Dindex = np.argmin(
        abp[MinDomain], axis=1
    )  # Get lowest value across 'Window' samples before beat onset
    P_sys = np.max(abp[MaxDomain], axis=1)
    Sindex = np.argmax(
        abp[MaxDomain], axis=1
    )  # Get highest value across 'Window' samples after beat onset

    DiasTime_index = np.ravel_multi_index(
        [np.arange(BeatQty, dtype=int).T, Dindex], MinDomain.shape, order="F"
    )  # Map offset indices Dindex to original indices
    SysTime_index = np.ravel_multi_index(
        [np.arange(BeatQty, dtype=int).T, Sindex], MaxDomain.shape, order="F"
    )  # Map offset indices Sindex to original indices

    DiasTime = MinDomain[
        np.unravel_index(DiasTime_index, MinDomain.shape, "F")
    ]
    SysTime = MaxDomain[np.unravel_index(SysTime_index, MaxDomain.shape, "F")]
    #############################################
    # Pulse Pressure [mmHG]
    PP = P_sys - P_dias

    # Beat Period [samples]
    BeatPeriod = np.diff(OnsetTimes)

    # Mean, StdDev, Median Deriv- (noise detector)
    dyneg = np.diff(abp)
    dyneg[dyneg > 0] = 0

    MAP = np.zeros(BeatQty)
    stddev = np.zeros(BeatQty)
    mean_dyneg = np.zeros(BeatQty)

    if OnsetTimes[-1] == np.size(abp):
        OnsetTimes[-1] = np.size(abp) - 1

    for i in range(0, BeatQty):
        interval = abp[int(OnsetTimes[i]) - 1 : int(OnsetTimes[i + 1])]
        MAP[i] = np.mean(interval)
        stddev[i] = np.std(interval)

        dyneg_interval = dyneg[int(OnsetTimes[i]) - 1 : int(OnsetTimes[i + 1])]
        dyneg_interval = np.delete(
            dyneg_interval, np.where(dyneg_interval == 0)
        )
        if np.min(np.shape(dyneg_interval)) == 0:
            dyneg_interval = 0

        mean_dyneg[i] = np.mean(dyneg_interval)

    # Systolic Area calculation using 'first minimum slope' method
    # Systolic Area calculation using 0.3*sqrt(RR)
    RR = BeatPeriod / Fs  # RR time in seconds
    sys_duration = 0.3 * np.sqrt(RR)
    EndOfSys1 = np.round(OT + sys_duration * Fs)
    SysArea1 = localfun_area(abp, OT, EndOfSys1.T, P_dias)

    ##############################################
    # Systolic Area calculation using 'first minimum slope' method
    SlopeWindow = int(np.ceil(0.28 * Fs))
    ST = EndOfSys1  # start 12 samples after P_sys

    if ST[-1] > (len(abp) - SlopeWindow):  # error protection
        ST[-1] = len(abp) - SlopeWindow

    SlopeDomain = np.zeros((BeatQty, SlopeWindow), dtype=int)
    for i in range(0, SlopeWindow):
        SlopeDomain[:, i] = ST + i - 1

    Slope = np.diff(abp[SlopeDomain], axis=1)
    Slope[Slope > 0] = 0  # Set positive slopes to zero

    # MinSlope = np.min(np.abs(Slope), axis = 1)
    index = np.argmin(np.abs(Slope), axis=1)  # Find first positive slope

    sub2ind = np.ravel_multi_index(
        [np.arange(BeatQty, dtype=int), index], SlopeDomain.shape, order="F"
    )
    ind2sub = np.unravel_index(sub2ind, SlopeDomain.shape, order="F")
    EndOfSys2 = SlopeDomain[ind2sub] + 1
    SysArea2 = localfun_area(abp, OT, EndOfSys2, P_dias)
    Pulse = 60.0 / BeatPeriod

    # OUTPUT
    # Ensure that there is no concatenation error by using
    r = np.concatenate(
        (
            SysTime,
            P_sys,
            DiasTime,
            P_dias,
            PP,
            MAP,
            BeatPeriod,
            mean_dyneg,
            EndOfSys1,
            SysArea1,
            EndOfSys2,
            SysArea2,
            Pulse,
        ),
        axis=0,
    )
    r = r.reshape((13, -1))

    return np.transpose(r).astype(int)


def localfun_area(abp, onset, EndSys, P_dias):
    """
    Calculate the systolic area under the ABP curve for each beat.

    Parameters
    ----------
    abp : array_like
        Array of ABP values.
    onset : array_like
        Array of onset indices for each beat.
    EndSys : array_like
        Array of end systole indices for each beat.
    P_dias : float
        Diastolic pressure value.

    Returns
    -------
    array_like
        Array of systolic areas for each beat.

    Notes
    -----
    The function calculates the systolic area under the ABP curve for each beat
    by summing the ABP values between the onset and end systole indices. The
    diastolic area under each systolic interval is subtracted from the total
    area, and the result is divided by 125 to obtain the area in [mmHg*sec].
    """

    BeatQty = len(onset)
    SysArea = np.zeros(BeatQty)
    for i in range(0, BeatQty):
        SysArea[i] = np.sum(
            abp[int(onset[i]) - 1: int(EndSys[i])]
        )  # faster than trapz below
    EndSys = EndSys[:]
    onset = onset[:]
    SysPeriod = EndSys - onset  # force col vectors

    # Time scale and subtract the diastolic area under each systolic interval
    SysArea = (SysArea - (P_dias * SysPeriod)) / 125  # Area [mmHG*sec]

    return SysArea


def jSQI(features, onset, abp):
    """Calculate the JSQI (ABP waveform signal quality index) for each beat in ABP.

    Parameters
    ----------
    features : numpy.ndarray
        Features extracted from ABP using abpfeature.m.
    onset : numpy.ndarray
        Onset times of ABP using wabp.m.
    abp : numpy.ndarray
        Arterial blood pressure waveform (125Hz sampled).

    Returns
    -------
    numpy.ndarray
        SQI (Signal Quality Index) of each beat: 0=good, 1=bad.
        - Col 1: logical OR of cols 2 thru 10
        - Col 2: P not physiologic (<20 or >300 mmHg)
        - Col 3: MAP not physiologic (<30 or >200 mmHg)
        - Col 4: HR not physiologic (<20 or >200 bpm)
        - Col 5: PP not physiologic (<30 mmHg)
        - Col 6: abnormal Psys (beat-to-beat change > 20 mmHg)
        - Col 7: abnormal Pdias (beat-to-beat change > 20 mmHg)
        - Col 8: abnormal period (beat-to-beat change > 1/2 sec)
        - Col 9: abnormal P(onset) (beat-to-beat change > 20 mmHg)
        - Col 10: noisy beat (mean of negative dP < -3)
    float
        Fraction of good beats in ABP.

    Notes
    -----
    - FEATURES must be obtained using abpfeature.m.
    - ONSET must be obtained using wabp.m.

    Written by James Sun (xinsun@mit.edu) on Nov 19, 2005.
    - v2.0 - 1/18/06 - thresholds updated to reduce false positives
    - v3.0 - 2/10/06 - added "..101..." detection - see lines 92-96

    LICENSE:
    This software is offered freely and without warranty under
    the GNU (v3 or later) public license. See license file for
    more information.
    """

    if len(onset) < 30:
        BeatQ = np.array([])
        r = np.array([])

    # thresholds
    rangeP = np.array([20, 300])  # mmHg
    rangeMAP = np.array([30, 200])  # mmHg
    rangeHR = np.array([20, 200])  # bpm
    rangePP = np.array([20, np.inf])  # mmHg

    dPsys = 20
    dPdias = 20
    dPeriod = 62.5
    dPOnset = 20

    noise = -3

    # get ABP features
    Psys = features[:, 1]
    Pdias = features[:, 3]
    PP = features[:, 4]
    MAP = features[:, 5]
    BeatPeriod = features[:, 6]
    mean_dyneg = features[:, 7]
    HR = 60 * 125.0 / BeatPeriod

    # absolute thresholding ( falg unphysiologic beats)

    badP = np.array(
        [(np.where(Pdias < rangeP[0])) or (np.where(Pdias > rangeP[1]))]
    )[0][0]
    badMAP = np.array(
        [(np.where(MAP < rangeMAP[0])) or (np.where(MAP > rangeMAP[1]))]
    )[0][0]
    badHR = np.array(
        [(np.nonzero(HR < rangeHR[0])) or (np.nonzero(HR > rangeHR[1]))]
    )[0][0]
    badPP = np.array([np.nonzero(PP < rangePP[0])])[0][0]

    # first difference thresholding (flag beat-to-beat variations)
    jerkPsys = 1 + np.where(np.abs(np.diff(Psys)) > dPsys)[0]
    jerkPdias = np.where(np.abs(np.diff(Pdias)) > dPdias)[0]
    jerkPeriod = 1 + np.where(np.abs(np.diff(BeatPeriod)) > dPeriod)[0]
    jerkPOnset = np.where(
        np.abs(np.diff(abp[onset.astype(np.int64)])) > dPOnset
    )[0]

    # noise detector
    noisy = np.where(mean_dyneg < noise)

    # SQI final
    bq = np.zeros((len(onset), 10))
    bq[badP, 1] = 1
    bq[badMAP, 2] = 1
    bq[badHR, 3] = 1
    bq[badPP, 4] = 1
    bq[jerkPsys, 5] = 1
    bq[jerkPdias, 6] = 1
    bq[jerkPeriod, 7] = 1
    bq[jerkPOnset, 8] = 1  #### 여기서 문제
    bq[noisy, 9] = 1

    bq[:, 0] = (
        bq[:, 1]
        + bq[:, 2]
        + bq[:, 3]
        + bq[:, 4]
        + bq[:, 5]
        + bq[:, 6]
        + bq[:, 7]
        + bq[:, 8]
        + bq[:, 9]
    )
    bq[:, 0][bq[:, 0] > 1] = 1

    ############################################################
    # make all "...101..." into "...111..."
    y = bq[:, 0]
    y[np.where(np.diff(y, 2) == 2)[0] + 1] = 1
    bq[:, 0] = y
    ############################################################

    logical = np.nonzero(bq)
    bool_bq = np.copy(bq)
    bool_bq[logical] = 1
    BeatQ = bool_bq
    BeatQ = BeatQ.astype(np.int64)

    # fraction of good beats overall
    r = len(np.where(bq[:, 0] == 0)[0]) / len(onset)
    return BeatQ, r


def calc_abp(time_abp, signal_abp, fs):
    """Calculate systolic and diastolic blood pressure from ABP signal.

    Parameters
    ----------
    time_abp : array_like
        Array of time values corresponding to the ABP signal.
    signal_abp : array_like
        Array of ABP signal values.
    fs : float
        Sampling frequency of the ABP signal.

    Returns
    -------
    time_sbp : array_like
        Array of time values corresponding to the systolic blood pressure (SBP) peaks.
    signal_sbp : array_like
        Array of SBP values.
    time_dbp : array_like
        Array of time values corresponding to the diastolic blood pressure (DBP) peaks.
    signal_dbp : array_like
        Array of DBP values.
    signal_pp : array_like
        Array of pulse pressure (PP) values.
    dP_dt_max : array_like
        Array of maximum rate of change of ABP signal (dP/dt) values.
    """
    onset_time = run_wabp(signal_abp)
    feat_abp = abpfeature(signal_abp, onset_time, 125)
    time_sbp = feat_abp[:, 0]
    signal_sbp = feat_abp[:, 1]
    time_dbp = feat_abp[:, 2]
    signal_dbp = feat_abp[:, 3]
    signal_pp = feat_abp[:, 4]

    dP_dt_max = calculate_dP_dt_max(signal_abp, onset_time, fs)
    if len(dP_dt_max) > len(signal_sbp):
        slice_idx = len(signal_sbp) - len(dP_dt_max)
        dP_dt_max = dP_dt_max[:slice_idx]
    elif len(dP_dt_max) < len(signal_sbp):
        slice_idx = len(dP_dt_max) - len(signal_sbp)
        dP_dt_max = dP_dt_max[:slice_idx]

    return time_sbp, signal_sbp, time_dbp, signal_dbp, signal_pp, dP_dt_max


def extract_features(ecg, ecg_feat, hls_feat, abp, abp_feat, time_segment):
    """Extracts features from the given ECG and ABP signals within the specified time segment.

    Parameters
    ----------
    ecg : dict
        Dictionary containing ECG signal data with "time" and "value" keys.
    ecg_feat : object
        Object containing ECG feature data.
    hls_feat : object
        Object containing HLS feature data.
    abp : dict
        Dictionary containing ABP signal data with "time" and "value" keys.
    abp_feat : object
        Object containing ABP feature data.
    time_segment : list
        List containing the start and end time of the segment of interest.

    Returns
    -------
    Features
        Object containing the extracted features.
    """
    # time_segment = [start_time, end_time]
    feat = Features()
    if not time_segment:
        time_segment = [ecg["time"][0], ecg["time"][-1]]
    else:
        print(time_segment)

    for i, _idx_r_peak in enumerate(ecg_feat.idx_r_peak):
        # find r peak in segment of time of interest
        if not (
            ecg["time"][_idx_r_peak] > time_segment[0]
            and ecg["time"][_idx_r_peak] < time_segment[1]
        ):
            continue

        # find first s1 peak time
        idx_ts1 = np.where(
            np.diff(np.sign(ecg["time"][_idx_r_peak] - hls_feat.time_s1))
        )[0]
        if len(idx_ts1) > 1:
            idx_ts1 = np.array([idx_ts1[0]])
        if not idx_ts1:
            continue

        # find time of sbp
        idx_sbp = np.where(
            np.diff(
                np.sign(
                    ecg["time"][_idx_r_peak]
                    - abp["time"][abp_feat.idx_time_sbp]
                )
            )
        )[0]
        if len(idx_sbp) > 1:
            idx_sbp = np.array([idx_sbp[0]])
        if not idx_sbp:
            continue
        idx_dbp = np.where(
            np.diff(
                np.sign(
                    ecg["time"][_idx_r_peak]
                    - abp["time"][abp_feat.idx_time_dbp]
                )
            )
        )[0]
        if len(idx_dbp) > 1:
            idx_dbp = np.array([idx_dbp[0]])
        if not idx_dbp:
            continue

        # outlier removal
        if i > 0:
            if np.abs(ecg_feat.hr[i] - ecg_feat.hr[i - 1]) > 10:
                continue
            if (
                np.abs(
                    abp_feat.val_sbp[idx_sbp] - abp_feat.val_sbp[idx_sbp - 1]
                )
                > 10
            ):
                continue
            if (
                np.abs(
                    abp_feat.val_dbp[idx_dbp] - abp_feat.val_dbp[idx_dbp - 1]
                )
                > 10
            ):
                continue

        feat.time_r_peaks.append(ecg["time"][_idx_r_peak])
        feat.value_r_peaks.append(ecg["value"][_idx_r_peak])
        feat.heart_rate.append(ecg_feat.hr[i])

        feat.time_s1_peaks.append(hls_feat.time_s1[idx_ts1])
        feat.value_s1_peaks.append(hls_feat.val_s1[idx_ts1])
        feat.time_s2_peaks.append(hls_feat.time_s2[idx_ts1])
        feat.value_s2_peaks.append(hls_feat.val_s2[idx_ts1])
        feat.interval.append(hls_feat.intv[idx_ts1])
        feat.interval_hr.append(hls_feat.intv[idx_ts1] * ecg_feat.hr[i])
        feat.s1s2_ratio.append(
            hls_feat.val_s1[idx_ts1] / hls_feat.val_s2[idx_ts1]
        )

        feat.time_sbp.append(abp["time"][abp_feat.idx_time_sbp[idx_sbp]])
        feat.value_sbp.append(abp_feat.val_sbp[idx_sbp])
        feat.time_dbp.append(abp["time"][abp_feat.idx_time_dbp[idx_dbp]])
        feat.value_dbp.append(abp_feat.val_dbp[idx_dbp])
        feat.pulse_pressure.append(abp_feat.val_pp[idx_sbp])
        feat.dpdt_max.append(abp_feat.val_dpdtmax[idx_sbp])

    feat.time_s1_peaks = np.squeeze(feat.time_s1_peaks)
    feat.time_s2_peaks = np.squeeze(feat.time_s2_peaks)
    feat.value_s1_peaks = np.squeeze(feat.value_s1_peaks)
    feat.value_s2_peaks = np.squeeze(feat.value_s2_peaks)
    feat.interval = np.squeeze(feat.interval)
    feat.s1s2_ratio = np.squeeze(feat.s1s2_ratio)

    feat.time_sbp = np.squeeze(feat.time_sbp)
    feat.time_dbp = np.squeeze(feat.time_dbp)
    feat.value_sbp = np.squeeze(feat.value_sbp)
    feat.value_dbp = np.squeeze(feat.value_dbp)
    feat.pulse_pressure = np.squeeze(feat.pulse_pressure)
    feat.dpdt_max = np.squeeze(feat.dpdt_max)

    return feat

import numpy as np
from scipy import signal
from scipy.integrate import simps

def getBandPower(psd, freqs, freq_band, freq_resolution):
    # Find values within the selected freq band
    idx = np.logical_and(freqs >= freq_band[0] , freqs <= freq_band[1])

    # Compute the absolute power by approximating the area under the curve
    bp = simps(psd[:,idx], dx=freq_resolution)        
    return bp

def getBandPower_Pool(data, freq_bands, sfreq, relative=True):
    min_freq = freq_bands[0][0]
    win_len = 2/min_freq
    s_win = sfreq*win_len
    freqs, psd = signal.welch(data, sfreq, nperseg=s_win, detrend='linear')
    
#     freqs, psd = getPSD(data, sfreq, min_freq=freq_bands[0][0])
    freq_resolution = freqs[1]-freqs[0]
    bps = [getBandPower(psd, freqs, freq_band, freq_resolution) for freq_band in freq_bands]
    bps = np.array(bps)
    if relative:
        total_bp = simps(psd, dx=freq_resolution)
        bps /= total_bp
    return bps
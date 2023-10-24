import numpy as np

def power(signal):
    return 1/len(signal) * sum(signal**2)

def SNR_dB(util, noised):
    return 10*np.log10(power(util)/power(noised))
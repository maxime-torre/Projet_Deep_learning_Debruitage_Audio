import numpy as np
from math import log10, sqrt 


def power(signal):
    return 1/len(signal) * sum(signal**2)

def SNRin_dB(util, noised):
    return 10*np.log10(power(util)/power(noised))

def SDRout_dB(util, predicted):
    return 10*np.log10(power(util)/power(predicted - util))

def gain(SNRin, SDRout):
    return SDRout - SNRin
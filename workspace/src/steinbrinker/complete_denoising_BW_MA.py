import numpy as np
import pandas
import wfdb
from matplotlib import pyplot as plt
from tqdm import tqdm

from dictionary_based_learning.detect_QRS import detect_QRS
from dictionary_based_learning.BPDN import BPDN
from dictionary_based_learning.bandwith_filter import bandpass_filter
from dictionary_based_learning.ASMF import asmf, asmf_with_qrs

def MA_denoising(signal, lambda_param=100, thresh=1e-6):

    dictionary = np.load('dictionary_QRS_complete.npy')
    dictionary_QRS = dictionary[:12]
    dictionary_nQRS = dictionary[12]

    signal_length = len(signal)
    denoised = np.empty_like(signal)
    start = 0
    while start + 132 < signal_length:
        stop = start + 132
        signal_clip = signal[start:stop].copy()

        if detect_QRS(signal_clip): # true if qrs is detected
            #find position of the R-Peak and select section it is in
            max_index = np.argmax(np.abs(signal_clip))
            section = (max_index) // 11
            #approximate with corresponding Dictionary and BPDN
            alpha = BPDN(signal_clip, dictionary_QRS[section, :, :], lambda_param=lambda_param)[1]
            alpha[np.abs(alpha) < thresh] = 0
            denoised[start:stop] = dictionary_QRS[section]@alpha
            denoised[start:stop] = asmf_with_qrs(denoised[start:stop], max_index)

        else:                       # no qrs is detected
            alpha = BPDN(signal_clip, dictionary_nQRS, lambda_param=lambda_param)[1]
            alpha[np.abs(alpha) < thresh] = 0
            denoised[start:stop] = dictionary_nQRS@alpha
            denoised[start:stop] = asmf(denoised[start:stop])

        start += 132

    # for last signal clip
    signal_clip = signal[start:].copy()
    remained_length = len(signal_clip)
    if detect_QRS(signal_clip):  # true if qrs is detected
        max_index = np.argmax(np.abs(signal_clip))
        section = (max_index) // 11
        alpha = BPDN(signal_clip, dictionary_QRS[section, :remained_length, :], lambda_param=lambda_param)[1]
        alpha[np.abs(alpha) < thresh] = 0
        denoised[start:] = dictionary_QRS[section, :remained_length, :] @ alpha
        denoised[start:] = asmf_with_qrs(denoised[start:], max_index)

    else:  # no qrs is detecte
        alpha = BPDN(signal_clip, dictionary_nQRS[:remained_length, :], lambda_param=lambda_param)[1]
        alpha[np.abs(alpha) < thresh] = 0
        denoised[start:] = dictionary_nQRS[:remained_length, :] @ alpha
        denoised[start:] = asmf(denoised[start:])

    return denoised


def BW_denoising(signal, output=True, lambda_param=1, thresh=1e-6):
    # The function takes an input vector (signal) that is a 10s ECG sampled at 250Hz i.e. a vector of length 2500. 
    dictionary = np.load('dictionary_BW_real_data.npy')
    # dictionary has the dimensions (2500,100)
    alpha = BPDN(signal, dictionary, lambda_param=lambda_param)[1]
    alpha[np.abs(alpha) < thresh] = 0
    bw_approx = dictionary@alpha
    denoised = signal - bw_approx

    if output:
        plt.plot(signal)
        plt.plot(denoised)
        plt.show()

    return denoised

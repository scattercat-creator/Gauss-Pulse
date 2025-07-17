

# '''
# 1. Find power spectral density of gaussian pulse
# 2. Do a zero equalization of transfer
# 3. Do transfer function times zero equalization filter
# 4. Should get flat thing
# 5. Do transfer function times wiener filter
# '''

import skrf as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

def Create_Gaussian_Pulse(amplitude=1.0, center_time=0.0, standard_deviation=.02, range=10, num_points=1024):
    time = np.linspace(-range, range, num_points)
    gaussian_pulse = amplitude * np.exp(-((time - center_time)**2) / (2 * standard_deviation**2))
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, gaussian_pulse)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Gaussian Pulse (σ = {standard_deviation}s)')
    plt.grid(True)
    plt.show()
    
    return time, gaussian_pulse

def Load_Transfer_Function(filename):
    # load s2p
    network = rf.Network(filename)
    # grab s21 values and frequency
    s21 = network.s[:,1,0]
    frequency = network.f
    return frequency, s21

def Power_Spectral_Density(signal, time):
    X = fft(signal)
    N = len(signal)
    fs = 1/(time[1] - time[0])
    freqs = fftfreq(N, d=1/fs)
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    psd_fft = (np.abs(X[mask])**2) / (fs * N)


"""
Fixed Wiener Deconvolution with Zero Equalization
Authors: Nicholas David-John White, Nathan Campos Pimenten Garcia
"""

import skrf as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft, fftfreq

def Create_Gaussian_Pulse(amplitude=1.0, center_time=0.0, standard_deviation=0.02, range=10, num_points=1024):
    """Create Gaussian pulse - using narrower pulse for visible spectrum"""
    time = np.linspace(-range, range, num_points)
    gaussian_pulse = amplitude * np.exp(-((time - center_time)**2) / (2 * standard_deviation**2))
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, gaussian_pulse)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Gaussian Pulse (σ = {standard_deviation}s)')
    plt.grid(True)
    plt.show()
    
    return time, gaussian_pulse

def Load_Transfer_Function(filename):
    """Load S2P file and return frequency and S21"""
    network = rf.Network(filename)
    s21 = network.s[:,1,0]
    frequency = network.f
    return frequency, s21

def Power_Spectral_Density(signal, time):
    """Calculate PSD using FFT method"""
    X = fft(signal)
    N = len(signal)
    fs = 1/(time[1] - time[0])
    freqs = fftfreq(N, d=1/fs)
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    psd_fft = (np.abs(X[mask])**2) / (fs * N)

    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs_pos, psd_fft)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V²/Hz)')
    plt.title('Power Spectral Density')
    plt.grid(True)
    plt.show()
    
    return freqs_pos, psd_fft

def Zero_Equalization(transfer_function, transfer_frequencies, regularization=0):
    transfer_function = np.array(transfer_function)

    max_tf = np.max(np.abs(transfer_function))
    reg_term = regularization * max_tf
    
    equalizer = 1/(transfer_function + reg_term)
    
    flat_response = equalizer * transfer_function

    plt.figure(figsize=(12, 4))

    plt.subplot(1,2,1)
    plt.plot(transfer_frequencies/1e6, equalizer)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Zero Equalizer')
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(transfer_frequencies/1e6, flat_response)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('TF × Equalizer (Should be Flat)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return equalizer

def Create_Noise(signal, time, noise_std =0.02):
    noise = np.random.normal(0, noise_std, size=signal.shape)
    noisy_signal = signal + noise

    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, 'b-', label='Original Signal')
    plt.plot(time, noisy_signal, 'r--', alpha=0.7, label='Noisy Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Signal with Noise (σ = {noise_std})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return noise, noisy_signal


def Wiener_Filter(transfer_function, transfer_frequencies, psd_signal, psd_signal_freq, psd_noise=None, psd_noise_freq=None):
    # INTERPOLATE
    
    psd_signal = np.interp(transfer_frequencies, psd_signal_freq, psd_signal)

    w_filter = []
    if psd_noise is not None and psd_noise_freq is not None:
        psd_noise = np.interp(transfer_frequencies, psd_noise_freq, psd_noise)
        w_filter = (np.conj(transfer_function) * np.array(psd_signal)) / (np.abs(transfer_function)**2 * psd_signal + psd_noise)
    else:
        w_filter = (np.conj(transfer_function) * np.array(psd_signal)) / (np.abs(transfer_function)**2 * psd_signal)

    # wiener deconvolution formula
    
# Plot results
    plt.figure(figsize=(15, 4))
    
    # Plot Wiener filter
    plt.subplot(1,2,1)
    plt.plot(transfer_frequencies/1e6, 20*np.log10(np.abs(w_filter)))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Wiener Filter W(f)')
    plt.grid(True)
    
    # Plot combined response W(f) * H(f)
    plt.subplot(1,2,2)
    combined = w_filter * transfer_function
    plt.plot(transfer_frequencies/1e6, 20*np.log10(np.abs(transfer_function)), 'b-', label='Original H(f)')
    plt.plot(transfer_frequencies/1e6, 20*np.log10(np.abs(combined)), 'r--', label='H(f) × W(f)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Combined Response')
    plt.legend()
    plt.grid(True)

    
    plt.tight_layout()
    plt.show()

    # return the filter
    return w_filter

def Apply_Filter_to_Signal(signal, time, filter, filter_freqs):
    X = fft(signal)
    N = len(signal)
    fs = 1/(time[1] - time[0])
    freqs = fftfreq(N, d=1/fs)
    filter = np.interp(freqs, filter_freqs, filter)
    filtered = X * filter
    filtered = ifft(filtered)
    plt.figure()
    plt.plot(time, filtered)
    plt.show()


def main():
    # shows gaussian pulse
    time, gauss_pulse = Create_Gaussian_Pulse()

    # takes s2p and grabs gain
    transfer_frequencies, s21 = Load_Transfer_Function('converted_filter.s2p')

    # calculates power spectral density of the original signal
    psd_signal_freqs, psd_signal = Power_Spectral_Density(gauss_pulse, time)

    # creates an equalizer for the transfer function
    equalizer = Zero_Equalization(s21, transfer_frequencies)

    # creates noise and adds it to signal
    noise, noisy_signal = Create_Noise(gauss_pulse, time)

    # grabs the power spectral density of the noise
    psd_noise_freqs, psd_noise = Power_Spectral_Density(noise, time)

    # wiener filter given no noise
    Wiener_Filter(s21, transfer_frequencies, psd_signal, psd_signal_freqs)

    # wiener filter with noise
    w_filter = Wiener_Filter(s21, transfer_frequencies, psd_signal, psd_signal_freqs, psd_noise, psd_noise_freqs)
    Apply_Filter_to_Signal(noisy_signal, time, w_filter, transfer_frequencies)


 
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import skrf as rf

def Power_Spectral_Density(signal, time, name):
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
    plt.title(f'Power Spectral Density of {name}')
    plt.grid(True)
    plt.show()
    
    return freqs_pos, psd_fft

def Load_Transfer_Function(filename):
    """Load S2P file and return frequency and S21"""
    network = rf.Network(filename)
    s21 = network.s[:,1,0]
    frequency = network.f
  # Plot results
    plt.figure()
    
    # Plot Wiener filter
    plt.plot(frequency/1e6, 20*np.log10(np.abs(s21)))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Transfer Function H(f)')
    plt.grid(True)
    plt.show()

    return frequency, s21

def Wiener_Filter(transfer_function, transfer_frequencies, psd_signal, psd_signal_freq, psd_noise=None, psd_noise_freq=None):
    # INTERPOLATE
    
    #psd_signal = np.interp(transfer_frequencies, psd_signal_freq, psd_signal)
#def Wiener_Filter(transfer_function, transfer_frequencies, psd_signal, psd_signal_freq, psd_noise=None, psd_noise_freq=None):
    
    psd_signal_interp = np.interp(transfer_frequencies, psd_signal_freq, psd_signal)
    w_filter = []
    if psd_noise is None:
        w_filter = (np.conj(transfer_function) * psd_signal_interp) / (np.abs(transfer_function)**2 * psd_signal_interp)
        combined = w_filter * transfer_function
    
        
    # Rest of your code...
    if psd_noise is not None and psd_noise_freq is not None:
        psd_noise = np.interp(transfer_frequencies, psd_noise_freq, psd_noise)
        w_filter = (np.conj(transfer_function) * np.array(psd_signal_interp)) / (np.abs(transfer_function)**2 * psd_signal_interp + psd_noise)
    else:
        w_filter = (np.conj(transfer_function) * np.array(psd_signal_interp)) / (np.abs(transfer_function)**2 * psd_signal_interp)

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

def Apply_Filter_to_Signal(signal, time, filter, filter_freqs, name):
    # --- frequency–domain filtering (robust version) ---------------------------
    X = fft(signal)
    N = len(signal)
    fs = 1 / (time[1] - time[0])           # sampling-rate
    freqs = fftfreq(N, d=1/fs)             # ±-Nyquist grid

    # 1)  absolute–value grid for interpolation
    freqs_abs = np.abs(freqs)

    # 2)  interpolate H(f) (or Wiener filter) onto full FFT grid
    #     – left = value at f=0
    #     – right = keep the last measured value (no hard zero)
    filter_interp = np.interp(
            freqs_abs,
            filter_freqs,
            filter,                     # measured / designed filter (complex!)
            left=filter[0],
            right=filter[-1]
    )

    # 3)  optional smooth roll-off beyond last measured frequency
    f_stop = filter_freqs[-1]
    f_max  = freqs_abs.max()
    mask   = freqs_abs > f_stop
    if mask.any():
        taper = 0.5 * (1 + np.cos(np.pi * (freqs_abs[mask] - f_stop) / (f_max - f_stop)))
        filter_interp[mask] *= taper          # cosine taper instead of brick-wall

    # 4)  apply filter
    Y = X * filter_interp
    filtered_signal = ifft(Y)
    filtered_signal = np.real(filtered_signal)        # drop tiny imag component
    filtered_signal -= np.mean(filtered_signal)       # remove DC shift

    filtered_fft = fft(filtered_signal)
    return np.real(filtered_signal)
'''
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1,2,1)
    plt.plot(time*1e9, signal, 'b--', label='Input Signal')
    plt.plot(time*1e9, np.real(filtered_signal), 'r-', label='Filtered Signal')
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title(f'Time Domain: Before and After Filtering Through {name}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(freqs_abs[:N//2]/1e6, 20*np.log10(np.abs(X[:N//2])), 'b--', label='Input Spectrum')
    plt.plot(freqs_abs[:N//2]/1e6, 20*np.log10(np.abs(filtered_fft[:N//2])), 'r-', label='Filtered Spectrum')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Domain: Before and After Filtering')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
# added
    print(f"Before final filtering:")
    print(f"Signal max: {np.max(filtered_signal):.6f}")
    print(f"Signal min: {np.min(filtered_signal):.6f}")
    print(f"Signal mean: {np.mean(filtered_signal):.6f}")
    print(f"Signal std: {np.std(filtered_signal):.6f}")

    # Also check your Wiener filter
    print(f"Wiener filter max: {np.max(np.abs(filter)):.6f}")
    print(f"Wiener filter min: {np.min(np.abs(filter)):.6f}")
# end added
'''
    

def Apply_To_Long_Signal(signal, time, filter, filter_freqs, name):
    sig = signal 
    necessary_length = len(filter)
    plt.figure()
    while (len(sig) > necessary_length):
        current_segment = sig[:necessary_length]
        filtered_segment = Apply_Filter_to_Signal(current_segment, time[:necessary_length], filter, filter_freqs, name)
        sig = sig[necessary_length:]
        # do something with filtered_segment
        plt.plot(time[:necessary_length]*1e9, filtered_segment)
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title(f'Filtered Segments of {name}')
    plt.grid(True)
    plt.show()
    return sig


def main():
    # read a lil bit of the data
    df = pd.read_csv('WCM_4turns.csv', header=None, names=['time', 'amplitude'], usecols=[0, 1], nrows=30000000)
    # plot the entire signal
    plt.figure(figsize=(15, 5))
    plt.plot(df['time'], df['amplitude'])
    plt.show()

    # plot a noisy segment of the signal
    plt.figure(figsize=(15, 5))
    plt.plot(df['time'][5875001:7375001], df['amplitude'][5875001:7375001])
    plt.show()

    noise_segment = df['amplitude'].iloc[5_875_001 : 7_375_001].to_numpy()
    noise_t_seg = df['time'].iloc[5_875_001 : 7_375_001].to_numpy()
    # get the PSD of the noisy segment
    noise_freqs, noise_psds = Power_Spectral_Density(noise_segment, noise_t_seg, 'Noisy Segment')

    # get the PSD of a hill segment
    hill_segment = df['amplitude'].iloc[22_125_001 : 23_625_001].to_numpy()
    hill_t_seg = df['time'].iloc[22_125_001 : 23_625_001].to_numpy()

    signal_freqs, signal_psds = Power_Spectral_Density(hill_segment, hill_t_seg, 'Hill Segment')

    transfer_frequencies, transfer_function = Load_Transfer_Function('converted_filter.s2p')

    w_filter = Wiener_Filter(transfer_function, transfer_frequencies, signal_psds, signal_freqs, noise_psds, noise_freqs)

    Apply_To_Long_Signal(hill_segment, hill_t_seg, w_filter, transfer_frequencies, 'Wiener Filter')

if __name__ == "__main__":
    main()
# goal 1: read s2p data into panda dataframe
import skrf as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# goal: turn s2p file into pandas dataframe

network = rf.Network('filter_time_series.s2p')

# save all parameters in s2p file
freq = network.f
s11 = network.s[:,0,0] # how much signal at port 1 bounces back
s12 = network.s[:,0,1] # gain / loss 
s21 = network.s[:,1,0] # how much signal leaks backward through device
s22 = network.s[:,1,1] # how much signal bounces back from output

# create panda dataframe

df = pd.DataFrame({
    'frequency': freq,
    's11_real': s11.real,
    's11_imag': s11.imag,
    's12_real': s12.real,
    's12_imag': s12.imag,
    's21_real': s21.real,
    's21_imag': s21.imag,
    's22_real': s22.real,
    's22_imag': s22.imag
})

# Goal: plot the gain (s21)

# get rid of points i don't want
df = df.drop(df.index[-75:])

# get the s21 magnitude
s21_magnitude_db = 20 * np.log10(np.sqrt(df['s21_real']**2 + df['s21_imag']**2))

# plot it
plt.figure()
plt.plot(df['frequency']/1e9, s21_magnitude_db)
plt.xlabel('frequency (GHz)')
plt.ylabel('s21 Magnitude (dB)')
plt.show()

# Goal: create a guassian pulse

A = 1.0 # pulse amplitude
FWHM = 200e-9
SIGMA = FWHM / 2.355
t0 = 0

# time array
t = np.linspace(-600e-9,600e-9,1000)

# gaussian pulse
pulse = A * np.exp(-((t - t0) ** 2) / (2 * SIGMA ** 2))

# check it
plt.figure()
plt.plot(t * 1e9, pulse)
plt.xlabel('time (ns)')
plt.ylabel('amplitude')
plt.show()

# Goal: filter the pulse using the data from the Network Analyzer

def FilterSignal(my_signal, s21_real, s21_imagin, frequency, time):
    # get the fft of the pulse
    my_signal_fft = np.fft.fft(my_signal)
    freq_signal = np.fft.fftfreq(len(my_signal), time[1]-time[0])

    # get the phase of s21 in radians and the linear magnitude of s21
    s21_phase = np.arctan2(s21_imagin,s21_real)
    s21_complex = s21_real + 1j * s21_imagin
    s21_magnitude_linear = np.abs(s21_complex)

    # interpolate s21 to get the same number of points as the pulse
    s21_interp = np.interp(freq_signal, frequency, s21_magnitude_linear) * \
        np.exp(1j * np.interp(freq_signal, frequency, s21_phase))

    # filter the pulse
    filtered_fft = my_signal_fft * s21_interp
    filtered_signal = np.fft.ifft(filtered_fft)

    # checks
    plt.figure()
    plt.plot(t, np.abs(filtered_signal))
    plt.show()

    print(f"S21 magnitude range: {s21_magnitude_linear.min():.3f} to {s21_magnitude_linear.max():.3f}")
    print(f"Original pulse max: {np.max(np.abs(my_signal)):.3f}")
    print(f"Filtered pulse max: {np.max(np.abs(filtered_signal)):.3f}")
    
    return filtered_signal

# call filtered signal on pulse
filtered_pulse = FilterSignal(pulse, df['s21_real'], df['s21_imag'], df['frequency'], t)

# Goal: create a FIR-wiener filter

def WienerFIR(time, frequency, s21_real, s21_imaginary, num_taps=128, snr=100):
    # define variables
    N = num_taps
    fs = 1/(time[1] - time[0])
    freq_signal = np.fft.fftfreq(N, 1/fs)
    
    # get the phase of s21 in radians and the linear magnitude of s21
    s21_phase = np.arctan2(s21_imaginary,s21_real)
    s21_complex = s21_real + 1j * s21_imaginary
    s21_magnitude_linear = np.abs(s21_complex)

    # interpolate s21 to get the same number of points as the pulse
    s21_interp = np.interp(freq_signal, frequency, s21_magnitude_linear) * \
        np.exp(1j * np.interp(freq_signal, frequency, s21_phase))
    
    # Goal: create the wiener filter (formula)
    s21_conj = np.conj(s21_interp) # complex conjugate
    s21_power = np.abs(s21_interp)**2  # square conjugate
    noise_power = 1.0 / snr # noise for distortion trade off
    
    H_wiener = s21_conj / (s21_power + noise_power)

    # make symmetric
    H_wiener[N//2:] = np.conj(H_wiener[N//2-1::-1])

    # convert to time domain
    fir_coeffs = np.fft.ifft(H_wiener).real
    fir_coeffs = np.roll(fir_coeffs, N//2)

    # add a window for smoothing
    window = signal.windows.hamming(N)
    fir_coeffs *= window
    
    return fir_coeffs

# get coeffs for pulse
pulse_coeffs = WienerFIR(t, df['frequency'], df['s21_real'], df['s21_imag'])

# get the fully recovered pulse
recovered_pulse = signal.lfilter(pulse_coeffs, 1, filtered_pulse.real)

plt.figure()
plt.plot(t, recovered_pulse)
plt.show()
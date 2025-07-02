# goal 1: read s2p data into panda dataframe
import skrf as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# plot the gain (s21)

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

# create a guassian pulse

A = 1.0 # pulse amplitude
FWHM = 200e-9
SIGMA = FWHM / 2.355
t0 = 0

# time array
t = np.linspace(-600e-9,600e-9,1000)

# gaussian pulse
pulse = A * np.exp(-((t - t0) ** 2) / (2 * SIGMA ** 2))

plt.figure()
plt.plot(t * 1e9, pulse)
plt.xlabel('time (ns)')
plt.ylabel('amplitude')
plt.show()

# filter the pulse using the data

# get the fft of the pulse
pulse_fft = np.fft.fft(pulse)
freq_pulse = np.fft.fftfreq(len(pulse), t[1]-t[0])

# get the phase of s21 in radians and the linear magnitude of s21
s21_phase = np.arctan2(df['s21_imag'],df['s21_real'])
s21_complex = df['s21_real'] + 1j * df['s21_imag']
s21_magnitude_linear = np.abs(s21_complex)

# interpolate s21 to get the same number of points as the pulse
s21_interp = np.interp(freq_pulse, df['frequency'], s21_magnitude_linear) * \
    np.exp(1j * np.interp(freq_pulse, df['frequency'], s21_phase))

# filter the pulse
filtered_fft = pulse_fft * s21_interp
filtered_pulse = np.fft.ifft(filtered_fft)

# checks
plt.figure()
plt.plot(t, np.abs(filtered_pulse))
plt.show()

print(f"S21 magnitude range: {s21_magnitude_linear.min():.3f} to {s21_magnitude_linear.max():.3f}")
print(f"Original pulse max: {np.max(np.abs(pulse)):.3f}")
print(f"Filtered pulse max: {np.max(np.abs(filtered_pulse)):.3f}")



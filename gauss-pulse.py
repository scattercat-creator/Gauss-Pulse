import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
center_time = 0
pulse_width = 0.5
time_range = 5

time = np.linspace(-time_range, time_range, num_points)

gaussian_pulse = np.exp(-((time - center_time) ** 2) / (2 * pulse_width ** 2))

# FFT
sampling_rate = 500
fft_results = np.fft.fft(gaussian_pulse)
freqs = np.fft.fftfreq(num_points, d=1/sampling_rate)

plt.figure()
plt.plot(freqs[:num_points//2], np.abs(fft_results[:num_points//2]))
plt.title('FFT of Gaussian Pulse')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.figure()
plt.plot(time, gaussian_pulse)
plt.title('Gaussian Pulse')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()



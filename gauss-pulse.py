# # import skrf as rf
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt

# # def load_s2p_filter(filename):
# #     # load s2p 
# #     network = rf.Network(filename)
    
# #     # Grab s21 values an frequency
# #     s21 = network.s[:,1,0]
# #     frequency = network.f

# #     return frequency, s21

# # # create a train of gaussian pulses
# # def create_gaussian_pulse_train(duration=2e-6, num_points=2000, fwhm=100e-9, amplitude=1.0, num_pulses=5, spacing=300e-9):
    
# #     t = np.linspace(-duration/2, duration/2, num_points)
# #     sigma = fwhm / 2.355

# #     pulse_train = np.zeros_like(t)
# #     center_start = -((num_pulses - 1) / 2) * spacing  # centers in time window

# #     for i in range(num_pulses):
# #         center = center_start + i * spacing
# #         pulse_train += amplitude * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))

# #     return t, pulse_train

# # # create regular gaussian pulse
# # def create_gaussian_pulse(duration=1200e-9, num_points=1000, fwhm=200e-9, amplitude=1.0):
    
# #     t = np.linspace(-duration/2, duration/2, num_points)

# #     sigma = fwhm / 2.355
    
# #     pulse = amplitude * np.exp(-(t ** 2) / (2 * sigma ** 2))
    
# #     return t, pulse

# # # applies the s2p to a signal
# # def apply_s2p_filter(pulse, time, frequency, s21):
# #     # Get FFT of pulse
# #     pulse_fft = np.fft.fft(pulse)
# #     freq_signal = np.fft.fftfreq(len(pulse), time[1] - time[0]) # look for length of frequ signal <- should be num bins
# #                                                                 # check settings for np.ffft
    
# #     # Interpolate S21 to match signal frequency grid
# #     freq_abs = np.abs(freq_signal)                              # plot both imaginary and real
# #     s21_interp = np.interp(freq_abs, frequency, s21)            # signal
    
# #     # Set to zero beyond S21 frequency range
# #     s21_interp[freq_abs > frequency[-1]] = 0
    
# #     # Apply filter
# #     filtered_fft = pulse_fft * s21_interp
# #     filtered_pulse = np.fft.ifft(filtered_fft)
    
# #     return filtered_pulse

# # def add_noise_to_signal(signal, snr_db):
# #     # Convert SNR from dB to linear scale
# #     snr_linear = 10 ** (snr_db / 10)

# #     # Calculate signal power
# #     signal_power = np.mean(np.abs(signal)**2)

# #     # Calculate noise power
# #     noise_power = signal_power / snr_linear

# #     # Generate white Gaussian noise
# #     noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)

# #     # Add noise to signal
# #     noisy_signal = signal + noise

# #     return noisy_signal 


# # def wiener_deconvolution(filtered_pulse, time, frequency, s21, snr=10):
# #     # Get FFT of filtered signal
# #     filtered_fft = np.fft.fft(filtered_pulse)
# #     freq_signal = np.fft.fftfreq(len(filtered_pulse), time[1] - time[0])
    
# #     # Interpolate S21 to match numpoints of signal
# #     freq_abs = np.abs(freq_signal)
# #     s21_interp = np.interp(freq_abs, frequency, s21)
    
# #     # Set to zero beyond S21 frequency range
# #     s21_interp[freq_abs > frequency[-1]] = 0
    
# #     # Calculate Wiener filter
# #     s21_power = np.abs(s21_interp)**2

# #     noise_power = np.mean(s21_power) / snr
# #     regularization =1e-6 #1e-10
    
# #     # Wiener deconvolution formula
# #     H_wiener = np.conj(s21_interp) / (s21_power + noise_power + regularization) # wrong S, and should grab noise from signal

# #     # Apply Wiener filter
# #     recovered_fft = filtered_fft * H_wiener
# #     recovered_pulse = np.fft.ifft(recovered_fft)
# #     return recovered_pulse
# # #    return recovered_pulse.real

# # def plot_results(time, original_pulse, filtered_pulse, recovered_pulse, frequency, s21):
# #     """Plot all results"""
# #     fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
# #     # Plot 1: S21 magnitude response
# #     axes[0, 0].plot(frequency/1e9, 20*np.log10(np.abs(s21)), 'b-', linewidth=2)
# #     axes[0, 0].set_xlabel('Frequency (GHz)')
# #     axes[0, 0].set_ylabel('S21 Magnitude (dB)')
# #     axes[0, 0].set_title('Filter Response (S21)')
# #     axes[0, 0].grid(True)
    
# #     # Plot 2: Original vs Filtered pulse
# #     axes[0, 1].plot(time*1e9, original_pulse, 'b-', linewidth=2, label='Original Pulse')
# #     axes[0, 1].plot(time*1e9, np.abs(filtered_pulse)/np.max(filtered_pulse), 'r-', linewidth=2, label='Filtered Pulse')
# #     axes[0, 1].set_xlabel('Time (ns)')
# #     axes[0, 1].set_ylabel('Amplitude')
# #     axes[0, 1].set_title('Original vs Filtered Pulse')
# #     axes[0, 1].legend()
# #     axes[0, 1].grid(True)
    
# #     # Plot 3: All three signals
# #     axes[1, 0].plot(time*1e9, original_pulse, 'b-', linewidth=2, label='Original Pulse')
# #     axes[1, 0].plot(time*1e9, np.abs(filtered_pulse), 'r-', linewidth=2, label='Filtered Pulse')
# #     axes[1, 0].plot(time*1e9, recovered_pulse, 'g-', linewidth=2, label='Recovered Pulse')
# #     axes[1, 0].set_xlabel('Time (ns)')
# #     axes[1, 0].set_ylabel('Amplitude')
# #     axes[1, 0].set_title('Complete Process: Original → Filtered → Recovered')
# #     axes[1, 0].legend()
# #     axes[1, 0].grid(True)

# #     # [Time Domain Fidelity]
# #     # Plot 4: Recovery comparison (zoomed) 
# #     axes[1, 1].plot(time*1e9, original_pulse, 'b-', linewidth=2, label='Original Pulse')
# #     axes[1, 1].plot(time*1e9, recovered_pulse, 'g-', linewidth=2, label='Recovered Pulse')
# #     axes[1, 1].set_xlabel('Time (ns)')
# #     axes[1, 1].set_ylabel('Amplitude')
# #     axes[1, 1].set_title('Recovery Quality Check')
# #     axes[1, 1].legend()
# #     axes[1, 1].grid(True)
    
# #     plt.tight_layout()
# #     plt.show()

# # def plot_ffts(pulse, time, recovered, filtered_pulse):
# #     pulse_fft = np.fft.fft(pulse)
# #     freq_signal = np.fft.fftfreq(len(pulse), time[1] - time[0])
# #     recovered_fft = np.fft.fft(recovered)
# #     filtered_pulse_fft = np.fft.fft(filtered_pulse)

# #     n = len(pulse)
# #     freq_signal = freq_signal[:n//2]
# #     pulse_fft = pulse_fft[:n//2]
# #     recovered_fft = recovered_fft[:n//2]
# #     filtered_pulse_fft = filtered_pulse_fft[:n//2]
# #     # FFT of original pulse
# #     fig, axes = plt.subplots(2, 2, figsize=(15, 5))

# #     axes[0,0].plot(freq_signal, np.abs(pulse_fft), 'b-', linewidth=2, label='original pulse fft')
# #     axes[0,0].set_xlabel('frequency')
# #     axes[0,0].set_ylabel('Amplitude')
# #     axes[0,0].set_title('Original FFT')
# #     axes[0,0].grid(True)

# #     # plot conjugate 
    
# #     # FFT of recovered pulse
# #     axes[1,0].plot(freq_signal, np.abs(recovered_fft), 'g-', linewidth=2, label="Recovered pulse fft")
# #     axes[1,0].set_xlabel('frequency')
# #     axes[1,0].set_ylabel('amplitude')
# #     axes[1,0].set_title('Recovered Signal\'s FFT')
# #     axes[1,0].grid(True)

# #     axes[1,1].plot(freq_signal, np.abs(pulse_fft), 'b-', linewidth=2, label="Original pulse fft")
# #     axes[1,1].plot(freq_signal, np.abs(recovered_fft), 'g--', linewidth=2, label="Recovered pulse fft")

# #     axes[1,1].set_xlabel('frequency')
# #     axes[1,1].set_ylabel('amplitude')
# #     axes[1,1].set_title('Recovered FFT vs Original FFT')
# #     axes[1,1].legend()
# #     axes[1,1].grid(True)

# #     axes[0,1].plot(freq_signal, np.abs(pulse_fft), 'b-', linewidth=2, label="Original pulse fft")
# #     axes[0,1].plot(freq_signal, np.abs(recovered_fft), 'g--', linewidth=2, label="Recovered pulse fft")
# #     axes[0,1].plot(freq_signal, np.abs(filtered_pulse_fft), 'r-', linewidth=2, label="Filtered pulse fft")

# #     axes[0,1].set_xlabel('frequency')
# #     axes[0,1].set_ylabel('amplitude')
# #     axes[0,1].set_title('Recovered FFT vs Original FFT')
# #     axes[0,1].legend()
# #     axes[0,1].grid(True)

# #     plt.tight_layout()
# #     plt.show()



# # def main():
# #     # Parameters
# #     S2P_FILENAME = 'converted_filter.s2p' # Can change s2p to use different frequency responses
# #     # PULSE_DURATION = 1e-6       # shorter total duration (optional, helps if you want better time resolution)
# #     # NUM_POINTS = 2**14          # = 16,384 → higher FFT resolution
# #     # PULSE_FWHM = 20e-9           # very sharp pulse → high frequency content
# #     # SPACING = 200e-9 

# #     PULSE_DURATION = 2e-6
# #     NUM_POINTS = 8192
# #     PULSE_FWHM = 1e-9 #50e-9      
# #     SPACING= 400e-9         
# #     NUM_PULSES=3            
# #     PULSE_AMPLITUDE = 1.0    
# #     SNR = 40 #100 # Signal-to-noise ratio for Wiener filter
    

# #     print("=== S2P Filter and Wiener Recovery Program ===\n")
    
# #     # Step 1: Load S2P file
# #     print("1. Loading S2P filter...")
# #     frequency, s21 = load_s2p_filter(S2P_FILENAME)
# #     print(f"Loaded {len(frequency)} frequency points from {frequency[0]/1e9:.2f} to {frequency[-1]/1e9:.2f} GHz")
    
    

# #     # Step 2: Create Gaussian pulse
# #     print("2. Creating Gaussian pulse...")
# #     #time, original_pulse = create_gaussian_pulse(PULSE_DURATION, NUM_POINTS, PULSE_FWHM, PULSE_AMPLITUDE)
# #     time, original_pulse = create_gaussian_pulse_train(PULSE_DURATION, NUM_POINTS, PULSE_FWHM, PULSE_AMPLITUDE, NUM_PULSES, SPACING)

# #     print(f"Created pulse: {PULSE_FWHM*1e9:.0f} ns FWHM, {len(original_pulse)} points")
    
# #     # Step 3: Apply S2P filter
# #     print("3. Applying S2P filter to pulse...")
# #     filtered_pulse = apply_s2p_filter(original_pulse, time, frequency, s21) 

# #     plt.figure(figsize=(15,7))
# #     s21_power = np.abs(s21)**2
# #     snr = SNR
# #     plt.plot(frequency, s21 * ((np.conj(s21) / (s21_power + (np.mean(s21_power) / snr) +10e-9))))
# #     plt.show()
# #     print(f"Original pulse peak: {np.max(original_pulse):.3f}")
# #     print(f"Filtered pulse peak: {np.max(np.abs(filtered_pulse)):.3f}")
    
# #     # add noise here
# #     filtered_pulse = add_noise_to_signal(filtered_pulse, 40)
  
# #     # Step 4: Apply Wiener deconvolution
# #     print("4. Applying Wiener deconvolution...")
# #     recovered_pulse = wiener_deconvolution(filtered_pulse, time, frequency, s21, SNR)
# #     print(f"   Recovered pulse peak: {np.max(np.abs(recovered_pulse)):.3f}")
    
# #     # Step 5: Calculate recovery quality
# #     print("5. Analyzing recovery quality...")
    
# #     # Find pulse centers for alignment
# #     orig_center = np.argmax(original_pulse)
# #     rec_center = np.argmax(np.abs(recovered_pulse))
    
# #     # Calculate correlation coefficient
# #     correlation = np.corrcoef(original_pulse, recovered_pulse)[0, 1]
    
# #     # Calculate RMS error
# #     rms_error = np.sqrt(np.mean((np.abs(original_pulse - filtered_pulse))**2))
# #     print(f'og error: {rms_error}')

# #     rms_error = np.sqrt(np.mean((original_pulse - recovered_pulse)**2))
    
# #     print(f"Correlation coefficient: {correlation:.3f}")
# #     print(f"RMS error: {rms_error:.3f}")
# #     print(f"Peak recovery ratio: {np.max(np.abs(recovered_pulse))/np.max(original_pulse):.3f}")
    
# #     # Step 6: Plot results
# #     print("6. Plotting results...")
# #     plot_results(time, original_pulse, filtered_pulse, recovered_pulse, frequency, s21)
# #     plot_ffts(original_pulse, time, recovered_pulse, filtered_pulse)


# # if __name__ == "__main__":
# #     main()


# import numpy as np
# import matplotlib.pyplot as plt

# # def create_amplitude_modulated_gaussian_train(amplitude=1.0, sigma=10e-9,
# #                                                f_c=500e6, num_pulses=1,
# #                                                pulse_spacing=80e-9, 
# #                                                total_time=600e-9, 
# #                                                num_points=10000):
# #     t = np.linspace(0, total_time, num_points)
# #     pulse_train = np.zeros_like(t)

# #     for i in range(num_pulses):
# #         t_i = i * pulse_spacing + sigma * 2  # center of each pulse
# #         envelope = amplitude * np.exp(-((t - t_i) ** 2) / (2 * sigma ** 2))
# #         modulation = 1 + np.cos(2 * np.pi * f_c * (t - t_i))  # non-negative
# #         pulse = envelope * modulation
# #         pulse_train += pulse

# #     return t, pulse_train


# def create_amplitude_modulated_gaussian_train(amplitude=1.0, sigma=10e-9,
#                                   f_c=500e6, total_time=100e-9, num_points=5000):
#     t = np.linspace(0, total_time, num_points)
#     t_i = total_time / 2  # center the pulse in the time range

#     envelope = amplitude * np.exp(-((t - t_i) ** 2) / (2 * sigma ** 2))
#     modulation = 1 + np.cos(2 * np.pi * f_c * (t - t_i))  # centered mod
#     pulse = envelope * modulation

#     plt.plot(t * 1e9, pulse)
#     plt.xlabel("Time (ns)")
#     plt.ylabel("Amplitude")
#     plt.title("Single Modulated Gaussian Pulse")
#     plt.grid(True)
#     plt.show()

#     return t, pulse


# import numpy as np

# def insert_deadspace_chunks(signal, t, samples_per_chunk=5, deadspace_len=20):
#     new_signal = []
#     new_time = []
#     chunk_duration = t[1] - t[0]
#     time_cursor = 0.0

#     for i in range(0, len(signal), samples_per_chunk):
#         chunk = signal[i:i+samples_per_chunk]
#         chunk_time = t[i:i+samples_per_chunk]

#         # Add the chunk to the signal
#         new_signal.extend(chunk)
#         new_time.extend(time_cursor + (chunk_time - chunk_time[0]))

#         # Add deadspace (zeros)
#         time_cursor = new_time[-1] + chunk_duration
#         deadspace_times = np.linspace(time_cursor,
#                                       time_cursor + chunk_duration * (deadspace_len - 1),
#                                       deadspace_len)
#         new_signal.extend([0.0] * deadspace_len)
#         new_time.extend(deadspace_times)

#         # Update cursor for next chunk
#         time_cursor = deadspace_times[-1] + chunk_duration

#     return np.array(new_time), np.array(new_signal)




# # Main execution
# if __name__ == "__main__":
#     t, signal = create_amplitude_modulated_gaussian_train()
#     t_spaced, signal_spaced = insert_deadspace_chunks(signal, t, samples_per_chunk=5, deadspace_len=20)

#     plt.figure(figsize=(12, 4))
#     plt.plot(t_spaced * 1e9, signal_spaced)
#     plt.xlabel("Time (ns)")
#     plt.ylabel("Amplitude")
#     plt.title("Gaussian-Modulated Pulse Train with Deadspace Between Ripples")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()









# # 3-4x pulse width

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def create_amplitude_modulated_gaussian_train(amplitude=1.0, sigma=20e-9,
                                              f_c=900e6, total_time=300e-9, num_points=5000):
    t = np.linspace(0, total_time, num_points)
    t_i = total_time / 2  # center the pulse in the time range

    envelope = amplitude * np.exp(-((t - t_i) ** 2) / (2 * sigma ** 2))
    modulation = 1 + np.cos(2 * np.pi * f_c * (t - t_i))  # centered mod
    pulse = envelope * modulation

    return t, pulse

def insert_deadspace_at_all_minima(t, pulse, deadspace_len=100):
    minima_indices = argrelextrema(pulse, np.less)[0]
    dt = t[1] - t[0]

    new_signal = []
    new_time = []

    prev_index = 0
    time_offset = 0.0

    for idx in minima_indices:
        # Append segment before minimum
        segment = pulse[prev_index:idx + 1]
        time_segment = t[prev_index:idx + 1] + time_offset

        new_signal.extend(segment)
        new_time.extend(time_segment)

        # Insert deadspace zeros and times
        deadspace_times = np.linspace(new_time[-1] + dt, new_time[-1] + dt * deadspace_len, deadspace_len)
        new_signal.extend([0.0] * deadspace_len)
        new_time.extend(deadspace_times)

        # Update offsets and prev_index
        time_offset = deadspace_times[-1] - t[idx]
        prev_index = idx + 1

    # Append the remaining part of the pulse after last minimum
    if prev_index < len(pulse):
        segment = pulse[prev_index:]
        time_segment = t[prev_index:] + time_offset

        new_signal.extend(segment)
        new_time.extend(time_segment)

    return np.array(new_time), np.array(new_signal)

# Generate original pulse
t, pulse = create_amplitude_modulated_gaussian_train()

# Insert deadspace at every minimum
t_new, pulse_new = insert_deadspace_at_all_minima(t, pulse, deadspace_len=100)

# Plotting the result
plt.figure(figsize=(12, 4))
plt.plot(t_new * 1e9, pulse_new)
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.title("Amplitude-Modulated Gaussian Pulse with Deadspace at Every Minimum")
plt.grid(True)
plt.tight_layout()
plt.show()

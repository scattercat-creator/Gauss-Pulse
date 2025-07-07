import skrf as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_s2p_filter(filename):
    """Load S2P file and extract S21 parameters"""
    network = rf.Network(filename)
    
    # Extract S21 (forward transmission)
    s21 = network.s[:,1,0]
    frequency = network.f
    
    # Remove last 75 points if needed (adjust as necessary)
    s21 = s21[:-75]
    frequency = frequency[:-75]
    
    return frequency, s21

def create_gaussian_pulse_train(duration=2e-6, num_points=2000, fwhm=100e-9, amplitude=1.0, num_pulses=5, spacing=300e-9):
    """
    Create a train of non-overlapping Gaussian pulses.
    
    duration: total time window in seconds
    num_points: number of time points in signal
    fwhm: full-width at half max of each Gaussian pulse
    amplitude: height of each pulse
    num_pulses: number of pulses to generate
    spacing: time between pulse centers in seconds
    """
    t = np.linspace(-duration/2, duration/2, num_points)
    sigma = fwhm / 2.355

    pulse_train = np.zeros_like(t)
    center_start = -((num_pulses - 1) / 2) * spacing  # centers centered in time window

    for i in range(num_pulses):
        center = center_start + i * spacing
        pulse_train += amplitude * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))

    return t, pulse_train


def create_gaussian_pulse(duration=1200e-9, num_points=1000, fwhm=200e-9, amplitude=1.0):
    """Create a Gaussian pulse"""
    t = np.linspace(-duration/2, duration/2, num_points)
    sigma = fwhm / 2.355
    pulse = amplitude * np.exp(-(t ** 2) / (2 * sigma ** 2))
    return t, pulse

def apply_s2p_filter(pulse, time, frequency, s21):
    """Apply S2P filter to the pulse"""
    # Get FFT of pulse
    pulse_fft = np.fft.fft(pulse)
    freq_signal = np.fft.fftfreq(len(pulse), time[1] - time[0])
    
    # Interpolate S21 to match signal frequency grid
    freq_abs = np.abs(freq_signal)
    s21_interp = np.interp(freq_abs, frequency, s21)
    
    # Set to zero beyond S21 frequency range
    s21_interp[freq_abs > frequency[-1]] = 0
    
    # Apply filter
    filtered_fft = pulse_fft * s21_interp
    filtered_pulse = np.fft.ifft(filtered_fft)
    
    return filtered_pulse

def add_noise_to_signal(signal, snr_db):
    """
    Add white Gaussian noise to a signal at a specified SNR (in dB).

    Parameters:
    - signal: the original time-domain signal (1D numpy array)
    - snr_db: desired signal-to-noise ratio in decibels

    Returns:
    - noisy_signal: signal with added noise
    - noise: the actual noise added (can be useful for debugging)
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)

    # Calculate required noise power
    noise_power = signal_power / snr_linear

    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)

    # Add noise to signal
    noisy_signal = signal + noise

    return noisy_signal 


def wiener_deconvolution(filtered_pulse, time, frequency, s21, snr=10):
    """Apply Wiener deconvolution to recover original pulse"""
    # Get FFT of filtered signal
    filtered_fft = np.fft.fft(filtered_pulse)
    freq_signal = np.fft.fftfreq(len(filtered_pulse), time[1] - time[0])
    
    # Interpolate S21 to match signal frequency grid
    freq_abs = np.abs(freq_signal)
    s21_interp = np.interp(freq_abs, frequency, s21)
    
    # Set to zero beyond S21 frequency range
    s21_interp[freq_abs > frequency[-1]] = 0
    
    # Calculate Wiener filter
    s21_power = np.abs(s21_interp)**2
    noise_power = np.mean(s21_power) / snr
    regularization = 1e-10
    
    # Wiener deconvolution formula
    H_wiener = np.conj(s21_interp) / (s21_power + noise_power + regularization)
    
    # Apply Wiener filter
    recovered_fft = filtered_fft * H_wiener
    recovered_pulse = np.fft.ifft(recovered_fft)
    
    return recovered_pulse.real

def plot_results(time, original_pulse, filtered_pulse, recovered_pulse, frequency, s21):
    """Plot all results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: S21 magnitude response
    axes[0, 0].plot(frequency/1e9, 20*np.log10(np.abs(s21)), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Frequency (GHz)')
    axes[0, 0].set_ylabel('S21 Magnitude (dB)')
    axes[0, 0].set_title('Filter Response (S21)')
    axes[0, 0].grid(True)
    
    # Plot 2: Original vs Filtered pulse
    axes[0, 1].plot(time*1e9, original_pulse, 'b-', linewidth=2, label='Original Pulse')
    axes[0, 1].plot(time*1e9, np.abs(filtered_pulse)/np.max(filtered_pulse), 'r-', linewidth=2, label='Filtered Pulse')
    axes[0, 1].set_xlabel('Time (ns)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Original vs Filtered Pulse')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: All three signals
    axes[1, 0].plot(time*1e9, original_pulse, 'b-', linewidth=2, label='Original Pulse')
    axes[1, 0].plot(time*1e9, np.abs(filtered_pulse), 'r-', linewidth=2, label='Filtered Pulse')
    axes[1, 0].plot(time*1e9, recovered_pulse, 'g-', linewidth=2, label='Recovered Pulse')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Complete Process: Original → Filtered → Recovered')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 4: Recovery comparison (zoomed)
    axes[1, 1].plot(time*1e9, original_pulse, 'b-', linewidth=2, label='Original Pulse')
    axes[1, 1].plot(time*1e9, recovered_pulse, 'g-', linewidth=2, label='Recovered Pulse')
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Recovery Quality Check')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main program"""
    # Parameters
    S2P_FILENAME = 'filter_time_series.s2p'  # Change this to your S2P file
    PULSE_DURATION = 2e-6
    NUM_POINTS = 8192
    PULSE_FWHM = 50e-9      # 50ns instead of 2ns
    SPACING=400e-9          # 400ns spacing instead of 100ns
    NUM_PULSES=3            # Fewer pulses for clarity
    PULSE_AMPLITUDE = 1.0     # Pulse amplitude
    SNR = 100                  # Signal-to-noise ratio for Wiener filter
    
    print("=== S2P Filter and Wiener Recovery Program ===\n")
    
    # Step 1: Load S2P file
    print("1. Loading S2P filter...")
    frequency, s21 = load_s2p_filter(S2P_FILENAME)
    print(f"   Loaded {len(frequency)} frequency points from {frequency[0]/1e9:.2f} to {frequency[-1]/1e9:.2f} GHz")
    
    # Step 2: Create Gaussian pulse
    print("2. Creating Gaussian pulse...")
    #time, original_pulse = create_gaussian_pulse(PULSE_DURATION, NUM_POINTS, PULSE_FWHM, PULSE_AMPLITUDE)
    time, original_pulse = create_gaussian_pulse_train(PULSE_DURATION, NUM_POINTS, PULSE_FWHM, PULSE_AMPLITUDE, NUM_PULSES, SPACING)

    print(f"   Created pulse: {PULSE_FWHM*1e9:.0f} ns FWHM, {len(original_pulse)} points")
    
    # Step 3: Apply S2P filter
    print("3. Applying S2P filter to pulse...")
    filtered_pulse = apply_s2p_filter(original_pulse, time, frequency, s21)
    print(f"   Original pulse peak: {np.max(original_pulse):.3f}")
    print(f"   Filtered pulse peak: {np.max(np.abs(filtered_pulse)):.3f}")
    
    filtered_pulse = add_noise_to_signal(filtered_pulse, 40)
    # Step 4: Apply Wiener deconvolution
    print("4. Applying Wiener deconvolution...")
    recovered_pulse = wiener_deconvolution(filtered_pulse, time, frequency, s21, SNR)
    print(f"   Recovered pulse peak: {np.max(np.abs(recovered_pulse)):.3f}")
    
    # Step 5: Calculate recovery quality
    print("5. Analyzing recovery quality...")
    
    # Find pulse centers for alignment
    orig_center = np.argmax(original_pulse)
    rec_center = np.argmax(np.abs(recovered_pulse))
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(original_pulse, recovered_pulse)[0, 1]
    
    # Calculate RMS error
    rms_error = np.sqrt(np.mean((original_pulse - recovered_pulse)**2))
    
    print(f"   Correlation coefficient: {correlation:.3f}")
    print(f"   RMS error: {rms_error:.3f}")
    print(f"   Peak recovery ratio: {np.max(np.abs(recovered_pulse))/np.max(original_pulse):.3f}")
    
    # Step 6: Plot results
    print("6. Plotting results...")
    plot_results(time, original_pulse, filtered_pulse, recovered_pulse, frequency, s21)
    
    print("\n=== Process Complete ===")
    print("Tips for improving recovery:")
    print("- Increase SNR for less noise but potential ringing")
    print("- Decrease SNR for more smoothing but less accuracy")
    print("- Check that your S2P file covers the pulse bandwidth")

if __name__ == "__main__":
    main()
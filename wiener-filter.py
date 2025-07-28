'''
1. Find power spectral density of gaussian pulse
2. Do a zero equalization of transfer
3. Do transfer function times zero equalization filter
4. Should get flat thing
5. Do transfer function times wiener filter
'''

import skrf as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import argrelextrema


def Create_Gaussian_Pulse(amplitude=1.0, center_time=0.0, standard_deviation=20e-9, range=200e-9, num_points=1024):
    """Create Gaussian pulse - using narrower pulse for visible spectrum"""
    time = np.linspace(-range, range, num_points)
    gaussian_pulse = amplitude * np.exp(-((time - center_time)**2) / (2 * standard_deviation**2))
    
    plt.figure()
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
  # Plot results
    plt.figure()
    
    # Plot Wiener filter
    plt.plot(frequency/1e6, 20*np.log10(np.abs(s21)))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Transfer Function H(f)')
    plt.grid(True)

    return frequency, s21

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
    plt.ylabel('Magnitude (Linear)')
    plt.title('Zero Equalizer')
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(transfer_frequencies/1e6, flat_response)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (Linear)')
    plt.title('TF × Equalizer (Should be Flat)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return equalizer

def Create_Noise(signal, time, noise_std =0.05):
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
    # Get signal's frequency grid
    X = fft(signal)
    N = len(signal)
    fs = 1/(time[1] - time[0])
    freqs = fftfreq(N, d=1/fs)
    bins = fs/N
    print(f'bins: {bins}')
    # Interpolate filter to signal's frequency grid
    # Handle positive and negative frequencies properly
    freqs_abs = np.abs(freqs)
    filter_interp = np.interp(freqs_abs, filter_freqs, filter)
    
    # Set to zero beyond filter frequency range
    filter_interp[freqs_abs > filter_freqs[-1]] = 0
    
    # Apply filter
    filtered_fft = X * filter_interp
    filtered_signal = ifft(filtered_fft)

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
    return np.real(filtered_signal)

def Before_VS_After(original_signal, new_signal, time):
    originall_fft = fft(original_signal)
    new_fft = fft(new_signal)
    N = len(original_signal)
    fs = 1/(time[1] - time[0])
    freqs = fftfreq(N, d=1/fs)
    freqs_abs = np.abs(freqs)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1,2,1)
    plt.plot(time*1e9, original_signal, 'b--', label='Original Pulse')
    plt.plot(time*1e9, np.real(new_signal), 'r-', label='Recovered Pulse')
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain: Origianal Vs Recovered Signal')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(freqs_abs[:N//2]/1e6, 20*np.log10(np.abs(originall_fft[:N//2])), 'b--', label='Original Spectrum')
    plt.plot(freqs_abs[:N//2]/1e6, 20*np.log10(np.abs(new_fft[:N//2])), 'r-', label='Recovered Spectrum')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Domain: Origianal Vs Recovered Signal')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def Comprehensive_Error_Analysis(original_signal, recovered_signal, time, label="Recovery"):
    """
    Comprehensive error analysis for signal recovery performance
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    
    # Ensure signals are real and same length
    original_signal = np.real(original_signal)
    recovered_signal = np.real(recovered_signal)
    
    if len(original_signal) != len(recovered_signal):
        min_len = min(len(original_signal), len(recovered_signal))
        original_signal = original_signal[:min_len]
        recovered_signal = recovered_signal[:min_len]
    
    # Time domain error metrics
    error_signal = original_signal - recovered_signal
    
    # 1. Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean(error_signal**2))
    
    # 2. Normalized RMSE (as percentage)
    signal_power = np.sqrt(np.mean(original_signal**2))
    nrmse_percent = (rmse / signal_power) * 100
    
    # 3. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(error_signal))
    
    # 4. Peak Signal-to-Noise Ratio (PSNR)
    max_signal = np.max(np.abs(original_signal))
    if rmse > 0:
        psnr = 20 * np.log10(max_signal / rmse)
    else:
        psnr = float('inf')
    
    # 5. Correlation coefficient
    correlation = np.corrcoef(original_signal, recovered_signal)[0, 1]
    
    # 6. Signal-to-Noise Ratio (SNR)
    signal_power_db = 10 * np.log10(np.mean(original_signal**2))
    noise_power_db = 10 * np.log10(np.mean(error_signal**2))
    snr_db = signal_power_db - noise_power_db
    
    # 7. Peak recovery ratio
    original_peak = np.max(np.abs(original_signal))
    recovered_peak = np.max(np.abs(recovered_signal))
    peak_ratio = recovered_peak / original_peak if original_peak != 0 else 0
    
    # 8. Energy recovery ratio
    original_energy = np.sum(original_signal**2)
    recovered_energy = np.sum(recovered_signal**2)
    energy_ratio = recovered_energy / original_energy if original_energy != 0 else 0
    
    # 9. Phase analysis (for complex signals)
    N = len(original_signal)
    fs = 1/(time[1] - time[0])
    freqs = fftfreq(N, d=1/fs)
    
    original_fft = fft(original_signal)
    recovered_fft = fft(recovered_signal)
    
    # Frequency domain metrics (for positive frequencies only)
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    original_fft_pos = original_fft[mask]
    recovered_fft_pos = recovered_fft[mask]
    
    # Magnitude error in frequency domain
    original_mag = np.abs(original_fft_pos)
    recovered_mag = np.abs(recovered_fft_pos)
    mag_error = np.mean((original_mag - recovered_mag)**2)
    
    # Phase error (where magnitude is significant)
    threshold = 0.1 * np.max(original_mag)
    significant_mask = original_mag > threshold
    
    if np.any(significant_mask):
        original_phase = np.angle(original_fft_pos[significant_mask])
        recovered_phase = np.angle(recovered_fft_pos[significant_mask])
        phase_error = np.mean(np.abs(np.angle(np.exp(1j*(original_phase - recovered_phase)))))
        phase_error_degrees = np.degrees(phase_error)
    else:
        phase_error_degrees = 0
    
    # Print comprehensive results
    print(f"\n=== {label} Error Analysis ===")
    print(f"Time Domain Metrics:")
    print(f"  RMSE:                    {rmse:.6f}")
    print(f"  Normalized RMSE:         {nrmse_percent:.2f}%")
    print(f"  Mean Absolute Error:     {mae:.6f}")
    print(f"  Correlation Coefficient: {correlation:.6f}")
    print(f"  SNR:                     {snr_db:.2f} dB")
    print(f"  PSNR:                    {psnr:.2f} dB")
    
    print(f"\nAmplitude Recovery:")
    print(f"  Original Peak:           {original_peak:.6f}")
    print(f"  Recovered Peak:          {recovered_peak:.6f}")
    print(f"  Peak Recovery Ratio:     {peak_ratio:.6f} ({peak_ratio*100:.1f}%)")
    print(f"  Energy Recovery Ratio:   {energy_ratio:.6f} ({energy_ratio*100:.1f}%)")
    
    print(f"\nFrequency Domain Metrics:")
    print(f"  Magnitude Error (RMS):   {np.sqrt(mag_error):.6f}")
    print(f"  Phase Error:             {phase_error_degrees:.2f} degrees")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    if correlation > 0.95:
        quality = "Excellent"
    elif correlation > 0.9:
        quality = "Good"
    elif correlation > 0.8:
        quality = "Fair"
    elif correlation > 0.6:
        quality = "Poor"
    else:
        quality = "Very Poor"
    print(f"  Overall Quality:         {quality}")
    
    # Detailed plotting
    plt.figure(figsize=(15, 8))
    
    # Time domain comparison
    plt.subplot(2, 3, 1)
    plt.plot(time*1e9, original_signal, 'b-', linewidth=2, label='Original')
    plt.plot(time*1e9, recovered_signal, 'r--', linewidth=2, label='Recovered')
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title('Signal Comparison')
    plt.legend()
    plt.grid(True)
    
    # Error signal
    plt.subplot(2, 3, 2)
    plt.plot(time*1e9, error_signal, 'k-', linewidth=1)
    plt.xlabel('Time (ns)')
    plt.ylabel('Error')
    plt.title(f'Error Signal (RMSE: {rmse:.4f})')
    plt.grid(True)
    
    # Correlation plot
    plt.subplot(2, 3, 3)
    plt.scatter(original_signal, recovered_signal, alpha=0.6, s=20)
    min_val = min(np.min(original_signal), np.min(recovered_signal))
    max_val = max(np.max(original_signal), np.max(recovered_signal))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Recovery')
    plt.xlabel('Original Signal')
    plt.ylabel('Recovered Signal')
    plt.title(f'Correlation Plot (r={correlation:.3f})')
    plt.legend()
    plt.grid(True)
    
    # Frequency domain magnitude comparison
    plt.subplot(2, 3, 4)
    plt.semilogy(freqs_pos[:N//4]/1e6, original_mag[:N//4], 'b-', linewidth=2, label='Original')
    plt.semilogy(freqs_pos[:N//4]/1e6, recovered_mag[:N//4], 'r--', linewidth=2, label='Recovered')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Error histogram
    plt.subplot(2, 3, 5)
    plt.hist(error_signal, bins=50, alpha=0.7, density=True)
    plt.xlabel('Error Value')
    plt.ylabel('Probability Density')
    plt.title('Error Distribution')
    plt.grid(True)
    
    # Metrics summary
    plt.subplot(2, 3, 6)
    metrics = ['Correlation', 'Peak Ratio', 'Energy Ratio', 'NRMSE(%)', 'SNR(dB)']
    values = [correlation, peak_ratio, energy_ratio, nrmse_percent/100, snr_db/50]  # Normalized for display
    colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Normalized Value')
    plt.title('Performance Metrics')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val, orig in zip(bars, values, [correlation, peak_ratio, energy_ratio, nrmse_percent, snr_db]):
        if 'NRMSE' in metrics[bars.index(bar)]:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{orig:.1f}%', 
                    ha='center', va='bottom')
        elif 'SNR' in metrics[bars.index(bar)]:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{orig:.1f}dB', 
                    ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{orig:.3f}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Return metrics dictionary for further analysis
    return {
        'rmse': rmse,
        'nrmse_percent': nrmse_percent,
        'mae': mae,
        'correlation': correlation,
        'snr_db': snr_db,
        'psnr': psnr,
        'peak_ratio': peak_ratio,
        'energy_ratio': energy_ratio,
        'phase_error_deg': phase_error_degrees,
        'quality': quality
    }

def create_amplitude_modulated_gaussian_train(amplitude=1.0, sigma=30e-9,
                                              f_c=200e6, total_time=300e-9, num_points=6000):
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
    t_new, pulse_new = np.array(new_time), np.array(new_signal)
    plt.figure(figsize=(12, 4))
    plt.plot(t_new * 1e9, pulse_new)
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.title("Simulated Beam Signal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return t_new, pulse_new




def main():
    
    t, signal = create_amplitude_modulated_gaussian_train()
    time, gauss_pulse = insert_deadspace_at_all_minima(t, signal, deadspace_len=100)
    
    # shows gaussian pulse
    #time, gauss_pulse = Create_Gaussian_Pulse()

    # takes s2p and grabs gain
    transfer_frequencies, s21 = Load_Transfer_Function('converted_filter.s2p')

   
    # calculates power spectral density of the original signal
    psd_signal_freqs, psd_signal = Power_Spectral_Density(gauss_pulse, time, "Signal")

    # creates an equalizer for the transfer function
    equalizer = Zero_Equalization(s21, transfer_frequencies)

    # apply s21 filter to signal 
    filtered_signal = Apply_Filter_to_Signal(gauss_pulse, time, s21, transfer_frequencies, "Transfer Function")

    # creates noise and adds it to signal
    noise, filtered_signal = Create_Noise(filtered_signal, time)

    # grabs the power spectral density of the noise
    psd_noise_freqs, psd_noise = Power_Spectral_Density(noise, time, "Gaussian White Noise")
    
    # wiener filter with noise
    w_filter = Wiener_Filter(s21, transfer_frequencies, psd_signal, psd_signal_freqs)
    w_filter = Wiener_Filter(s21, transfer_frequencies, psd_signal, psd_signal_freqs, psd_noise, psd_noise_freqs)
    new_signal = Apply_Filter_to_Signal(filtered_signal, time, w_filter, transfer_frequencies, "Wiener Filter")

    Before_VS_After(gauss_pulse, new_signal, time)

    Comprehensive_Error_Analysis(gauss_pulse, new_signal, time)



if __name__ == "__main__":
    main()
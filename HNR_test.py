import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

def autocorrelation(signal):
    n = len(signal)
    signal = signal - np.mean(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:]
    autocorr = autocorr / np.arange(n, 0, -1)
    return autocorr

def estimate_f0_autocorr(signal, sr, f0_min=75, f0_max=400):
    autocorr = autocorrelation(signal)
    lag_min = int(sr / f0_max)
    lag_max = int(sr / f0_min)
    if lag_max >= len(autocorr):
        lag_max = len(autocorr) - 1
    autocorr_search = autocorr[lag_min:lag_max]
    if len(autocorr_search) == 0:
        return None, autocorr, 0
    max_lag_local = np.argmax(autocorr_search)
    max_lag = max_lag_local + lag_min
    f0 = sr / max_lag
    return f0, autocorr, max_lag

def calculate_hnr(signal, sr, f0_min=75, f0_max=400):
    f0, autocorr, T0 = estimate_f0_autocorr(signal, sr, f0_min, f0_max)
    if f0 is None or T0 == 0:
        return None, None
    R0 = autocorr[0]
    RT0 = autocorr[T0]
    noise_energy = R0 - RT0
    if noise_energy <= 0 or RT0 <= 0:
        return None, f0
    hnr = RT0 / noise_energy
    hnr_db = 10 * np.log10(hnr)
    return hnr_db, f0

def apply_window(signal, window_type='hamming'):
    if window_type == 'hamming':
        window = np.hamming(len(signal))
    elif window_type == 'hann':
        window = np.hanning(len(signal))
    elif window_type == 'blackman':
        window = np.blackman(len(signal))
    else:
        window = np.ones(len(signal))
    return signal * window

def cepstral_filtering(signal, sr, quefrency_min=None, quefrency_max=None):
    if quefrency_min is None:
        quefrency_min = (1000 / 400) * sr / 1000
    if quefrency_max is None:
        quefrency_max = (1000 / 75) * sr / 1000
    q_min_samples = int(quefrency_min * sr / 1000)
    q_max_samples = int(quefrency_max * sr / 1000)
    q_min_samples = max(q_min_samples, 1)
    q_max_samples = min(q_max_samples, len(signal) // 2)
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    spectrum = np.maximum(spectrum, 1e-10)
    cepstrum = np.fft.irfft(np.log(spectrum))
    lifter = np.zeros_like(cepstrum)
    lifter[q_min_samples:q_max_samples] = 1
    cepstrum_filtered = cepstrum * lifter
    spectrum_filtered = np.exp(np.fft.rfft(cepstrum_filtered).real)
    phase_original = np.angle(np.fft.rfft(signal))
    spectrum_complex = spectrum_filtered * np.exp(1j * phase_original)
    signal_filtered = np.fft.irfft(spectrum_complex)
    return signal_filtered[:len(signal)]

def apply_preprocessing(signal, sr, method='none', **kwargs):
    if method == 'none':
        return signal
    elif method == 'lowpass':
        cutoff_hz = kwargs.get('cutoff_hz', 1000)
        print(f"Applicazione: Filtro passa-basso ({cutoff_hz} Hz)")
        return apply_lowpass_filter(signal, sr, cutoff_hz)
    elif method == 'cepstral':
        quefrency_min = kwargs.get('quefrency_min', None)
        quefrency_max = kwargs.get('quefrency_max', None)
        print(f"Applicazione: Cepstral filtering")
        return cepstral_filtering(signal, sr, quefrency_min, quefrency_max)
    elif method == 'preemphasis':
        coeff = kwargs.get('coeff', 0.97)
        print(f"Applicazione: Pre-enfasi (coeff={coeff})")
        return apply_preemphasis(signal, coeff)
    else:
        print(f"Metodo sconosciuto: {method}")
        return signal

def apply_lowpass_filter(signal, sr, cutoff_hz=1000):
    from scipy.signal import butter, filtfilt
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    b, a = butter(4, normalized_cutoff, btype='low')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_preemphasis(signal, coeff=0.97):
    emphasized = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    return emphasized

def analyze_hnr_windowed(signal, sr, window_length_ms=40, hop_length_ms=10, 
                         f0_min=75, f0_max=400, window_type='hamming',
                         preprocessing_method='none', preprocessing_params=None, verbose=False):
    if preprocessing_params is None:
        preprocessing_params = {}
    window_length = int(window_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)
    if verbose:
        print("\n=== Preprocessing ===")
        if preprocessing_method != 'none':
            print(f"Metodo: {preprocessing_method}")
            signal = apply_preprocessing(signal, sr, preprocessing_method, **preprocessing_params)
        else:
            print("Nessun preprocessing")
    else:
        if preprocessing_method != 'none':
            signal = apply_preprocessing(signal, sr, preprocessing_method, **preprocessing_params)
    num_frames = int((len(signal) - window_length) / hop_length) + 1
    hnr_values = []
    f0_values = []
    time_centers = []
    valid_frames = 0
    for i in range(num_frames):
        start = i * hop_length
        end = start + window_length
        if end > len(signal):
            break
        frame = signal[start:end]
        frame_windowed = apply_window(frame, window_type)
        hnr_db, f0 = calculate_hnr(frame_windowed, sr, f0_min, f0_max)
        time_center = (start + window_length/2) / sr
        hnr_values.append(hnr_db)
        f0_values.append(f0)
        time_centers.append(time_center)
        if hnr_db is not None:
            valid_frames += 1
    return {
        'hnr_values': np.array(hnr_values),
        'f0_values': np.array(f0_values),
        'time_centers': np.array(time_centers),
        'valid_frames': valid_frames,
        'total_frames': len(hnr_values),
        'preprocessing_method': preprocessing_method
    }

def moving_average(data, window_size=5):
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    return smoothed

def median_filter(data, window_size=5):
    from scipy.ndimage import median_filter as scipy_median
    filtered = scipy_median(data, size=window_size)
    return filtered

def savitzky_golay_filter(data, window_size=11, polyorder=2):
    from scipy.signal import savgol_filter
    if window_size % 2 == 0:
        window_size += 1
    if polyorder >= window_size:
        polyorder = window_size - 2
    filtered = savgol_filter(data, window_size, polyorder)
    return filtered

def apply_postprocessing(results, smoothing_method='moving_average', 
                        smoothing_window=5, verbose=False):
    results_processed = results.copy()
    hnr_values = results['hnr_values'].astype(float)
    f0_values = results['f0_values'].astype(float)
    hnr_valid = hnr_values[hnr_values != None]
    f0_valid = f0_values[f0_values != None]
    if len(hnr_valid) == 0 or len(f0_valid) == 0:
        return results_processed
    hnr_filled = hnr_values.copy()
    f0_filled = f0_values.copy()
    hnr_filled[hnr_values == None] = np.mean(hnr_valid)
    f0_filled[f0_values == None] = np.mean(f0_valid)
    if verbose:
        print("\n=== Postprocessing ===")
        print(f"Metodo: {smoothing_method} (window={smoothing_window})")
    if smoothing_method == 'none':
        results_processed['hnr_smoothed'] = hnr_filled
        results_processed['f0_smoothed'] = f0_filled
    elif smoothing_method == 'moving_average':
        results_processed['hnr_smoothed'] = moving_average(hnr_filled, smoothing_window)
        results_processed['f0_smoothed'] = moving_average(f0_filled, smoothing_window)
    elif smoothing_method == 'median':
        results_processed['hnr_smoothed'] = median_filter(hnr_filled, smoothing_window)
        results_processed['f0_smoothed'] = median_filter(f0_filled, smoothing_window)
    elif smoothing_method == 'savitzky_golay':
        results_processed['hnr_smoothed'] = savitzky_golay_filter(hnr_filled, smoothing_window)
        results_processed['f0_smoothed'] = savitzky_golay_filter(f0_filled, smoothing_window)
    else:
        results_processed['hnr_smoothed'] = hnr_filled
        results_processed['f0_smoothed'] = f0_filled
    results_processed['smoothing_method'] = smoothing_method
    results_processed['smoothing_window'] = smoothing_window
    return results_processed

def plot_hnr_analysis(results, audio_file, preprocessing_name, postprocessing_name):
    time_centers = results['time_centers']
    if 'hnr_smoothed' in results:
        hnr = results['hnr_smoothed']
        label = 'HNR smoothed'
    else:
        hnr = results['hnr_values'].astype(float)
        label = 'HNR'
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    ax.plot(time_centers, hnr, color='b', linewidth=2.2, label=label)
    ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Molto armonico (>20 dB)')
    ax.axhline(y=15, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Moderato (>15 dB)')
    ax.axhline(y=13, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Basso (>13 dB)')
    ax.fill_between(time_centers, 15, 25, color='yellow', alpha=0.07)
    ax.set_xlabel('Tempo (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('HNR (dB)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'HNR nel tempo - {audio_file}\n'
        f'Preprocessing: {preprocessing_name} | Postprocessing: {postprocessing_name}',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='best', fontsize=11, framealpha=0.92)
    ax.set_ylim([-20, 15])
    plt.tight_layout()
    if not os.path.exists('figures'):
        os.makedirs('figures')
    filename = f'figures/hnr_{preprocessing_name}_{postprocessing_name}.png'.replace(' ', '_').replace('-', '_').replace('.', '')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"{filename}")
    plt.close()

def find_peaks_adaptive(hnr_array, method='std', k=2.5, percentile=98, prominence=0.1):
    score = np.array(hnr_array, dtype=float)
    valid_mask = ~np.isnan(score)
    valid_score = score[valid_mask]
    if len(valid_score) == 0:
        print("Nessun valore valido trovato!")
        return np.array([], dtype=int), None
    if method == 'std':
        threshold = np.mean(valid_score) + k * np.std(valid_score)
        peaks, _ = find_peaks(score, height=threshold)
        print(f"Adaptive threshold (mean + {k}·std): {threshold:.3f} dB")
    elif method == 'percentile':
        threshold = np.percentile(valid_score, percentile)
        peaks, _ = find_peaks(score, height=threshold)
        print(f"Adaptive threshold (percentile {percentile}%): {threshold:.3f} dB")
    elif method == 'prominence':
        prom_threshold = prominence * np.max(valid_score)
        peaks, props = find_peaks(score, prominence=prom_threshold)
        print(f"Prominence-based detection (threshold: {prom_threshold:.3f} dB)")
        threshold = prom_threshold
    else:
        raise ValueError("method must be 'std', 'percentile', or 'prominence'")
    print(f"Trovati {len(peaks)} picchi")
    return peaks, threshold

if __name__ == "__main__":
    audio_file = "whooping_raw.wav"
    print("\n" + "="*80)
    print("WHOOPING SIGNALS - ANALISI COMPLETE COMBINAZIONI PRE/POST PROCESSING")
    print("="*80)
    window_length_ms = 40
    hop_length_ms = 7
    f0_min = 250
    f0_max = 700
    window_type = 'hamming'
    print(f"\nCaricamento: {audio_file}")
    signal, sr = sf.read(audio_file)
    if len(signal) > sr * 10:
        signal = signal[:sr*10]
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    print(f"Sample rate: {sr} Hz | Durata: {len(signal)/sr:.2f}s")
    preprocessing_configs = {
        'none': {
            'method': 'none',
            'params': {}
        },
        'lowpass_1000Hz': {
            'method': 'lowpass',
            'params': {'cutoff_hz': 1000}
        },
        'lowpass_2500Hz': {
            'method': 'lowpass',
            'params': {'cutoff_hz': 2500}
        },
        'cepstral': {
            'method': 'cepstral',
            'params': {'quefrency_min': 2.0, 'quefrency_max': 10.0}
        },
        'preemphasis': {
            'method': 'preemphasis',
            'params': {'coeff': 0.97}
        }
    }
    postprocessing_configs = {
        'moving_average_5': {
            'method': 'moving_average',
            'window': 5
        },
        'median_filter_5': {
            'method': 'median',
            'window': 5
        },
        'savitzky_golay_5': {
            'method': 'savitzky_golay',
            'window': 5
        }
    }
    print(f"\nConfigurazione:")
    print(f"   Window: {window_length_ms}ms | Hop: {hop_length_ms}ms")
    print(f"   F0 range: {f0_min}-{f0_max} Hz")
    print(f"   Preprocessing methods: {len(preprocessing_configs)}")
    print(f"   Postprocessing methods: {len(postprocessing_configs)}")
    print(f"   Total combinations: {len(preprocessing_configs) * len(postprocessing_configs)}")
    print("\nGenerazione grafici...")
    print("="*80)
    combination_count = 0
    total_combinations = len(preprocessing_configs) * len(postprocessing_configs)
    original_results = analyze_hnr_windowed(
        signal, sr,
        window_length_ms=window_length_ms,
        hop_length_ms=hop_length_ms,
        f0_min=f0_min,
        f0_max=f0_max,
        window_type=window_type,
        preprocessing_method='none',
        preprocessing_params='none',
        verbose=False
    )
    peaks, threshold = find_peaks_adaptive(original_results['hnr_values'], method='std', k=1.9)
    print(f"Found peaks at: {peaks}\n")
    print("threshold:", threshold)
    plot_hnr_analysis(
        original_results,
        audio_file,
        'none',
        'none'
    )
    for prep_name, prep_config in preprocessing_configs.items():
        results = analyze_hnr_windowed(
            signal, sr,
            window_length_ms=window_length_ms,
            hop_length_ms=hop_length_ms,
            f0_min=f0_min,
            f0_max=f0_max,
            window_type=window_type,
            preprocessing_method=prep_config['method'],
            preprocessing_params=prep_config['params'],
            verbose=False
        )
        for post_name, post_config in postprocessing_configs.items():
            combination_count += 1
            results_processed = apply_postprocessing(
                results,
                smoothing_method=post_config['method'],
                smoothing_window=post_config['window'],
                verbose=False
            )
            peaks, threshold = find_peaks_adaptive(results_processed['hnr_values'], method='std', k=1.9)
            print(f"Found peaks at: {peaks}\n")
            print("threshold:", threshold)
            plot_hnr_analysis(
                results_processed,
                audio_file,
                prep_name,
                post_name
            )
            print(f"[{combination_count:2d}/{total_combinations}] {prep_name:20s} + {post_name:20s}")
    print("="*80)
    print(f"COMPLETATO! {total_combinations} grafici generati nella cartella 'figures/'")
    print("="*80)
    print("="*80)
    print("WHOOPING SIGNALS - ANALISI VINCITRICE COMBINAZIONE PRE/POST PROCESSING lowpass_2500 + savitzky_golay_5")
    print("="*80)

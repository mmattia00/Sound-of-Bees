import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter as scipy_median
from scipy.signal import savgol_filter
import sounddevice as sd

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

def apply_lowpass_filter(signal, sr, cutoff_hz=15000):
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    b, a = butter(4, normalized_cutoff, btype='low')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_highpass_filter(signal, sr, cutoff_hz=2500):
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    b, a = butter(4, normalized_cutoff, btype='high')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_preprocessing(signal, sr):
    # Applica filtro low-pass a 15k Hz
    preprocessed_signal = apply_lowpass_filter(signal, sr, cutoff_hz=15000)
    preprocessed_signal = apply_highpass_filter(preprocessed_signal, sr, cutoff_hz=2500)
    return preprocessed_signal


def analyze_hnr_windowed(signal, sr, window_length_ms=40, hop_length_ms=10, 
                         f0_min=75, f0_max=400, window_type='hamming'):
    
    window_length = int(window_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)

    # APPLY PREPROCESSING: low-pass filter at 3500 Hz
    signal = apply_preprocessing(signal, sr)

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
        'total_frames': len(hnr_values)
    }


def savitzky_golay_filter(data, window_size=11, polyorder=2):
    if window_size % 2 == 0:
        window_size += 1
    if polyorder >= window_size:
        polyorder = window_size - 2
    filtered = savgol_filter(data, window_size, polyorder)
    return filtered

def apply_postprocessing(results, smoothing_window=5):
    results_processed = results.copy()
    hnr_values = results['hnr_values'].astype(float) # None -> np.nan
    f0_values = results['f0_values'].astype(float)

    # USA np.isnan invece di confronto con None
    hnr_valid = hnr_values[~np.isnan(hnr_values)]  # ← FIX
    f0_valid = f0_values[~np.isnan(f0_values)]      # ← FIX

    if len(hnr_valid) == 0 or len(f0_valid) == 0:
        return results_processed
    
    hnr_filled = hnr_values.copy()
    f0_filled = f0_values.copy()

    # Riempi i NaN con la media
    hnr_filled[np.isnan(hnr_values)] = np.mean(hnr_valid)  # ← FIX
    f0_filled[np.isnan(f0_values)] = np.mean(f0_valid)      # ← FIX

    results_processed['hnr_smoothed'] = savitzky_golay_filter(hnr_filled, smoothing_window)
    results_processed['f0_smoothed'] = savitzky_golay_filter(f0_filled, smoothing_window)
    return results_processed

def plot_hnr_analysis(results, ch_num=None, peaks=None, peak_windows=None, threshold=None):
    time_centers = results['time_centers']
    hnr = results['hnr_smoothed']
    
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(time_centers, hnr, color='b', linewidth=2.2, label='HNR')
    ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Molto armonico (>20 dB)')
    ax.axhline(y=15, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='Moderato (>15 dB)')
    ax.axhline(y=13, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Basso (>13 dB)')
    ax.fill_between(time_centers, 15, 25, color='yellow', alpha=0.07)
    
    # Plotta la linea di threshold se fornita
    if threshold is not None:
        ax.axhline(y=threshold, color='purple', linestyle='-', linewidth=2.5, 
                  alpha=0.8, label=f'Threshold ({threshold:.2f} dB)')
    
    # Evidenzia le finestre dei picchi
    if peak_windows is not None and len(peak_windows) > 0:
        for start, end in peak_windows:
            ax.axvspan(start, end, alpha=0.15, color='red', label='Finestra picco' if peak_windows[0] == (start, end) else '')
    
    # Plotta i picchi
    if peaks is not None and len(peaks) > 0:
        ax.scatter(time_centers[peaks], hnr[peaks], 
                  color='red', s=100, marker='o', 
                  edgecolors='darkred', linewidths=2,
                  label=f'Picchi rilevati ({len(peaks)})', 
                  zorder=5, alpha=0.8)
    
    ax.set_xlabel('Tempo (s)', fontsize=7, fontweight='bold')
    ax.set_ylabel('HNR (dB)', fontsize=7, fontweight='bold')
    ax.set_title(f'HNR nel tempo - channel {ch_num}\n', fontsize=7, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='best', fontsize=5, framealpha=0.7)
    plt.tight_layout()
    
    plt.show()




def find_peaks_percentile(hnr_values, time_centers, percentile=99, offset=2):
    """
    Rileva outlier usando un percentile alto della distribuzione.
    
    Args:
        percentile: soglia percentile (95-99 per outlier veri)
        offset: margine aggiuntivo sopra il percentile in dB
    """
    hnr = np.array(hnr_values, dtype=float)
    valid = hnr[~np.isnan(hnr)]
    
    # Base sulla distribuzione
    percentile_value = np.percentile(valid, percentile)
    threshold = percentile_value + offset
    
    peaks, _ = find_peaks(hnr, height=threshold)
    
    print(f"Metodo: Percentile-based | {percentile}° percentile={percentile_value:.2f} dB")
    print(f"Threshold: {threshold:.2f} dB | Picchi trovati: {len(peaks)}")
    
    return peaks, time_centers[peaks], threshold

def merge_overlapping_windows(peak_windows, peaks, peaks_times):
    """
    Unisce finestre sovrapposte: se la fine della finestra i
    è maggiore dell'inizio della finestra i+1, le unisce.
    Aggiorna anche gli indici 'peaks' coerentemente.
    """
    # Converti peaks in lista se è un array numpy (per poter usare del)
    if isinstance(peaks, np.ndarray):
        peaks = peaks.tolist()
    
    # Converti peaks_times in lista se è un array numpy
    if isinstance(peaks_times, np.ndarray):
        peaks_times = peaks_times.tolist()

    i = 0
    while i < len(peak_windows) - 1:
        start_i, end_i = peak_windows[i]
        start_next, end_next = peak_windows[i + 1]

        # Se si sovrappongono o si toccano
        if end_i >= start_next:
            # Unisci le due finestre prendendo l'inizio della prima e la fine della seconda
            merged_window = (start_i, max(end_i, end_next))
            peak_windows[i] = merged_window

            # Elimina la finestra successiva e il picco corrispondente
            del peak_windows[i + 1]
            del peaks[i + 1]
            del peaks_times[i + 1]

            # Non incrementare i, perché la finestra unita potrebbe sovrapporsi anche alla successiva
        else:
            i += 1

    return peak_windows, peaks, peaks_times

def normalize_signal_rms(signal, target_rms=0.1):
    """
    Normalizza il segnale a un RMS target
    target_rms=0.1 significa normalizza a 0.1 (scala lineare)
    """
    current_rms = np.sqrt(np.mean(signal**2))
    if current_rms < 1e-6:  # protezione da divisione per zero
        return signal
    scaling_factor = target_rms / current_rms
    return signal * scaling_factor

if __name__ == "__main__":
    # audio_file = "reduced_whooping_raw.wav"
    audio_file = "sounds/whoop_examples/whooping_collection.wav"

    print("\n" + "="*80)
    print("WHOOPING SIGNALS - INIZIO ANALISI HNR BASED")
    print("="*80)

    # setup parameters
    # --- OPTION A
    # window_length_ms = 50  # più robusto per frequenze basse
    # hop_length_ms = 10     # standard, efficiente
    # overlap = 80%

    # --- OPTION B Già molto dettagliato, eccellente time resolution
    window_length_ms = 40
    hop_length_ms = 7
    # overlap = 82.5%

    #  --- OPTION C da provare se perdo whooping veloci
    # window_length_ms = 35  # più agile per transients
    # hop_length_ms = 5      # massima risoluzione pratica
    # overlap = 85.7%

    f0_min = 250
    f0_max = 700
    window_type = 'hamming'
    print(f"\nCaricamento: {audio_file}")
    signal, sr = sf.read(audio_file)


    # Limita a 10 secondi per test veloce
    if len(signal) > sr * 10:
        signal = signal[:sr*10]

    # tieni solo primo canale se stereo
    if len(signal.shape) > 1:
        signal = signal[:, 0]


  
    print(f'INIZIO ANALISI')

    # normalization through channel
    signal = normalize_signal_rms(signal, target_rms=0.1)

    print(f"Sample rate: {sr} Hz | Durata: {len(signal)/sr:.2f}s")
    
    print(f"\nConfigurazione:")
    print(f"  Window: {window_length_ms}ms | Hop: {hop_length_ms}ms")
    print(f"   F0 range: {f0_min}-{f0_max} Hz")
    print(f"   Preprocessing methods: low pass filter at 15 kHz, high pass filter at 2.5 kHz, normalization")
    print(f"   Postprocessing methods: savitzky_golay filter with 5 windows")
    print("="*80)
    
    results = analyze_hnr_windowed(
        signal, sr,
        window_length_ms=window_length_ms,
        hop_length_ms=hop_length_ms,
        f0_min=f0_min,
        f0_max=f0_max,
        window_type=window_type,
    )

    # Postprocessing: savitzky_golay filter
    smoothed_results = apply_postprocessing(results, smoothing_window=5)


    peaks, peak_times, threshold = find_peaks_percentile(
            smoothed_results['hnr_smoothed'],
            smoothed_results['time_centers'],
            percentile=99.5,
            offset=4
    )

    # merge overlapping peaks based on time proximity
    playback_window_sec = 2  # 2 secondi (molto più lungo!)
    peak_windows = []
    for peak_time in peak_times:
        start_time = max(0, peak_time - playback_window_sec / 2)
        end_time = min(len(signal) / sr, peak_time + playback_window_sec / 2)
        peak_windows.append((start_time, end_time))

    # mostra intervalli trovati
    print(f'{len(peak_windows)} picchi trovati prima del merge:')
    # print("Detected time windows before merging:")
    # for (s, e) in peak_windows:
    #     print(f"Time window: {s:.3f}s to {e:.3f}s")
    
    # merge overlapping windows
    peak_windows, peaks, peak_times = merge_overlapping_windows(peak_windows, peaks, peak_times)

    # mostra intervalli trovati dopo il merge
    print(f'{len(peak_windows)} picchi trovati dopo il merge:')
    # for (s, e) in peak_windows:
    #     print(f"Time window: {s:.3f}s to {e:.3f}s")
        
    
    print(f"Riproduzione {len(peaks)} picchi...")

    print("="*80) 

    for i, peak_window in enumerate(peak_windows):
        start_time = peak_window[0]
        end_time = peak_window[1]

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        print(f"\nPicco {i+1}: {peak_times[i]:.3f}s")
        print(f"Durata riproduzione: {(end_time - start_time)*1000:.0f}ms")
        print(f"Campioni: {start_sample} - {end_sample} (lunghezza: {end_sample - start_sample})")
        
        segment = signal[start_sample:end_sample]
        
        if len(segment) > 0:
            rms = np.sqrt(np.mean(segment**2))
            print(f"Volume RMS: {rms:.6f}")
            
            if rms < 0.01:
                print("Avviso: volume molto basso, amplificazione...")
                segment = segment * 5  # Amplifica di 5x
            
            print("Riproduzione...")
            sd.play(segment, sr)
            sd.wait()
            sd.sleep(500)

        print("="*80)

    
   
    # # print all the hnr values 
    # for i, hnr_value in enumerate(smoothed_results['hnr_smoothed']):
    #     time_center = smoothed_results['time_centers'][i]
    #     print(f"Time: {time_center:.3f}s | HNR: {hnr_value:.2f} dB")
    
    plot_hnr_analysis(smoothed_results, ch_num=None, peaks=peaks, peak_windows=peak_windows, threshold=threshold)


    print("="*80)
    print(f"ANALISI COMPLETATA !")
    print("="*80)
    

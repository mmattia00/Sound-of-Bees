import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
import os
import soundfile as sf


class WhoopDetector:
    """
    Classe per la rilevazione di segnali whooping in registrazioni audio mono.
    
    Utilizza l'analisi HNR (Harmonic-to-Noise Ratio) su finestre temporali
    per identificare eventi acustici armonici caratteristici del whooping delle api.
    
    Parameters
    ----------
    signal : np.ndarray
        Segnale audio mono
    sr : int
        Sample rate del segnale audio
    window_length_ms : float, default=50
        Lunghezza della finestra di analisi in millisecondi
    hop_length_ms : float, default=10
        Hop length tra finestre consecutive in millisecondi
    f0_min : float, default=250
        Frequenza fondamentale minima attesa (Hz)
    f0_max : float, default=700
        Frequenza fondamentale massima attesa (Hz)
    window_type : str, default='hamming'
        Tipo di finestra ('hamming', 'hann', 'blackman')
    lowpass_cutoff : float, default=15000
        Frequenza di taglio del filtro passa-basso (Hz)
    highpass_cutoff : float, default=2500
        Frequenza di taglio del filtro passa-alto (Hz)
    normalize : bool, default=True
        Se True, normalizza il segnale a RMS target
    target_rms : float, default=0.1
        Valore RMS target per la normalizzazione
    
    Attributes
    ----------
    results_ : dict
        Risultati dell'ultima analisi HNR
    peaks_ : np.ndarray
        Indici dei picchi rilevati
    peak_times_ : np.ndarray
        Tempi centrali dei picchi (secondi)
    peak_windows_ : list of tuple
        Liste di tuple (start_time, end_time) per ogni finestra di picco
    threshold_ : float
        Valore di threshold HNR utilizzato per il rilevamento
    """
    
    def __init__(self, signal, sr, window_length_ms=50, hop_length_ms=10,
                 f0_min=250, f0_max=700, window_type='hamming',
                 lowpass_cutoff=15000, highpass_cutoff=2500,
                 normalize=True, target_rms=0.1):
        
        self.signal = signal
        self.sr = sr
        self.window_length_ms = window_length_ms
        self.hop_length_ms = hop_length_ms
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.window_type = window_type
        self.lowpass_cutoff = lowpass_cutoff
        self.highpass_cutoff = highpass_cutoff
        self.normalize = normalize
        self.target_rms = target_rms
        
        # Attributi per i risultati
        self.results_ = None
        self.peaks_ = None
        self.peak_times_ = None
        self.peak_windows_ = None
        self.threshold_ = None
    
    # ==================== METODI STATICI DI UTILITÀ ====================
    
    @staticmethod
    def autocorrelation(signal):
        """Calcola l'autocorrelazione del segnale."""
        n = len(signal)
        signal = signal - np.mean(signal)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / np.arange(n, 0, -1)
        return autocorr
    
    @staticmethod
    def estimate_f0_autocorr(signal, sr, f0_min=75, f0_max=400):
        """Stima la frequenza fondamentale tramite autocorrelazione."""
        autocorr = WhoopDetector.autocorrelation(signal)
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
    
    @staticmethod
    def calculate_hnr(signal, sr, f0_min=75, f0_max=400):
        """Calcola l'HNR (Harmonic-to-Noise Ratio) del segnale."""
        f0, autocorr, T0 = WhoopDetector.estimate_f0_autocorr(signal, sr, f0_min, f0_max)
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
    
    @staticmethod
    def apply_window(signal, window_type='hamming'):
        """Applica una finestra al segnale."""
        if window_type == 'hamming':
            window = np.hamming(len(signal))
        elif window_type == 'hann':
            window = np.hanning(len(signal))
        elif window_type == 'blackman':
            window = np.blackman(len(signal))
        else:
            window = np.ones(len(signal))
        return signal * window
    
    @staticmethod
    def apply_lowpass_filter(signal, sr, cutoff_hz=15000):
        """Applica un filtro passa-basso Butterworth."""
        nyquist = sr / 2
        normalized_cutoff = cutoff_hz / nyquist
        if normalized_cutoff >= 1:
            normalized_cutoff = 0.99
        b, a = butter(4, normalized_cutoff, btype='low')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    @staticmethod
    def apply_highpass_filter(signal, sr, cutoff_hz=2500):
        """Applica un filtro passa-alto Butterworth."""
        nyquist = sr / 2
        normalized_cutoff = cutoff_hz / nyquist
        if normalized_cutoff >= 1:
            normalized_cutoff = 0.99
        b, a = butter(4, normalized_cutoff, btype='high')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    @staticmethod
    def savitzky_golay_filter(data, window_size=11, polyorder=2):
        """Applica il filtro Savitzky-Golay per smoothing."""
        if window_size % 2 == 0:
            window_size += 1
        if polyorder >= window_size:
            polyorder = window_size - 2
        filtered = savgol_filter(data, window_size, polyorder)
        return filtered
    
    @staticmethod
    def normalize_signal_rms(signal, target_rms=0.1):
        """Normalizza il segnale a un RMS target."""
        current_rms = np.sqrt(np.mean(signal**2))
        if current_rms < 1e-6:
            return signal
        scaling_factor = target_rms / current_rms
        return signal * scaling_factor
    
    @staticmethod
    def merge_overlapping_windows(peak_windows, peaks, peaks_times):
        """Unisce finestre temporali sovrapposte."""
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()
        if isinstance(peaks_times, np.ndarray):
            peaks_times = peaks_times.tolist()
        
        i = 0
        while i < len(peak_windows) - 1:
            start_i, end_i = peak_windows[i]
            start_next, end_next = peak_windows[i + 1]
            
            if end_i >= start_next:
                merged_window = (start_i, max(end_i, end_next))
                peak_windows[i] = merged_window
                del peak_windows[i + 1]
                del peaks[i + 1]
                del peaks_times[i + 1]
            else:
                i += 1
        
        return peak_windows, peaks, peaks_times
    
    # ==================== METODI PRINCIPALI ====================
    
    def preprocess_signal(self, signal):
        """
        Applica preprocessing al segnale: filtri e normalizzazione.
        
        Parameters
        ----------
        signal : np.ndarray
            Segnale audio mono
        
        Returns
        -------
        np.ndarray
            Segnale preprocessato
        """
        # Normalizzazione
        if self.normalize:
            signal = self.normalize_signal_rms(signal, self.target_rms)
        
        # Filtri
        signal = self.apply_lowpass_filter(signal, self.sr, self.lowpass_cutoff)
        signal = self.apply_highpass_filter(signal, self.sr, self.highpass_cutoff)
        
        return signal
    
    def analyze_hnr(self, signal):
        """
        Analizza l'HNR del segnale su finestre temporali.
        
        Parameters
        ----------
        signal : np.ndarray
            Segnale audio preprocessato
        
        Returns
        -------
        dict
            Dizionario con chiavi: 'hnr_values', 'f0_values', 'time_centers',
            'valid_frames', 'total_frames'
        """
        window_length = int(self.window_length_ms * self.sr / 1000)
        hop_length = int(self.hop_length_ms * self.sr / 1000)
        
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
            frame_windowed = self.apply_window(frame, self.window_type)
            hnr_db, f0 = self.calculate_hnr(frame_windowed, self.sr, 
                                           self.f0_min, self.f0_max)
            time_center = (start + window_length/2) / self.sr
            
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
    
    def postprocess_results(self, results, smoothing_window=5):
        """
        Applica postprocessing ai risultati: riempimento NaN e smoothing.
        
        Parameters
        ----------
        results : dict
            Risultati dell'analisi HNR
        smoothing_window : int, default=5
            Dimensione della finestra per il filtro Savitzky-Golay
        
        Returns
        -------
        dict
            Risultati con chiavi aggiuntive 'hnr_smoothed' e 'f0_smoothed'
        """
        results_processed = results.copy()
        hnr_values = results['hnr_values'].astype(float)
        f0_values = results['f0_values'].astype(float)
        
        hnr_valid = hnr_values[~np.isnan(hnr_values)]
        f0_valid = f0_values[~np.isnan(f0_values)]
        
        if len(hnr_valid) == 0 or len(f0_valid) == 0:
            return results_processed
        
        hnr_filled = hnr_values.copy()
        f0_filled = f0_values.copy()
        
        hnr_filled[np.isnan(hnr_values)] = np.mean(hnr_valid)
        f0_filled[np.isnan(f0_values)] = np.mean(f0_valid)
        
        results_processed['hnr_smoothed'] = self.savitzky_golay_filter(
            hnr_filled, smoothing_window)
        results_processed['f0_smoothed'] = self.savitzky_golay_filter(
            f0_filled, smoothing_window)
        
        return results_processed
    
    def find_peaks(self, hnr_values, time_centers, percentile=99, offset=2):
        """
        Rileva i picchi HNR usando un approccio basato su percentili.
        
        Parameters
        ----------
        hnr_values : np.ndarray
            Valori HNR smoothed
        time_centers : np.ndarray
            Tempi centrali delle finestre
        percentile : float, default=99
            Percentile della distribuzione per definire la soglia
        offset : float, default=2
            Offset in dB da aggiungere al percentile
        
        Returns
        -------
        tuple
            (peaks, peak_times, threshold) dove:
            - peaks: indici dei picchi
            - peak_times: tempi dei picchi (secondi)
            - threshold: valore di soglia utilizzato (dB)
        """
        hnr = np.array(hnr_values, dtype=float)
        valid = hnr[~np.isnan(hnr)]
        
        percentile_value = np.percentile(valid, percentile)
        threshold = percentile_value + offset
        
        peaks, _ = find_peaks(hnr, height=threshold)
        peak_times = time_centers[peaks]
        
        return peaks, peak_times, threshold
    
    def create_peak_windows(self, peak_times, signal_duration, window_sec=3.0):
        """
        Crea finestre temporali centrate sui picchi rilevati.
        
        Parameters
        ----------
        peak_times : np.ndarray
            Tempi centrali dei picchi (secondi)
        signal_duration : float
            Durata totale del segnale (secondi)
        window_sec : float, default=3.0
            Durata della finestra attorno al picco (secondi)
        
        Returns
        -------
        list of tuple
            Lista di tuple (start_time, end_time) per ogni finestra
        """
        peak_windows = []
        for peak_time in peak_times:
            start_time = max(0, peak_time - window_sec / 2)
            end_time = min(signal_duration, peak_time + window_sec / 2)
            peak_windows.append((start_time, end_time))
        return peak_windows
    
    def detect(self, percentile=99.5, offset=4, window_sec=3.0, 
               merge_overlaps=True, smoothing_window=5):
        """
        Metodo principale per la rilevazione di segnali whooping.
        
        Parameters
        ----------
        percentile : float, default=99.5
            Percentile per il rilevamento dei picchi
        offset : float, default=4
            Offset in dB sopra il percentile
        window_sec : float, default=3.0
            Durata della finestra attorno ai picchi (secondi)
        merge_overlaps : bool, default=True
            Se True, unisce finestre sovrapposte
        smoothing_window : int, default=5
            Dimensione finestra per smoothing Savitzky-Golay
        
        Returns
        -------
        dict
            Dizionario con chiavi:
            - 'peaks': indici dei picchi
            - 'peak_times': tempi dei picchi (secondi)
            - 'peak_windows': lista di tuple (start_time, end_time)
            - 'threshold': valore soglia HNR (dB)
            - 'results': risultati completi dell'analisi HNR
        """
        # Preprocessing
        signal_processed = self.preprocess_signal(self.signal)
        
        # Analisi HNR
        results = self.analyze_hnr(signal_processed)
        
        # Postprocessing
        results_smoothed = self.postprocess_results(results, smoothing_window)
        
        # Rilevamento picchi
        peaks, peak_times, threshold = self.find_peaks(
            results_smoothed['hnr_smoothed'],
            results_smoothed['time_centers'],
            percentile=percentile,
            offset=offset
        )
        
        # Creazione finestre
        signal_duration = len(self.signal) / self.sr
        peak_windows = self.create_peak_windows(peak_times, signal_duration, window_sec)
        
        # Merge finestre sovrapposte
        if merge_overlaps and len(peak_windows) > 0:
            peak_windows, peaks, peak_times = self.merge_overlapping_windows(
                peak_windows, peaks, peak_times
            )
        
        # Salva risultati come attributi
        self.results_ = results_smoothed
        self.peaks_ = np.array(peaks)
        self.peak_times_ = np.array(peak_times)
        self.peak_windows_ = peak_windows
        self.threshold_ = threshold
        
        return {
            'peaks': self.peaks_,
            'peak_times': self.peak_times_,
            'peak_windows': self.peak_windows_,
            'threshold': self.threshold_,
            'results': self.results_
        }
    
    def save_segments(self, audio_filename, channel, output_dir="whooping_candidates"):
        """
        Salva un segmento audio candidato whooping come file WAV.
        
        Parameters:
        -----------
        audio_filename : str
            Nome del file audio originale (con path)
        channel : int
            Numero del canale
        output_dir : str
            Cartella principale di output
        """

        # Estrai il nome del file senza path ed estensione
        base_filename = os.path.splitext(os.path.basename(audio_filename))[0]

        # Crea la struttura di cartelle: whooping_candidates/nome_file_originale/
        output_folder = os.path.join(output_dir, base_filename)
        os.makedirs(output_folder, exist_ok=True)

        for i in range(len(self.peak_windows_)):
            start_time, end_time = self.peak_windows_[i]
            peak_time = self.peak_times_[i]
            segment = self.signal[int(start_time * self.sr):int(end_time * self.sr)]

            # Crea il nome del file: ch{channel}_{peak_time}s_{start}-{end}s.wav
            filename = f"ch{channel:02d}_{peak_time:.3f}s_{start_time:.3f}-{end_time:.3f}s.wav"
            output_path = os.path.join(output_folder, filename)

            # Salva il file WAV
            sf.write(output_path, segment, self.sr)

            print(f"Salvato: {output_path}")
    
    
    def extract_segments(self, peak_windows=None):
        """
        Estrae i segmenti audio corrispondenti alle finestre dei picchi.
        
        Parameters
        ----------
        signal : np.ndarray
            Segnale audio originale
        peak_windows : list of tuple, optional
            Lista di finestre (start_time, end_time). Se None, usa self.peak_windows_
        
        Returns
        -------
        list of np.ndarray
            Lista di segmenti audio
        """
        if peak_windows is None:
            peak_windows = self.peak_windows_
        
        if peak_windows is None:
            raise ValueError("Nessuna finestra disponibile. Esegui prima detect().")
        
        segments = []
        for start_time, end_time in peak_windows:
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            segment = self.signal[start_sample:end_sample]
            segments.append(segment)
        
        return segments
    
    def plot_analysis(self, ch_num=1, figsize=(10, 5), dpi=150):
        """
        Visualizza i risultati dell'analisi HNR con i picchi rilevati.
        
        Parameters
        ----------
        ch_num : int, default=1
            Numero del canale (solo per il titolo)
        figsize : tuple, default=(10, 5)
            Dimensione della figura
        dpi : int, default=150
            Risoluzione della figura
        """
        if self.results_ is None:
            raise ValueError("Nessun risultato disponibile. Esegui prima detect().")
        
        time_centers = self.results_['time_centers']
        hnr = self.results_['hnr_smoothed']
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(time_centers, hnr, color='b', linewidth=2.2, label='HNR')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Molto armonico (>20 dB)')
        ax.axhline(y=15, color='orange', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Moderato (>15 dB)')
        ax.axhline(y=13, color='red', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Basso (>13 dB)')
        ax.fill_between(time_centers, 15, 25, color='yellow', alpha=0.07)
        
        if self.threshold_ is not None:
            ax.axhline(y=self.threshold_, color='purple', linestyle='-', 
                      linewidth=2.5, alpha=0.8, 
                      label=f'Threshold ({self.threshold_:.2f} dB)')
        
        if self.peak_windows_ is not None and len(self.peak_windows_) > 0:
            for i, (start, end) in enumerate(self.peak_windows_):
                label = 'Finestra picco' if i == 0 else ''
                ax.axvspan(start, end, alpha=0.15, color='red', label=label)
        
        if self.peaks_ is not None and len(self.peaks_) > 0:
            ax.scatter(time_centers[self.peaks_], hnr[self.peaks_], 
                      color='red', s=100, marker='o', 
                      edgecolors='darkred', linewidths=2,
                      label=f'Picchi rilevati ({len(self.peaks_)})', 
                      zorder=5, alpha=0.8)
        
        ax.set_xlabel('Tempo (s)', fontsize=10, fontweight='bold')
        ax.set_ylabel('HNR (dB)', fontsize=10, fontweight='bold')
        ax.set_title(f'HNR nel tempo - channel {ch_num}', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='best', fontsize=8, framealpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def get_peak_info(self):
        """
        Restituisce informazioni sui picchi rilevati in formato leggibile.
        
        Returns
        -------
        list of dict
            Lista di dizionari con informazioni per ogni picco
        """
        if self.peaks_ is None or self.peak_times_ is None or self.peak_windows_ is None:
            return []
        
        peak_info = []
        for i, (peak_idx, peak_time, window) in enumerate(
            zip(self.peaks_, self.peak_times_, self.peak_windows_)):
            info = {
                'index': i,
                'peak_index': int(peak_idx),
                'peak_time': float(peak_time),
                'window_start': float(window[0]),
                'window_end': float(window[1]),
                'window_duration': float(window[1] - window[0])
            }
            peak_info.append(info)
        
        return peak_info
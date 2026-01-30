import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks




class HarmonicsAnalyzer:
    """
    Classe per l'analisi spettrale delle armoniche di un segnale whoop.
    
    Responsabilità:
    - Calcolo della FFT
    - Analisi dell'energia armonica
    - Visualizzazione e stampa dei risultati
    """
    
    def __init__(self, sr, nfft=4096):
        """
        Inizializza l'analizzatore.
        
        Parameters:
        -----------
        sr : int
            Sample rate dell'audio
        nfft : int
            Numero di punti per la FFT (default: 4096)
        """
        self.sr = sr
        self.nfft = nfft
    
    # ===== SPETTRO =====
    
    def compute_fft(self, frame, nfft=None):
        """
        Calcola la FFT del segnale.
        
        Parameters:
        -----------
        frame : np.ndarray
            Frame audio nel dominio del tempo
        nfft : int, optional
            Numero di punti per la FFT (usa self.nfft se non specificato)
        
        Returns:
        --------
        spectrum : np.ndarray
            Spettro complesso
        magnitude : np.ndarray
            Magnitudine dello spettro
        """
        if nfft is None:
            nfft = self.nfft
        
        spectrum = np.fft.rfft(frame, n=nfft)
        magnitude = np.abs(spectrum)
        
        return spectrum, magnitude
    
    def get_frequencies(self, nfft=None):
        """
        Calcola l'array delle frequenze corrispondenti alla FFT.
        
        Parameters:
        -----------
        nfft : int, optional
            Numero di punti per la FFT
        
        Returns:
        --------
        freqs : np.ndarray
            Array delle frequenze (Hz)
        """
        if nfft is None:
            nfft = self.nfft
        
        freqs = np.fft.rfftfreq(nfft, d=1/self.sr)
        return freqs


    def compute_signal_to_harmonic_ratio(self, whoop_segment, f0, center_sample,
                                      window_duration_ms=10, 
                                      num_harmonics=5,
                                      bandwidth_hz=50):
        """
        Calcola il Signal-to-Harmonic Ratio (SHR).
        
        Misura la frazione dell'energia totale del segnale che cade
        nelle bande armoniche teoriche.
        
        Parameters:
        -----------
        whoop_segment : np.ndarray
            Segmento audio da analizzare
        f0 : float
            Frequenza fondamentale stimata (Hz)
        center_sample : int
            Campione centrale per l'analisi
        window_duration_ms : float
            Durata della finestra (default: 10ms)
        num_harmonics : int
            Numero di armoniche da considerare (default: 5)
        bandwidth_hz : float
            Larghezza di banda attorno a ogni armonica (default: 50 Hz)
        
        Returns:
        --------
        shr_dict : dict
            {
                'shr': float (0-1),           # Main metric
                'harmonic_energy': float,     # Energia nelle bande armoniche
                'total_energy': float,        # Energia totale
                'noise_energy': float,        # Energia fuori dalle bande
                'per_harmonic_energy': array, # Breakdown per armonica
                'per_harmonic_ratio': array,  # Ratio per armonica
                'spectrum': array,            # Lo spettro calcolato
                'freqs': array,               # Frequenze
                'harmonic_frequencies': array # Frequenze teoriche armoniche
            }
        """
        
        # ===== CALCOLA SPETTRO =====
        window_length = int(window_duration_ms * self.sr / 1000)
        half_window = window_length // 2
        window_start = max(0, center_sample - half_window)
        window_end = min(len(whoop_segment), center_sample + half_window)
        
        analysis_window = whoop_segment[window_start:window_end]
        
        if len(analysis_window) < 2:
            return None
        
        # Applica finestra Hamming
        window = np.hamming(len(analysis_window))
        windowed_signal = analysis_window * window
        
        # FFT
        spectrum, magnitude = self.compute_fft(windowed_signal, nfft=self.nfft)
        freqs = self.get_frequencies(nfft=self.nfft)
        
        # ===== CALCOLA ENERGIA TOTALE =====
        # L'energia nel dominio della frequenza è la somma dei quadrati
        total_energy = np.sum(magnitude ** 2)
        
        # ===== CALCOLA ENERGIA NELLE BANDE ARMONICHE =====
        harmonic_frequencies = np.array([f0 * (n + 1) for n in range(num_harmonics)])
        
        harmonic_energies = np.zeros(num_harmonics)
        per_harmonic_ratios = np.zeros(num_harmonics)
        
        for i, harmonic_freq in enumerate(harmonic_frequencies):
            freq_min = harmonic_freq - bandwidth_hz / 2
            freq_max = harmonic_freq + bandwidth_hz / 2
            
            # Indici della banda
            idx_band = (freqs >= freq_min) & (freqs <= freq_max)
            
            if np.any(idx_band):
                harmonic_energies[i] = np.sum(magnitude[idx_band] ** 2)
                per_harmonic_ratios[i] = harmonic_energies[i] / (total_energy + 1e-10)
        
        # Energia totale nelle bande armoniche
        total_harmonic_energy = np.sum(harmonic_energies)
        
        # Energia nel rumore (fuori dalle bande armoniche)
        noise_energy = total_energy - total_harmonic_energy
        
        # ===== SIGNAL-TO-HARMONIC RATIO =====
        shr = total_harmonic_energy / (total_energy + 1e-10)
        
        return {
            'shr': shr,
            'harmonic_energy': total_harmonic_energy,
            'total_energy': total_energy,
            'noise_energy': noise_energy,
            'per_harmonic_energy': harmonic_energies,
            'per_harmonic_ratio': per_harmonic_ratios,
            'spectrum': magnitude,
            'freqs': freqs,
            'harmonic_frequencies': harmonic_frequencies,
            'window_start': window_start,
            'window_end': window_end,
            'window_duration_ms': (window_end - window_start) / self.sr * 1000,
            'f0': f0,
            'bandwidth_hz': bandwidth_hz,
            'num_harmonics': num_harmonics
        }
    

    def plot_spectrum_with_peaks(self, spectrum, freqs, harmonic_frequencies, 
                            peaks, bandwidth_hz=50, f0=None, freq_max=5000):
        """
        Plotta spettro con:
        - Picchi evidenziati (punti rossi)
        - Linee nere alle frequenze armoniche teoriche
        - Bande colorate attorno alle frequenze armoniche
        
        Parameters:
        -----------
        spectrum : np.ndarray
            Magnitudini dello spettro
        freqs : np.ndarray
            Array delle frequenze
        harmonic_frequencies : np.ndarray
            Frequenze teoriche armoniche [f0, 2*f0, 3*f0, ...]
        peaks : np.ndarray
            Indici dei picchi (output di find_peaks)
        bandwidth_hz : float
            Larghezza banda attorno a ogni armonica
        f0 : float, optional
            Frequenza fondamentale (per il titolo)
        freq_max : float
            Frequenza massima da visualizzare
        """
        
        
        # Limita le frequenze visualizzate
        idx_max = np.argmin(np.abs(freqs - freq_max))
        
        plt.figure(figsize=(14, 6))
        
        # Plot spettro
        plt.plot(freqs[:idx_max], spectrum[:idx_max], color='black', linewidth=1, alpha=0.6)
        plt.fill_between(freqs[:idx_max], spectrum[:idx_max], alpha=0.1, color='gray')
        
        # Colori per le bande
        colors = plt.cm.Set3(np.linspace(0, 1, len(harmonic_frequencies)))
        
        # Disegna bande e linee armoniche
        for i, (harm_freq, color) in enumerate(zip(harmonic_frequencies, colors)):
            if harm_freq > freq_max:
                break
            
            freq_min = harm_freq - bandwidth_hz / 2
            freq_max_band = harm_freq + bandwidth_hz / 2
            
            # Banda ombreggiata
            plt.axvspan(freq_min, freq_max_band, alpha=0.15, color=color)
            
            # Linea verticale al centro
            plt.axvline(harm_freq, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
        
        # Plot picchi
        peak_freqs = freqs[peaks]
        peak_mags = spectrum[peaks]
        plt.scatter(peak_freqs, peak_mags, color='red', s=80, marker='o', 
                edgecolors='darkred', linewidth=1.5, zorder=5, label='Detected peaks')
        
        # Formattazione
        plt.xlabel('Frequency (Hz)', fontsize=11)
        plt.ylabel('Magnitude', fontsize=11)
        
        title = 'Spectrum with Peaks and Harmonic Bands'
        if f0 is not None:
            title += f' (f₀ = {f0:.1f} Hz)'
        plt.title(title, fontsize=12, fontweight='bold')
        
        plt.xlim(0, freq_max)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


    def compute_weighted_shr(self, whoop_segment, best_queue_dic,
                            window_duration_ms=10,
                            num_harmonics=5,
                            bandwidth_hz=50,
                            prominence_threshold_ratio=0.1,
                            plot_core=False,
                            plot_verbose=False,
                            verbose=False):
        """
        Calcola metriche di allineamento spettrale per validare la struttura armonica.
        
        I pesi sono basati solo sul NUMERO di picchi allineati per frame.
        
        Returns:
        --------
        result : dict
            {
                'whar': float,                        # ← Score weighted SHR (metrica principale 1)
                'alignment_max_peaks': int,           # ← Picchi allineati nel frame migliore (metrica principale 2)
                'num_aligned_peaks_mean': float,      # Media numero picchi allineati
                'aligned_peaks_per_frame': array,     # Numero picchi allineati per frame
                'weights': array,                     # Pesi per ogni frame
                'shr_values': array,                  # SHR per ogni frame
                'best_f0': float,                     # f0 del frame con più picchi allineati
                'best_sample': int,                   # Sample del frame con più picchi allineati
                'num_highly_aligned_frames': int,     # Quanti frame hanno ≥3 picchi allineati
                'alignment_details': list,
            }
        """
        
        shr_values = []
        aligned_peaks_per_frame = []
        alignment_details_list = []
        
        # === FASE 1: Calcola SHR e conta picchi allineati per ogni frame ===
        for idx in range(len(best_queue_dic['frame_indices'])):
            f0 = best_queue_dic['f0_values'][idx]
            sample = best_queue_dic['samples'][idx]
            
            # Calcola SHR
            shr_info = self.compute_signal_to_harmonic_ratio(
                whoop_segment=whoop_segment,
                f0=f0,
                center_sample=sample,
                window_duration_ms=window_duration_ms,
                num_harmonics=num_harmonics,
                bandwidth_hz=bandwidth_hz
            )
            
            shr_values.append(shr_info['shr'])
            
            # Trova picchi nello spettro
            peaks, _ = find_peaks(
                shr_info['spectrum'], 
                prominence=np.max(shr_info['spectrum']) * prominence_threshold_ratio,
                distance=5
            )
            
            # Conta quanti picchi cadono nelle bande armoniche
            peak_freqs = shr_info['freqs'][peaks]
            num_aligned = 0
            
            for peak_freq in peak_freqs:
                for harm_freq in shr_info['harmonic_frequencies']:
                    freq_min = harm_freq - bandwidth_hz / 2
                    freq_max = harm_freq + bandwidth_hz / 2
                    
                    if freq_min <= peak_freq <= freq_max:
                        num_aligned += 1
                        break
            
            aligned_peaks_per_frame.append(num_aligned)
            
            # Plot verbose opzionale
            if plot_verbose:
                self.plot_spectrum_with_peaks(
                    spectrum=shr_info['spectrum'],
                    freqs=shr_info['freqs'],
                    harmonic_frequencies=shr_info['harmonic_frequencies'],
                    peaks=peaks,
                    bandwidth_hz=bandwidth_hz,
                    f0=f0,
                    freq_max=5000
                )
            
            alignment_details_list.append({
                'frame_idx': idx,
                'f0': f0,
                'shr': shr_info['shr'],
                'num_aligned_peaks': num_aligned,
                'num_total_peaks': len(peaks)
            })
        
        shr_values = np.array(shr_values)
        aligned_peaks_per_frame = np.array(aligned_peaks_per_frame)
        
        # === FASE 2: Calcola pesi dal numero di picchi allineati ===
        max_aligned = np.max(aligned_peaks_per_frame)
        
        if max_aligned > 0:
            # Normalizza 0-1 basandoti sul massimo
            normalized_peaks = aligned_peaks_per_frame / max_aligned
        else:
            normalized_peaks = np.zeros_like(aligned_peaks_per_frame)
        
        # Aggiungi peso minimo per evitare zero
        min_weight = 0.05
        weights = min_weight + (1.0 - min_weight) * normalized_peaks
        
        # === FASE 3: Calcola WHAR (Weighted Harmonic-to-Acoustic Ratio) ===
        weighted_shr_values = shr_values * weights
        whar = np.mean(weighted_shr_values)
        
        # === FASE 4: Metriche principali ===
        best_idx = np.argmax(aligned_peaks_per_frame)
        alignment_max_peaks = int(aligned_peaks_per_frame[best_idx])
        
        num_highly_aligned = np.sum(aligned_peaks_per_frame >= 3)  # almeno 3 picchi
        num_aligned_peaks_mean = np.mean(aligned_peaks_per_frame)
        
        # === FASE 5: Visualizzazione opzionale ===
        if verbose:
            self._print_weighted_shr_table(
                shr_values=shr_values,
                aligned_peaks_per_frame=aligned_peaks_per_frame,
                weights=weights,
                alignment_details_list=alignment_details_list
            )
                
        if plot_core:
            self._plot_weighted_shr_analysis(
                shr_values=shr_values,
                alignment_scores=normalized_peaks,
                aligned_peaks_per_frame=aligned_peaks_per_frame,
                weights=weights,
                f0_values=best_queue_dic['f0_values'],
                alignment_details_list=alignment_details_list
            )


        return {
            'whar': whar,
            'alignment_max_peaks': alignment_max_peaks,
            'num_aligned_peaks_mean': num_aligned_peaks_mean,
            'aligned_peaks_per_frame': aligned_peaks_per_frame,
            'weights': weights,
            'shr_values': shr_values,
            'best_f0': best_queue_dic['f0_values'][best_idx],
            'best_sample': best_queue_dic['samples'][best_idx],
            'num_highly_aligned_frames': int(num_highly_aligned),
            'alignment_details': alignment_details_list,
            'num_frames': len(best_queue_dic['frame_indices'])
        }



    def _plot_weighted_shr_analysis(self, shr_values, alignment_scores, aligned_peaks_per_frame,
                                weights, f0_values, alignment_details_list):
        """
        Visualizza l'analisi WHAR con 3 grafici:
        1. SHR per frame
        2. Numero picchi allineati per frame
        3. SHR pesati (WHAR components) per frame
        """
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        frame_idx = np.arange(len(shr_values))
        
        # --- Subplot 1: SHR ---
        axes[0].bar(frame_idx, shr_values, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axhline(y=np.mean(shr_values), color='blue', linestyle='--', 
                        linewidth=2, label=f'Mean: {np.mean(shr_values):.3f}')
        axes[0].set_ylabel('SHR', fontsize=11)
        axes[0].set_title('SHR per Frame', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_xticks(frame_idx)
        axes[0].set_ylim(0, 1.0)
        axes[0].legend(loc='upper right')
        
        # --- Subplot 2: Numero picchi allineati ---
        axes[1].bar(frame_idx, aligned_peaks_per_frame, color='darkgreen', alpha=0.7, edgecolor='black')
        axes[1].axhline(y=np.mean(aligned_peaks_per_frame), color='red', linestyle='--', 
                        linewidth=2, label=f'Mean: {np.mean(aligned_peaks_per_frame):.1f}')
        axes[1].axhline(y=np.max(aligned_peaks_per_frame), color='green', linestyle=':', 
                        linewidth=2, label=f'Max: {int(np.max(aligned_peaks_per_frame))}')
        axes[1].set_ylabel('Number of Aligned Peaks', fontsize=11)
        axes[1].set_title('Aligned Peaks per Frame', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_xticks(frame_idx)
        axes[1].legend(loc='upper right')
        
        # --- Subplot 3: SHR pesati (weighted SHR) ---
        weighted_shr = shr_values * weights
        colors_weight = plt.cm.RdYlGn(weights)
        axes[2].bar(frame_idx, weighted_shr, color=colors_weight, edgecolor='black', alpha=0.8)
        axes[2].axhline(y=np.mean(weighted_shr), color='black', linestyle='--', 
                        linewidth=2, label=f'WHAR: {np.mean(weighted_shr):.3f}')
        axes[2].set_ylabel('Weighted SHR', fontsize=11)
        axes[2].set_xlabel('Frame Index', fontsize=11)
        axes[2].set_title('Weighted SHR per Frame (WHAR Components)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_xticks(frame_idx)
        axes[2].set_ylim(0, 1.0)
        axes[2].legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()


    def _print_weighted_shr_table(self, shr_values, aligned_peaks_per_frame, weights, 
                                alignment_details_list):
        """
        Stampa tabella riassuntiva dell'analisi WHAR frame per frame.
        """
        
        print(f"\n{'='*95}")
        print(f"WEIGHTED SHR ANALYSIS - FRAME BY FRAME")
        print(f"{'='*95}")
        print(f"{'Frame':<7} {'f₀ (Hz)':<12} {'SHR':<10} {'Peaks':<10} {'Weight':<10} {'W-SHR':<10}")
        print(f"{'-'*95}")
        
        for detail in alignment_details_list:
            idx = detail['frame_idx']
            w_shr = shr_values[idx] * weights[idx]
            peaks_str = f"{detail['num_aligned_peaks']}/{detail['num_total_peaks']}"
            
            print(f"{detail['frame_idx']:<7} {detail['f0']:<12.1f} {detail['shr']:<10.3f} "
                f"{peaks_str:<10} {weights[idx]:<10.3f} {w_shr:<10.4f}")
        
        print(f"{'-'*95}")
        print(f"{'MEDIE:':<7} {'-':<12} {np.mean(shr_values):<10.3f} "
            f"{np.mean(aligned_peaks_per_frame):<10.1f} {np.mean(weights):<10.3f} "
            f"{np.mean(shr_values * weights):<10.4f}")
        
        print(f"Max Aligned Peaks: {np.max(aligned_peaks_per_frame)}")
        
        print(f"{'='*95}\n")







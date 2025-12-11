import numpy as np
import matplotlib.pyplot as plt


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
    
    # ===== ANALISI ARMONICHE =====
    
    def analyze_harmonics(self, whoop_segment, f0, center_sample, 
                         window_duration_ms=10, num_harmonics=10, bandwidth_hz=50):
        """
        Analizza le armoniche di un segnale whoop in una finestra breve centrata 
        attorno al punto di f0 mediano (per evitare il drift di frequenza).
        
        Parameters:
        -----------
        whoop_segment : np.ndarray
            Segmento audio contenente il whoop
        f0 : float
            Frequenza fondamentale stimata (Hz)
        center_sample : int
            Indice del campione centrale (relativo a whoop_segment) attorno al quale analizzare
        window_duration_ms : float
            Durata della finestra di analisi in millisecondi (default: 10ms)
        num_harmonics : int
            Numero di armoniche da analizzare (default: 10)
        bandwidth_hz : float
            Larghezza di banda in Hz attorno a ciascuna armonica (default: 50 Hz)
        
        Returns:
        --------
        harmonics_info : dict or None
            Dizionario con le informazioni sull'analisi armonica, o None se errore
        """
        
        # Calcola la lunghezza della finestra in campioni
        window_length = int(window_duration_ms * self.sr / 1000)
        
        # Calcola inizio e fine della finestra centrata
        half_window = window_length // 2
        window_start = max(0, center_sample - half_window)
        window_end = min(len(whoop_segment), center_sample + half_window)
        
        # Estrai la finestra
        analysis_window = whoop_segment[window_start:window_end]
        
        # Verifica che ci siano abbastanza campioni
        if len(analysis_window) < 2:
            print(f"⚠️  Finestra troppo corta ({len(analysis_window)} campioni), impossibile analizzare")
            return None
        
        # Applica finestra (Hamming) per ridurre leakage spettrale
        window = np.hamming(len(analysis_window))
        windowed_signal = analysis_window * window
        
        # Calcola FFT
        spectrum, magnitude = self.compute_fft(windowed_signal, nfft=self.nfft)
        freqs = self.get_frequencies(nfft=self.nfft)
        
        # Calcola le frequenze armoniche teoriche (f0, 2*f0, 3*f0, ...)
        harmonic_frequencies = np.array([f0 * (n + 1) for n in range(num_harmonics)])
        
        # Array per memorizzare i risultati
        harmonic_energies = np.zeros(num_harmonics)
        harmonic_magnitudes = np.zeros(num_harmonics)
        
        # Per ogni armonica, calcola l'energia in una banda attorno alla frequenza teorica
        for i, harmonic_freq in enumerate(harmonic_frequencies):
            # Definisci la banda di frequenza attorno all'armonica
            freq_min = harmonic_freq - bandwidth_hz / 2
            freq_max = harmonic_freq + bandwidth_hz / 2
            
            # Trova gli indici corrispondenti
            idx_band = (freqs >= freq_min) & (freqs <= freq_max)
            
            if np.any(idx_band):
                # Energia: somma dei quadrati delle magnitudini nella banda
                harmonic_energies[i] = np.sum(magnitude[idx_band] ** 2)
                
                # Magnitudine di picco nella banda
                harmonic_magnitudes[i] = np.max(magnitude[idx_band])
            else:
                harmonic_energies[i] = 0
                harmonic_magnitudes[i] = 0
        
        # Calcola i ratio rispetto alla fondamentale (prima armonica)
        if harmonic_energies[0] > 0:
            harmonic_energy_ratios = harmonic_energies / harmonic_energies[0]
        else:
            harmonic_energy_ratios = np.zeros(num_harmonics)
        
        # Energia armonica totale (somma di tutte le armoniche)
        total_harmonic_energy = np.sum(harmonic_energies)
        
        # Calcola durata effettiva della finestra
        actual_window_duration_ms = (window_end - window_start) / self.sr * 1000
        
        # Assembla il dizionario dei risultati
        harmonics_info = {
            'f0': f0,
            'center_sample': center_sample,
            'window_start': window_start,
            'window_end': window_end,
            'window_duration_ms': actual_window_duration_ms,
            'harmonic_frequencies': harmonic_frequencies,
            'harmonic_energies': harmonic_energies,
            'harmonic_magnitudes': harmonic_magnitudes,
            'harmonic_energy_ratios': harmonic_energy_ratios,
            'total_harmonic_energy': total_harmonic_energy,
            'spectrum': magnitude,
            'freqs': freqs
        }
        
        return harmonics_info
    
    # ===== OUTPUT =====
    
    def print_harmonics_summary(self, harmonics_info, top_n=5):
        """
        Stampa un riassunto testuale dell'analisi delle armoniche.
        
        Parameters:
        -----------
        harmonics_info : dict
            Dizionario restituito da analyze_harmonics
        top_n : int
            Numero di top armoniche da stampare (default: 5)
        """
        if harmonics_info is None:
            print("Nessuna informazione armonica disponibile")
            return
        
        print(f"\n{'='*70}")
        print(f"ANALISI ARMONICHE (Finestra centrata su f0 mediano)")
        print(f"{'='*70}")
        print(f"Frequenza fondamentale (f₀): {harmonics_info['f0']:.2f} Hz")
        print(f"Campione centrale: {harmonics_info['center_sample']}")
        print(f"Finestra analisi: [{harmonics_info['window_start']}, {harmonics_info['window_end']}]")
        print(f"Durata finestra: {harmonics_info['window_duration_ms']:.2f} ms")
        print(f"Energia armonica totale: {harmonics_info['total_harmonic_energy']:.2e}")
        print(f"\n{'Top ' + str(top_n) + ' Armoniche per Energia':^70}")
        print(f"{'-'*70}")
        print(f"{'#':<5} {'Freq (Hz)':<12} {'Energia':<15} {'Magnitudine':<15} {'Ratio vs f₀':<15}")
        print(f"{'-'*70}")
        
        # Ordina per energia (descending)
        sorted_indices = np.argsort(harmonics_info['harmonic_energies'])[::-1]
        
        for rank, idx in enumerate(sorted_indices[:top_n], 1):
            freq = harmonics_info['harmonic_frequencies'][idx]
            energy = harmonics_info['harmonic_energies'][idx]
            magnitude = harmonics_info['harmonic_magnitudes'][idx]
            ratio = harmonics_info['harmonic_energy_ratios'][idx]
            
            print(f"H{idx+1:<3} {freq:<12.1f} {energy:<15.2e} {magnitude:<15.2f} {ratio:<15.3f}")
        
        print(f"{'='*70}\n")
    
    def plot_harmonics_analysis(self, harmonics_info, freq_max=5000, highlight_first_n=3):
        """
        Visualizza l'analisi delle armoniche con 3 subplot.
        
        Parameters:
        -----------
        harmonics_info : dict
            Dizionario restituito da analyze_harmonics
        freq_max : float
            Frequenza massima da visualizzare nello spettro (Hz, default: 5000)
        highlight_first_n : int
            Numero di armoniche da evidenziare (default: 3)
        """
        
        if harmonics_info is None:
            print("Nessuna informazione armonica disponibile per il plot")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # --- SUBPLOT 1: Spettro con armoniche evidenziate ---
        freqs = harmonics_info['freqs']
        spectrum = harmonics_info['spectrum']
        
        # Limita la visualizzazione alla freq_max
        idx_max = np.argmin(np.abs(freqs - freq_max))
        
        ax1.plot(freqs[:idx_max], spectrum[:idx_max], linewidth=0.8, color='gray', 
                alpha=0.7, label='Spettro')
        
        # Colori per le armoniche
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
        
        # Disegna le armoniche
        for i, (freq, mag) in enumerate(zip(harmonics_info['harmonic_frequencies'], 
                                             harmonics_info['harmonic_magnitudes'])):
            color = colors[i % len(colors)]
            
            if freq <= freq_max:
                if i < highlight_first_n:
                    # Armonica evidenziata
                    ax1.axvline(x=freq, color=color, linestyle='--', linewidth=2, alpha=0.8)
                    # Etichetta
                    ax1.text(freq, ax1.get_ylim()[1] * 0.95, f'H{i+1}\n{freq:.0f}Hz', 
                            rotation=0, ha='center', va='top', fontsize=9, color=color,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    alpha=0.8, edgecolor=color))
                else:
                    # Armonica debole
                    ax1.axvline(x=freq, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax1.set_xlabel('Frequenza (Hz)', fontsize=11)
        ax1.set_ylabel('Magnitudine', fontsize=11)
        ax1.set_title(f'Spettro con Armoniche Evidenziate (f₀ = {harmonics_info["f0"]:.1f} Hz)', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, freq_max)
        ax1.legend(loc='upper right')
        
        # --- SUBPLOT 2: Energia delle armoniche ---
        harmonic_numbers = np.arange(1, len(harmonics_info['harmonic_energies']) + 1)
        
        bars = ax2.bar(harmonic_numbers, harmonics_info['harmonic_energies'], 
                      color=colors[:len(harmonic_numbers)], alpha=0.7, edgecolor='black')
        
        # Evidenzia le prime N armoniche con bordo più spesso
        for i in range(min(highlight_first_n, len(bars))):
            bars[i].set_linewidth(2.5)
        
        ax2.set_xlabel('Numero Armonica', fontsize=11)
        ax2.set_ylabel('Energia', fontsize=11)
        ax2.set_title('Energia delle Armoniche', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(harmonic_numbers)
        
        # Aggiungi valori sopra le barre per le prime N
        for i in range(min(highlight_first_n, len(bars))):
            height = bars[i].get_height()
            ax2.text(bars[i].get_x() + bars[i].get_width()/2., height,
                    f'{height:.1e}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # --- SUBPLOT 3: Ratio rispetto alla fondamentale ---
        ax3.bar(harmonic_numbers, harmonics_info['harmonic_energy_ratios'], 
               color=colors[:len(harmonic_numbers)], alpha=0.7, edgecolor='black')
        
        # Linea di riferimento per la fondamentale
        ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label='Fondamentale (riferimento)')
        
        ax3.set_xlabel('Numero Armonica', fontsize=11)
        ax3.set_ylabel('Ratio rispetto a f₀', fontsize=11)
        ax3.set_title('Ratio Energia Armoniche rispetto alla Fondamentale', 
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticks(harmonic_numbers)
        ax3.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()


# ===== ESEMPIO DI UTILIZZO =====

if __name__ == "__main__":
    import librosa
    
    # Carica un file audio
    audio_path = "path/to/whoop.wav"
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Crea l'analizzatore
    analyzer = HarmonicsAnalyzer(sr=sr, nfft=4096)
    
    # Analizza le armoniche
    f0 = 300.0  # Hz (esempio)
    center_sample = len(y) // 2
    
    harmonics_info = analyzer.analyze_harmonics(
        whoop_segment=y,
        f0=f0,
        center_sample=center_sample,
        window_duration_ms=10,
        num_harmonics=10,
        bandwidth_hz=50
    )
    
    # Stampa il riassunto
    analyzer.print_harmonics_summary(harmonics_info, top_n=5)
    
    # Visualizza i risultati
    analyzer.plot_harmonics_analysis(harmonics_info, freq_max=5000, highlight_first_n=3)

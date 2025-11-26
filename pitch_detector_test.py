import sounddevice as sd
import os
from whoop_detector import WhoopDetector
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import csv
from datetime import datetime


def compute_fft(y, sr, nfft=None):
    # Calcola la FFT
    if nfft is None:
        nfft = len(y)
    
    spectrum = np.fft.rfft(y, n=nfft)  # reducted FFT solo frequenze positive, zero padding se nfft>len(y) per incrementare risoluzione
    magnitude = np.abs(spectrum)   # Magnitudine

    return spectrum, magnitude

def plot_spectrum(magnitude, sr, freq_min=2500, freq_max=15000, nfft=None, fund_freq_grand_truth=250):
    # Calcola le frequenze corrispondenti
    if nfft is None:
        nfft = (len(magnitude) - 1) * 2
    
    freqs = np.fft.rfftfreq(nfft, d=1/sr)

    # Trova gli indici corrispondenti alla banda di interesse
    idx_min = np.argmin(np.abs(freqs - freq_min))
    idx_max = np.argmin(np.abs(freqs - freq_max))
    
    # Estrai solo la banda di interesse
    freqs_band = freqs[idx_min:idx_max+1]
    magnitude_band = magnitude[idx_min:idx_max+1]

    # print max value in band which frequency
    max_idx = np.argmax(magnitude_band)
    print(f"Max magnitude in band: {magnitude_band[max_idx]:.2f} at frequency {freqs_band[max_idx]:.2f} Hz")
    
    # Plot
    plt.figure(figsize=(14, 7))
    
    # Plot spettro
    plt.plot(freqs_band, magnitude_band, linewidth=0.8, label='Spettro')
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('Magnitudine')
    plt.title(f'Spettro del frame ({freq_min/1000:.1f}-{freq_max/1000:.1f} kHz) - Risoluzione: {sr/nfft:.1f} Hz/bin')
    plt.grid(True, alpha=0.3, which='both')
    plt.xlim(freq_min, freq_max)
    
    # Aggiungi linee verticali per le armoniche della fondamentale
    # Calcola quali armoniche cadono nella banda visualizzata
    harmonic_number = int(np.ceil(freq_min / fund_freq_grand_truth))  # Prima armonica nella banda
    max_harmonic = int(np.floor(freq_max / fund_freq_grand_truth))     # Ultima armonica nella banda
    
    # for n in range(harmonic_number, max_harmonic + 1):
    #     harmonic_freq = n * fund_freq_grand_truth
    #     plt.axvline(x=harmonic_freq, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    #     # Aggiungi etichetta per ogni armonica
    #     plt.text(harmonic_freq, plt.ylim()[1] * 0.95, f'{n}×f₀\n{harmonic_freq:.0f}Hz', 
    #              rotation=0, ha='center', va='top', fontsize=8, color='red',
    #              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'))
    
    # Aumenta i tick sull'asse X
    plt.xticks(np.arange(freq_min, freq_max + 1, 500))
    plt.minorticks_on()
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()




def estimate_f0_cepstrum(magnitude, sr, f0_min=200, f0_max=400, nfft=None):
    """
    Stima F0 usando cepstrum (versione ottimizzata).
    Prende il frame temporale direttamente.
    
    Parameters
    ----------
    frame : np.ndarray
        Frame audio nel dominio del tempo
    sr : int
        Sample rate
    f0_min, f0_max : float
        Range di ricerca per F0
    
    Returns
    -------
    f0 : float
        Frequenza fondamentale stimata (Hz)
    cepstrum : np.ndarray
        Cepstrum calcolato (per debug/visualizzazione)
    quefrency : np.ndarray
        Quefrency corrispondenti (per debug/visualizzazione)
    """
    
    
    # FFT → log → IFFT (Real Cepstrum)
    # spectrum = np.fft.rfft(signal)
    # magnitude_spectrum = np.abs(spectrum)
    log_spectrum = np.log(magnitude + 1e-10)
    cepstrum = np.fft.irfft(log_spectrum, n=nfft)
    
    # Quefrency (tempo) e F0 candidati
    quefrency = np.arange(len(cepstrum)) / sr
    
    # Range di quefrency corrispondente al range F0
    # f0_min=200Hz → quefrency_max = 1/200 = 0.005s
    # f0_max=400Hz → quefrency_min = 1/400 = 0.0025s
    quefrency_min = 1.0 / f0_max
    quefrency_max = 1.0 / f0_min
    
    valid_range = (quefrency >= quefrency_min) & (quefrency <= quefrency_max)
    
    if not np.any(valid_range):
        return None, cepstrum, quefrency
    
    # Trova il picco nel cepstrum
    cepstrum_valid = np.abs(cepstrum[valid_range])
    quefrency_valid = quefrency[valid_range]
    
    peak_idx = np.argmax(cepstrum_valid)
    peak_quefrency = quefrency_valid[peak_idx]
    
    # Converti quefrency → F0
    f0 = 1.0 / peak_quefrency
    
    return f0, cepstrum, quefrency


def plot_fundamental_frequencies(fundamental_frequencies, method_name):
    plt.figure(figsize=(15, 5)) 
    plt.plot(fundamental_frequencies, marker='o')
    plt.title(f'Frequenza Fondamentale Stimata per Frame - {method_name}')
    plt.xlabel('Frame')
    plt.ylabel('Frequenza Fondamentale (Hz)')
    plt.ylim(200, 500)
    plt.grid(True)
    plt.show()

def analyze_f0_array(freq_array, length_queue=5, hz_threshold=10, threshold_increment=1.5, 
                     plot=False, hop_length=None, sr=None, segment_start_sample=0,
                     padding_start_ms=20, padding_end_ms=20, audio_length=None):
    """
    Analizza un array di frequenze per estrarre f0 trovando la coda più lunga
    di valori con trend discendente e variazione controllata.
    
    Parameters:
    -----------
    freq_array : array-like
        Array di frequenze da analizzare
    length_queue : int
        Numero di punti precedenti da controllare per validare una coda
    hz_threshold : float
        Soglia iniziale in Hz per la differenza con il punto immediatamente precedente
    threshold_increment : float
        Fattore moltiplicativo per incrementare la soglia man mano che si guardano
        punti più lontani (es: 1.5 significa +50% per ogni step indietro)
    plot : bool
        Se True, mostra un plot dell'array originale e della best queue identificata
    hop_length : int, optional
        Hop length in campioni (necessario per calcolare il tempo)
    sr : int, optional
        Sample rate (necessario per calcolare il tempo)
    segment_start_sample : int, optional
        Posizione (in campioni) dell'inizio del segmento nell'audio originale
    padding_start_ms : float, optional
        Millisecondi di padding da aggiungere PRIMA del whoop rilevato (default: 20ms)
    padding_end_ms : float, optional
        Millisecondi di padding da aggiungere DOPO il whoop rilevato (default: 20ms)
    audio_length : int, optional
        Lunghezza totale dell'audio originale in campioni (per limitare il padding)
    
    Returns:
    --------
    f0 : float or None
        Frequenza fondamentale stimata (mediana della coda più lunga)
    best_queue : list
        Lista degli indici della coda migliore trovata
    all_queues : list of lists
        Tutte le code candidate trovate
    whoop_info : dict or None
        Dizionario con informazioni temporali del whoop:
        - 'start_time': tempo di inizio del whoop (sec, senza padding)
        - 'end_time': tempo di fine del whoop (sec, senza padding)
        - 'duration_ms': durata del whoop (millisecondi, senza padding)
        - 'start_sample': campione di inizio del whoop nell'audio originale (senza padding)
        - 'end_sample': campione di fine del whoop nell'audio originale (senza padding)
        - 'start_sample_padded': campione di inizio CON padding
        - 'end_sample_padded': campione di fine CON padding
        - 'duration_padded_ms': durata totale con padding (millisecondi)
        - 'padding_start_ms': padding applicato all'inizio
        - 'padding_end_ms': padding applicato alla fine
        - 'frame_indices': indici dei frame nella best queue
        - 'num_frames': numero di frame nella best queue
        - 'f0_median_frame_idx': indice del frame corrispondente alla mediana di f0
        - 'f0_median_sample': campione assoluto nell'audio originale corrispondente alla mediana
        - 'f0_median_sample_relative': campione relativo al segmento estratto (con padding)
    """
    
    freq_array = np.asarray(freq_array)
    
    # Rimuovi NaN o valori non validi
    valid_mask = np.isfinite(freq_array) & (freq_array > 0)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        if plot:
            _plot_results(freq_array, None, [], None, hop_length, sr)
        return None, [], [], None
    
    all_queues = []
    
    # Scorri ogni punto dell'array come potenziale inizio di una coda
    for start_idx in valid_indices:
        current_queue = [start_idx]
        current_freq = freq_array[start_idx]
        
        # Cerca punti successivi che soddisfano i criteri
        for next_idx in valid_indices[valid_indices > start_idx]:
            next_freq = freq_array[next_idx]
            
            # Verifica compatibilità con tutti i punti nella coda corrente
            is_valid = True
            
            # Controlla la differenza con ogni punto precedente nella coda
            for queue_position, prev_idx in enumerate(reversed(current_queue)):
                prev_freq = freq_array[prev_idx]
                
                # Calcola la soglia incrementata in base alla distanza
                adjusted_threshold = hz_threshold * (threshold_increment ** queue_position)
                
                # Calcola la differenza assoluta
                diff = abs(next_freq - prev_freq)
                
                # Verifica se supera la soglia
                if diff > adjusted_threshold:
                    is_valid = False
                    break
            
            # Se il punto è valido, aggiungilo alla coda
            if is_valid:
                current_queue.append(next_idx)
                current_freq = next_freq
            else:
                # Se fallisce il test, termina questa coda
                break
        
        # Salva la coda se ha almeno length_queue punti
        if len(current_queue) >= length_queue:
            all_queues.append(current_queue)
    
    # Trova la coda più lunga
    if not all_queues:
        if plot:
            _plot_results(freq_array, None, [], None, hop_length, sr)
        return None, [], [], None
    
    best_queue = max(all_queues, key=len)
    
    # Calcola f0 come mediana delle frequenze nella coda migliore
    f0 = np.median(freq_array[best_queue])

    # ===== NUOVA PARTE: Trova il frame più vicino alla mediana =====
    # Trova l'indice del frame nella best_queue il cui valore è più vicino alla mediana
    freq_values_in_queue = freq_array[best_queue]
    differences = np.abs(freq_values_in_queue - f0)
    median_position_in_queue = np.argmin(differences)
    f0_median_frame_idx = best_queue[median_position_in_queue]
    # ================================================================
    
    # Calcola informazioni temporali se disponibili
    whoop_info = None
    if hop_length is not None and sr is not None:
        # Tempo di inizio e fine del whoop (relativo al segmento) - SENZA PADDING
        start_frame_idx = best_queue[0]
        end_frame_idx = best_queue[-1]
        
        # Converti indici frame in campioni (relativi al segmento) - SENZA PADDING
        start_sample_rel = start_frame_idx * hop_length
        end_sample_rel = end_frame_idx * hop_length
        
        # Converti in tempo (secondi) - SENZA PADDING
        start_time = start_sample_rel / sr
        end_time = end_sample_rel / sr
        duration_ms = (end_time - start_time) * 1000

        # ===== CALCOLA POSIZIONE DEL CAMPIONE CORRISPONDENTE ALLA MEDIANA =====
        f0_median_sample_in_segment = f0_median_frame_idx * hop_length
        f0_median_sample_absolute = segment_start_sample + f0_median_sample_in_segment
        # ========================================================================
        
        # Campioni assoluti nell'audio originale - SENZA PADDING
        start_sample_abs = segment_start_sample + start_sample_rel
        end_sample_abs = segment_start_sample + end_sample_rel
        
        # CALCOLA PADDING IN CAMPIONI
        padding_start_samples = int(padding_start_ms * sr / 1000)
        padding_end_samples = int(padding_end_ms * sr / 1000)
        
        # APPLICA PADDING CON CONTROLLO DEI LIMITI
        start_sample_padded = max(0, start_sample_abs - padding_start_samples)
        
        if audio_length is not None:
            end_sample_padded = min(audio_length, end_sample_abs + padding_end_samples)
        else:
            end_sample_padded = end_sample_abs + padding_end_samples
        
        # Calcola durata con padding
        duration_padded_ms = (end_sample_padded - start_sample_padded) / sr * 1000
        
        whoop_info = {
            # Informazioni senza padding
            'start_time': start_time,
            'end_time': end_time,
            'duration_ms': duration_ms,
            'start_sample': start_sample_abs,
            'end_sample': end_sample_abs,
            
            # Informazioni con padding
            'start_sample_padded': start_sample_padded,
            'end_sample_padded': end_sample_padded,
            'duration_padded_ms': duration_padded_ms,

            # NUOVE INFORMAZIONI: Posizione della mediana
            'f0_median_frame_idx': f0_median_frame_idx,
            'f0_median_sample_absolute': f0_median_sample_absolute,
            
            # Altri dati
            'frame_indices': best_queue,
            'num_frames': len(best_queue)
        }
    
    # Genera il plot se richiesto
    if plot:
        _plot_results(freq_array, best_queue, all_queues, f0, hop_length, sr)
    
    return f0, best_queue, all_queues, whoop_info




def _plot_results(freq_array, best_queue, all_queues, f0, hop_length=None, sr=None):
    """
    Funzione helper per generare il plot dei risultati dell'analisi.
    """
    plt.figure(figsize=(12, 6))
    
    # Calcola l'asse X (tempo o indice)
    if hop_length is not None and sr is not None:
        x_axis = np.arange(len(freq_array)) * hop_length / sr  # Tempo in secondi
        x_label = 'Tempo (s)'
        title_suffix = f' - Hop: {hop_length/sr*1000:.1f}ms'
    else:
        x_axis = np.arange(len(freq_array))
        x_label = 'Indice del frame'
        title_suffix = ''
    
    # Plot dell'array originale
    plt.plot(x_axis, freq_array, 'o-', color='lightgray', 
             linewidth=1, markersize=6, alpha=0.6, label='Array originale')
    
    # Plot di tutte le code candidate (in trasparenza)
    if all_queues:
        for i, queue in enumerate(all_queues):
            if queue != best_queue:
                plt.plot(x_axis[queue], freq_array[queue], 'o-', 
                        color='orange', alpha=0.3, linewidth=1, 
                        markersize=4, label='Code candidate' if i == 0 else '')
    
    # Evidenzia la best queue
    if best_queue is not None and len(best_queue) > 0:
        plt.plot(x_axis[best_queue], freq_array[best_queue], 'o-', 
                color='red', linewidth=2.5, markersize=8, 
                label=f'Best queue (n={len(best_queue)})', zorder=5)
        
        # Linea orizzontale per f0
        if f0 is not None:
            plt.axhline(y=f0, color='red', linestyle='--', 
                       linewidth=1.5, alpha=0.7, 
                       label=f'f0 = {f0:.2f} Hz')
            
        # Aggiungi ombreggiatura per il whoop
        if hop_length is not None and sr is not None:
            start_time = x_axis[best_queue[0]]
            end_time = x_axis[best_queue[-1]]
            plt.axvspan(start_time, end_time, alpha=0.2, color='red', label='Whoop region')
    else:
        plt.text(0.5, 0.5, 'Nessuna coda valida trovata', 
                transform=plt.gca().transAxes, 
                fontsize=14, color='red', ha='center', va='center')
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Frequenza (Hz)', fontsize=12)
    plt.title(f'Analisi f0: Array originale e Best Queue identificata{title_suffix}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show()


def analyze_harmonics(whoop_segment, sr, f0, center_sample, window_duration_ms=10, 
                      nfft=4096, num_harmonics=10, bandwidth_hz=50):
    """
    Analizza le armoniche di un segnale whoop in una finestra breve centrata 
    attorno al punto di f0 mediano (per evitare il drift di frequenza).
    
    Parameters:
    -----------
    whoop_segment : np.ndarray
        Segmento audio contenente il whoop
    sr : int
        Sample rate
    f0 : float
        Frequenza fondamentale stimata (Hz)
    center_sample : int
        Indice del campione centrale (relativo a whoop_segment) attorno al quale analizzare
    window_duration_ms : float
        Durata della finestra di analisi in millisecondi (default: 10ms)
        Una finestra breve cattura il whoop in un punto stabile
    nfft : int
        Numero di punti per la FFT
    num_harmonics : int
        Numero di armoniche da analizzare (default: 10)
    bandwidth_hz : float
        Larghezza di banda in Hz attorno a ciascuna armonica per integrare l'energia (default: 50 Hz)
    
    Returns:
    --------
    harmonics_info : dict
        Dizionario contenente:
        - 'f0': frequenza fondamentale
        - 'center_sample': campione centrale utilizzato
        - 'window_start': inizio della finestra (campioni)
        - 'window_end': fine della finestra (campioni)
        - 'window_duration_ms': durata effettiva della finestra
        - 'harmonic_frequencies': array delle frequenze armoniche teoriche
        - 'harmonic_energies': array delle energie di ciascuna armonica
        - 'harmonic_magnitudes': array delle magnitudini di picco
        - 'harmonic_energy_ratios': ratio rispetto alla fondamentale
        - 'total_harmonic_energy': energia totale delle armoniche
        - 'spectrum': spettro completo
        - 'freqs': array delle frequenze
    """
    
    # Calcola la lunghezza della finestra in campioni
    window_length = int(window_duration_ms * sr / 1000)
    
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
    
    # Applica finestra per ridurre leakage
    window = np.hamming(len(analysis_window))
    windowed_signal = analysis_window * window
    
    # Calcola FFT
    spectrum, magnitude = compute_fft(windowed_signal, sr, nfft=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1/sr)
    
    # Calcola le frequenze armoniche teoriche
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
    
    # Calcola i ratio rispetto alla fondamentale
    if harmonic_energies[0] > 0:
        harmonic_energy_ratios = harmonic_energies / harmonic_energies[0]
    else:
        harmonic_energy_ratios = np.zeros(num_harmonics)
    
    # Energia armonica totale
    total_harmonic_energy = np.sum(harmonic_energies)
    
    # Calcola durata effettiva della finestra
    actual_window_duration_ms = (window_end - window_start) / sr * 1000
    
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


def print_harmonics_summary(harmonics_info, top_n=5):
    """
    Stampa un riassunto testuale dell'analisi delle armoniche.
    """
    print(f"\n{'='*70}")
    print(f"ANALISI ARMONICHE (Finestra centrata su f0 mediano)")
    print(f"{'='*70}")
    print(f"Frequenza fondamentale (f₀): {harmonics_info['f0']:.2f} Hz")
    print(f"Campione centrale: {harmonics_info['center_sample']}")
    print(f"Finestra analisi: [{harmonics_info['window_start']}, {harmonics_info['window_end']}]")
    print(f"Durata finestra: {harmonics_info['window_duration_ms']:.2f} ms")
    print(f"Energia armonica totale: {harmonics_info['total_harmonic_energy']:.2e}")
    print(f"\n{'Top ' + str(top_n) + ' Armoniche per Energia:':^70}")
    print(f"{'-'*70}")
    print(f"{'#':<5} {'Freq (Hz)':<12} {'Energia':<15} {'Magnitudine':<15} {'Ratio vs f₀':<15}")
    print(f"{'-'*70}")
    
    # Ordina per energia
    sorted_indices = np.argsort(harmonics_info['harmonic_energies'])[::-1]
    
    for rank, idx in enumerate(sorted_indices[:top_n], 1):
        freq = harmonics_info['harmonic_frequencies'][idx]
        energy = harmonics_info['harmonic_energies'][idx]
        magnitude = harmonics_info['harmonic_magnitudes'][idx]
        ratio = harmonics_info['harmonic_energy_ratios'][idx]
        
        print(f"H{idx+1:<3} {freq:<12.1f} {energy:<15.2e} {magnitude:<15.2f} {ratio:<15.3f}")
    
    print(f"{'='*70}\n")

def plot_harmonics_analysis(harmonics_info, freq_max=5000, highlight_first_n=3):
    """
    Visualizza l'analisi delle armoniche con spettro e grafico a barre.
    
    Parameters:
    -----------
    harmonics_info : dict
        Dizionario restituito da analyze_harmonics
    freq_max : float
        Frequenza massima da visualizzare nello spettro (Hz)
    highlight_first_n : int
        Numero di armoniche da evidenziare nello spettro (default: 3)
    """
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # --- SUBPLOT 1: Spettro con armoniche evidenziate ---
    freqs = harmonics_info['freqs']
    spectrum = harmonics_info['spectrum']
    
    # Limita la visualizzazione
    idx_max = np.argmin(np.abs(freqs - freq_max))
    
    ax1.plot(freqs[:idx_max], spectrum[:idx_max], linewidth=0.8, color='gray', alpha=0.7, label='Spettro')
    
    # Evidenzia le prime N armoniche
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
    
    for i, (freq, mag) in enumerate(zip(harmonics_info['harmonic_frequencies'], 
                                         harmonics_info['harmonic_magnitudes'])):
        if i < len(colors):
            color = colors[i]
        else:
            color = 'gray'
        
        if freq <= freq_max:
            # Linea verticale
            if i < highlight_first_n:
                ax1.axvline(x=freq, color=color, linestyle='--', linewidth=2, alpha=0.8)
                # Etichetta
                ax1.text(freq, ax1.get_ylim()[1] * 0.95, f'H{i+1}\n{freq:.0f}Hz', 
                        rotation=0, ha='center', va='top', fontsize=9, color=color,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))
            else:
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
    
    # Evidenzia le prime 3
    for i in range(min(highlight_first_n, len(bars))):
        bars[i].set_linewidth(2.5)
    
    ax2.set_xlabel('Numero Armonica', fontsize=11)
    ax2.set_ylabel('Energia', fontsize=11)
    ax2.set_title('Energia delle Armoniche', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(harmonic_numbers)
    
    # Aggiungi valori sopra le barre per le prime 3
    for i in range(min(highlight_first_n, len(bars))):
        height = bars[i].get_height()
        ax2.text(bars[i].get_x() + bars[i].get_width()/2., height,
                f'{height:.1e}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # --- SUBPLOT 3: Ratio rispetto alla fondamentale ---
    ax3.bar(harmonic_numbers, harmonics_info['harmonic_energy_ratios'], 
            color=colors[:len(harmonic_numbers)], alpha=0.7, edgecolor='black')
    
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Fondamentale (riferimento)')
    
    ax3.set_xlabel('Numero Armonica', fontsize=11)
    ax3.set_ylabel('Ratio rispetto a f₀', fontsize=11)
    ax3.set_title('Ratio Energia Armoniche rispetto alla Fondamentale', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(harmonic_numbers)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def export_final_results_to_csv(f0_frequencies_final, duration_ms_final, time_stamp):
    if f0_frequencies_final and duration_ms_final:
        
        csv_filename = f"whoop_analysis_results_{timestamp}.csv"
        csv_path = os.path.join(main_folder_path, csv_filename)
        
        # Salva i risultati
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Whoop_Index', 'f0_Hz', 'Duration_ms'])
            
            # Dati
            for idx, (f0, duration) in enumerate(zip(f0_frequencies_final, duration_ms_final), start=1):
                writer.writerow([idx, f"{f0:.2f}", f"{duration:.2f}"])
            
            # Aggiungi statistiche
            writer.writerow([])  # Riga vuota
            writer.writerow(['Statistics', '', ''])
            writer.writerow(['Mean f0', f"{np.mean(f0_frequencies_final):.2f}", 'Hz'])
            writer.writerow(['Std f0', f"{np.std(f0_frequencies_final):.2f}", 'Hz'])
            writer.writerow(['Mean Duration', f"{np.mean(duration_ms_final):.2f}", 'ms'])
            writer.writerow(['Std Duration', f"{np.std(duration_ms_final):.2f}", 'ms'])
            writer.writerow(['Total Whoops', len(f0_frequencies_final), ''])
        
        print(f"\nRisultati salvati in: {csv_path}")
    else:
        print("Nessun risultato da salvare.")


def plot_final_visualization(f0_frequencies_final, duration_ms_final, save=False, timestamp=""):
    if f0_frequencies_final and duration_ms_final:
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # --- GRAFICO 1: Istogramma Frequenze Fondamentali ---
        
        # Calcola statistiche
        mean_f0 = np.mean(f0_frequencies_final)
        std_f0 = np.std(f0_frequencies_final)
        
        # Crea istogramma con bins automatici (o specificali manualmente)
        n_bins_f0 = min(20, len(f0_frequencies_final) // 2) if len(f0_frequencies_final) > 10 else len(set(f0_frequencies_final))
        counts_f0, bins_f0, patches_f0 = ax1.hist(f0_frequencies_final, bins=n_bins_f0, 
                                                    color='steelblue', alpha=0.7, 
                                                    edgecolor='black', linewidth=1.2)
        
        # Linea media
        ax1.axvline(x=mean_f0, color='red', linestyle='--', linewidth=2, 
                    label=f'Media: {mean_f0:.2f} Hz')
        
        # Linee per ±1 SD
        ax1.axvline(x=mean_f0 - std_f0, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f'±1 SD: {std_f0:.2f} Hz')
        ax1.axvline(x=mean_f0 + std_f0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax1.set_xlabel('Frequenza Fondamentale (Hz)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Numero di Occorrenze', fontsize=12, fontweight='bold')
        ax1.set_title(f'Distribuzione Frequenze Fondamentali (n={len(f0_frequencies_final)} whoops)', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(loc='upper right', fontsize=10)
        
        # Aggiungi conteggi sopra le barre
        for count, patch in zip(counts_f0, patches_f0):
            if count > 0:  # Solo se ci sono occorrenze
                height = patch.get_height()
                ax1.text(patch.get_x() + patch.get_width()/2., height,
                        f'{int(count)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # --- GRAFICO 2: Istogramma Durate ---
        
        # Calcola statistiche
        mean_dur = np.mean(duration_ms_final)
        std_dur = np.std(duration_ms_final)
        
        # Crea istogramma
        n_bins_dur = min(20, len(duration_ms_final) // 2) if len(duration_ms_final) > 10 else len(set(duration_ms_final))
        counts_dur, bins_dur, patches_dur = ax2.hist(duration_ms_final, bins=n_bins_dur, 
                                                      color='darkorange', alpha=0.7, 
                                                      edgecolor='black', linewidth=1.2)
        
        # Linea media
        ax2.axvline(x=mean_dur, color='red', linestyle='--', linewidth=2, 
                    label=f'Media: {mean_dur:.2f} ms')
        
        # Linee per ±1 SD
        ax2.axvline(x=mean_dur - std_dur, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f'±1 SD: {std_dur:.2f} ms')
        ax2.axvline(x=mean_dur + std_dur, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Durata (ms)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Numero di Occorrenze', fontsize=12, fontweight='bold')
        ax2.set_title(f'Distribuzione Durate (n={len(duration_ms_final)} whoops)', 
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(loc='upper right', fontsize=10)
        
        # Aggiungi conteggi sopra le barre
        for count, patch in zip(counts_dur, patches_dur):
            if count > 0:
                height = patch.get_height()
                ax2.text(patch.get_x() + patch.get_width()/2., height,
                        f'{int(count)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plot_filename = f"whoop_analysis_histogram_{timestamp}.png"
            plot_path = os.path.join(main_folder_path, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Istogramma salvato in: {plot_path}")
        
        plt.show()
        
        # Stampa statistiche aggiuntive
        print(f"\n{'='*60}")
        print(f"DISTRIBUZIONE FREQUENZE FONDAMENTALI")
        print(f"{'='*60}")
        print(f"Range: {np.min(f0_frequencies_final):.2f} - {np.max(f0_frequencies_final):.2f} Hz")
        print(f"Mediana: {np.median(f0_frequencies_final):.2f} Hz")
        print(f"Bins utilizzati: {n_bins_f0}")
        
        print(f"\n{'='*60}")
        print(f"DISTRIBUZIONE DURATE")
        print(f"{'='*60}")
        print(f"Range: {np.min(duration_ms_final):.2f} - {np.max(duration_ms_final):.2f} ms")
        print(f"Mediana: {np.median(duration_ms_final):.2f} ms")
        print(f"Bins utilizzati: {n_bins_dur}")
        print(f"{'='*60}\n")
        
    else:
        print("Nessun dato disponibile per la visualizzazione finale.")


def print_statistic_summary(f0_frequencies_final, duration_ms_final, total_whoops):
    if f0_frequencies_final and duration_ms_final:
        print(f"\n{'='*60}")
        print(f"STATISTICHE RIASSUNTIVE")
        print(f"{'='*60}")
        print(f"Numero totale di whoops analizzati: {total_whoops}")
        print(f"di cui considerati validi: {len(f0_frequencies_final)}")
        print(f"\nFrequenza Fondamentale:")
        print(f"  Media:    {np.mean(f0_frequencies_final):.2f} Hz")
        print(f"  Std Dev:  {np.std(f0_frequencies_final):.2f} Hz")
        print(f"  Min:      {np.min(f0_frequencies_final):.2f} Hz")
        print(f"  Max:      {np.max(f0_frequencies_final):.2f} Hz")
        print(f"  Mediana:  {np.median(f0_frequencies_final):.2f} Hz")
        print(f"\nDurata:")
        print(f"  Media:    {np.mean(duration_ms_final):.2f} ms")
        print(f"  Std Dev:  {np.std(duration_ms_final):.2f} ms")
        print(f"  Min:      {np.min(duration_ms_final):.2f} ms")
        print(f"  Max:      {np.max(duration_ms_final):.2f} ms")
        print(f"  Mediana:  {np.median(duration_ms_final):.2f} ms")
        print(f"{'='*60}\n")
    else:
        print("Nessun dato disponibile per le statistiche riassuntive.")

if __name__ == "__main__":

    # main_folder_path = "sounds/fake_whoops_for_testing"
    # main_folder_path = "sounds/whoop"
    main_folder_path = "sounds/best_whoops"
    # main_folder_path = "D:/whooping_candidates/audio_recording_2025-09-10T08_00_42.172886Z"
    # main_folder_path = "sounds/whoop_examples"


    
    # Estrai tutti i file audio nella cartella
    audio_files = sorted([f for f in os.listdir(main_folder_path) if f.endswith('.wav')])
    
    if not audio_files:
        print(f"Nessun file .wav trovato in {main_folder_path}")
        exit(1)
    
    print(f"\nTrovati {len(audio_files)} file audio candidati whoop")
    print("="*80)

    f0_frequencies_final = []
    duration_ms_final = []

    for file in audio_files:
        print(f"\n\nAnalisi del file: {file}")
        
        audio_path = os.path.join(main_folder_path, file)
        

        y, sr = librosa.load(audio_path, sr=None, mono=True)    
    
        # Inizializza il detector
        detector = WhoopDetector(
            signal=y,
            sr=sr,
            window_length_ms=50,
            hop_length_ms=10,
            f0_min=250,
            f0_max=700,
            window_type='hamming',
            lowpass_cutoff=15000,
            highpass_cutoff=2500,
            normalize=True,
            target_rms=0.1
        )

        # Esegui la rilevazione
        detection_results = detector.detect(
            percentile=80,
            offset=4,
            window_sec=0.7, # lunghezza finestra di analisi intorno al picco
            merge_overlaps=True
        )

        # Accedi ai risultati
        print(f"Picchi rilevati: {len(detector.peak_times_)}")
        print(f"Threshold HNR: {detector.threshold_:.2f} dB")

        # Ottieni informazioni dettagliate sui picchi
        peak_info = detector.get_peak_info()
        for info in peak_info:
            print(f"Picco {info['index']}: t={info['peak_time']:.3f}s, "
                f"finestra=[{info['window_start']:.3f}, {info['window_end']:.3f}]s")

        # Estrai i segmenti audio
        segments = detector.extract_segments()

            
        
        # Visualizza i risultati
        # detector.plot_analysis(ch_num=1)

        # Accesso diretto agli array
        # detector.peaks_         -> indici dei picchi nell'array HNR
        # detector.peak_times_    -> tempi centrali dei picchi (secondi)
        # detector.peak_windows_  -> lista di tuple (start_time, end_time)

        # Stima frequenza fondamentale 


        window_duration = 0.050   # 0.050sec = 50 ms
        hop_duration = 0.010      # 0.010sec = 10 ms
        nfft = 4096                # Numero di punti per la FFT (zero padding per migliorare risoluzione frequenza)
        # converti in campioni
        window_length = int(window_duration * sr) # sec * (samples/sec) = samples 
        hop_length = int(hop_duration * sr)

        # detector.plot_analysis(ch_num=1)


        
        for segment_idx, segment in enumerate(segments):

            # play segment
            # sd.play(segment, sr)
            # sd.wait()

            # Ottieni l'inizio del segmento nell'audio originale (in campioni)
            segment_start_time = detector.peak_windows_[segment_idx][0]  # in secondi
            segment_start_sample = int(segment_start_time * sr)
            
            # Calcola il numero di finestre
            num_frames = int((len(segment) - window_length) / hop_length) + 1
            
            fundamental_frequencies_cep = []

            print("Inizio analisi segmento...")
            for i in range(num_frames):
                start = i * hop_length
                end = start + window_length
                frame = segment[start:end]

                # Applica finestra per ridurre leakage
                frame_windowed = frame * np.hamming(len(frame))

                spectrum, magnitude = compute_fft(frame_windowed, sr, nfft=nfft)
                                                                
                f0_estimated,_,_ = estimate_f0_cepstrum(magnitude, sr, f0_min=200, f0_max=600, nfft=nfft)
                fundamental_frequencies_cep.append(f0_estimated)

                

                
                # if i > 20:
                #     plot_spectrum(magnitude, sr, freq_min=1000, freq_max=15000, nfft=nfft, fund_freq_grand_truth=300)
                #     print(f"Frame {i+1}/{num_frames} analizzato.") 

            print("Fine analisi segmento.")

                     
            # plot_fundamental_frequencies(fundamental_frequencies_cep, method_name="Cepstrum")
            

            # Analisi con informazioni temporali E PADDING INTEGRATO
            f0_final, queue, all_queues, whoop_info = analyze_f0_array(
                fundamental_frequencies_cep, 
                length_queue=5, 
                hz_threshold=25, 
                threshold_increment=1.3,
                plot=True,
                hop_length=hop_length,
                sr=sr,
                segment_start_sample=segment_start_sample,
                padding_start_ms=5,    # 5ms prima
                padding_end_ms=25,      # 25ms dopo
                audio_length=len(y)     # Lunghezza audio originale
            )
            
            # Stampa informazioni sul whoop
            if whoop_info is not None:
                print(f"\n{'='*60}")
                print(f"WHOOP RILEVATO - Segmento {segment_idx + 1}")
                print(f"{'='*60}")
                print(f"f0 stimato: {f0_final:.2f} Hz")
                print(f"Durata whoop: {whoop_info['duration_padded_ms']:.2f} ms")
                print(f"Campioni nell'audio originale: {whoop_info['start_sample_padded']} - {whoop_info['end_sample_padded']}")
                print(f"Numero di frame: {whoop_info['num_frames']}")
                print(f"{'='*60}\n")
                
                
                
                # ==========================================
                # ANALISI DELLE ARMONICHE SU FINESTRA BREVE CENTRATA
                # ==========================================
                harmonics_info = analyze_harmonics(
                    whoop_segment=y,
                    sr=sr,
                    f0=f0_final,
                    center_sample=whoop_info['f0_median_sample_absolute'], # Usa il campione assoluto corrispondente alla mediana
                    window_duration_ms=10,  # Finestra di 10ms (o 5ms per essere ancora più precisi)
                    nfft=8192,
                    num_harmonics=10,
                    bandwidth_hz=50
                )
                
                if harmonics_info is not None:
                    # Stampa riassunto
                    print_harmonics_summary(harmonics_info, top_n=5)
                    
                    # Visualizza grafici
                    # plot_harmonics_analysis(harmonics_info, freq_max=5000, highlight_first_n=3)

                # # ESTRAI E SALVA IL WHOOP
                # whoop_segment = y[whoop_info['start_sample_padded']:whoop_info['end_sample_padded']]
                
                # # Salva il whoop
                # output_filename = f"whoop_extracted_{file[:-4]}_seg{segment_idx+1}_f0_{f0_final:.0f}Hz.wav"
                # output_path = os.path.join("sounds/extracted_whoops", output_filename)
                
                # # Crea la cartella se non esiste
                # os.makedirs("sounds/extracted_whoops", exist_ok=True)
                
                # # Salva usando soundfile o scipy
                
                # sf.write(output_path, whoop_segment, sr)
                # print(f"✓ Whoop salvato in: {output_path}\n")
                
                # Riproduci solo il whoop estratto
                # sd.play(whoop_segment, sr)
                # sd.wait()

                f0_frequencies_final.append(f0_final)
                duration_ms_final.append(whoop_info['duration_padded_ms'])
            else:
                print("Nessun whoop valido trovato in questo segmento.")
                    

            

            
            
            # sd.play(segment, sr)
            # sd.wait()
            # print("Do you want to hear it again=? (y/n)")
            # while(input().lower() == 'y'):
            #     sd.play(segment, sr)
            #     sd.wait()
    print("\n\nAnalisi completata per tutti i file.")
    print("="*80)
    
    # save results to csv...
    # Crea nome file con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # export_final_results_to_csv(f0_frequencies_final, duration_ms_final, timestamp)
    # plot final visualization
    plot_final_visualization(f0_frequencies_final, duration_ms_final, save=False, timestamp=timestamp)
    # print statistic summary
    print_statistic_summary(f0_frequencies_final, duration_ms_final, len(audio_files))
    
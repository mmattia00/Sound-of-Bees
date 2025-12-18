import numpy as np
import soundfile as sf
import re
from pathlib import Path
import time
import sounddevice as sd
from scipy.signal import filtfilt, butter
import os
import matplotlib.pyplot as plt



import re

def parse_filename(filename):
    """
    Estrae il numero di canale, il raw path e le informazioni temporali dal nome del file.
    Formato atteso: RAWPATH_chXX_X.XXXs_START-ENDs.wav
    
    Esempio:
        audio_recording_2025-09-15T06_22_43.635117Z_ch06_52.855s_51.355-54.395s.wav
        → {
            'raw_path': 'audio_recording_2025-09-15T06_22_43.635117Z',
            'channel': 6,
            'start_time': 51.355,
            'end_time': 54.395
          }
    
    Args:
        filename (str): Nome del file WAV
    
    Returns:
        dict: Contiene 'raw_path', 'channel', 'start_time' e 'end_time' in secondi
    """
    # Regex completa:
    #  (.+?)         -> raw path (tutto prima di "_ch")
    #  _ch(\d+)_     -> canale
    #  \d+\.\d+s_    -> offset/marker intermedio (es. "52.855s_"), qui ignorato
    #  (\d+\.\d+)    -> start_time
    #  -             -> separatore
    #  (\d+\.\d+)s   -> end_time + "s"
    #  \.wav$        -> estensione finale
    pattern = r'(.+?)_ch(\d+)_\d+\.\d+s_(\d+\.\d+)-(\d+\.\d+)s\.wav$'
    match = re.search(pattern, filename)
    
    if not match:
        raise ValueError(f"Impossibile parsare il filename: {filename}")
    
    raw_path = match.group(1)
    channel = int(match.group(2))
    start_time = float(match.group(3))
    end_time = float(match.group(4))

    print(f"✓ Parsing filename: RAW={raw_path}, CH={channel}, START={start_time}s, END={end_time}s")
    return {
        'raw_path': raw_path,
        'channel': channel,
        'start_time': start_time,
        'end_time': end_time
    }


def load_audio_file(filepath):
    """
    Carica il file WAV completo.
    
    Args:
        filepath (str): Percorso del file WAV
    
    Returns:
        data (np.ndarray): Array audio (samples, channels)
        samplerate (int): Frequenza di campionamento
    """
    try:
        data, samplerate = sf.read(filepath, dtype='float32')
        
        # Gestisci caso mono
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        print(f"✓ File caricato: {Path(filepath).name}")
        print(f"  Samplerate: {samplerate} Hz")
        print(f"  Shape: {data.shape} (samples × channels)")
        
        return data, samplerate
    except Exception as e:
        print(f"✗ Errore nel caricamento: {e}")
        raise


def time_to_samples(time_sec, samplerate):
    """
    Converte tempo in secondi a numero di campioni.
    
    Args:
        time_sec (float): Tempo in secondi
        samplerate (int): Frequenza di campionamento
    
    Returns:
        int: Numero di campioni
    """
    return int(time_sec * samplerate)

def apply_lowpass_filter(signal, sr, cutoff_hz=15000):
    """Applica un filtro passa-basso Butterworth."""
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    b, a = butter(4, normalized_cutoff, btype='low')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def apply_highpass_filter(signal, sr, cutoff_hz=2500):
    """Applica un filtro passa-alto Butterworth."""
    nyquist = sr / 2
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    b, a = butter(4, normalized_cutoff, btype='high')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def normalize_signal_rms(signal, target_rms=0.1):
    """Normalizza il segnale a un RMS target."""
    current_rms = np.sqrt(np.mean(signal**2))
    if current_rms < 1e-6:
        return signal
    scaling_factor = target_rms / current_rms
    return signal * scaling_factor

def preprocess_signal(signal, sr, lowpass_cutoff, highpass_cutoff, target_rms):
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
    signal = normalize_signal_rms(signal, target_rms)
    
    # Filtri
    signal = apply_lowpass_filter(signal, sr, lowpass_cutoff)
    signal = apply_highpass_filter(signal, sr, highpass_cutoff)
    
    return signal
    

def extract_and_play_channels(data, samplerate, start_time, end_time, 
                               num_channels=16, pause_between=0.5, channel_broken=None):
    """
    Estrae la finestra di interesse da ciascuno dei primi N canali
    e la riproduce in sequenza.
    
    Args:
        data (np.ndarray): Array audio (samples, channels)
        samplerate (int): Frequenza di campionamento
        start_time (float): Tempo di inizio in secondi
        end_time (float): Tempo di fine in secondi
        num_channels (int): Numero di canali da elaborare (default 16)
        pause_between (float): Pausa tra riproduzioni in secondi
        channel_broken (int or None): Canali rotti da escludere (default None)
    """

    lowpass_cutoff = 20000
    highpass_cutoff = 2000
    target_rms = 0.1

    # Converti i tempi in campioni
    start_sample = time_to_samples(start_time, samplerate)
    end_sample = time_to_samples(end_time, samplerate)
    
    window_duration = end_sample - start_sample
    window_duration_ms = (window_duration / samplerate) * 1000
    
    print(f"\n{'='*60}")
    print(f"ESTRAZIONE E RIPRODUZIONE WHOOP SIGNAL")
    print(f"{'='*60}")
    print(f"Intervallo temporale: {start_time}s - {end_time}s")
    print(f"Intervallo campioni: {start_sample} - {end_sample}")
    print(f"Durata finestra: {window_duration_ms:.1f} ms ({window_duration} campioni)")
    print(f"Numero canali da riprodurre: {num_channels}")
    print(f"{'='*60}\n")
    
    # Verifica che il numero di canali sia sufficiente
    if data.shape[1] < num_channels:
        print(f"⚠ Attenzione: file ha {data.shape[1]} canali, richiesti {num_channels}")
        num_channels = data.shape[1]
    

        
    # Loop sui primi N canali
    for ch in range(num_channels):
        if ch in channel_broken:
            print(f"▶ Canale {ch+1:2d}: ✗ Canale rotto, salto")
            continue
        else:
            # Estrai la finestra dal canale corrente
            tmp = data[start_sample:end_sample, ch]
            
            
            # preprocess
            # tmp = preprocess_signal(tmp, samplerate, lowpass_cutoff, highpass_cutoff, target_rms)

            # Normalizza per evitare clipping
            max_val = np.max(np.abs(tmp))
            if max_val > 0:
                tmp = tmp / (max_val * 1.1)
            
            # Output
            print(f"▶ Canale {ch+1:2d}: ", end='', flush=True)
            print(f"min={np.min(tmp):7.4f}, max={np.max(tmp):7.4f}, ", end='', flush=True)
            print(f"RMS={np.sqrt(np.mean(tmp**2)):7.4f} ", end='')
            
            # Riproduce
            try:
                sd.play(tmp, samplerate=samplerate, blocking=False)
                # Attendi la fine della riproduzione
                duration = len(tmp) / samplerate
                time.sleep(duration)
                print("✓")
            except Exception as e:
                print(f"✗ Errore: {e}")
            
            # Pausa tra i canali (se non è l'ultimo)
            if ch < num_channels - 1:
                time.sleep(pause_between)
    
    print(f"\n{'='*60}")
    print("✓ Riproduzione completata")
    print(f"{'='*60}\n")







def gcc_phat(sig1, sig2, sr, max_tau_ms=100):
    """
    Stima dello time delay usando GCC-PHAT (Generalized Cross-Correlation with Phase Transform).
    
    Args:
        sig1 (np.ndarray): Primo segnale audio (reference)
        sig2 (np.ndarray): Secondo segnale audio (neighbor)
        sr (int): Sample rate in Hz
        max_tau_ms (float): Massimo delay da cercare in millisecondi
    
    Returns:
        dict: Contiene delay_samples, delay_ms, delay_sec, correlation, lags, peak_value
    """
    # Lunghezza di FFT (padding per evitare aliasing)
    fft_len = 2 * max(len(sig1), len(sig2))
    
    # FFT dei due segnali
    X1 = np.fft.rfft(sig1, n=fft_len)
    X2 = np.fft.rfft(sig2, n=fft_len)
    
    # Cross-spettro
    Gxy = X1 * np.conj(X2)
    
    # GCC-PHAT: normalizzazione per la magnitude (whitening spettrale)
    mag = np.abs(Gxy)
    mag[mag < 1e-10] = 1e-10
    Gxy_phat = Gxy / mag
    
    # Inverse FFT per ottenere la correlazione generalizzata
    correlation = np.fft.irfft(Gxy_phat, n=fft_len)
    correlation = np.fft.fftshift(correlation)
    
    # Taglia il range di delay da cercare
    max_samples = int((max_tau_ms / 1000) * sr)
    center = len(correlation) // 2
    start_idx = center - max_samples
    end_idx = center + max_samples + 1
    correlation_windowed = correlation[start_idx:end_idx]
    
    # Trova il picco della correlazione
    peak_idx = np.argmax(correlation_windowed)
    delay_samples = peak_idx - max_samples
    
    # Converti in unità diverse
    delay_ms = (delay_samples / sr) * 1000
    delay_sec = delay_samples / sr
    
    # Lags array (per plotting)
    lags = (np.arange(-max_samples, max_samples + 1) / sr) * 1000  # in ms

    # plot del risultato 
    plt.figure(figsize=(6, 3))
    plt.plot(lags, correlation_windowed)
    plt.xlabel("Lag (ms)")
    plt.ylabel("Normalized Correlation")
    plt.grid()
    plt.show()

    
    return {
        'delay_samples': delay_samples,
        'delay_ms': delay_ms,
        'delay_sec': delay_sec,
        'correlation': correlation_windowed,
        'lags': lags,
        'peak_value': correlation_windowed[delay_samples + max_samples]
    }


def make_delayed_copy(raw_audio, sr, reference_ch, start_time, end_time, delay_ms):
    """
    Estrae un segnale ritardato da raw_audio usando uno shift temporale.

    Supporta sia delay positivi che negativi.

    Args:
        raw_audio (np.ndarray): Array audio multicanale (samples, channels)
        sr (int): Sample rate in Hz
        reference_ch (int): Numero di canale (zero-based)
        start_time (float): Tempo di inizio originale in secondi
        end_time (float): Tempo di fine originale in secondi
        delay_ms (float): Ritardo desiderato in millisecondi
                          > 0 → shift a destra (arriva dopo)
                          < 0 → shift a sinistra (arriva prima)

    Returns:
        np.ndarray: Segnale estratto e ritardato
    """
    # Converti delay da ms a secondi
    delay_sec = delay_ms / 1000.0

    # Applica lo shift temporale
    delayed_start_time = start_time + delay_sec
    delayed_end_time = end_time + delay_sec

    # Converti in campioni
    delayed_start_sample = time_to_samples(delayed_start_time, sr)
    delayed_end_sample = time_to_samples(delayed_end_time, sr)

    # Controlla che l'intervallo sia valido (sia inizio che fine)
    if delayed_start_sample < 0:
        raise ValueError(
            f"Intervallo ritardato esce dai limiti: "
            f"start_sample={delayed_start_sample} (< 0)"
        )

    if delayed_end_sample > raw_audio.shape[0]:
        raise ValueError(
            f"Intervallo ritardato esce dai limiti: "
            f"end_sample={delayed_end_sample}, audio length={raw_audio.shape[0]}"
        )

    # Estrai il canale nella finestra ritardata
    delayed_signal = raw_audio[delayed_start_sample:delayed_end_sample, reference_ch]

    return delayed_signal





# main per testare gcc-phat tra due whoop estratti da canali vicini oppure tra whoop di riferimento e una sua copia ritardata
def main():

    reference_whoop_folder = "sounds/best_whoop_available/"
    reference_whoop_path = "audio_recording_2025-09-15T06_40_43.498109Z_ch04_5.005s_3.505-6.505s.wav"
    raw_audio_folder = "E:/soundofbees/"
    channel_broken = [2, 3, 8, 13, 21, 25, 27, 28, 31]
    channel_broken = [x - 1 for x in channel_broken]  # zero-based index

    reference_whoop_info = parse_filename(reference_whoop_path)

    
    # load reference whoop file -----------------------------------------------------------------------------
    reference_whoop, sr = sf.read(reference_whoop_folder + reference_whoop_path, dtype='float32')
    reference_ch = reference_whoop_info['channel'] - 1  # zero-based index

    # load neighboring whoop file -----------------------------------------------------------------------------
    raw_audio, sr = sf.read(raw_audio_folder + reference_whoop_info['raw_path'] + ".wav", dtype='float32')
    
    # Calcola la finestra temporale PRIMA
    start_sample = time_to_samples(reference_whoop_info['start_time'], sr)
    end_sample = time_to_samples(reference_whoop_info['end_time'], sr)
    
    
    # # Trova neighbor channel
    # neighbor_ch = reference_ch + 3
    # while neighbor_ch < raw_audio.shape[1] and neighbor_ch in channel_broken:
    #     neighbor_ch += 1
    # print(f"Reference ch: {reference_ch}, Neighbor ch: {neighbor_ch}")
    
    # ESTRAI CORRETTAMENTE: prima tempo, poi canale
    # neighboring_whoop = raw_audio[start_sample:end_sample, neighbor_ch]  # (samples, 1) → mono!

    true_delay_ms = -3.5  # delay in milliseconds
    neighboring_whoop = make_delayed_copy(raw_audio, sr, reference_ch, reference_whoop_info['start_time'], reference_whoop_info['end_time'], true_delay_ms)

    # play for test
    # Normalizza per evitare clipping
    max_val = np.max(np.abs(reference_whoop))
    if max_val > 0:
        reference_whoop = reference_whoop / (max_val * 1.1)
    sd.play(reference_whoop, sr)
    sd.wait()
    # Normalizza per evitare clipping
    max_val = np.max(np.abs(neighboring_whoop))
    if max_val > 0:
        neighboring_whoop = neighboring_whoop / (max_val * 1.1)
    sd.play(neighboring_whoop, sr)
    sd.wait()

    # ========== APPLICA GCC-PHAT ==========
    print(f"\n{'='*60}")
    print(f"GCC-PHAT TIME DELAY ESTIMATION")
    print(f"{'='*60}")
    
    result = gcc_phat(reference_whoop, neighboring_whoop, sr, max_tau_ms=50)
    
    print(f"Delay (samples): {result['delay_samples']}")
    print(f"Delay (ms):      {result['delay_ms']:.3f}")
    print(f"Delay (sec):     {result['delay_sec']:.6f}")
    print(f"Peak correlation: {result['peak_value']:.6f}")
    print(f"{'='*60}\n")

    print(f"True delay:      {true_delay_ms:.3f} ms")



# # main che dato il canale di riferimento estrare la stessa finestra temporale da tutti i canali e li riproduce uno alla volta
# def main():

#     num_channels_default = 16
#     pause_between_default = 0.5

    

#     # Canali rotti
#     channel_broken = [2, 3, 8, 13, 21, 25, 27, 28, 31]
#     channel_broken = [x - 1 for x in channel_broken]  # zero-based index

#     raw_folder = "sounds/tmp/"
#     whoops_folder = "sounds/best_whoop_available"
#     starting_whoop = "audio_recording_2025-09-15T06_40_43.498109Z_ch04_5.005s_3.505-6.505s.wav"

#     # Trova tutti i file .wav nella cartella
#     whoops = sorted([f for f in os.listdir(whoops_folder) if f.endswith('.wav')])
    
#     # Trova l’indice dell audio file di partenza
#     try:
#         start_index = whoops.index(starting_whoop)
#     except ValueError:
#         start_index = -1

#     # Prendi tutti i file successivi
#     if start_index != -1:
#         whoops = whoops[start_index:]
    


#     if not whoops:
#         print("Nessun file .wav trovato in " + whoops_folder)
#         exit(1)
    
#     print("Trovati " + str(len(whoops)) + " file audio")
#     print("="*80 + "\n")
    
#     # Processa ogni file
#     for file_idx, whoop_filename in enumerate(whoops, 1):

#         channel = whoop_filename.split("_")[5]
#         channel_num = int(channel.split("ch")[1])

#         if channel_num > 16:
#             continue
#         else:
#             print(whoop_filename)
#             print(f"Expected whoop at channel {channel_num}") 

#             # print(whoop_filename.split("_ch"))  # -> ['audio_recording_2025-09-15T06_00_43.664505Z', '06_24.035s_22.535-25.535s.wav']
#             # Splitta sul pattern "_ch" e prendi la parte prima
#             rawfilepath = f"{raw_folder}{whoop_filename.split('_ch')[0]}.wav"



#             try:
#                 # Carica il file audio raw
#                 signal_multichannel, sr = sf.read(rawfilepath)

                
                
#             except Exception as e:
#                 print("Errore nel caricamento di " + rawfilepath + ": " + str(e))
            
            
#             # Parsa il filename per estrarre start/end
#             print(f"\n{'='*60}")
#             print(f"PARSING FILENAME")
#             print(f"{'='*60}")
#             time_info = parse_filename(whoop_filename)
            
            
#             # Estrai e riproduci i canali
#             extract_and_play_channels(
#                 signal_multichannel, 
#                 sr, 
#                 time_info['start_time'],
#                 time_info['end_time'],
#                 num_channels=num_channels_default,
#                 pause_between=pause_between_default,
#                 channel_broken=channel_broken
#             )

#             input("vai avanti?")





if __name__ == '__main__':
    main()
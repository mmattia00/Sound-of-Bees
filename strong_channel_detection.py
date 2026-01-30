from classes.strong_channel_detector_OLD import StrongChannelDetector
import soundfile as sf
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
from classes.whoop_detector import WhoopDetector
from classes.pitch_detector import PitchDetector

def load_coordinates_with_labels(filename: str):
    """
    Legge un file di testo nel formato:
    x, y, # commento

    La prima riga contiene i boundaries nel commento, del tipo:
    0.0, 0.205, 0.0, 0.32 # Boundaries: (x_min, x_max, y_min, y_max)

    Ritorna:
    - coords: array numpy strutturato con campi 'x', 'y', 'label'
    - boundaries: [x_max, x_min, y_max, y_min]
    """
    records = []
    boundaries = None

    with open(filename, "r", encoding="utf-8") as f:
        # Leggi e parse della prima riga (boundaries)
        first_line = f.readline()
        second_line = f.readline()
        if "#" in second_line and "Boundaries" in second_line:
            data_part, comment_part = second_line.split("#", 1)
            data_part = data_part.strip()
            parts = [p.strip() for p in data_part.split(",")]

            if len(parts) == 4:
                x_min, x_max, y_min, y_max = map(float, parts)
                boundaries = [x_min, x_max, y_min, y_max]

        # Parse delle righe successive (canali)
        for line in f:
            if "#" in line:
                data_part, comment_part = line.split("#", 1)
                label = comment_part.strip()
            else:
                data_part = line
                label = ""

            data_part = data_part.strip()
            if not data_part:
                continue

            parts = [p.strip() for p in data_part.split(",")]
            if len(parts) < 2:
                continue

            x = float(parts[0])
            y = float(parts[1])
            records.append((x, y, label))

    dtype = np.dtype([("x", "f8"), ("y", "f8"), ("label", "U64")])
    coords = np.array(records, dtype=dtype)
    return coords, boundaries

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
    

    if channel_broken is None:
        channel_broken = []

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



def main():

    
    num_channels_default = 16
    pause_between_default = 0.5

    # Canali rotti
    channel_broken = [2, 3, 8, 13, 21, 25, 27, 28, 31]
    channel_broken = [x - 1 for x in channel_broken]  # zero-based index

    # raw_folder = "sounds/tmp/"
    raw_folder = "E:\\soundofbees"
    whoops_folder = "sounds/best_whoop_available"
    starting_whoop = "audio_recording_2025-09-15T06_40_43.498109Z_ch04_5.005s_3.505-6.505s.wav"
    # starting_whoop = "audio_recording_2025-09-15T07_18_43.462247Z_ch12_47.135s_45.635-49.295s.wav"
    # starting_whoop = "audio_recording_2025-09-15T06_00_43.664505Z_ch06_24.035s_22.535-25.535s.wav"
    # starting_whoop = "audio_recording_2025-09-15T07_09_45.544480Z_ch07_30.245s_28.745-31.745s.wav" # weird behavior on ch 7

    whoops = sorted([f for f in os.listdir(whoops_folder) if f.endswith('.wav')])

    try:
        start_index = whoops.index(starting_whoop)
    except ValueError:
        start_index = -1

    if start_index != -1:
        whoops = whoops[start_index:]

    if not whoops:
        print("Nessun file .wav trovato in " + whoops_folder)
        exit(1)

    print("Trovati " + str(len(whoops)) + " file audio")
    print("="*80 + "\n")

    for file_idx, whoop_filename in enumerate(whoops, 1):
        # extract info encoded in the whoop filename (channel number, raw path, start-end time)
        whoop_filename_parsed = parse_filename(whoop_filename)

        channel_num = int(whoop_filename_parsed['channel'])

        if channel_num > num_channels_default:
            continue

        print(whoop_filename)
        print(f"Expected whoop at channel {channel_num}")

        rawfilepath = whoop_filename_parsed['raw_path']
        rawfilepath = os.path.join(raw_folder, rawfilepath + ".wav")

        try:
            signal_multichannel, sr = sf.read(rawfilepath)
        except Exception as e:
            print("Errore nel caricamento di " + rawfilepath + ": " + str(e))
            continue

        start_time = whoop_filename_parsed['start_time']
        end_time = whoop_filename_parsed['end_time']
        start_sample = time_to_samples(start_time, sr)
        end_sample = time_to_samples(end_time, sr) 

        ref_signal = signal_multichannel[start_sample:end_sample, channel_num - 1]

        # play ref signal
        print(f"\n[{file_idx}/{len(whoops)}] Playing reference signal from channel {channel_num}...")
        max_val = np.max(np.abs(ref_signal))
        if max_val > 0:
            ref_signal = ref_signal / (max_val * 1.1)
        sd.play(ref_signal, sr)
        sd.wait()

        # # salva ref signal
        # dir = "sounds/bees_symposium_audios"
        # sf.write(f"{dir}/listening_test_ref_ch{channel_num:02d}.wav", ref_signal, sr)
        # print("Reference signal saved.")

        # # 1) Riproduce la finestra su tutti i canali (come prima)
        # extract_and_play_channels(
        #     signal_multichannel,
        #     sr,
        #     start_time,
        #     end_time,
        #     num_channels=num_channels_default,
        #     pause_between=pause_between_default,
        #     channel_broken=channel_broken
        # )
    
        # # Carica audio
        # signal, sr = sf.read("audio.wav")
        # ref_signal = signal[start_sample:end_sample, reference_channel]

        # Crea detector e analizza
        detector = StrongChannelDetector(signal_multichannel, sr, channel_num - 1, start_time, end_time, broken_channels=channel_broken, num_channels=16, verbose=True,plot=True)
        detector.analyze_reference()
        results = detector.detect_strong_channels()

        # Accedi risultati
        if results['strongest_channel'] is not None:
            print(f"Canale più forte: {results['strongest_channel'].channel_num + 1}")
            print(f"HNR: {results['strongest_channel'].hnr_db:.2f} dB")
            print(f"Numero canali con evento: {results['num_channels_with_whoop']}")
        else:
            print("Nessun canale con whoop rilevato.")

        # Ottieni l'array HNR lineare (IDENTICO al tuo channel_levels)
        channel_levels = detector.get_channel_levels_array()

        # Plot
        # coordinate dei microfoni
        channel_coords, boundaries = load_coordinates_with_labels("coordinates/mic_positions_16_ch.txt")
        # remove labels from coordinates
        channel_coords = np.array([(coord['x'], coord['y']) for coord in channel_coords])
        # detector.plot_voronoi_2d(
        #     channel_coords[:num_channels_default], 
        #     boundaries=boundaries,  # ← QUI!
        #     use_db=False, 
        #     cmap="hot" 
        # )

        # Dopo aver eseguito detect_strong_channels()
        detector.plot_hexagon_hnr_map(
            mic_positions=channel_coords,
            hexagon_radius=0.025,
            use_db=False,
            title="HNR Hexagon Map"
        )


if __name__ == "__main__":
    main()
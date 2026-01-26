from classes.whoop_localizator import SRPLocalizator, MicrophoneArray, SearchGrid, plot_power_map_2d
import numpy as np
import re
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from classes.pitch_detector import PitchDetector
from classes.whoop_detector import WhoopDetector

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

def validate_signals(signals: np.ndarray, sr: int, broken_channels: list, f0_ref: float, tolerance: float = 50.0) -> list:
    """
    controlla il pitch di ogni canale e salva in un array gli indici dei canali validi (con pitch nel range atteso)
    scarta anche i canali rotti in broken_channels
    """
    print(f"Validating signals against reference f0: {f0_ref} Hz with tolerance {tolerance} Hz")
    valid_channels = []
    for ch in range(signals.shape[1]):
        if ch in broken_channels:
            print(f"  → Channel {ch+1} is broken, skipping.")
            continue

        valid_channels.append(ch)
        print(f"  → Channel {ch+1} valid")
        # pitch_detector = PitchDetector(audio_segment=signals[:, ch], sr=sr)
        # f0,_,_,_ = pitch_detector.estimate_f0(
        #     length_queue=3, 
        #     hz_threshold=25, 
        #     threshold_increment=1.3,
        #     plot=True,
        #     padding_start_ms=20,    # 20ms prima
        #     padding_end_ms=25,      # 25ms dopo
        #     freq_min=200,
        #     freq_max=600
        # )
        # if f0 is not None and abs(f0 - f0_ref) <= tolerance:
        #     valid_channels.append(ch)
        #     print(f"  → Channel {ch+1} valid with pitch {f0:.2f} Hz")
        # else:
        #     print(f"  → Channel {ch+1} invalid with pitch {f0} Hz")

    print(f"Valid channels based on pitch detection: {[ch+1 for ch in valid_channels]}")

    return valid_channels



def main():

    # PIPELINE 
    # 1) partenza da segmento grezzo che dovrebbe contenere al suo interno un whoop (file mono con inizio e fine window nell'intestazione)
    # 2) analisi più fine all'interno di questo segmento con WhoopDetector per ridurre la finestra attorno al whoop
    # 3) nella finestra ridotta, si estrae una finestra ancora più piccola e precisa tramite PitchDetector (non funziona su finestre che comprendono anche il rumore)
    # 4) si calcolano i sample assoluti di inizio e fine whoop nel file grezzo originale
    # 5) si estraggono i segnali multicanale corrispondenti (16 segnali estratti dal raw multicanale che iniziano e finiscono nei sample calcolati al punto 4)
    # 6) si usano questi segnali estratti come input per la localizzazione SRP
    # 7) ....


    # 1. Definisci il tuo array di microfoni
    # mic_positions = np.array([...])  # shape (32, 2)
    # MIC_POSITIONS_16CH = np.array([
    #     [0.0000, 0.16425],  # Ch  1
    #     [0.0000, 0.11225],  # Ch  2
    #     [0.0000, 0.06025],  # Ch  3
    #     [0.0000, 0.00825],  # Ch  4
    #     [0.0500, 0.00825],  # Ch  5
    #     [0.0500, 0.06025],  # Ch  6
    #     [0.0500, 0.11225],  # Ch  7
    #     [0.0500, 0.16425],  # Ch  8
    #     [0.1000, 0.16425],  # Ch  9
    #     [0.1000, 0.11225],  # Ch 10
    #     [0.1000, 0.06025],  # Ch 11
    #     [0.1000, 0.00825],  # Ch 12
    #     [0.1500, 0.00825],  # Ch 13
    #     [0.1500, 0.06025],  # Ch 14
    #     [0.1500, 0.11225],  # Ch 15
    #     [0.1500, 0.16425],  # Ch 16
    # ])

    MIC_POSITIONS_16CH = np.array([
        [0.054, 0.253],  # Ch  1 (A) 
        [0.054, 0.202],  # Ch  2 BROKEN
        [0.055, 0.149],  # Ch  3 BROKEN
        [0.055, 0.093],  # Ch  4 (B)
        [0.101, 0.067],  # Ch  5 (C)
        [0.10, 0.12],  # Ch  6 (D)
        [0.099, 0.172],  # Ch  7 (E)
        [0.098, 0.226],  # Ch  8 BROKEN
        [0.142, 0.253],  # Ch  9 (F)
        [0.142, 0.201],  # Ch  10 (G)
        [0.142, 0.148],  # Ch 11 (H)
        [0.142, 0.097],  # Ch 12 (I)
        [0.187, 0.070],  # Ch 13 BROKEN
        [0.187, 0.121],  # Ch 14 (J)
        [0.187, 0.173],  # Ch 15 (K)
        [0.188, 0.225],  # Ch 16 (L)
    ])

    MIC_POSITIONS_32CH = np.array([
        [0.054, 0.253],  # Ch  1 (A) 
        [0.054, 0.202],  # Ch  2 BROKEN
        [0.055, 0.149],  # Ch  3 BROKEN
        [0.055, 0.093],  # Ch  4 (B)
        [0.101, 0.067],  # Ch  5 (C)
        [0.10, 0.12],  # Ch  6 (D)
        [0.099, 0.172],  # Ch  7 (E)
        [0.098, 0.226],  # Ch  8 BROKEN
        [0.142, 0.253],  # Ch  9 (F)
        [0.142, 0.201],  # Ch  10 (G)
        [0.142, 0.148],  # Ch 11 (H)
        [0.142, 0.097],  # Ch 12 (I)
        [0.187, 0.070],  # Ch 13 BROKEN
        [0.187, 0.121],  # Ch 14 (J)
        [0.187, 0.173],  # Ch 15 (K)
        [0.188, 0.225],  # Ch 16 (L)
        [0.230, 0.253],  # Ch  17 (M) 
        [0.231, 0.200],  # Ch  18 (N)
        [0.231, 0.149],  # Ch  19 (O) 
        [0.231, 0.097],  # Ch  20 (P)
        [0.272, 0.068],  # Ch  21 BROKEN
        [0.272, 0.120],  # Ch  22 (Q)
        [0.273, 0.172],  # Ch  23 (R) 
        [0.274, 0.224],  # Ch  24 (S)
        [0.320, 0.254],  # Ch  25 BROKEN
        [0.319, 0.200],  # Ch  26 (T)
        [0.320, 0.150],  # Ch 27 BROKEN
        [0.319, 0.098],  # Ch 28 BROKEN
        [0.356, 0.063],  # Ch 29 (U) 
        [0.356, 0.120],  # Ch 30 (V)
        [0.356, 0.175],  # Ch 31 BROKEN
        [0.356, 0.224],  # Ch 32 (W)
    ])

    channel_broken = [2, 3, 8, 13, 21, 25, 27, 28, 31]


    boundaries = [0.0, 0.37, 0.0, 0.275]  # (x_min, x_max, y_min, y_max)


    


    # 2. Carica i segnali
    reference_whoop_folder = "sounds/best_whoop_available/"
    reference_whoop_path = "audio_recording_2025-09-15T06_40_43.498109Z_ch04_5.005s_3.505-6.505s.wav"
    # reference_whoop_path = "audio_recording_2025-09-15T07_18_43.462247Z_ch12_47.135s_45.635-49.295s.wav"
    # reference_whoop_path = "audio_recording_2025-09-15T08_04_43.560131Z_ch06_44.705s_43.205-46.205s.wav"
    raw_audio_folder = "sounds/tmp/"

    reference_whoop_info = parse_filename(reference_whoop_path)

    # carica il segnale con 32 canali
    raw_audio, sr = sf.read(raw_audio_folder + reference_whoop_info['raw_path'] + ".wav", dtype='float32')

    reference_whoop, _ = sf.read(reference_whoop_folder + reference_whoop_path, dtype='float32')

    # estrai finestra più fine attorno al whoop con HNR analysis abbassando window_length_ms e hop_length_ms
    # Inizializza il detector
    detector = WhoopDetector(
        signal=reference_whoop,
        sr=sr,
        window_length_ms=15,
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
        percentile=90,
        offset=4,
        window_sec=0.7, # lunghezza finestra di analisi intorno al picco
        merge_overlaps=True
    )

    # Accedi ai risultati
    print(f"Picchi rilevati: {len(detector.peak_times_)}")

    # Estrai i segmenti audio
    segments = detector.extract_segments()

    detector.plot_analysis()

    for segment_idx, segment in enumerate(segments):
        pitch_detector = PitchDetector(audio_segment=segment, sr=sr)
        f0, queue, all_queues, whoop_info = pitch_detector.estimate_f0(
            length_queue=5, 
            hz_threshold=25, 
            threshold_increment=1.3,
            plot=True,
            padding_start_ms=20,    # 20ms prima
            padding_end_ms=25,      # 25ms dopo
            freq_min=200,
            freq_max=600
        )

        if whoop_info is not None:
            pitch_detector.print_whoop_info()

            # to get the absolute start and end samples in the original audio we need outer segment start (present in the name),
            # the inner segment start (present in peak_windows_) and finally the whoop_info start_sample_padded and end_sample_padded

            m = re.search(r"_([0-9]+\.[0-9]+)-[0-9]+\.[0-9]+s\.wav$", reference_whoop_path)
            if m:
                outer_offset = float(m.group(1)) 

            inner_offset = detector.peak_windows_[segment_idx][0]  # in seconds

            absolute_start_time = inner_offset + outer_offset + whoop_info['start_sample_padded'] / sr
            absolute_end_time = inner_offset + outer_offset + whoop_info['end_sample_padded'] / sr
            print(f"Whoop absolute window: {absolute_start_time:.3f}s - {absolute_end_time:.3f}s")

            absoulte_start_time_expanded = absolute_start_time - 0.1  # 100ms before
            absoulte_end_time_expanded = absolute_end_time + 0.1  # 100ms after

            print(f"Expanded absolute window for localization: {absoulte_start_time_expanded:.3f}s - {absoulte_end_time_expanded:.3f}s")
            
            absolute_start_sample = time_to_samples(absoulte_start_time_expanded, sr)
            absolute_end_sample = time_to_samples(absoulte_end_time_expanded, sr)
        


            channels = np.arange(0, 16)

            signals = raw_audio[absolute_start_sample:absolute_end_sample, channels]  # shape (n_samples, 32)

            validated_channels = validate_signals(signals, sr,channel_broken, f0_ref=f0, tolerance=50.0)

            # for i in range(16):
            #     # print(f"Preprocessing channel {i+1}...")
            #     signals[:, i] = preprocess_signal(
            #         signals[:, i],
            #         sr,
            #         lowpass_cutoff=20000,
            #         highpass_cutoff=2000,
            #         target_rms=0.1
            #     )
            
            

            # 3. Crea la griglia di ricerca
            mic_array = MicrophoneArray(
                positions=MIC_POSITIONS_16CH,
                sample_rate=sr,
                validated_channels=validated_channels,  # -> zero-based
                boundaries=boundaries,
                margin=0.01
            )

            localizer = SRPLocalizator(mic_array, c=343.0)
            # localizer = SRPLocalizator(mic_array, c=343.0, reference_channel=strongest_channel)

            strongest_channel=reference_whoop_info['channel'] - 1  # zero-based index

            grid = localizer.create_search_grid_centered_on_channel(
                x_width=0.3,      # ← parametri configurabili
                y_height=0.4,
                resolution=0.02,
                reference_channel=strongest_channel
            )
            


            # 4. Localizza!
            result = localizer.localize(signals, grid, max_tau_ms=100)
            print(f"Whoop position: {result.estimated_position}")
            print(f"Power: {result.max_power}")

            # Visualizza il risultato
            plot_power_map_2d(result, mic_array, grid)
            plt.show()



if __name__ == "__main__":
    main()
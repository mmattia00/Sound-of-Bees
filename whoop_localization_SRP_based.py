from whoop_localizator import SRPLocalizator, MicrophoneArray, SearchGrid, plot_power_map_2d
import numpy as np
import re
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

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



def main():
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
        [0.052, 0.249],  # Ch  1 (A) 
        [0.0000, 0.00000],  # Ch  2 BROKEN
        [0.0000, 0.00000],  # Ch  3 BROKEN
        [0.055, 0.093],  # Ch  4 (B)
        [0.101, 0.067],  # Ch  5 (C)
        [0.10, 0.12],  # Ch  6 (D)
        [0.099, 0.172],  # Ch  7 (E)
        [0.0000, 0.00000],  # Ch  8 BROKEN
        [0.142, 0.253],  # Ch  9 (F)
        [0.142, 0.201],  # Ch  10 (G)
        [0.142, 0.148],  # Ch 11 (H)
        [0.142, 0.097],  # Ch 12 (I)
        [0.0000, 0.00000],  # Ch 13 BROKEN
        [0.187, 0.121],  # Ch 14 (J)
        [0.187, 0.173],  # Ch 15 (K)
        [0.188, 0.225],  # Ch 16 (L)
    ])

    channel_broken = [2, 3, 8, 13, 21, 25, 27, 28, 31]

    mic_array = MicrophoneArray(
        positions=MIC_POSITIONS_16CH,
        sample_rate=48000,
        channel_broken=[ch-1 for ch in channel_broken],  # -> zero-based
    )


    # 2. Carica i segnali
    reference_whoop_folder = "sounds/best_whoop_available/"
    reference_whoop_path = "audio_recording_2025-09-15T06_40_43.498109Z_ch04_5.005s_3.505-6.505s.wav"
    # reference_whoop_path = "audio_recording_2025-09-15T17_12_49.403810Z_ch12_49.895s_48.395-51.395s.wav"
    # reference_whoop_path = "audio_recording_2025-09-15T08_04_43.560131Z_ch06_44.705s_43.205-46.205s.wav"
    raw_audio_folder = "D:/soundofbees/"

    reference_whoop_info = parse_filename(reference_whoop_path)

    # carica il segnale con 32 canali
    raw_audio, sr = sf.read(raw_audio_folder + reference_whoop_info['raw_path'] + ".wav", dtype='float32')

    # Calcola la finestra temporale PRIMA
    start_sample = time_to_samples(reference_whoop_info['start_time'], sr)
    end_sample = time_to_samples(reference_whoop_info['end_time'], sr)
    channels = np.arange(0, 16)

    signals = raw_audio[start_sample:end_sample, channels]  # shape (n_samples, 32)

    # for i in range(16):
    #     # print(f"Preprocessing channel {i+1}...")
    #     signals[:, i] = preprocess_signal(
    #         signals[:, i],
    #         sr,
    #         lowpass_cutoff=20000,
    #         highpass_cutoff=2000,
    #         target_rms=0.1
    #     )
    
    strongest_channel=reference_whoop_info['channel'] - 1  # zero-based index

    # 3. Crea la griglia di ricerca

    localizer = SRPLocalizator(mic_array, c=343.0)
    # localizer = SRPLocalizator(mic_array, c=343.0, reference_channel=strongest_channel)


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
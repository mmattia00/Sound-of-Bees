import librosa
import numpy as np

def estimate_fundamental(audio_path, duration=2.0, fmin=100, fmax=500, energy_threshold=0.01):
    y, sr = librosa.load(audio_path, sr=None, mono=True, duration=duration)
    # Calcolo dello spettro pitch usando il metodo Probabilistic Interframe Pitch (PIP)
    # restituisce due matrici 2D: pitches e magnitudes per entrambe il numero di colonne corrisponde ai frame temporali
    # per pitches il numero di righe corrisponde alle possibili frequenze
    # per magnitudes il numero di righe corrisponde alle confidenze associate a ciascuna frequenza
    # esempio per uno slice temporale
    # pitches[:, 50] = [0, 0, 45.3, 0, 250.2, 0, ..., 0]  # frequenze candidate (Hz)
    # magnitudes[:, 50] = [0.001, 0.0005, 0.15, 0.002, 0.85, 0.001, ..., 0]  # confidenze
    # la più probabile frequenza in questo frame è 250.2 Hz con confidenza 0.85

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=fmin, fmax=fmax)

    print(pitches )
    
    # Costruzione di lista frequenze fondamentali dai frame più energetici
    fundamental_freqs = []
    for i in range(magnitudes.shape[1]):
        # estrai magnitudini e pitch per il frame i (una colonna)
        mag_slice = magnitudes[:, i]
        pitch_slice = pitches[:, i]
        
        # Considera solo frame con energia significativa
        if mag_slice.max() > energy_threshold:
            # Frequenza con magnitudine (confidenza) massima in questo frame
            index = mag_slice.argmax()
            freq = pitch_slice[index]
            if freq > 0:
                fundamental_freqs.append(freq)
    
    if fundamental_freqs:
        # analizza tutte le frequenze fondamentali trovate e prendi quella centrale
        # Mediana (più robusta della media) come stima fondamentale
        fundamental = np.median(fundamental_freqs)
    else:
        fundamental = None
        
    return fundamental, sr

# Uso esempio
audio_file = "sounds/whoop/audio_recording_2025-09-10T08_10_42.196890Z_ch09_31.565s_30.065-33.145s.wav"
fundamental_freq, sample_rate = estimate_fundamental(audio_file)
print(f"Frequenza fondamentale stimata: {fundamental_freq} Hz")
import os
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import threading


base_folder = r"D:\whooping_candidates"
starting_audiofile_name = "audio_recording_2025-09-10T13_43_42.212901Z"
whoop_folder = "whoop"
weird_sounds_folder = "weird_sounds"
whoop_spec_folder = "whoop_spectrograms"
weird_spec_folder = "weird_spectrograms"


# Crea cartelle di output
os.makedirs(whoop_folder, exist_ok=True)
os.makedirs(whoop_spec_folder, exist_ok=True)
os.makedirs(weird_sounds_folder, exist_ok=True)
os.makedirs(weird_spec_folder, exist_ok=True)

# Attiva modalità interattiva di Matplotlib
plt.ion()

# Prendi l'elenco ordinato delle sottocartelle
folders = sorted([f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))])

# Trova l'indice della cartella di partenza
try:
    start_index = folders.index(starting_audiofile_name)
except ValueError:
    start_index = -1

# Prendi tutte le cartelle successive
if start_index != -1:
    successive_folders = folders[start_index:]
else:
    successive_folders = []


def compute_spectrogram(data, samplerate):
    """Calcola lo spettrogramma una sola volta"""
    nperseg = 1024
    noverlap = nperseg // 2
    nfft = nperseg * 4
    window = 'hann'
    
    frequencies, times, spectrogram_data = signal.spectrogram(
        data, 
        fs=samplerate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window,
        scaling='density'
    )
    
    # Converti in dB con guadagno di 20
    spectrogram_db = 20 * np.log10(spectrogram_data + 1e-10)
    
    return frequencies, times, spectrogram_db


def plot_spectrogram_from_data(frequencies, times, spectrogram_db, title="Spectrogram"):
    """Visualizza uno spettrogramma già calcolato"""
    plt.figure(figsize=(12, 6))
    
    # Visualizza con intervallo di 80 dB
    vmin = np.nanmax(spectrogram_db) - 80
    vmax = np.nanmax(spectrogram_db)
    
    im = plt.pcolormesh(times, frequencies, spectrogram_db, 
                        shading='gouraud', cmap='hot', vmin=vmin, vmax=vmax)
    
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    plt.title(title, fontsize=14)
    cbar = plt.colorbar(im, label='Power (dB)')
    
    plt.ylim([0, 20000])
    plt.xlim([times[0], times[-1]])
    
    plt.draw()
    plt.pause(0.1)


def save_spectrogram_from_data(frequencies, times, spectrogram_db, output_path):
    """Salva uno spettrogramma già calcolato"""
    plt.figure(figsize=(12, 6))
    
    vmin = np.nanmax(spectrogram_db) - 80
    vmax = np.nanmax(spectrogram_db)
    
    im = plt.pcolormesh(times, frequencies, spectrogram_db, 
                        shading='gouraud', cmap='hot', vmin=vmin, vmax=vmax)
    
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    plt.title(os.path.basename(output_path), fontsize=14)
    cbar = plt.colorbar(im, label='Power (dB)')
    
    plt.ylim([0, 20000])
    plt.xlim([times[0], times[-1]])
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def play_audio_in_thread(data, samplerate):
    """Riproduci audio in un thread separato (non bloccante)"""
    def _play():
        sd.play(data, samplerate)
        sd.wait()
    
    thread = threading.Thread(target=_play, daemon=True)
    thread.start()

for folder in successive_folders:
    print("-" * 80)
    print(f"Analyzing the candidates contained in the audio: {folder}")
    print("-" * 80)

    folder_path = os.path.join(base_folder, folder)
    
    # Trova tutti i file .wav nella cartella corrente
    candidates = sorted([f for f in os.listdir(folder_path) 
                        if f.endswith('.wav') and os.path.isfile(os.path.join(folder_path, f))])
    
    for candidate in candidates:
        candidate_path = os.path.join(folder_path, candidate)
        print(f"\nProcessing candidate: {candidate}")
        
        # Carica il file audio
        data, samplerate = sf.read(candidate_path)
        
        print("Computing spectrogram...")
        # Calcola lo spettrogramma UNA SOLA VOLTA
        frequencies, times, spectrogram_db = compute_spectrogram(data, samplerate)
        
        # Visualizza lo spettrogramma (non bloccante)
        plot_spectrogram_from_data(frequencies, times, spectrogram_db, 
                                   title=f"Candidate: {candidate}")
        
        # Riproduci l'audio in thread separato (non bloccante)
        play_audio_in_thread(data, samplerate)
        
        # Chiedi all'utente
        user_input = input(f"Is this a whoop (1), a weird sound (2) or noise (3) or skip all the remaining channels (4), play again (5)? ").strip().lower()
        
        # Chiudi la finestra del grafico
        plt.close('all')
        
        while user_input == '5':
            # Riproduci di nuovo l'audio
            play_audio_in_thread(data, samplerate)
            user_input = input(f"Is this a whoop (1), a weird sound (2) or noise (3) or skip all the remaining channels (4), play again (5)? ").strip().lower()
            plt.close('all')
        
        if user_input == '1':
            output_filename = f"{folder}_{candidate}"
            audio_output_path = os.path.join(whoop_folder, output_filename)
            spec_output_path = os.path.join(whoop_spec_folder, 
                                           output_filename.replace('.wav', '.png'))
            
            sf.write(audio_output_path, data, samplerate)
            print(f"✓ Audio saved as: {audio_output_path}")
            
            save_spectrogram_from_data(frequencies, times, spectrogram_db, spec_output_path)
            print(f"✓ Spectrogram saved as: {spec_output_path}")
            
        elif user_input == '2':
            output_filename = f"{folder}_{candidate}"
            audio_output_path = os.path.join(weird_sounds_folder, output_filename)
            spec_output_path = os.path.join(weird_spec_folder, 
                                           output_filename.replace('.wav', '.png'))
            
            sf.write(audio_output_path, data, samplerate)
            print(f"✓ Audio saved as: {audio_output_path}")
            
            save_spectrogram_from_data(frequencies, times, spectrogram_db, spec_output_path)
            print(f"✓ Spectrogram saved as: {spec_output_path}")
            
        elif user_input == '3':
            print("✗ Candidate rejected")
            continue
            
        elif user_input == '4':
            print("Skipping all remaining channels.")
            break


print("\n" + "=" * 80)
print("Labeling complete!")
print("=" * 80)

import os
import soundfile as sf
import sounddevice as sd
import shutil
import threading
import numpy as np


def normalize_signal_rms(signal, target_rms=0.1):
        """Normalizza il segnale a un RMS target."""
        current_rms = np.sqrt(np.mean(signal**2))
        if current_rms < 1e-6:
            return signal
        scaling_factor = target_rms / current_rms
        return signal * scaling_factor


# Configurazione
whoop_folder = "sounds/whoop"
validated_folder = os.path.join(whoop_folder, "validated_whoop")
other_behaviour_folder = os.path.join(whoop_folder, "other_behaviour")
trash_folder = os.path.join(whoop_folder, "trash")

# Crea cartelle di output
os.makedirs(validated_folder, exist_ok=True)
os.makedirs(other_behaviour_folder, exist_ok=True)
os.makedirs(trash_folder, exist_ok=True)

def play_audio_in_thread(data, samplerate):
    """Riproduci audio in un thread separato (non bloccante)"""
    def _play():
        sd.play(data, samplerate)
        sd.wait()
    
    thread = threading.Thread(target=_play, daemon=True)
    thread.start()

# Trova tutti i file .wav nella cartella whoop (escludendo le sottocartelle)
audio_files = sorted([f for f in os.listdir(whoop_folder) 
                     if f.endswith('.wav') and os.path.isfile(os.path.join(whoop_folder, f))])

print(f"Found {len(audio_files)} audio files to validate")
print("=" * 80)
print("Instructions:")
print("  w = Validated whoop")
print("  o = Other behaviour")
print("  t = Trash (noise/irrelevant)")
print("  r = Repeat playback")
print("=" * 80)

for idx, audio_file in enumerate(audio_files, 1):
    audio_path = os.path.join(whoop_folder, audio_file)
    
    print(f"\n[{idx}/{len(audio_files)}] Processing: {audio_file}")
    
    # Carica il file audio
    data, samplerate = sf.read(audio_path)

    # Normalizza il segnale
    data = normalize_signal_rms(data, target_rms=0.1)
    
    # Riproduci l'audio
    play_audio_in_thread(data, samplerate)
    
    # Chiedi all'utente
    while True:
        user_input = input("Whoop (w), Other behaviour (o), Trash (t), or Repeat (r)? ").strip().lower()
        
        if user_input == 'r':
            # Riproduci di nuovo
            play_audio_in_thread(data, samplerate)
            continue
        
        elif user_input == 'w':
            # Copia in validated_whoop
            dest_path = os.path.join(validated_folder, audio_file)
            shutil.copy(audio_path, dest_path)
            print(f"✓ Copied to validated_whoop")
            break
            
        elif user_input == 'o':
            # Copia in other_behaviour
            dest_path = os.path.join(other_behaviour_folder, audio_file)
            shutil.copy(audio_path, dest_path)
            print(f"✓ Copied to other_behaviour")
            break
            
        elif user_input == 't':
            # Copia in trash
            dest_path = os.path.join(trash_folder, audio_file)
            shutil.copy(audio_path, dest_path)
            print(f"✓ Copied to trash")
            break
        
        else:
            print("Invalid input. Use 'w' for whoop, 'o' for other behaviour, 't' for trash, 'r' to repeat")

print("\n" + "=" * 80)
print("Validation complete!")
print(f"Validated whoops: {len(os.listdir(validated_folder))} files")
print(f"Other behaviour: {len(os.listdir(other_behaviour_folder))} files")
print(f"Trash: {len(os.listdir(trash_folder))} files")
print("=" * 80)

import os
import soundfile as sf
import sounddevice as sd
import shutil
import threading

# Configurazione
whoop_folder = "sounds/whoop"
strong_folder = os.path.join(whoop_folder, "strong")
faint_folder = os.path.join(whoop_folder, "faint")

# Crea cartelle di output
os.makedirs(strong_folder, exist_ok=True)
os.makedirs(faint_folder, exist_ok=True)

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

print(f"Found {len(audio_files)} audio files to classify")
print("=" * 80)

for audio_file in audio_files:
    audio_path = os.path.join(whoop_folder, audio_file)
    
    print(f"\nProcessing: {audio_file}")
    
    # Carica il file audio
    data, samplerate = sf.read(audio_path)
    
    # Riproduci l'audio
    play_audio_in_thread(data, samplerate)
    
    # Chiedi all'utente
    while True:
        user_input = input("Strong (s), Faint (f), or Repeat (r)? ").strip().lower()
        
        if user_input == 'r':
            # Riproduci di nuovo
            play_audio_in_thread(data, samplerate)
            continue
        
        elif user_input == 's':
            # Copia in cartella strong
            dest_path = os.path.join(strong_folder, audio_file)
            shutil.copy(audio_path, dest_path)
            print(f"✓ Copied to strong: {dest_path}")
            break
            
        elif user_input == 'f':
            # Copia in cartella faint
            dest_path = os.path.join(faint_folder, audio_file)
            shutil.copy(audio_path, dest_path)
            print(f"✓ Copied to faint: {dest_path}")
            break
        
        else:
            print("Invalid input. Use 's' for strong, 'f' for faint, 'r' to repeat")

print("\n" + "=" * 80)
print("Classification complete!")
print(f"Strong: {len(os.listdir(strong_folder))} files")
print(f"Faint: {len(os.listdir(faint_folder))} files")
print("=" * 80)

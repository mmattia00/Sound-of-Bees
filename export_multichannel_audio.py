import os
import soundfile as sf

def split_multichannel_audio(input_path, output_folder):
    # Legge l'audio multicanale
    data, samplerate = sf.read(input_path)

    # Verifica che l'audio abbia 32 canali
    if data.ndim < 2 or data.shape[1] != 32:
        raise ValueError("L'audio non ha 32 canali")

    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Esporta ogni canale come file WAV separato
    for i in range(32):
        out_path = os.path.join(output_folder, f"track_{i+1}.wav")
        sf.write(out_path, data[:, i], samplerate)

    return f"Esportate 32 tracce in {output_folder}"

if __name__ == "__main__":
    audio_name = "audio_recording_2025-09-15T13_57_49.409772Z.wav"
    base_input_folder = "D:/soundofbees"
    input_audio_path = os.path.join(base_input_folder, audio_name)
    output_dir = f"C:/Users/ACE71542GR175/Desktop/TESI/Sound-of-Bees/significative_parts/{audio_name}"  # Cartella di output per le tracce separate

    result = split_multichannel_audio(input_audio_path, output_dir)
    print(result)
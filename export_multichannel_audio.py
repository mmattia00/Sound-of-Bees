import os
import soundfile as sf


def export_channels(input_path, output_folder, channels):
    """
    input_path    : path al file wav multicanale
    output_folder : cartella di output
    channels      : lista di indici canale 0-based (es. [0, 8, 15])
    """
    data, samplerate = sf.read(input_path)

    if data.ndim < 2:
        raise ValueError("Il file audio è mono, nessun canale da estrarre")

    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for ch in channels:
        if ch >= data.shape[1]:
            print(f"Canale {ch} non esiste (il file ha {data.shape[1]} canali), skip")
            continue
        out_path = os.path.join(output_folder, f"{base_name}_ch_{ch:02d}.wav")
        sf.write(out_path, data[:, ch], samplerate)
        print(f"Esportato: {out_path}")


if __name__ == "__main__":
    input_base_folder = "E:/soundofbees"
    input_audio_name = "audio_recording_2025-09-15T16_12_49.177436Z.wav"
    input_audio_path = os.path.join(input_base_folder, input_audio_name)
    output_dir = "E:/training_dataset_finetuning_raw"
    channels_to_export = [19, 13, 12, ]  # indici 0-based

    export_channels(input_audio_path, output_dir, channels_to_export)

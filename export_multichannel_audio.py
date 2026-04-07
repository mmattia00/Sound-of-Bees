import os
import soundfile as sf


def parse_job_file(job_file_path):
    """
    Legge il file di testo e restituisce una lista di tuple (audio_name, channels).
    """
    jobs = []
    with open(job_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            # parts[0] = nome file, parts[1] = peaktime (skip), parts[2:] = canali
            audio_name = parts[0] + ".wav"
            channels = []
            for p in parts[2:]:
                if p == "":
                    continue
                try:
                    channels.append(int(p))
                except ValueError:
                    print(f"Valore canale non valido ignorato: '{p}'")
            jobs.append((audio_name, channels))
    return jobs


def get_output_path(output_folder, base_name, ch):
    """
    Restituisce il path di output, aggiungendo '_bis' se il file esiste già.
    """
    candidate = os.path.join(output_folder, f"{base_name}_ch_{ch:02d}.wav")
    if not os.path.exists(candidate):
        return candidate
    # File già esistente → aggiungi _bis
    return os.path.join(output_folder, f"{base_name}_ch_{ch:02d}_bis.wav")


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
        out_path = get_output_path(output_folder, base_name, ch)
        sf.write(out_path, data[:, ch], samplerate)
        print(f"Esportato: {out_path}")


if __name__ == "__main__":
    input_base_folder  = "E:/soundofbees"
    output_dir         = "E:/training_dataset_finetuning_raw"
    job_file           = "E:/training_dataset_finetuning_raw/jobs.txt"  # il tuo file di testo

    jobs = parse_job_file(job_file)

    for audio_name, channels in jobs:
        input_audio_path = os.path.join(input_base_folder, audio_name)

        if not os.path.exists(input_audio_path):
            print(f"File non trovato, skip: {input_audio_path}")
            continue

        print(f"\n--- Elaborazione: {audio_name} | Canali: {channels} ---")
        export_channels(input_audio_path, output_dir, channels)

    print("\nDone.")

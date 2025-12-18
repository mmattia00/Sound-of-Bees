import numpy as np
import soundfile as sf
import sounddevice as sd
from whoop_detector import WhoopDetector
from whoop_localizator import MicrophoneArray, SRPLocalizator, plot_power_map_2d
import matplotlib.pyplot as plt
import os


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

def get_ground_truth_coordinates(filename, localization_points_coords):

    label = filename.split("_")[-1].replace(".wav", "")
    print(f"Label estratta dal filename: {label}")
    for coord in localization_points_coords[0]:
        if coord['label'] == label:
            print(f"Coordinate di ground truth per il punto '{label}': x={coord['x']}, y={coord['y']}")
            return coord['x'], coord['y']



def main():

    num_channels_analyzed = 16
    channel_coords, boundaries = load_coordinates_with_labels("coordinates/mic_positions_16_ch.txt")
    # remove labels from coordinates
    channel_coords = np.array([(coord['x'], coord['y']) for coord in channel_coords])
    localization_points_coords = load_coordinates_with_labels("coordinates/localization_points.txt")
    print("Loaded coordinates")

    # check to see if coordinates are loaded correctly
    # print("Coordinate caricate:")
    # for coord in channel_coords:
    #     print(f"x: {coord[0]}, y: {coord[1]}")
    # for point in localization_points_coords[0]:
    #     print(f"Localization point - x: {point['x']}, y: {point['y']}, label: {point['label']}")
    # print(boundaries)

    channels_broken = [2, 8, 21, 24, 25, 27, 28] # 1-based indexing
    channels_broken = [x - 1 for x in channels_broken]  # zero-based indexing
    # all the channels except the broken ones
    validated_channels = [i for i in range(num_channels_analyzed) if i not in channels_broken]

    
    # esegui la localizzazione su tutti i file di test nella cartella sounds/localization_test
    folder = "sounds/localization_test/noisy"
    starting_audiofile_name = "noisy_Point_i.wav"
    
    # Trova tutti i file .wav nella cartella
    audio_files = sorted([f for f in os.listdir(folder) if f.endswith('.wav')])
    
    # Trova l’indice dell audio file di partenza
    try:
        start_index = audio_files.index(starting_audiofile_name)
    except ValueError:
        start_index = -1

    # Prendi tutti i file successivi
    if start_index != -1:
        audio_files = audio_files[start_index:]
    


    if not audio_files:
        print("Nessun file .wav trovato in " + folder)
        exit(1)
    
    print("Trovati " + str(len(audio_files)) + " file audio")
    print("="*80 + "\n")

    for file in audio_files:

        raw_multichannel_audio_filename = os.path.join(folder, file)
        raw_multichannel_audio, sr = sf.read(raw_multichannel_audio_filename, dtype='float32')

        ground_truth_coords = get_ground_truth_coordinates(raw_multichannel_audio_filename, localization_points_coords)
        print(f"Ground truth coordinates: {ground_truth_coords}")
        # estrai la finestra attorno al whoop
        windows = []

        for idx in range(num_channels_analyzed):
            if idx in channels_broken:
                continue
            
            # prendi i primi 50 secondi di audio per ogni canale
            channel_data = raw_multichannel_audio[0:int(50*sr), idx]

            # Inizializza il detector
            detector = WhoopDetector(\
                signal=channel_data,
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
                percentile=99,
                offset=4,
                window_sec=0.7, # lunghezza finestra di analisi intorno al picco
                merge_overlaps=True
            )

            # Accedi ai risultati
            # print(f"Picchi rilevati: {len(detector.peak_times_)}")

            # print(detector.peak_windows_)

            windows.append(detector.peak_windows_)

            # detector.plot_analysis()

        # if windows contains null elements, remove them
        windows = [w for w in windows if w is not None and len(w) > 0]
        final_window = np.average(windows, axis=0)
        final_window = np.squeeze(final_window)  # da (1, 2) a (2,)
        start_sec, end_sec = final_window
        print(f"Finestra finale calcolata: {start_sec:.3f}s - {end_sec:.3f}s")

        for idx in range(num_channels_analyzed):
            if idx in channels_broken:
                continue

            channel_data = raw_multichannel_audio[int(start_sec*sr):int(end_sec*sr), idx]
            sd.play(channel_data, sr)
            sd.wait()




        mic_array = MicrophoneArray(
                    positions=channel_coords[:num_channels_analyzed],
                    sample_rate=sr,
                    validated_channels=validated_channels,  # -> zero-based
                    boundaries=boundaries,
                    margin=0.00
                )
        
        localizer = SRPLocalizator(mic_array, c=343.0)

        grid = localizer.create_search_grid_full(resolution=0.02)


        signals = raw_multichannel_audio[int(start_sec*sr):int(end_sec*sr), :num_channels_analyzed]
        # 4. Localizza!
        result = localizer.localize(signals, grid, max_tau_ms=100)
        print(f"Whoop position: {result.estimated_position}")
        print(f"Power: {result.max_power}")

        # Visualizza il risultato
        plot_power_map_2d(result, mic_array, grid, ground_truth=ground_truth_coords)
        plt.show()
    

if __name__ == "__main__":
    main()

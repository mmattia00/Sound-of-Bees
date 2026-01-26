from classes.pipeline_draft import PipelineDraft
import os
import numpy as np

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
        coords = []
        labels = []
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
                coords.append((x, y))
                labels.append(label)

        coords = np.array(coords)
        labels = np.array(labels, dtype="U64")
        return coords, boundaries, labels


if __name__ == "__main__":


    mics_coords_namefile = "coordinates/mic_positions_32_ch.txt"
    mics_coords, frame_boundaries, mics_labels = load_coordinates_with_labels(mics_coords_namefile)

    # Canali rotti
    channel_broken = [2, 3, 8, 13, 21, 25, 27, 28, 31]
    channel_broken = [x - 1 for x in channel_broken]  # zero-based index

    # Cartella contenente i file audio raw
    raw_audio_folder = "E:/soundofbees"

    root_candidates_dir = "sounds/whoop_candidates_splitted_significative"
    # cerca tutte le cartelle dentro alla cartella root
    candidate_folders = [f for f in os.listdir(root_candidates_dir)]
    for candidate_folder in candidate_folders:
        # cerca tutti i file .wav dentro alla cartella del candidato
        candidate_files = [f for f in os.listdir(os.path.join(root_candidates_dir, candidate_folder)) if f.endswith('.wav')]
        for candidate_file in candidate_files:
            print(f"\nProcessing candidate file: {candidate_file} in folder: {candidate_folder}")
            
            # Inizializza la pipeline
            pipeline = PipelineDraft(candidate_file, candidate_folder, raw_audio_folder, mics_coords=mics_coords, 
                                     frame_boundaries=frame_boundaries, channel_broken=channel_broken)
            
            # Esegui la pipeline
            feats = pipeline.run_full_pipeline(plot=True, verbose=True, listening_test=False)

            print(feats)

            pipeline.save_features_to_database(hdf5_path='whoop_database.h5')
            
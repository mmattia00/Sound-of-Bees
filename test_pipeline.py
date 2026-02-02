from classes.pipeline_draft import PipelineDraft
from classes.pipeline_wav_grouper import WAVPipelineGrouper
import os
import numpy as np
import re
import time
from datetime import datetime


def get_sort_key(filename):
    """Estrae (canale_int, peaktime_float) per ordinamento."""
    ch_match = re.search(r'ch_(\d+)', filename)
    peak_match = re.search(r'peaktime_([\d.]+)', filename)
    ch = int(ch_match.group(1)) if ch_match else 999
    peaktime = float(peak_match.group(1)) if peak_match else float('inf')
    return (ch, peaktime)


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

    figures_characteristics = {
        'fig_size': (12, 5),
        'label_fontsize': 14,
        'title_fontsize': 16,
        'legend_fontsize': 12,
        'tick_fontsize': 14,
        'colorbar_labelsize': 14,    # Label "Power (dB)"
        'colorbar_ticksize': 12,     # Numeri sulla barra (es: -40, -30...)
    }

    peaktime_tolerance = 0.02  # secondi
    wait_time_minutes = 30  # tempo di attesa tra i cicli

    mics_coords_namefile = "coordinates/mic_positions_32_ch.txt"
    mics_coords, frame_boundaries, mics_labels = load_coordinates_with_labels(mics_coords_namefile)

    # Canali rotti
    channel_broken = [2, 3, 8, 13, 21, 25, 27, 28, 31]
    channel_broken = [x - 1 for x in channel_broken]  # zero-based index

    # Cartella contenente i file audio raw
    # raw_audio_folder = "E:/soundofbees"
    # raw_audio_folder = "/media/uni-konstanz/My Passport/soundofbees" # for testing in the lab using the main hard disk
    raw_audio_folder = "D:/Sound-of-Bees_BACKUP_pt1" # for testing on my laptop using the backup hard disk


    root_candidates_dir = "sounds/whoop_candidates_splitted_significative"
    
    # Variabile per tracciare l'ultimo candidato processato
    last_processed_folder = None
    
    print("="*80)
    print("CONSUMER SCRIPT - Elaborazione candidati in loop continuo")
    print("="*80)
    print(f"Tempo di attesa tra i cicli: {wait_time_minutes} minuti")
    print(f"Root directory: {root_candidates_dir}")
    print("="*80 + "\n")
    
    cycle_count = 0
    
    # LOOP INFINITO
    while True:
        cycle_count += 1
        print(f"\n{'='*80}")
        print(f"CICLO #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Leggi tutte le cartelle presenti nella root
        all_candidate_folders = sorted([f for f in os.listdir(root_candidates_dir) 
                                       if os.path.isdir(os.path.join(root_candidates_dir, f))])
        
        print(f"Cartelle totali trovate: {len(all_candidate_folders)}")
        
        # Se c'è un ultimo processato, parti da lì
        if last_processed_folder is not None:
            try:
                last_index = all_candidate_folders.index(last_processed_folder)
                candidate_folders_to_process = all_candidate_folders[last_index + 1:]
                print(f"Ripresa da: {last_processed_folder}")
                print(f"Cartelle da processare in questo ciclo: {len(candidate_folders_to_process)}")
            except ValueError:
                # L'ultimo processato non è più nella lista (cartella eliminata?)
                print(f"⚠ Attenzione: cartella '{last_processed_folder}' non trovata. Ripartenza dall'inizio.")
                candidate_folders_to_process = all_candidate_folders
        else:
            # Prima esecuzione, processa tutto
            candidate_folders_to_process = all_candidate_folders
            print(f"Prima esecuzione - Processing di tutte le {len(candidate_folders_to_process)} cartelle")
        
        # Se non ci sono cartelle da processare
        if not candidate_folders_to_process:
            print(f"\n⏳ Nessuna nuova cartella da processare.")
            print(f"   Attesa di {wait_time_minutes} minuti prima del prossimo check...")
            print(f"   (Prossimo check: {datetime.now().strftime('%H:%M:%S')} + {wait_time_minutes} min)")
            time.sleep(wait_time_minutes * 60)
            continue
        
        print(f"\n{'─'*80}")
        print(f"Inizio elaborazione di {len(candidate_folders_to_process)} cartelle")
        print(f"{'─'*80}\n")
        
        # Processa ogni cartella
        for folder_idx, candidate_folder in enumerate(candidate_folders_to_process, 1):
            candidate_folder_complete_path = os.path.join(root_candidates_dir, candidate_folder)
            
            print(f"\n[{folder_idx}/{len(candidate_folders_to_process)}] Processing candidate folder: {candidate_folder}")
            print(f"Path: {candidate_folder_complete_path}\n")

            try:
                # Inizializza il grouper
                grouper = WAVPipelineGrouper(candidate_folder_complete_path, peaktime_tolerance=peaktime_tolerance)
                
                # Carica i file e estrai metadati
                if grouper.load_files():
                    # Raggruppa per peaktime simile
                    grouper.group_by_peaktime()
                    
                    # Stampa sommario
                    best_files = grouper.print_summary()
                    
                    # Ottieni la lista di file da processare
                    candidates_to_be_processed_file_paths, metadata = grouper.get_processing_list()
                    
                    print(f"📁 File da processare: {len(candidates_to_be_processed_file_paths)}")
                    
                    for path, meta, idx in zip(candidates_to_be_processed_file_paths, metadata, 
                                               range(len(candidates_to_be_processed_file_paths))):
                        print(f"\n  ├─ [{idx+1}/{len(candidates_to_be_processed_file_paths)}] {os.path.basename(path)}")
                        print(f"  │  → Ch {meta['channel']}, Peaktime {meta['peaktime']:.3f}s, HNR {meta['hnrvalue']:.2f}")

                        # Inizializza la pipeline
                        pipeline = PipelineDraft(path, candidate_folder, raw_audio_folder, 
                                               mics_coords=mics_coords, 
                                               frame_boundaries=frame_boundaries, 
                                               channel_broken=channel_broken)
                        
                        # Esegui la pipeline
                        feats = pipeline.run_full_pipeline(plot_core=True, plot_verbose=False, 
                                                          verbose=True, listening_test=False, 
                                                          save_to_database=False, **figures_characteristics)
                        
                        print(f"  │  ✓ Salvato nel database")
                        # print(feats)  # Decommenta se vuoi vedere i dettagli
                    
                    print(f"  └─ ✓ Cartella '{candidate_folder}' completata\n")
                
                else:
                    print(f"  ⚠ Nessun file caricato per '{candidate_folder}'\n")
                
                # Aggiorna l'ultimo processato
                last_processed_folder = candidate_folder
            
            except Exception as e:
                print(f"  ✗ ERRORE durante il processing di '{candidate_folder}': {e}\n")
                # Continua con la prossima cartella senza aggiornare last_processed_folder
                continue
        
        print(f"\n{'='*80}")
        print(f"CICLO #{cycle_count} COMPLETATO")
        print(f"Ultimo candidato processato: {last_processed_folder}")
        print(f"{'='*80}\n")
        
        # Attendi prima del prossimo ciclo
        print(f"⏳ Attesa di {wait_time_minutes} minuti prima del prossimo ciclo...")
        print(f"   (Prossimo ciclo: ~{datetime.now().strftime('%H:%M:%S')} + {wait_time_minutes} min)")
        print(f"   Press Ctrl+C per terminare\n")
        
        time.sleep(wait_time_minutes * 60)

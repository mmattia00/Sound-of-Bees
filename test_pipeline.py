from classes.pipeline_draft import PipelineDraft
from classes.pipeline_wav_grouper import WAVPipelineGrouper
from classes.utils import Utils, DEFAULT_FIGURE_CHARACTERISTICS, BROKEN_CHANNELS_ZERO_BASED
import os
import numpy as np
import re
import time
from datetime import datetime



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
    mics_coords, frame_boundaries, mics_labels = Utils.load_coordinates_with_labels(mics_coords_namefile)


    # Cartella contenente i file audio raw
    # raw_audio_folder = "E:/soundofbees"
    raw_audio_folder = "sounds/whoop_examples/testing_pipeline" # for testing on example files
    # raw_audio_folder = "/media/uni-konstanz/My Passport/soundofbees" # for testing in the lab using the main hard disk
    # raw_audio_folder = "D:/Sound-of-Bees_BACKUP_pt1" # for testing on my laptop using the backup hard disk


    # root_candidates_dir = "sounds/whoop_candidates_splitted_significative"
    root_candidates_dir = "sounds/whoop_examples/whoop_candidates_splitted" # for testing on example files
    
    # Variabile per tracciare l'ultimo candidato processato
    last_processed_folder = "audio_recording_2025-09-15T00_00_43.625108Z"
    # last_processed_folder = None  # Inizialmente nessun candidato processato
    
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
                                               channel_broken=BROKEN_CHANNELS_ZERO_BASED)
                        
                        # Esegui la pipeline
                        feats = pipeline.run_full_pipeline(plot_core=True, plot_verbose=False, 
                                                          verbose=True, listening_test=True, 
                                                          save_to_database=False, **DEFAULT_FIGURE_CHARACTERISTICS)
                        
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

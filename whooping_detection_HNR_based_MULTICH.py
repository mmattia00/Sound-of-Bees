import soundfile as sf
import sounddevice as sd
from classes.whoop_detector import WhoopDetector
import os



if __name__ == "__main__":
    # Cartella contenente i file audio
    raw_audio_folder = "E:\soundofbees"
    candidates_folder = "sounds\whoop_candidates_splitted" # nuova struttura di cartelle dove ogni .wav è più breve e centralizzato attorno al picco e hopefully contiene esattamente solo un whoop
    # starting_audiofile_name = "audio_recording_2025-09-15T00_00_43.625108Z.wav" # il primo file di quelli presenti nel hard disk in locale
    starting_audiofile_name = "audio_recording_2025-09-16T01_35_48.799810Z.wav" # ultimo file analizzato prima di interrompere la precedente esecuzione
    # starting_audiofile_name = "audio_recording_2025-09-15T07_09_45.544480Z.wav" # file significativo per test

    
    # Canali rotti
    channel_broken = [2, 3, 8, 13, 21, 25, 27, 28, 31]
    channel_broken = [x - 1 for x in channel_broken]  # zero-based index
    
    print("\n" + "="*80)
    print("WHOOPING SIGNALS - BATCH ANALYSIS HNR BASED")
    print("="*80)
    
    # Trova tutti i file .wav nella cartella
    audio_files = sorted([f for f in os.listdir(raw_audio_folder) if f.endswith('.wav')])
    
    # Trova l’indice dell audio file di partenza
    try:
        start_index = audio_files.index(starting_audiofile_name)
    except ValueError:
        start_index = -1

    # Prendi tutti i file successivi
    if start_index != -1:
        audio_files = audio_files[start_index:]
    


    if not audio_files:
        print("Nessun file .wav trovato in " + raw_audio_folder)
        exit(1)
    
    print("Trovati " + str(len(audio_files)) + " file audio")
    print("="*80 + "\n")
    
    # Processa ogni file
    for file_idx, audio_filename in enumerate(audio_files, 1):
        audio_file = os.path.join(raw_audio_folder, audio_filename)
        
        print("\n[" + str(file_idx) + "/" + str(len(audio_files)) + "] Processing: " + audio_filename)
        print("-" * 80)
        
        try:
            # Carica il file audio
            signal_multichannel, sr = sf.read(audio_file)
            
            print("Caricato: " + str(sr) + " Hz, " + str(len(signal_multichannel)/sr) + "s")
            
        except Exception as e:
            print("Errore nel caricamento di " + audio_filename + ": " + str(e))
            continue
        
        # Processa ogni canale
        for j in range(signal_multichannel.shape[1]):
            
            if j in channel_broken:
                continue
            
            print("  CHANNEL " + str(j+1).zfill(2), end=" ")
            
            try:
                signal = signal_multichannel[:, j]

                # init whoop detector
                detector = WhoopDetector(
                    signal=signal,
                    sr=sr,
                    window_length_ms=50,
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
                    percentile=95,
                    offset=4,
                    window_sec=0.5, # lunghezza finestra di analisi intorno al picco
                    merge_overlaps=True
                )

                # Accedi ai risultati
                print(f"Picchi rilevati: {len(detector.peak_times_)}")
                print(f"Threshold HNR: {detector.threshold_:.2f} dB")

                # Ottieni informazioni dettagliate sui picchi
                peak_info = detector.get_peak_info()
                for info in peak_info:
                    print(f"Picco {info['index']}: t={info['peak_time']:.3f}s, "
                        f"finestra=[{info['window_start']:.3f}, {info['window_end']:.3f}]s, "
                        f"HNR={info['peak_hnr_value']:.2f} dB")
                    

                # Accesso diretto agli array
                # detector.peaks_         -> indici dei picchi nell'array HNR
                # detector.peak_times_    -> tempi centrali dei picchi (secondi)
                # detector.peak_windows_  -> lista di tuple (start_time, end_time)
                
                # # optional playback
                # # Estrai i segmenti audio
                # segments = detector.extract_segments()
                # for segment in segments:
                #     print(f"   - Riproduzione segmento di {len(segment)/sr:.3f} secondi")
                #     sd.play(segment, sr)
                #     sd.wait()
                #     # aspetta mezzo secondo prima di continuare
                #     sd.sleep(500)
                
                # Salva i segmenti rilevati
                detector.save_segments(audio_file, j+1, output_dir=candidates_folder)
                
                # # Optional: Visualizza i risultati
                # detector.plot_analysis(ch_num=j+1)
                
            except Exception as e:
                print("Errore: " + str(e))
                continue
        
        print("\n" + "="*80)
    
    print("\nBATCH ANALYSIS COMPLETATA!")
    print("Risultati salvati in: whooping_candidates/")

    

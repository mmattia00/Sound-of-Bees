from whoop_detector import WhoopDetector
from harmonics_analyzer import HarmonicsAnalyzer
from pitch_detector import PitchDetector
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime


def plot_final_visualization(f0_frequencies_final, duration_ms_final, save=False, timestamp=""):
    if f0_frequencies_final and duration_ms_final:
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # --- GRAFICO 1: Istogramma Frequenze Fondamentali ---
        
        # Calcola statistiche
        mean_f0 = np.mean(f0_frequencies_final)
        std_f0 = np.std(f0_frequencies_final)
        
        # Crea istogramma con bins automatici (o specificali manualmente)
        n_bins_f0 = min(20, len(f0_frequencies_final) // 2) if len(f0_frequencies_final) > 10 else len(set(f0_frequencies_final))
        counts_f0, bins_f0, patches_f0 = ax1.hist(f0_frequencies_final, bins=n_bins_f0, 
                                                    color='steelblue', alpha=0.7, 
                                                    edgecolor='black', linewidth=1.2)
        
        # Linea media
        ax1.axvline(x=mean_f0, color='red', linestyle='--', linewidth=2, 
                    label=f'Media: {mean_f0:.2f} Hz')
        
        # Linee per ±1 SD
        ax1.axvline(x=mean_f0 - std_f0, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f'±1 SD: {std_f0:.2f} Hz')
        ax1.axvline(x=mean_f0 + std_f0, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax1.set_xlabel('Frequenza Fondamentale (Hz)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Numero di Occorrenze', fontsize=12, fontweight='bold')
        ax1.set_title(f'Distribuzione Frequenze Fondamentali (n={len(f0_frequencies_final)} whoops)', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(loc='upper right', fontsize=10)
        
        # Aggiungi conteggi sopra le barre
        for count, patch in zip(counts_f0, patches_f0):
            if count > 0:  # Solo se ci sono occorrenze
                height = patch.get_height()
                ax1.text(patch.get_x() + patch.get_width()/2., height,
                        f'{int(count)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # --- GRAFICO 2: Istogramma Durate ---
        
        # Calcola statistiche
        mean_dur = np.mean(duration_ms_final)
        std_dur = np.std(duration_ms_final)
        
        # Crea istogramma
        n_bins_dur = min(20, len(duration_ms_final) // 2) if len(duration_ms_final) > 10 else len(set(duration_ms_final))
        counts_dur, bins_dur, patches_dur = ax2.hist(duration_ms_final, bins=n_bins_dur, 
                                                      color='darkorange', alpha=0.7, 
                                                      edgecolor='black', linewidth=1.2)
        
        # Linea media
        ax2.axvline(x=mean_dur, color='red', linestyle='--', linewidth=2, 
                    label=f'Media: {mean_dur:.2f} ms')
        
        # Linee per ±1 SD
        ax2.axvline(x=mean_dur - std_dur, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f'±1 SD: {std_dur:.2f} ms')
        ax2.axvline(x=mean_dur + std_dur, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Durata (ms)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Numero di Occorrenze', fontsize=12, fontweight='bold')
        ax2.set_title(f'Distribuzione Durate (n={len(duration_ms_final)} whoops)', 
                      fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(loc='upper right', fontsize=10)
        
        # Aggiungi conteggi sopra le barre
        for count, patch in zip(counts_dur, patches_dur):
            if count > 0:
                height = patch.get_height()
                ax2.text(patch.get_x() + patch.get_width()/2., height,
                        f'{int(count)}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plot_filename = f"whoop_analysis_histogram_{timestamp}.png"
            plot_path = os.path.join(main_folder_path, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Istogramma salvato in: {plot_path}")
        
        plt.show()
        
        # Stampa statistiche aggiuntive
        print(f"\n{'='*60}")
        print(f"DISTRIBUZIONE FREQUENZE FONDAMENTALI")
        print(f"{'='*60}")
        print(f"Range: {np.min(f0_frequencies_final):.2f} - {np.max(f0_frequencies_final):.2f} Hz")
        print(f"Mediana: {np.median(f0_frequencies_final):.2f} Hz")
        print(f"Bins utilizzati: {n_bins_f0}")
        
        print(f"\n{'='*60}")
        print(f"DISTRIBUZIONE DURATE")
        print(f"{'='*60}")
        print(f"Range: {np.min(duration_ms_final):.2f} - {np.max(duration_ms_final):.2f} ms")
        print(f"Mediana: {np.median(duration_ms_final):.2f} ms")
        print(f"Bins utilizzati: {n_bins_dur}")
        print(f"{'='*60}\n")
        
    else:
        print("Nessun dato disponibile per la visualizzazione finale.")


def print_statistic_summary(f0_frequencies_final, duration_ms_final, total_whoops):
    if f0_frequencies_final and duration_ms_final:
        print(f"\n{'='*60}")
        print(f"STATISTICHE RIASSUNTIVE")
        print(f"{'='*60}")
        print(f"Numero totale di whoops analizzati: {total_whoops}")
        print(f"di cui considerati validi: {len(f0_frequencies_final)}")
        print(f"\nFrequenza Fondamentale:")
        print(f"  Media:    {np.mean(f0_frequencies_final):.2f} Hz")
        print(f"  Std Dev:  {np.std(f0_frequencies_final):.2f} Hz")
        print(f"  Min:      {np.min(f0_frequencies_final):.2f} Hz")
        print(f"  Max:      {np.max(f0_frequencies_final):.2f} Hz")
        print(f"  Mediana:  {np.median(f0_frequencies_final):.2f} Hz")
        print(f"\nDurata:")
        print(f"  Media:    {np.mean(duration_ms_final):.2f} ms")
        print(f"  Std Dev:  {np.std(duration_ms_final):.2f} ms")
        print(f"  Min:      {np.min(duration_ms_final):.2f} ms")
        print(f"  Max:      {np.max(duration_ms_final):.2f} ms")
        print(f"  Mediana:  {np.median(duration_ms_final):.2f} ms")
        print(f"{'='*60}\n")
    else:
        print("Nessun dato disponibile per le statistiche riassuntive.")

def export_final_results_to_csv(f0_frequencies_final, duration_ms_final, time_stamp):
    if f0_frequencies_final and duration_ms_final:
        
        csv_filename = f"whoop_analysis_results_{timestamp}.csv"
        csv_path = os.path.join(main_folder_path, csv_filename)
        
        # Salva i risultati
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            writer.writerow(['Whoop_Index', 'f0_Hz', 'Duration_ms'])
            
            # Dati
            for idx, (f0, duration) in enumerate(zip(f0_frequencies_final, duration_ms_final), start=1):
                writer.writerow([idx, f"{f0:.2f}", f"{duration:.2f}"])
            
            # Aggiungi statistiche
            writer.writerow([])  # Riga vuota
            writer.writerow(['Statistics', '', ''])
            writer.writerow(['Mean f0', f"{np.mean(f0_frequencies_final):.2f}", 'Hz'])
            writer.writerow(['Std f0', f"{np.std(f0_frequencies_final):.2f}", 'Hz'])
            writer.writerow(['Mean Duration', f"{np.mean(duration_ms_final):.2f}", 'ms'])
            writer.writerow(['Std Duration', f"{np.std(duration_ms_final):.2f}", 'ms'])
            writer.writerow(['Total Whoops', len(f0_frequencies_final), ''])
        
        print(f"\nRisultati salvati in: {csv_path}")
    else:
        print("Nessun risultato da salvare.")



if __name__ == "__main__":

    # main_folder_path = "sounds/fake_whoops_for_testing"
    # main_folder_path = "sounds/whoop"
    main_folder_path = "sounds/best_whoops_not_available"
    # main_folder_path = "D:/whooping_candidates/audio_recording_2025-09-10T08_00_42.172886Z"
    # main_folder_path = "sounds/whoop_examples"


    # Estrai tutti i file audio nella cartella
    audio_files = sorted([f for f in os.listdir(main_folder_path) if f.endswith('.wav')])
    
    if not audio_files:
        print(f"Nessun file .wav trovato in {main_folder_path}")
        exit(1)
    
    print(f"\nTrovati {len(audio_files)} file audio candidati whoop")
    print("="*80)

    f0_frequencies_final = []
    duration_ms_final = []

    for file in audio_files:
        print(f"\n\nAnalisi del file: {file}")
        
        audio_path = os.path.join(main_folder_path, file)
        

        y, sr = librosa.load(audio_path, sr=None, mono=True)    
    
        # Inizializza il detector
        detector = WhoopDetector(
            signal=y,
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
            percentile=80,
            offset=4,
            window_sec=0.7, # lunghezza finestra di analisi intorno al picco
            merge_overlaps=True
        )

        # Accedi ai risultati
        print(f"Picchi rilevati: {len(detector.peak_times_)}")
        print(f"Threshold HNR: {detector.threshold_:.2f} dB")

        # Ottieni informazioni dettagliate sui picchi
        peak_info = detector.get_peak_info()
        for info in peak_info:
            print(f"Picco {info['index']}: t={info['peak_time']:.3f}s, "
                f"finestra=[{info['window_start']:.3f}, {info['window_end']:.3f}]s")

        # Estrai i segmenti audio
        segments = detector.extract_segments()

        for segment_idx, segment in enumerate(segments):

            # Ottieni l'inizio del segmento nell'audio originale (in campioni)
            segment_start_time = detector.peak_windows_[segment_idx][0]  # in secondi
            segment_start_sample = int(segment_start_time * sr)

            pitch_detector = PitchDetector(audio_segment=segment, sr=sr)

            f0, queue, all_queues, whoop_info = pitch_detector.estimate_f0(
                length_queue=5, 
                hz_threshold=25, 
                threshold_increment=1.3,
                plot=False,
                padding_start_ms=5,    # 5ms prima
                padding_end_ms=25,      # 25ms dopo
                freq_min=200,
                freq_max=600
            )


            # Stampa informazioni sul whoop
            if whoop_info is not None:
                pitch_detector.print_whoop_info()


                # esegui analisi armonica
                # Crea l'analizzatore
                harmonic_analyzer = HarmonicsAnalyzer(sr=sr, nfft=8192)
                
                # Analizza le armoniche
                harmonics_info = harmonic_analyzer.analyze_harmonics(
                    whoop_segment=segment,
                    f0=f0,
                    center_sample=whoop_info['f0_median_sample'],
                    window_duration_ms=10,
                    num_harmonics=10,
                    bandwidth_hz=50
                )
                
                if harmonics_info is not None:
                    # Stampa il riassunto
                    harmonic_analyzer.print_harmonics_summary(harmonics_info, top_n=5)
                    
                    # Visualizza i risultati
                    harmonic_analyzer.plot_harmonics_analysis(harmonics_info, freq_max=5000, highlight_first_n=3)

                    f0_frequencies_final.append(f0)
                    duration_ms_final.append(whoop_info['duration_padded_ms'])
        else:
            print("Nessun whoop rilevato in questo file.")

    print("\n\nAnalisi completata per tutti i file.")
    print("="*80)
    
    # save results to csv...
    # Crea nome file con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # export_final_results_to_csv(f0_frequencies_final, duration_ms_final, timestamp)
    # plot final visualization
    plot_final_visualization(f0_frequencies_final, duration_ms_final, save=False, timestamp=timestamp)
    # print statistic summary
    print_statistic_summary(f0_frequencies_final, duration_ms_final, len(audio_files))
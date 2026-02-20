import pandas as pd
from classes.database_analyzer import DatabaseAnalyzer
from classes.utils import Utils
import platform
import socket


def get_database_path():
    """Auto-detect path basato su hostname/OS."""
    hostname = socket.gethostname()
    
    # Laptop
    if platform.system() == "Windows":
        return "E:/whoop_database.h5"
    
    # Lab desktop
    elif platform.system() == "Linux":
        return "database_analysis/whoop_database.h5"
    
    raise ValueError("Percorso database non definito per questo sistema: " + hostname)



if __name__ == "__main__":

    filename_whoop_test = 'audio_recording_2025-09-15T06_40_43.498109Z_ch_04_peaktime_5.005_windowstart_4.755_windowend_5.255_hnrvalue_7.53'
    # database_path = 'E:/whoop_database.h5' # for running the on the real database (in the hard drive) from my laptop
    # database_path = "database_analysis/whoop_database_test.h5" # for running tests on my laptop
    # database_path = "/media/uni-konstanz/My Passport/whoop_database_test.h5" # for testing in the lab computer
    # database_path = "/media/uni-konstanz/My Passport/whoop_database.h5" # for running on the real database but in the lab computer

    root_raw_audio_dir = "E:/soundofbees" # for running the on the real database (in the hard drive) from my laptop

    database_path = get_database_path()

    # load mic coordinates with labels
    mics_coords_namefile = "coordinates/mic_positions_32_ch.txt"
    mics_coords, frame_boundaries, mics_labels = Utils.load_coordinates_with_labels(mics_coords_namefile)

    database_analyzer = DatabaseAnalyzer(database_path=database_path, mics_coords=mics_coords, frame_boundaries=frame_boundaries)
    #############################################################
    ########### ESTRAI SUBSET DI WHOOP CON f0 NON NaN ###########
    #############################################################

    # stats = database_analyzer.pretest_f0_filter('database_analysis/f0_not_nan_ids.csv')
    





    ########################################################################################################################################################################
    ########### RICAVA CSV CON id,f0,precise_duration,weighted_shr,max_alignments,hnr,num_channels_with_whoop PER OGNUNO DI QUESTI WHOOP (quelli con f0 non NaN) ###########
    ########################################################################################################################################################################

    # df = database_analyzer.extract_avg_values(ids_csv_path="database_analysis/f0_not_nan_ids.csv", verbose=10)
    # df.to_csv("database_analysis/whoop_raw_stats.csv", index=False)
    # print(f"Salvato CSV con stats di {len(df)} whoop.")

    # means = df[["f0", "precise_duration", "weighted_shr",
    #                 "max_alignments", "hnr", "num_channels_with_whoop"]].mean()
    # print("\nVALORI MEDI:")
    # for k, v in means.items():
    #     print(f"  {k}: {v:.3f}")

    # # Plots distribuzione
    # database_analyzer._plot_histogram_distribution(df, "f0", bins="auto")
    # database_analyzer._plot_histogram_distribution(df, "precise_duration", bins="auto")
    # database_analyzer._plot_histogram_distribution(df, "weighted_shr", bins="auto")
    # database_analyzer._plot_histogram_distribution(df, "max_alignments", bins="auto", is_discrete=True)
    # database_analyzer._plot_histogram_distribution(df, "hnr", bins="auto")
    # database_analyzer._plot_histogram_distribution(df, "num_channels_with_whoop", bins="auto", is_discrete=True)





    
    
    #################################################################
    ########### DA QUESTO CSV DOBBIAMO RICAVARE THRESHOLD ###########
    #################################################################

    # df = pd.read_csv("database_analysis/whoop_raw_stats.csv")

    # # STATISTICHE COMPLETE per ogni metrica
    # for col in ['f0', 'precise_duration', 'weighted_shr', 'hnr', 'max_alignments', 'num_channels_with_whoop']:
    #     print(f"\n{'='*50}")
    #     print(f"{col.upper()}")
    #     print(f"{'='*50}")
        
    #     data = df[col].dropna()
        
    #     # 1. Tendenza centrale
    #     print(f"Mean:    {data.mean():.3f}")
    #     print(f"Median:  {data.median():.3f}")  # ← Più robusto della media!
        
    #     # 2. Spread
    #     print(f"Std:     {data.std():.3f}")
    #     print(f"Min:     {data.min():.3f}")
    #     print(f"Max:     {data.max():.3f}")
        
    #     # 3. PERCENTILI (chiave per threshold!)
    #     print(f"\nPercentili:")
    #     print(f"  5%:    {data.quantile(0.05):.3f}")
    #     print(f" 10%:    {data.quantile(0.10):.3f}")
    #     print(f" 25% (Q1): {data.quantile(0.25):.3f}")
    #     print(f" 50% (Q2): {data.quantile(0.50):.3f}")  # = mediana
    #     print(f" 75% (Q3): {data.quantile(0.75):.3f}")
    #     print(f" 90%:    {data.quantile(0.90):.3f}")
    #     print(f" 95%:    {data.quantile(0.95):.3f}")
        
    #     # 4. IQR (outlier detection)
    #     Q1 = data.quantile(0.25)
    #     Q3 = data.quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_fence = Q1 - 1.5*IQR
    #     upper_fence = Q3 + 1.5*IQR
    #     print(f"\nIQR: {IQR:.3f}")
    #     print(f"Outlier fences: [{lower_fence:.3f}, {upper_fence:.3f}]")
        
    #     # 5. Forma distribuzione
    #     from scipy.stats import skew, kurtosis
    #     print(f"\nSkewness: {skew(data):.3f}")  # >0 = coda destra, <0 = coda sinistra
    #     print(f"Kurtosis: {kurtosis(data):.3f}")  # Quanto è 'peaky'

    # f0_range = [200, 400] # from Nies's paper
    # duration_range = [0.05, 0.2] # from Nies's paper
    # weighted_shr_min = 0.150 # terzo quartile della distribuzione su tutti i nostri dati (75% dei whoop hanno weighted_shr minore o uguale a 0.150)
    # max_alignments_min = 5 # terzo quartile della distribuzione su tutti i nostri dati (75% dei whoop hanno max_alignments minore o uguale a 5)
    # num_channels_with_whoop_min = 6 # terzo quartile della distribuzione su tutti i nostri dati (75% dei whoop hanno num_channels_with_whoop minore o uguale a 6)

    # print(f"\nTOTALE INIZIALE: {len(df):,} whoop\n")

    # # Statistiche iniziali
    # print("DISTRIBUZIONI INIZIALI:")
    # print(f"  F0:              mean={df.f0.mean():.1f}Hz, Q1={df.f0.quantile(0.25):.1f}, Q3={df.f0.quantile(0.75):.1f}")
    # print(f"  Duration:        mean={df.precise_duration.mean():.3f}s, Q1={df.precise_duration.quantile(0.25):.3f}, Q3={df.precise_duration.quantile(0.75):.3f}")
    # print(f"  Weighted SHR:    mean={df.weighted_shr.mean():.3f}, Q1={df.weighted_shr.quantile(0.25):.3f}, Q3={df.weighted_shr.quantile(0.75):.3f}")
    # print(f"  Max alignments:  mean={df.max_alignments.mean():.1f}, Q1={df.max_alignments.quantile(0.25):.0f}, Q3={df.max_alignments.quantile(0.75):.0f}")

    # print("\n" + "="*70)
    # print("APPLICAZIONE FILTRI (PROGRESSIVA)")
    # print("="*70)

    # # Copia per filtering progressivo
    # df_filtered = df.copy()

    # # Filtro 1: F0
    # mask_f0 = (df_filtered.f0 >= f0_range[0]) & (df_filtered.f0 <= f0_range[1])
    # df_filtered = df_filtered[mask_f0]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n1️⃣  F0 in [{f0_range[0]}, {f0_range[1]}] Hz")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati:  {len(df) - len(df_filtered):,} ({100-pct:.1f}%)")

    # # Filtro 2: Duration
    # mask_dur = (df_filtered.precise_duration >= duration_range[0]) & (df_filtered.precise_duration <= duration_range[1])
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_dur]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n2️⃣  Duration in [{duration_range[0]}, {duration_range[1]}] s")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # # Filtro 3: Weighted SHR
    # mask_shr = df_filtered.weighted_shr >= weighted_shr_min
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_shr]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n3️⃣  Weighted SHR >= {weighted_shr_min:.3f} (Q3)")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # # Filtro 4: Max alignments
    # mask_align = df_filtered.max_alignments >= max_alignments_min
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_align]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n4️⃣  Max alignments >= {max_alignments_min} (Q3)")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # # Filtro 5: Num channels with whoop
    # mask_channels = df_filtered.num_channels_with_whoop >= num_channels_with_whoop_min
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_channels]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n5️⃣  Num channels with whoop >= {num_channels_with_whoop_min} (Q3)")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # print("\n" + "="*70)
    # print("RISULTATO FINALE")
    # print("="*70)
    # print(f"\n✅ WHOOP GOOD: {len(df_filtered):,}/{len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")
    # print(f"❌ SCARTATI:   {len(df) - len(df_filtered):,}/{len(df):,} ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")

    # print("\n" + "="*70)
    # print("STATISTICHE WHOOP GOOD (dopo filtering)")
    # print("="*70)
    # print(f"  F0:              mean={df_filtered.f0.mean():.1f}Hz (range: {df_filtered.f0.min():.0f}-{df_filtered.f0.max():.0f})")
    # print(f"  Duration:        mean={df_filtered.precise_duration.mean():.3f}s (range: {df_filtered.precise_duration.min():.3f}-{df_filtered.precise_duration.max():.3f})")
    # print(f"  Weighted SHR:    mean={df_filtered.weighted_shr.mean():.3f} (range: {df_filtered.weighted_shr.min():.3f}-{df_filtered.weighted_shr.max():.3f})")
    # print(f"  Max alignments:  mean={df_filtered.max_alignments.mean():.1f} (range: {df_filtered.max_alignments.min():.0f}-{df_filtered.max_alignments.max():.0f})")
    # print(f"  HNR:             mean={df_filtered.hnr.mean():.3f} (range: {df_filtered.hnr.min():.3f}-{df_filtered.hnr.max():.3f})")
    # print(f"  Num channels:    mean={df_filtered.num_channels_with_whoop.mean():.1f} (range: {df_filtered.num_channels_with_whoop.min():.0f}-{df_filtered.num_channels_with_whoop.max():.0f})")

    # # save ids of good whoop in a new CSV
    # df_filtered[["id"]].to_csv("database_analysis/good_whoop_ids.csv", index=False)








    # ##########################################################
    # ########### ANALIZZA I MIGLIORI WHOOP ESTRATTI ###########
    # ##########################################################

    # df = pd.read_csv("database_analysis/good_whoop_ids.csv")
    # print(f"Analizzando i {len(df)} whoop GOOD estratti...")
    # list_of_ids = df.id.tolist()

    # for whoop_id in list_of_ids: 
    #     database_analyzer.complete_whoop_analysis_by_id(whoop_id, root_raw_audio_dir=root_raw_audio_dir)



    # ######################################################################################
    # ########### DAI WHOOP BUONI ESTRAI QUELLI CHE SI RIPETONO VICINI NEL TEMPO ###########
    # ######################################################################################

    # # === CREA IL MAPPING (una volta sola) ===
    # queue_mapping = database_analyzer.create_queue_mapping_file(
    #     input_file='database_analysis/good_whoop_ids.csv',
    #     output_json='database_analysis/queue_mapping_good_whoops.json',
    #     time_threshold=4.0
    # )

    # # === OTTIENI LISTA DI TUTTI I FIRST_ID ===
    # all_first_ids = database_analyzer.get_all_queue_first_ids(queue_mapping_file='database_analysis/queue_mapping_good_whoops.json')
    # print(f"Found {len(all_first_ids)} queues")

    # # === ANALIZZA UNA SPECIFICA QUEUE ===
    # first_id = all_first_ids[0]  # Prendi la prima queue
    # queue_data = database_analyzer.load_queue_by_first_id(first_id, queue_mapping_file='database_analysis/queue_mapping_good_whoops.json')
    # stats = database_analyzer.analyze_queue_statistics(queue_data)

    # print(f"\nQueue Statistics:")
    # print(f"  Size: {stats['queue_size']} whoops")
    # print(f"  F0: {stats['f0_mean']:.1f} ± {stats['f0_std']:.1f} Hz (range: {stats['f0_min']:.1f}-{stats['f0_max']:.1f})")
    # print(f"  Total duration: {stats['duration_total']:.2f} s")
    # print(f"  Time intervals: {stats['time_intervals']}")

    # # === PLAY ALCUNE QUEUES ===
    # database_analyzer.play_whoops_close_in_time(
    #     first_ids=all_first_ids[:5],  # Prime 5 queues
    #     root_raw_audio_dir=root_raw_audio_dir,
    #     queue_mapping_file='database_analysis/queue_mapping_good_whoops.json'
    # )
    
    


    # #################################################################################################
    # ########### DA TUTTI I WHOOP CANDIDATI ESTRAI QUELLI CHE SI RIPETONO VICINI NEL TEMPO ###########
    # #################################################################################################

    # # === CREA IL MAPPING (una volta sola) ===
    # queue_mapping = database_analyzer.create_queue_mapping_file(
    #     input_file='database_analysis/f0_not_nan_ids.csv',
    #     output_json='database_analysis/queue_mapping_all_whoops.json',
    #     time_threshold=4.0
    # )

    # # === OTTIENI LISTA DI TUTTI I FIRST_ID ===
    # all_first_ids = database_analyzer.get_all_queue_first_ids(queue_mapping_file='database_analysis/queue_mapping_all_whoops.json')
    # print(f"Found {len(all_first_ids)} queues")

    # # === ANALIZZA UNA SPECIFICA QUEUE ===
    # first_id = all_first_ids[0]  # Prendi la prima queue
    # queue_data = database_analyzer.load_queue_by_first_id(first_id, queue_mapping_file='database_analysis/queue_mapping_all_whoops.json')
    # stats = database_analyzer.analyze_queue_statistics(queue_data)

    # print(f"\nQueue Statistics:")
    # print(f"  Size: {stats['queue_size']} whoops")
    # print(f"  F0: {stats['f0_mean']:.1f} ± {stats['f0_std']:.1f} Hz (range: {stats['f0_min']:.1f}-{stats['f0_max']:.1f})")
    # print(f"  Total duration: {stats['duration_total']:.2f} s")
    # print(f"  Time intervals: {stats['time_intervals']}")

    # # === PLAY ALCUNE QUEUES ===
    # database_analyzer.play_whoops_close_in_time(
    #     first_ids=all_first_ids[:5],  # Prime 5 queues
    #     root_raw_audio_dir=root_raw_audio_dir,
    #     queue_mapping_file='database_analysis/queue_mapping_all_whoops.json'
    # )


    #################################################################################################
    ###################################### CLUSTER 1 ################################################
    #################################################################################################
    #################################################################################################
    ########### CERCA IN TUTTI I CANDIDATI WHOOP NON NULLI I PIU SIMILI A example_1.wav   ###########
    #################################################################################################
    # df = pd.read_csv("database_analysis/whoop_raw_stats.csv")
    
    # f0_range = [500, 600] 
    # duration_range = [0.100, 0.220]
    # weighted_shr_min = 0.05 
    # max_alignments_min = 5 

    # print(f"\nTOTALE INIZIALE: {len(df):,} whoop\n")

    # # Statistiche iniziali
    # print("DISTRIBUZIONI INIZIALI:")
    # print(f"  F0:              mean={df.f0.mean():.1f}Hz, Q1={df.f0.quantile(0.25):.1f}, Q3={df.f0.quantile(0.75):.1f}")
    # print(f"  Duration:        mean={df.precise_duration.mean():.3f}s, Q1={df.precise_duration.quantile(0.25):.3f}, Q3={df.precise_duration.quantile(0.75):.3f}")
    # print(f"  Weighted SHR:    mean={df.weighted_shr.mean():.3f}, Q1={df.weighted_shr.quantile(0.25):.3f}, Q3={df.weighted_shr.quantile(0.75):.3f}")
    # print(f"  Max alignments:  mean={df.max_alignments.mean():.1f}, Q1={df.max_alignments.quantile(0.25):.0f}, Q3={df.max_alignments.quantile(0.75):.0f}")
    # print(f"  HNR:             mean={df.hnr.mean():.1f}, Q1={df.hnr.quantile(0.25):.1f}, Q3={df.hnr.quantile(0.75):.1f}")
    # print("\n" + "="*70)
    # print("APPLICAZIONE FILTRI (PROGRESSIVA)")
    # print("="*70)

    # # Copia per filtering progressivo
    # df_filtered = df.copy()

    # # Filtro 1: F0
    # mask_f0 = (df_filtered.f0 >= f0_range[0]) & (df_filtered.f0 <= f0_range[1])
    # df_filtered = df_filtered[mask_f0]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n1️⃣  F0 in [{f0_range[0]}, {f0_range[1]}] Hz")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati:  {len(df) - len(df_filtered):,} ({100-pct:.1f}%)")

    # # Filtro 2: Duration
    # mask_dur = (df_filtered.precise_duration >= duration_range[0]) & (df_filtered.precise_duration <= duration_range[1])
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_dur]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n2️⃣  Duration in [{duration_range[0]}, {duration_range[1]}] s")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # # # Filtro 3 (raffinato): qualità = SHR alto OR max_alignments alto
    # # mask_quality = (
    # #     (df_filtered.weighted_shr >= weighted_shr_min) |
    # #     (df_filtered.max_alignments >= max_alignments_min)
    # # )
    # # prev_count = len(df_filtered)
    # # df_filtered = df_filtered[mask_quality]

    # # pct = len(df_filtered)/len(df)*100
    # # print(f"\n3️⃣  Quality gate: weighted_shr >= {weighted_shr_min:.3f} OR max_alignments >= {max_alignments_min}")
    # # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")




    # print("\n" + "="*70)
    # print("RISULTATO FINALE")
    # print("="*70)
    # print(f"\n✅ WHOOP GOOD: {len(df_filtered):,}/{len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")
    # print(f"❌ SCARTATI:   {len(df) - len(df_filtered):,}/{len(df):,} ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")

    # print("\n" + "="*70)
    # print("STATISTICHE WHOOP GOOD (dopo filtering)")
    # print("="*70)
    # print(f"  F0:              mean={df_filtered.f0.mean():.1f}Hz (range: {df_filtered.f0.min():.0f}-{df_filtered.f0.max():.0f})")
    # print(f"  Duration:        mean={df_filtered.precise_duration.mean():.3f}s (range: {df_filtered.precise_duration.min():.3f}-{df_filtered.precise_duration.max():.3f})")
    # print(f"  Weighted SHR:    mean={df_filtered.weighted_shr.mean():.3f} (range: {df_filtered.weighted_shr.min():.3f}-{df_filtered.weighted_shr.max():.3f})")
    # print(f"  Max alignments:  mean={df_filtered.max_alignments.mean():.1f} (range: {df_filtered.max_alignments.min():.0f}-{df_filtered.max_alignments.max():.0f})")
    # print(f"  HNR:             mean={df_filtered.hnr.mean():.3f} (range: {df_filtered.hnr.min():.3f}-{df_filtered.hnr.max():.3f})")

    # # save ids of good whoop in a new CSV THE NON VALIDATED ONE
    # df_filtered[["id"]].to_csv("database_analysis/CLUSTER_1_ids.csv", index=False)


    # # analizza tutti i whoop del cluster 1 per validarli manualmente e mettere in un nuovo CSV quelli validati come buoni (da ascoltare e verificare uno ad uno) 
    # database_analyzer.validate_cluster(
    #     input_csv_path="database_analysis/CLUSTER_1_ids.csv",
    #     output_csv_path="database_analysis/CLUSTER_1_VALIDATED_ids.csv",
    #     root_raw_audio_dir=root_raw_audio_dir
    # )

    # database_analyzer.extract_statistics_from_cluster(
    #     input_csv_path="database_analysis/CLUSTER_1_VALIDATED_ids.csv")

    # database_analyzer.make_collection_of_sounds_out_of_a_cluster(
    # input_csv_path="database_analysis/CLUSTER_1_VALIDATED_ids.csv",
    # output_dir="sounds/clusters_collections",
    # root_raw_audio_dir=root_raw_audio_dir,
    # extra_padding=0.25)

    
    #################################################################################################
    ###################################### CLUSTER 2 ################################################
    #################################################################################################
    #################################################################################################
    ########### CERCA IN TUTTI I CANDIDATI WHOOP NON NULLI I PIU SIMILI A example_2.wav   ###########
    #################################################################################################
    # df = pd.read_csv("database_analysis/whoop_raw_stats.csv")
    
    # f0_range = [430, 470] 
    # duration_range = [0.110, 0.190] 
    # weighted_shr_min = 0.1 
    # max_alignments_min = 7

    # print(f"\nTOTALE INIZIALE: {len(df):,} whoop\n")

    # # Statistiche iniziali
    # print("DISTRIBUZIONI INIZIALI:")
    # print(f"  F0:              mean={df.f0.mean():.1f}Hz, Q1={df.f0.quantile(0.25):.1f}, Q3={df.f0.quantile(0.75):.1f}")
    # print(f"  Duration:        mean={df.precise_duration.mean():.3f}s, Q1={df.precise_duration.quantile(0.25):.3f}, Q3={df.precise_duration.quantile(0.75):.3f}")
    # print(f"  Weighted SHR:    mean={df.weighted_shr.mean():.3f}, Q1={df.weighted_shr.quantile(0.25):.3f}, Q3={df.weighted_shr.quantile(0.75):.3f}")
    # print(f"  Max alignments:  mean={df.max_alignments.mean():.1f}, Q1={df.max_alignments.quantile(0.25):.0f}, Q3={df.max_alignments.quantile(0.75):.0f}")
    # print(f"  HNR:             mean={df.hnr.mean():.1f}, Q1={df.hnr.quantile(0.25):.1f}, Q3={df.hnr.quantile(0.75):.1f}")
    # print("\n" + "="*70)
    # print("APPLICAZIONE FILTRI (PROGRESSIVA)")
    # print("="*70)

    # # Copia per filtering progressivo
    # df_filtered = df.copy()

    # # Filtro 1: F0
    # mask_f0 = (df_filtered.f0 >= f0_range[0]) & (df_filtered.f0 <= f0_range[1])
    # df_filtered = df_filtered[mask_f0]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n1️⃣  F0 in [{f0_range[0]}, {f0_range[1]}] Hz")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati:  {len(df) - len(df_filtered):,} ({100-pct:.1f}%)")

    # # Filtro 2: Duration
    # mask_dur = (df_filtered.precise_duration >= duration_range[0]) & (df_filtered.precise_duration <= duration_range[1])
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_dur]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n2️⃣  Duration in [{duration_range[0]}, {duration_range[1]}] s")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # # Filtro 3 (raffinato): qualità = SHR alto OR max_alignments alto
    # mask_quality = (
    #     (df_filtered.weighted_shr >= weighted_shr_min) |
    #     (df_filtered.max_alignments >= max_alignments_min)
    # )
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_quality]

    # pct = len(df_filtered)/len(df)*100
    # print(f"\n3️⃣  Quality gate: weighted_shr >= {weighted_shr_min:.3f} OR max_alignments >= {max_alignments_min}")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")




    # print("\n" + "="*70)
    # print("RISULTATO FINALE")
    # print("="*70)
    # print(f"\n✅ WHOOP GOOD: {len(df_filtered):,}/{len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")
    # print(f"❌ SCARTATI:   {len(df) - len(df_filtered):,}/{len(df):,} ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")

    # print("\n" + "="*70)
    # print("STATISTICHE WHOOP GOOD (dopo filtering)")
    # print("="*70)
    # print(f"  F0:              mean={df_filtered.f0.mean():.1f}Hz (range: {df_filtered.f0.min():.0f}-{df_filtered.f0.max():.0f})")
    # print(f"  Duration:        mean={df_filtered.precise_duration.mean():.3f}s (range: {df_filtered.precise_duration.min():.3f}-{df_filtered.precise_duration.max():.3f})")
    # print(f"  Weighted SHR:    mean={df_filtered.weighted_shr.mean():.3f} (range: {df_filtered.weighted_shr.min():.3f}-{df_filtered.weighted_shr.max():.3f})")
    # print(f"  Max alignments:  mean={df_filtered.max_alignments.mean():.1f} (range: {df_filtered.max_alignments.min():.0f}-{df_filtered.max_alignments.max():.0f})")
    # print(f"  HNR:             mean={df_filtered.hnr.mean():.3f} (range: {df_filtered.hnr.min():.3f}-{df_filtered.hnr.max():.3f})")

    # # save ids of good whoop in a new CSV THE NON VALIDATED ONE
    # df_filtered[["id"]].to_csv("database_analysis/CLUSTER_2_ids.csv", index=False)


    # # analizza tutti i whoop del cluster 2 per validarli manualmente e mettere in un nuovo CSV quelli validati come buoni (da ascoltare e verificare uno ad uno) 
    # database_analyzer.validate_cluster(
    #     input_csv_path="database_analysis/CLUSTER_2_ids.csv",
    #     output_csv_path="database_analysis/CLUSTER_2_VALIDATED_ids.csv",
    #     root_raw_audio_dir=root_raw_audio_dir
    # )

    # database_analyzer.extract_statistics_from_cluster(
    #     input_csv_path="database_analysis/CLUSTER_2_VALIDATED_ids.csv")
    
    # database_analyzer.make_collection_of_sounds_out_of_a_cluster(
    #     input_csv_path="database_analysis/CLUSTER_2_VALIDATED_ids.csv",
    #     output_dir="sounds/clusters_collections",
    #     root_raw_audio_dir=root_raw_audio_dir,
    #     extra_padding=0.25)



    #################################################################################################
    ###################################### CLUSTER 3 ################################################
    #################################################################################################
    #################################################################################################
    ########### CERCA IN TUTTI I CANDIDATI WHOOP NON NULLI I PIU SIMILI A whooping_collection.wav   ###########
    #################################################################################################
    # df = pd.read_csv("database_analysis/whoop_raw_stats.csv")
    
    # f0_range = [310, 350] 
    # duration_range = [0.05, 0.100] 
    # weighted_shr_min = 0.2 
    # max_alignments_min = 5

    # print(f"\nTOTALE INIZIALE: {len(df):,} whoop\n")

    # # Statistiche iniziali
    # print("DISTRIBUZIONI INIZIALI:")
    # print(f"  F0:              mean={df.f0.mean():.1f}Hz, Q1={df.f0.quantile(0.25):.1f}, Q3={df.f0.quantile(0.75):.1f}")
    # print(f"  Duration:        mean={df.precise_duration.mean():.3f}s, Q1={df.precise_duration.quantile(0.25):.3f}, Q3={df.precise_duration.quantile(0.75):.3f}")
    # print(f"  Weighted SHR:    mean={df.weighted_shr.mean():.3f}, Q1={df.weighted_shr.quantile(0.25):.3f}, Q3={df.weighted_shr.quantile(0.75):.3f}")
    # print(f"  Max alignments:  mean={df.max_alignments.mean():.1f}, Q1={df.max_alignments.quantile(0.25):.0f}, Q3={df.max_alignments.quantile(0.75):.0f}")
    # print(f"  HNR:             mean={df.hnr.mean():.1f}, Q1={df.hnr.quantile(0.25):.1f}, Q3={df.hnr.quantile(0.75):.1f}")
    # print("\n" + "="*70)
    # print("APPLICAZIONE FILTRI (PROGRESSIVA)")
    # print("="*70)

    # # Copia per filtering progressivo
    # df_filtered = df.copy()

    # # Filtro 1: F0
    # mask_f0 = (df_filtered.f0 >= f0_range[0]) & (df_filtered.f0 <= f0_range[1])
    # df_filtered = df_filtered[mask_f0]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n1️⃣  F0 in [{f0_range[0]}, {f0_range[1]}] Hz")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati:  {len(df) - len(df_filtered):,} ({100-pct:.1f}%)")

    # # Filtro 2: Duration
    # mask_dur = (df_filtered.precise_duration >= duration_range[0]) & (df_filtered.precise_duration <= duration_range[1])
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_dur]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n2️⃣  Duration in [{duration_range[0]}, {duration_range[1]}] s")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # # Filtro 3 (raffinato): qualità = SHR alto  max_alignments alto
    # mask_quality = (
    #     (df_filtered.weighted_shr >= weighted_shr_min) & 
    #     (df_filtered.max_alignments >= max_alignments_min)
    # )
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_quality]

    # pct = len(df_filtered)/len(df)*100
    # print(f"\n3️⃣  Quality gate: weighted_shr >= {weighted_shr_min:.3f} OR max_alignments >= {max_alignments_min}")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")




    # print("\n" + "="*70)
    # print("RISULTATO FINALE")
    # print("="*70)
    # print(f"\n✅ WHOOP GOOD: {len(df_filtered):,}/{len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")
    # print(f"❌ SCARTATI:   {len(df) - len(df_filtered):,}/{len(df):,} ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")

    # print("\n" + "="*70)
    # print("STATISTICHE WHOOP GOOD (dopo filtering)")
    # print("="*70)
    # print(f"  F0:              mean={df_filtered.f0.mean():.1f}Hz (range: {df_filtered.f0.min():.0f}-{df_filtered.f0.max():.0f})")
    # print(f"  Duration:        mean={df_filtered.precise_duration.mean():.3f}s (range: {df_filtered.precise_duration.min():.3f}-{df_filtered.precise_duration.max():.3f})")
    # print(f"  Weighted SHR:    mean={df_filtered.weighted_shr.mean():.3f} (range: {df_filtered.weighted_shr.min():.3f}-{df_filtered.weighted_shr.max():.3f})")
    # print(f"  Max alignments:  mean={df_filtered.max_alignments.mean():.1f} (range: {df_filtered.max_alignments.min():.0f}-{df_filtered.max_alignments.max():.0f})")
    # print(f"  HNR:             mean={df_filtered.hnr.mean():.3f} (range: {df_filtered.hnr.min():.3f}-{df_filtered.hnr.max():.3f})")

    # # save ids of good whoop in a new CSV THE NON VALIDATED ONE
    # df_filtered[["id"]].to_csv("database_analysis/CLUSTER_3_ids.csv", index=False)


    # # analizza tutti i whoop del cluster 2 per validarli manualmente e mettere in un nuovo CSV quelli validati come buoni (da ascoltare e verificare uno ad uno) 
    # database_analyzer.validate_cluster(
    #     input_csv_path="database_analysis/CLUSTER_3_ids.csv",
    #     output_csv_path="database_analysis/CLUSTER_3_VALIDATED_ids.csv",
    #     root_raw_audio_dir=root_raw_audio_dir
    # )

    # database_analyzer.extract_statistics_from_cluster(
    #     input_csv_path="database_analysis/CLUSTER_3_VALIDATED_ids.csv")
    
    # database_analyzer.make_collection_of_sounds_out_of_a_cluster(
    #     input_csv_path="database_analysis/CLUSTER_3_VALIDATED_ids.csv",
    #     output_dir="sounds/clusters_collections",
    #     root_raw_audio_dir=root_raw_audio_dir,
    #     extra_padding=0.5)


    # ###########################################################################################################
    # ########### CERCA IN TUTTI I CANDIDATI WHOOP NON NULLI I PIU SIMILI A whooping_collection.wav   ###########
    # ###########################################################################################################
    # df = pd.read_csv("database_analysis/whoop_raw_stats.csv")
    
    # f0_range = [317, 345] 
    # duration_range = [0.060, 0.130] 
    # weighted_shr_min = 0.3 
    # max_alignments_min = 9

    # print(f"\nTOTALE INIZIALE: {len(df):,} whoop\n")

    # # Statistiche iniziali
    # print("DISTRIBUZIONI INIZIALI:")
    # print(f"  F0:              mean={df.f0.mean():.1f}Hz, Q1={df.f0.quantile(0.25):.1f}, Q3={df.f0.quantile(0.75):.1f}")
    # print(f"  Duration:        mean={df.precise_duration.mean():.3f}s, Q1={df.precise_duration.quantile(0.25):.3f}, Q3={df.precise_duration.quantile(0.75):.3f}")
    # print(f"  Weighted SHR:    mean={df.weighted_shr.mean():.3f}, Q1={df.weighted_shr.quantile(0.25):.3f}, Q3={df.weighted_shr.quantile(0.75):.3f}")
    # print(f"  Max alignments:  mean={df.max_alignments.mean():.1f}, Q1={df.max_alignments.quantile(0.25):.0f}, Q3={df.max_alignments.quantile(0.75):.0f}")

    # print("\n" + "="*70)
    # print("APPLICAZIONE FILTRI (PROGRESSIVA)")
    # print("="*70)

    # # Copia per filtering progressivo
    # df_filtered = df.copy()

    # # Filtro 1: F0
    # mask_f0 = (df_filtered.f0 >= f0_range[0]) & (df_filtered.f0 <= f0_range[1])
    # df_filtered = df_filtered[mask_f0]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n1️⃣  F0 in [{f0_range[0]}, {f0_range[1]}] Hz")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati:  {len(df) - len(df_filtered):,} ({100-pct:.1f}%)")

    # # Filtro 2: Duration
    # mask_dur = (df_filtered.precise_duration >= duration_range[0]) & (df_filtered.precise_duration <= duration_range[1])
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_dur]
    # pct = len(df_filtered)/len(df)*100
    # print(f"\n2️⃣  Duration in [{duration_range[0]}, {duration_range[1]}] s")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # # # Filtro 3: Weighted SHR
    # # mask_shr = df_filtered.weighted_shr >= weighted_shr_min
    # # prev_count = len(df_filtered)
    # # df_filtered = df_filtered[mask_shr]
    # # pct = len(df_filtered)/len(df)*100
    # # print(f"\n3️⃣  Weighted SHR >= {weighted_shr_min:.3f} (Q3)")
    # # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # # # Filtro 4: Max alignments
    # # mask_align = df_filtered.max_alignments >= max_alignments_min
    # # prev_count = len(df_filtered)
    # # df_filtered = df_filtered[mask_align]
    # # pct = len(df_filtered)/len(df)*100
    # # print(f"\n4️⃣  Max alignments >= {max_alignments_min} (Q3)")
    # # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")

    # # Filtro 3 (raffinato): qualità = SHR alto OR max_alignments alto
    # mask_quality = (
    #     (df_filtered.weighted_shr >= weighted_shr_min) |
    #     (df_filtered.max_alignments >= max_alignments_min)
    # )
    # prev_count = len(df_filtered)
    # df_filtered = df_filtered[mask_quality]

    # pct = len(df_filtered)/len(df)*100
    # print(f"\n3️⃣  Quality gate: weighted_shr >= {weighted_shr_min:.3f} OR max_alignments >= {max_alignments_min}")
    # print(f"    Rimangono: {len(df_filtered):,}/{len(df):,} ({pct:.1f}%)")
    # print(f"    Scartati in questo step: {prev_count - len(df_filtered):,}")



    # print("\n" + "="*70)
    # print("RISULTATO FINALE")
    # print("="*70)
    # print(f"\n✅ WHOOP GOOD: {len(df_filtered):,}/{len(df):,} ({len(df_filtered)/len(df)*100:.1f}%)")
    # print(f"❌ SCARTATI:   {len(df) - len(df_filtered):,}/{len(df):,} ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")

    # print("\n" + "="*70)
    # print("STATISTICHE WHOOP GOOD (dopo filtering)")
    # print("="*70)
    # print(f"  F0:              mean={df_filtered.f0.mean():.1f}Hz (range: {df_filtered.f0.min():.0f}-{df_filtered.f0.max():.0f})")
    # print(f"  Duration:        mean={df_filtered.precise_duration.mean():.3f}s (range: {df_filtered.precise_duration.min():.3f}-{df_filtered.precise_duration.max():.3f})")
    # print(f"  Weighted SHR:    mean={df_filtered.weighted_shr.mean():.3f} (range: {df_filtered.weighted_shr.min():.3f}-{df_filtered.weighted_shr.max():.3f})")
    # print(f"  Max alignments:  mean={df_filtered.max_alignments.mean():.1f} (range: {df_filtered.max_alignments.min():.0f}-{df_filtered.max_alignments.max():.0f})")
    # print(f"  HNR:             mean={df_filtered.hnr.mean():.3f} (range: {df_filtered.hnr.min():.3f}-{df_filtered.hnr.max():.3f})")

    # # save ids of good whoop in a new CSV
    # df_filtered[["id"]].to_csv("database_analysis/whoop_similar_to_whooping_collection_ids.csv", index=False)

    # # analizza i whoop simili a example_2.wav
    # ids = pd.read_csv("database_analysis/whoop_similar_to_whooping_collection_ids.csv").id.tolist()
    # for id in ids:
    #     database_analyzer.complete_whoop_analysis_by_id(id, root_raw_audio_dir=root_raw_audio_dir)





    # for whoop_id in list_of_ids: 
    #     database_analyzer.complete_whoop_analysis_by_id(whoop_id, root_raw_audio_dir=root_raw_audio_dir)


    

    # 1. Carica singolo
    # data = database_analyzer.load_whoop_by_id(filename_whoop_test)
    # print(f"Ch {data['ch']}, F0 {data['f0_mean']:.1f}Hz")

    # 2. Plot spectrogram
    # database_analyzer.plot_spectrogram_from_db(filename_whoop_test, sr=data['sr']) 
    

    data = database_analyzer.load_whoop_by_id("audio_recording_2025-09-18T14_17_47.924832Z_ch_16_peaktime_55.875_windowstart_55.625_windowend_56.125_hnrvalue_4.16", verbose=True)



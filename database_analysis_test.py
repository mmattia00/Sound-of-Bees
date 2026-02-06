import pandas as pd
from classes.database_analyzer import DatabaseAnalyzer
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

    database_path = get_database_path()

    database_analyzer = DatabaseAnalyzer(database_path)
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








    ##########################################################
    ########### ANALIZZA I MIGLIORI WHOOP ESTRATTI ###########
    ##########################################################

    df = pd.read_csv("database_analysis/good_whoop_ids.csv")
    print(f"Analizzando i {len(df)} whoop GOOD estratti...")
    list_of_ids = df.id.tolist()

    for whoop_id in list_of_ids[:10]: # analizza i primi 10 per test
        database_analyzer.complete_whoop_analysis_by_id(whoop_id)


    # 1. Carica singolo
    # data = database_analyzer.load_whoop_by_id(filename_whoop_test)
    # print(f"Ch {data['ch']}, F0 {data['f0_mean']:.1f}Hz")

    # 2. Plot spectrogram
    # database_analyzer.plot_spectrogram_from_db(filename_whoop_test, sr=data['sr']) 
    




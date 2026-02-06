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

    # stats = database_analyzer.pretest_f0_filter('database_analysis/f0_not_nan_ids.csv')
    
    df = database_analyzer.extract_avg_values(ids_csv_path="database_analysis/f0_not_nan_ids.csv", verbose=10)
    df.to_csv("database_analysis/whoop_raw_stats.csv", index=False)
    print(f"Salvato CSV con stats di {len(df)} whoop.")

    means = df[["f0", "precise_duration", "weighted_shr",
                    "max_alignments", "hnr", "num_channels_with_whoop"]].mean()
    print("\nVALORI MEDI:")
    for k, v in means.items():
        print(f"  {k}: {v:.3f}")

    # Plots distribuzione
    database_analyzer._plot_histogram_distribution(df, "f0", bins="auto")
    database_analyzer._plot_histogram_distribution(df, "precise_duration", bins="auto")
    database_analyzer._plot_histogram_distribution(df, "weighted_shr", bins="auto")
    database_analyzer._plot_histogram_distribution(df, "max_alignments", bins="auto", is_discrete=True)
    database_analyzer._plot_histogram_distribution(df, "hnr", bins="auto")
    database_analyzer._plot_histogram_distribution(df, "num_channels_with_whoop", bins="auto", is_discrete=True)



    # 1. Carica singolo
    # data = database_analyzer.load_whoop_by_id(filename_whoop_test)
    # print(f"Ch {data['ch']}, F0 {data['f0_mean']:.1f}Hz")

    # 2. Plot spectrogram
    # database_analyzer.plot_spectrogram_from_db(filename_whoop_test, sr=data['sr']) 
    




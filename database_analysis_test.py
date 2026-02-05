from classes.database_analyzer import DatabaseAnalyzer



# def get_all_whoops_df(hdf5_path: str):
#     """Ritorna DataFrame con scalari di TUTTI i whoop."""
#     data_list = []
    
#     with h5py.File(hdf5_path, 'r') as f:
#         for group_name in f.keys():
#             grp = f[group_name]
            
#             data = load_whoop_by_id(hdf5_path, group_name)
#             data['group_name'] = group_name
#             data_list.append(data)
            
#     df = pd.DataFrame(data_list)
#     return df




if __name__ == "__main__":

    filename_whoop_test = 'audio_recording_2025-09-15T06_40_43.498109Z_ch_04_peaktime_5.005_windowstart_4.755_windowend_5.255_hnrvalue_7.53'
    # database_path = 'E:/whoop_database.h5' # the real one on the hard drive
    # database_path = "database_analysis/whoop_database_test.h5" # for running on the real database on my laptop
    # database_path = "/media/uni-konstanz/My Passport/whoop_database_test.h5" # for testing in the lab computer
    database_path = "/media/uni-konstanz/My Passport/whoop_database.h5" # for running on the real database but in the lab computer



    database_analyzer = DatabaseAnalyzer(database_path)

    stats = database_analyzer.pretest_f0_filter('database_analysis/f0_not_nan_ids.csv')
    

    # 1. Carica singolo
    # data = database_analyzer.load_whoop_by_id(filename_whoop_test)
    # print(f"Ch {data['ch']}, F0 {data['f0_mean']:.1f}Hz")

    # 2. Plot spectrogram
    # database_analyzer.plot_spectrogram_from_db(filename_whoop_test, sr=data['sr']) 
    # 3. Analisi tutti
    # df = get_all_whoops_df(database_path)
    # check how many rows in the dataframe
    # print(f"Totale whoop nel DB: {len(df)}")
    # printa le prime 10 righe
    # high_f0 = df[df.f0_mean > 400]
    # print(f"Whoop forti: {len(high_f0)}")




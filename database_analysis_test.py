import pandas as pd


import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa

def safe_get_scalar_integer(grp, key, dtype=int):  # o float
    val = grp[key][()]
    if isinstance(val, float) and np.isnan(val):
        return np.nan  # → pandas NaN
    if key not in grp:
        return np.nan
    return dtype(val)

def load_whoop_by_id(hdf5_path: str, group_name: str):
    """
    Carica TUTTI i dati di un whoop specifico (18+ features).
    """
    data = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        if group_name not in f:
            raise ValueError(f"Whoop '{group_name}' non trovato!")
        
        grp = f[group_name]
        
        # ===== ATTRS (stringhe) =====
        data['date'] = grp.attrs.get('date', '')
        data['time'] = grp.attrs.get('time', '')
        
        # ===== SCALARI ===== (tutti i tuoi!) sempre presenti
        data['ch'] = int(grp['ch'][()])
        data['peak_time'] = float(grp['peak_time'][()])
        data['start_peak'] = float(grp['start_peak'][()])
        data['end_peak'] = float(grp['end_peak'][()])
        data['rough_duration'] = float(grp['rough_duration'][()])
        data['sr'] = int(grp['sr'][()])
        
        # SCALARI NON SEMPRE PRESENTI (POTREBBERO ESSERE Nan e creare problemi)
        # con float non ci sono problemi a leggere NaN
        data['f0_mean'] = float(grp['f0_mean'][()])
        data['precise_start_peak'] = float(grp['precise_start_peak'][()])
        data['precise_end_peak'] = float(grp['precise_end_peak'][()])
        data['precise_duration'] = float(grp['precise_duration'][()])
        data['weighted_shr'] = float(grp['weighted_shr'][()])

        # con int usiamo funzione apposta
        data['max_aligned_peaks'] = safe_get_scalar_integer(grp, 'max_aligned_peaks', int)
        data['strongest_channel'] = safe_get_scalar_integer(grp, 'strongest_channel', int)
        data['num_channels_with_whoop'] = safe_get_scalar_integer(grp, 'num_channels_with_whoop', int)
        
        # ===== ARRAY ===== (tutti i tuoi!) alcuni potrebbero essere vuoti
            
        data['hnr_levels'] = np.array(grp['hnr_levels'])
        data['precise_localization'] = np.array(grp['precise_localization'])
        data['spectrogram_dB'] = np.array(grp['spectrogram_dB'])
        data['spec_frequencies'] = np.array(grp['spec_frequencies'])
        data['spec_times'] = np.array(grp['spec_times'])
    
    print(f"✓ Caricato {group_name}: "
          f"F0={data['f0_mean']:.1f}Hz, Ch{data['ch']}→{data['strongest_channel']}, "
          f"Loc={data['precise_localization']}")
    return data



def plot_spectrogram_from_db(hdf5_path: str, group_name: str, sr: int = 48000):
    """Plotta spectrogram salvato."""
    data = load_whoop_by_id(hdf5_path, group_name)
    
    if 'spectrogram_dB' not in data:
        print("❌ Spectrogram non trovato")
        return
    
    plt.figure(figsize=(12, 6))

    spectrogram_db = data['spectrogram_dB']
    frequencies = data['spec_frequencies']
    times = data['spec_times']

    vmin = np.nanmax(spectrogram_db) - 80
    vmax = np.nanmax(spectrogram_db)

    im = plt.pcolormesh(times, frequencies, spectrogram_db, 
                        shading='gouraud', cmap='hot', vmin=vmin, vmax=vmax)
    
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    plt.title("title", fontsize=14)
    cbar = plt.colorbar(im, label='Power (dB)')
    
    plt.ylim([0, 20000])
    plt.xlim([times[0], times[-1]])
    plt.show()


def get_all_whoops_df(hdf5_path: str):
    """Ritorna DataFrame con scalari di TUTTI i whoop."""
    data_list = []
    
    with h5py.File(hdf5_path, 'r') as f:
        for group_name in f.keys():
            grp = f[group_name]
            
            data = load_whoop_by_id(hdf5_path, group_name)
            data['group_name'] = group_name
            data_list.append(data)
            
    df = pd.DataFrame(data_list)
    return df




if __name__ == "__main__":

    filename_whoop_test = 'audio_recording_2025-09-15T06_40_43.498109Z_ch_04_peaktime_5.005_windowstart_4.755_windowend_5.255_hnrvalue_7.53'
    
    # 1. Carica singolo
    data = load_whoop_by_id('whoop_database.h5', filename_whoop_test)
    print(f"Ch {data['ch']}, F0 {data['f0_mean']:.1f}Hz")

    # 2. Plot spectrogram
    plot_spectrogram_from_db('whoop_database.h5', filename_whoop_test, sr=data['sr'])

    # 3. Analisi tutti
    df = get_all_whoops_df('whoop_database.h5')
    # check how many rows in the dataframe
    print(f"Totale whoop nel DB: {len(df)}")
    # printa le prime 10 righe
    # high_f0 = df[df.f0_mean > 400]
    # print(f"Whoop forti: {len(high_f0)}")

    # # 4. Plotta tutti spectrogram di un canale
    # for name in df[df.ch == 9]['group_name'].head(3):
    #     plot_spectrogram_from_db('whoop_database.h5', name)



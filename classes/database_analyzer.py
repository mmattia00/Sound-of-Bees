import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import time

class DatabaseAnalyzer:
    def __init__(self, database_path):
        self.database_path = database_path
    
    def _safe_get_scalar_integer(self, grp, key, dtype=int):  # o float
        val = grp[key][()]
        if isinstance(val, float) and np.isnan(val):
            return np.nan  # → pandas NaN
        if key not in grp:
            return np.nan
        return dtype(val)

    def load_whoop_by_id(self, group_name: str):
        """
        Carica TUTTI i dati di un whoop specifico (18+ features).
        """
        data = {}
        
        with h5py.File(self.database_path, 'r') as f:
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
            data['max_aligned_peaks'] = self._safe_get_scalar_integer(grp, 'max_aligned_peaks', int)
            data['strongest_channel'] = self._safe_get_scalar_integer(grp, 'strongest_channel', int)
            data['num_channels_with_whoop'] = self._safe_get_scalar_integer(grp, 'num_channels_with_whoop', int)
            
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


    def plot_spectrogram_from_db(self, group_name: str, sr: int = 48000):
        """Plotta spectrogram salvato."""
        data = self.load_whoop_by_id(group_name)
        
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

    def pretest_f0_filter(self, output_csv='f0_not_nan_ids.csv'):
        """
        Scrematura f0 NaN → CSV good_ids per skip futuri.
        UNA VOLTA sola, poi tutto x10 più veloce.
        """
        stats = {'total_groups': 0, 'f0_nan': 0, 'f0_valid': 0}
        good_ids = []
        
        print("🔍 Pretest F0 filter... (1-2h su 130GB HDD)")
        start_time = time.time()
        
        with h5py.File(self.database_path, 'r') as f:
            all_gids = list(f.keys())  # Copia nomi RAM (evita degrado)
            stats['total_groups'] = len(all_gids)
            
            for i, gid in enumerate(all_gids, 1):
                grp = f[gid]
                
                # La tua logica sicura per f0
                f0 = float(grp['f0_mean'][()])
                
                if np.isnan(f0):
                    stats['f0_nan'] += 1
                else:
                    stats['f0_valid'] += 1
                    good_ids.append(gid)
                
                if i % 10000 == 0:
                    elapsed = (time.time() - start_time) / 60
                    print(f"⏳ {i:,}/{len(all_gids):,} ({i/len(all_gids)*100:.1f}%) "
                        f"- {elapsed:.1f}min - Good: {len(good_ids):,}")
        
        # Salva CSV
        pd.Series(good_ids).to_csv(output_csv, index=False, header=False)
        
        elapsed_total = (time.time() - start_time) / 60
        print(f"\n✅ Pretest finito: {elapsed_total:.1f}min")
        print(f"📊 Total: {stats['total_groups']:,} | "
            f"F0 NaN: {stats['f0_nan']:,} ({stats['f0_nan']/stats['total_groups']*100:.1f}%) | "
            f"Good: {stats['f0_valid']:,} → {output_csv}")
        
        return stats
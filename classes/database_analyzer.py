import multiprocessing as mp
from multiprocessing import Pool
from functools import wraps
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import h5py
from joblib import Parallel, delayed
from classes.strong_channel_detector import StrongChannelDetector


def parallelize_joblib(n_jobs=-1, default_verbose=10):
    """
    DECORATOR UNIVERSALE con joblib + progress bar.
    
    Args:
        n_jobs: Numero core (-1 = tutti, int = specifico)
        verbose: Progress bar (0=off, 10=standard, 50=dettagliato)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]  # Analyzer instance
            ids_csv_path = kwargs.pop('ids_csv_path', None)
            verbose = kwargs.pop('verbose', default_verbose)
            
            # Carica IDs
            if ids_csv_path and os.path.exists(ids_csv_path):
                group_ids = pd.read_csv(ids_csv_path, header=None)[0].tolist()
                print(f"📂 {len(group_ids):,} IDs da {ids_csv_path}")
            else:
                import h5py
                with h5py.File(self.database_path, 'r') as f:
                    group_ids = list(f.keys())
                print(f"📂 {len(group_ids):,} IDs totali")
            
            cores = mp.cpu_count() if n_jobs == -1 else n_jobs
            print(f"🚀 Parallel {cores} cores su {len(group_ids):,} IDs")
            
            start = time.time()
            
            # MAGIA JOBLIB con progress bar!
            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(
                delayed(func)(self, gid, *args[1:], **kwargs) for gid in group_ids
            )
            
            elapsed = time.time() - start
            print(f"✅ {elapsed:.0f}s ({len(group_ids)/elapsed:.0f} IDs/s)")
            
            # Aggrega risultati
            results = [r for r in results if r is not None]
            if results and isinstance(results[0], dict):
                return pd.DataFrame(results)
            elif results and isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            return results
        
        return wrapper
    return decorator

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

    @parallelize_joblib(n_jobs=-1,  default_verbose=10)  # ← MAGIA!
    def extract_avg_values(self, gid):  # UN SOLO ID!
        """Processa 1 whoop, restituisce dict."""
        with h5py.File(self.database_path, "r") as f:
            if gid not in f: return None
            grp = f[gid]

            # Leggi scalari (riusa la tua logica)
            f0 = float(grp["f0_mean"][()])
            precise_duration = float(grp["precise_duration"][()])
            weighted_shr = float(grp["weighted_shr"][()])
            max_alignments = self._safe_get_scalar_integer(grp, "max_aligned_peaks", int)
            num_channels = self._safe_get_scalar_integer(grp, "num_channels_with_whoop", int)

            # HNR: ad es. massimo sul canale strong
            hnr_levels = np.array(grp["hnr_levels"])
            hnr = float(np.nanmax(hnr_levels)) if hnr_levels.size > 0 else np.nan
            
            return {
                'id': gid,
                'f0': f0,
                'precise_duration': precise_duration,
                'weighted_shr': weighted_shr,
                'max_alignments': max_alignments,
                'hnr': hnr,
                'num_channels_with_whoop': num_channels
            }

    def _plot_histogram_distribution(self, df, column: str, bins="auto", is_discrete=False):
        """
        Istogramma tipo 'grafico a barre' per la colonna indicata.
        - se is_discrete=True: barre per valori interi (0,1,2,...)
        - altrimenti: bins continui stile istogramma.
        """
        data = df[column].dropna().values

        if is_discrete:
            # Per interi, usa value_counts ordinato → barre pulite
            counts = pd.Series(data).value_counts().sort_index()
            x = counts.index.values
            y = counts.values
            plt.figure(figsize=(10, 5))
            plt.bar(x, y, width=0.8, align="center")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.title(f"Distribuzione {column}")
            plt.grid(axis="y", alpha=0.3)
            plt.show()
        else:
            # Continua: np.histogram → barre
            hist, edges = np.histogram(data, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            width = edges[1] - edges[0]

            plt.figure(figsize=(10, 5))
            plt.bar(centers, hist, width=width, align="center")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.title(f"Distribuzione {column}")
            plt.grid(axis="y", alpha=0.3)
            plt.show()
    
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


    def plot_spectrogram_from_db(self, db_data: dict):
        """Plotta spectrogram salvato."""
        
        if 'spectrogram_dB' not in db_data:
            print("❌ Spectrogram non trovato")
            return
        
        plt.figure(figsize=(12, 6))

        spectrogram_db = db_data['spectrogram_dB']
        frequencies = db_data['spec_frequencies']
        times = db_data['spec_times']

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

    def plot_localization(self, db_data: dict):
        pass  # TODO: plot della localizzazione precisa (es. canali vs tempo)


    def complete_whoop_analysis_by_id(self, group_name: str):
        """Analisi completa di un whoop specifico."""
        data = self.load_whoop_by_id(group_name)
        
        # Qui puoi aggiungere tutte le analisi che vuoi, ad es.:
        print(f"📊 Analisi completa per {group_name}:")
        print(f"F0: {data['f0_mean']:.1f} Hz")
        print(f"Durata precisa: {data['precise_duration']:.2f} s")
        print(f"Weighted SHR: {data['weighted_shr']:.2f}")
        print(f"Max Alignments: {data['max_aligned_peaks']}")
        print(f"HNR max: {np.nanmax(data['hnr_levels']):.2f} dB")
        print(f"Num canali con whoop: {data['num_channels_with_whoop']}")
        
        # Esempio: plot dello spectrogram
        self.plot_spectrogram_from_db(data)

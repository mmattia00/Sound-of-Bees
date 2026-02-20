import json
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
from classes.utils import Utils, DEFAULT_FIGURE_CHARACTERISTICS, BROKEN_CHANNELS_ZERO_BASED
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import numpy as np
from datetime import datetime, timedelta
import pytz
from matplotlib.patches import Patch
import soundfile as sf



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
    def __init__(self, database_path, mics_coords=None, frame_boundaries=None):
        self.database_path = database_path
        self.mics_coords = mics_coords
        self.frame_boundaries = frame_boundaries

    
    
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

    def _plot_histogram_distribution(self, values: list, column: str, bins="auto", is_discrete=False):
        """
        Istogramma tipo 'grafico a barre' per la colonna indicata.
        - se is_discrete=True: barre per valori interi (0,1,2,...)
        - altrimenti: bins continui stile istogramma.
        """
        data = pd.Series(values).dropna().values
        mean_val = np.mean(data)

        fig, ax = plt.subplots(figsize=(10, 5))

        if is_discrete:
            counts = pd.Series(data).value_counts().sort_index()
            x = counts.index.values
            y = counts.values

            ax.bar(x, y, width=0.8, align="center",
                edgecolor="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(x)

        else:
            hist, edges = np.histogram(data, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            width = edges[1] - edges[0]

            ax.bar(centers, hist, width=width, align="center",
                edgecolor="black", linewidth=0.8)
            ax.set_xticks(centers)
            ax.set_xticklabels([f"{c:.2f}" for c in centers], rotation=45, ha="right", fontsize=8)

        # linea tratteggiata sulla media
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_val:.3f}")
        ax.legend()

        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribuzione {column}")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_peak_time_distribution(self, peak_times: list, start_time, end_time, timezone: str = "Europe/Berlin"):
        """
        Visualizza la distribuzione temporale dei whoop peaks su una timeline.
        - peak_times: lista di timestamp datetime (UTC o naive)
        - start_time: datetime di inizio registrazione
        - end_time:   datetime di fine registrazione
        - timezone: timezone per visualizzazione "UTC" o "Europe/Berlin" (con conversione)
        """

        # --- conversione UTC → Europe/Berlin ---
        utc = pytz.utc
        berlin = pytz.timezone(timezone)

        def to_berlin(dt):
            if dt.tzinfo is None:
                dt = utc.localize(dt)
            return dt.astimezone(berlin)

        start_time = to_berlin(start_time)
        end_time   = to_berlin(end_time)
        peak_times = [to_berlin(pt) for pt in peak_times]

        fig, ax = plt.subplots(figsize=(16, 3))

        # --- timeline orizzontale ---
        ax.hlines(0, start_time, end_time, colors="steelblue", linewidth=3, zorder=1)

        # --- tick asse x: ogni 3 ore (con timezone Berlin) ---
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3, tz=berlin))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=berlin))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1, tz=berlin))
        plt.xticks(rotation=45, ha="right", fontsize=8)

        # --- secondo livello: data del giorno a ogni mezzanotte Berlin ---
        current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        if current < start_time:
            current += timedelta(days=1)

        while current <= end_time:
            total_seconds = (end_time - start_time).total_seconds()
            x_pos = (current - start_time).total_seconds() / total_seconds

            ax.annotate(
                current.strftime("%d %b %Y"),
                xy=(x_pos, 0),
                xycoords=("axes fraction", "axes fraction"),
                xytext=(x_pos, -0.28),
                textcoords=("axes fraction", "axes fraction"),
                ha="center", va="top",
                fontsize=8, color="black",
                fontweight="bold",
                annotation_clip=False,
                arrowprops=None
            )
            ax.axvline(current, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            current += timedelta(days=1)

        # --- fasce giorno/notte ---
        day_start_h, day_end_h = 6, 20
        current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        while current < end_time:
            dawn = current.replace(hour=day_start_h)
            dusk = current.replace(hour=day_end_h)

            night_start = max(start_time, current)
            night_end   = min(end_time, dawn)
            if night_start < night_end:
                ax.axvspan(night_start, night_end, ymin=0.0, ymax=1.0,
                        color="navy", alpha=0.12, label="_night")

            day_s = max(start_time, dawn)
            day_e = min(end_time, dusk)
            if day_s < day_e:
                ax.axvspan(day_s, day_e, ymin=0.0, ymax=1.0,
                        color="gold", alpha=0.10, label="_day")

            night2_start = max(start_time, dusk)
            night2_end   = min(end_time, current + timedelta(days=1))
            if night2_start < night2_end:
                ax.axvspan(night2_start, night2_end, ymin=0.0, ymax=1.0,
                        color="navy", alpha=0.12, label="_night2")

            current += timedelta(days=1)

        # --- linea mezzanotte Berlin ---
        midnight = start_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        while midnight < end_time:
            ax.axvline(midnight, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.text(midnight, 0.62, "00:00", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color="gray")
            midnight += timedelta(days=1)

        # --- eventi whoop ---
        for pt in peak_times:
            ax.vlines(pt, -0.4, 0.4, colors="crimson", linewidth=1.0, alpha=0.7, zorder=3)

        # --- legenda ---
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="gold", alpha=0.4, label="Giorno (06:00–20:00)"),
            Patch(facecolor="navy", alpha=0.3, label="Notte"),
            plt.Line2D([0], [0], color="crimson", linewidth=1.5, label=f"Whoop (n={len(peak_times)})"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        ax.set_xlim(start_time, end_time)
        ax.set_ylim(-0.8, 0.8)
        ax.set_yticks([])
        ax.set_title(
            f"Distribuzione temporale whoops  |  "
            f"{start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%Y-%m-%d %H:%M')}  "
            f"[{timezone}, UTC{start_time.strftime('%z')}]"
        )
        ax.set_xlabel("Ora")

        plt.subplots_adjust(bottom=0.25)
        plt.show()


    def _plot_peak_spatial_distribution(self, strong_channels: list, mics_coords: np.ndarray, frame_boundaries: dict):
        """
        Mappa spaziale della distribuzione dei whoops per canale.
        - strong_channels: lista di interi (1-based) del canale più forte per ogni whoop
        - mics_coords:     coordinate microfoni shape [N, 2], indici 0-based
        - frame_boundaries: dict con 'xmin', 'xmax', 'ymin', 'ymax'
        """

        n_mics = mics_coords.shape[0]

        # --- conta occorrenze per canale (1-based → 0-based) ---
        counts = np.zeros(n_mics, dtype=int)
        for ch in strong_channels:
            if ch is not None and not (isinstance(ch, float) and np.isnan(ch)):
                counts[int(ch) - 1] += 1

        # --- normalizzazione per colormap ---
        max_count = counts.max() if counts.max() > 0 else 1
        norm = plt.Normalize(vmin=0, vmax=max_count)
        cmap = plt.cm.YlOrRd  # giallo (0) → arancione → rosso (max)
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        fig, ax = plt.subplots(figsize=(12, 10))

        hexagon_radius = 0.025

        for i, (x, y) in enumerate(mics_coords):
            hexagon = Utils._create_hexagon(x, y, hexagon_radius)
            color = scalar_map.to_rgba(counts[i])[:3]

            is_broken = i in BROKEN_CHANNELS_ZERO_BASED

            if is_broken:
                ax.fill(hexagon[:, 0], hexagon[:, 1],
                        color="darkgray", alpha=0.9,
                        hatch="///", edgecolor="red", linewidth=2.0, zorder=1)
            else:
                ax.fill(hexagon[:, 0], hexagon[:, 1],
                        color=color, alpha=0.85,
                        edgecolor="black", linewidth=1.2, zorder=1)

            # puntino microfono
            dot_color = "red" if is_broken else "black"
            ax.scatter(x, y, c=dot_color, s=30, zorder=10, linewidths=0)

            # numero canale + count (solo se non rotto e count > 0)
            label = f"{i+1}" if is_broken or counts[i] == 0 else f"{i+1}\n({counts[i]})"
            ax.text(x, y + hexagon_radius * 0.35, label,
                    ha="center", va="center", fontsize=6,
                    fontweight="bold", color="black", zorder=11)

        # --- colorbar ---
        cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Numero di whoops", fontsize=12, fontweight="bold", labelpad=12)
        cbar.ax.tick_params(labelsize=10)
        # tick interi sulla colorbar
        tick_vals = np.unique(np.linspace(0, max_count, min(max_count + 1, 8), dtype=int))
        cbar.set_ticks(tick_vals)

        # --- legenda ---
        legend_elements = [
            Patch(facecolor="darkgray", alpha=0.9, hatch="///",
                edgecolor="red", linewidth=2, label="Canale rotto"),
            Patch(facecolor=scalar_map.to_rgba(0)[:3], edgecolor="black",
                linewidth=1.2, label="0 whoops"),
            Patch(facecolor=scalar_map.to_rgba(max_count)[:3], edgecolor="black",
                linewidth=1.2, label=f"Max whoops ({max_count})"),
        ]
        ax.legend(handles=legend_elements, loc="upper left",
                bbox_to_anchor=(1.18, 1.0), fontsize=10,
                frameon=True, fancybox=True, shadow=True)

        # --- limiti e stile ---
        ax.set_xlim(frame_boundaries[0], frame_boundaries[1])
        ax.set_ylim(frame_boundaries[2], frame_boundaries[3])
        ax.set_aspect("equal")
        ax.set_title(f"Distribuzione spaziale whoops  |  n={len([c for c in strong_channels if c is not None and not (isinstance(c, float) and np.isnan(c))])} eventi",
                    fontsize=14, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.show()




    def _fmt_scalar(self, v, unit=""):
        """Formatta scalari e NaN in modo robusto."""
        if v is None:
            return "None"
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "NaN"
            return f"{v:.4f}{unit}"
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}{unit}"
        return str(v)

    def _fmt_array(self, a, name="", max_elems=12, precision=3):
        """Stampa array in modo leggibile: shape, dtype e preview (non tutto)."""
        if a is None:
            return f"{name}: None"
        a = np.asarray(a)
        if a.size == 0:
            return f"{name}: empty, shape={a.shape}, dtype={a.dtype}"
        # Preview: primi/ultimi elementi per non intasare (soprattutto per spectrogram)
        edge = max(1, max_elems // 2)
        with np.printoptions(precision=precision, suppress=True, edgeitems=edge, threshold=max_elems):
            preview = np.array2string(a)
        return f"{name}: shape={a.shape}, dtype={a.dtype}, preview={preview}"
    
    def load_whoop_by_id(self, group_name: str, verbose=False):
        """
        Carica TUTTI i dati di un whoop specifico (18+ features).
        """
        data = {}
        
        with h5py.File(self.database_path, 'r') as f:
            if group_name not in f:
                raise ValueError(f"Whoop '{group_name}' non trovato!")
            
            grp = f[group_name] # eg audio_recording_2025-09-15T01_55_43.359927Z_ch_19_peaktime_10.235_windowstart_9.985_windowend_10.485_hnrvalue_-2.02
            # keep audio_recording_2025-09-15T01_55_43.359927Z
            data['parent_filename'] = group_name.split('_ch_')[0]  # estrae la parte prima di _ch_
            
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
        
        if verbose:
            print(f"✓ Caricato: {group_name}")
            print(f"  parent_filename: {data['parent_filename']}")
            print(f"  date: {data['date']}")
            print(f"  time: {data['time']}")

            # --- SCALARI ---
            scalar_keys = [
                ("ch", ""),
                ("sr", " Hz"),
                ("peak_time", " s"),
                ("start_peak", " s"),
                ("end_peak", " s"),
                ("rough_duration", " s"),
                ("f0_mean", " Hz"),
                ("precise_start_peak", " s"),
                ("precise_end_peak", " s"),
                ("precise_duration", " s"),
                ("weighted_shr", ""),
                ("max_aligned_peaks", ""),
                ("strongest_channel", ""),
                ("num_channels_with_whoop", ""),
            ]
            print("  --- scalari ---")
            for k, unit in scalar_keys:
                print(f"  {k:24s}: {self._fmt_scalar(data.get(k), unit)}")

            # --- ARRAY ---
            print("  --- array ---")
            print(" ", self._fmt_array(data.get("hnr_levels"), "hnr_levels", max_elems=20, precision=3))
            print(" ", self._fmt_array(data.get("precise_localization"), "precise_localization", max_elems=20, precision=3))
            print(" ", self._fmt_array(data.get("spec_frequencies"), "spec_frequencies", max_elems=12, precision=2))
            print(" ", self._fmt_array(data.get("spec_times"), "spec_times", max_elems=12, precision=3))

            # spectrogram: solo shape (o preview piccolissimo)
            spec = data.get("spectrogram_dB")
            if spec is None:
                print("  spectrogram_dB: None")
            else:
                spec = np.asarray(spec)
                print(f"  spectrogram_dB          : shape={spec.shape}, dtype={spec.dtype}, "
                    f"min={np.nanmin(spec):.1f} dB, max={np.nanmax(spec):.1f} dB")

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

        if db_data['ch'] - 1 < 16:
            channels_of_interest = list(range(16))
        else:
            channels_of_interest = list(range(16, 32))

        precise_loc = db_data['precise_localization']

        print(f"Precise localization: {precise_loc}")
        if len(precise_loc) == 0:
            precise_loc = None
        
        Utils.plot_hexagon_hnr_map(
            hnr_levels=db_data['hnr_levels'],
            broken_channels=BROKEN_CHANNELS_ZERO_BASED,
            channels_of_interest=channels_of_interest,
            mic_positions=self.mics_coords,
            hexagon_radius=0.025,
            use_db = False,
            cmap = 'hot',
            boundaries=self.frame_boundaries,
            show_colorbar=True,
            precise_localization=precise_loc,
            **DEFAULT_FIGURE_CHARACTERISTICS
        )

    def complete_whoop_analysis_by_id(self, group_name: str, root_raw_audio_dir: str):
        """Analisi completa di un whoop specifico."""
        data = self.load_whoop_by_id(group_name)
        
        # Qui puoi aggiungere tutte le analisi che vuoi, ad es.:
        print(f"📊 Analisi completa per {group_name}:")
        print(f"Canale originale: {data['ch']} → Canale più forte: {data['strongest_channel']}")
        print(f"F0: {data['f0_mean']:.1f} Hz")
        print(f"Durata precisa: {data['precise_duration']:.2f} s")
        print(f"Weighted SHR: {data['weighted_shr']:.2f}")
        print(f"Max Alignments: {data['max_aligned_peaks']}")
        # print(f"HNR max: {np.nanmax(data['hnr_levels']):.2f} dB")
        print(f"Num canali con whoop: {data['num_channels_with_whoop']}")


        if data['strongest_channel'] is None or np.isnan(data['strongest_channel']):
            print("No strongest channel found in database for this whoop. The candidate is super weak and might be a false positive. Skipping audio segment extraction and spectrogram plotting.")
        else:
            # Esempio: plot dello spectrogram
            print(f"Playing strongest channel audio segment ({data['strongest_channel']})...")
            Utils.extract_and_play_audio_segment_from_raw_multichannel_audio(f"{root_raw_audio_dir}/{data['parent_filename']}.wav",
                                                                    data['start_peak'],
                                                                    data['end_peak'],
                                                                    data['strongest_channel'] - 1,
                                                                    lenght_extension=1.0)
            self.plot_spectrogram_from_db(data)
            self.plot_localization(data)

    def _parse_whoop_id(self, whoop_id):
        """Parse whoop ID to extract all components"""
        pattern = r'(audio_recording_[\d\-TZ_\.]+)_ch_(\d+)_peaktime_([\d\.]+)_windowstart_([\d\.]+)_windowend_([\d\.]+)_hnrvalue_([\-\d\.]+)'
        match = re.match(pattern, whoop_id)

        if not match:
            return None

        return {
            'original_id': whoop_id,
            'timestamp_cluster': match.group(1),
            'ch': int(match.group(2)),
            'peak_time': float(match.group(3)),
            'window_start': float(match.group(4)),
            'window_end': float(match.group(5)),
            'hnr_value': float(match.group(6))
        }

    def _create_time_queues(self, whoops, time_threshold=2.0):
        """
        Create queues of whoops that are within time_threshold seconds of each other.
        Whoops must be sorted by peak_time before calling this function.
        """
        if len(whoops) == 0:
            return []

        queues = []
        current_queue = [whoops[0]]

        for i in range(1, len(whoops)):
            time_diff = whoops[i]['peak_time'] - current_queue[-1]['peak_time']

            if 0.85 <= time_diff <= time_threshold:
                current_queue.append(whoops[i])
            else:
                if len(current_queue) >= 2:
                    queues.append(current_queue)
                current_queue = [whoops[i]]

        if len(current_queue) >= 2:
            queues.append(current_queue)

        return queues

    def _get_strongest_channel_from_hdf5(self, group_name):
        """
        Get the strongest_channel from HDF5 database for a given whoop ID
        """
        try:
            with h5py.File(self.database_path, 'r') as f:
                if group_name not in f:
                    return None

                grp = f[group_name]

                if 'strongest_channel' in grp:
                    strongest_channel = int(grp['strongest_channel'][()])
                    if not np.isnan(strongest_channel):
                        return strongest_channel

                return None
        except Exception as e:
            print(f"Warning: Could not read strongest_channel for {group_name}: {e}")
            return None

    def _generate_new_id(self, queue, strongest_channel):
        """Generate new ID for a queue"""
        first = queue[0]
        last = queue[-1]

        timestamp_cluster = first['timestamp_cluster']
        peak_time = first['peak_time']
        window_start = first['window_start']
        window_end = last['window_end']

        new_id = f"{timestamp_cluster}_ch_{strongest_channel}_peaktime_{peak_time:.3f}_windowstart_{window_start:.3f}_windowend_{window_end:.3f}"

        return new_id

    def create_queue_mapping_file(self, input_file, output_json='database_analysis/queue_mapping.json', time_threshold=4.0):
        """
        Crea un file JSON con il mapping: first_id -> [list_of_queue_ids]
        
        Args:
            input_file: Path to file containing whoop IDs (one per line)
            output_json: Path to output JSON file with queue mapping
            time_threshold: Time threshold in seconds for grouping whoops
        
        Returns:
            dict: queue_mapping = {first_id: [list_of_all_ids_in_queue]}
        """
        # Read input IDs
        with open(input_file, 'r') as f:
            whoop_ids = [line.strip() for line in f if line.strip()]

        print(f"📂 Read {len(whoop_ids)} whoop IDs from {input_file}")

        # Parse all IDs
        parsed_whoops = []
        for whoop_id in whoop_ids:
            parsed = self._parse_whoop_id(whoop_id)
            if parsed:
                parsed_whoops.append(parsed)
            else:
                print(f"Warning: Could not parse ID: {whoop_id}")

        print(f"✓ Successfully parsed {len(parsed_whoops)} IDs")

        # Group by timestamp cluster
        df = pd.DataFrame(parsed_whoops)
        grouped_by_cluster = df.groupby('timestamp_cluster')

        # Dizionario: {first_id: [list_of_all_ids_in_queue]}
        queue_mapping = {}
        
        stats = {
            'total_clusters': 0,
            'total_queues': 0,
            'total_whoops_in_queues': 0
        }

        # Process each cluster
        for cluster_name, cluster_group in grouped_by_cluster:
            stats['total_clusters'] += 1
            print(f"\n📦 Processing cluster: {cluster_name} ({len(cluster_group)} whoops)")

            # Split into two channel groups
            group_1_16 = cluster_group[cluster_group['ch'].between(1, 16)].copy()
            group_17_32 = cluster_group[cluster_group['ch'].between(17, 32)].copy()

            for group_name, group_df in [("Ch 1-16", group_1_16), ("Ch 17-32", group_17_32)]:
                if len(group_df) == 0:
                    continue

                print(f"  {group_name}: {len(group_df)} whoops")

                # Sort by peak_time
                group_df_sorted = group_df.sort_values('peak_time').to_dict('records')

                # Create time queues
                queues = self._create_time_queues(group_df_sorted, time_threshold=time_threshold)

                print(f"    Found {len(queues)} queues with 2+ elements")

                # Generate mapping for each queue
                for queue_idx, queue in enumerate(queues):
                    stats['total_queues'] += 1
                    stats['total_whoops_in_queues'] += len(queue)

                    # Prendi il primo ID della queue
                    first_id = queue[0]['original_id']

                    # Salva TUTTI gli ID della queue (compreso il primo)
                    all_ids_in_queue = [whoop['original_id'] for whoop in queue]
                    queue_mapping[first_id] = all_ids_in_queue

                    print(f"      Queue {queue_idx + 1}: {len(queue)} whoops (first: Ch{queue[0]['ch']}, peak {queue[0]['peak_time']:.3f}s)")

        # Salva come JSON
        with open(output_json, 'w') as f:
            json.dump(queue_mapping, f, indent=2)

        print(f"\n{'='*60}")
        print(f"✅ SUMMARY")
        print(f"{'='*60}")
        print(f"Total clusters processed: {stats['total_clusters']}")
        print(f"Total queues created: {stats['total_queues']}")
        print(f"Total whoops in queues: {stats['total_whoops_in_queues']}")
        print(f"Queue mapping saved to: {output_json}")

        return queue_mapping

    def load_queue_by_first_id(self, first_id, queue_mapping_file='database_analysis/queue_mapping.json'):
        """
        Carica tutti i dati dei whoop in una queue dato il primo ID.
        
        Args:
            first_id: ID del primo elemento della queue
            queue_mapping_file: Path al file JSON con il mapping
        
        Returns:
            list: Lista di dict con tutti i dati dei whoops nella queue
        """
        import json
        
        # Carica mapping
        with open(queue_mapping_file, 'r') as f:
            queue_mapping = json.load(f)
        
        if first_id not in queue_mapping:
            print(f"⚠️  {first_id} is not a first element of any queue")
            return None
        
        queue_ids = queue_mapping[first_id]
        
        print(f"📊 Queue contains {len(queue_ids)} whoops")
        
        # Carica tutti i dati
        queue_data = []
        for whoop_id in queue_ids:
            data = self.load_whoop_by_id(whoop_id)
            queue_data.append(data)
        
        return queue_data


    def analyze_queue_statistics(self, queue_data):
        """
        Calcola statistiche aggregate per una queue di whoops.
        
        Args:
            queue_data: Lista di dict con dati dei whoops (output di load_queue_by_first_id)
        
        Returns:
            dict: Statistiche della queue
        """
        f0_values = [w['f0_mean'] for w in queue_data if not np.isnan(w['f0_mean'])]
        durations = [w['precise_duration'] for w in queue_data if not np.isnan(w['precise_duration'])]
        shr_values = [w['weighted_shr'] for w in queue_data if not np.isnan(w['weighted_shr'])]
        
        stats = {
            'queue_size': len(queue_data),
            'f0_mean': np.mean(f0_values) if f0_values else np.nan,
            'f0_std': np.std(f0_values) if f0_values else np.nan,
            'f0_min': np.min(f0_values) if f0_values else np.nan,
            'f0_max': np.max(f0_values) if f0_values else np.nan,
            'duration_mean': np.mean(durations) if durations else np.nan,
            'duration_total': sum(durations) if durations else np.nan,
            'shr_mean': np.mean(shr_values) if shr_values else np.nan,
            'channels': [w['ch'] for w in queue_data],
            'strongest_channels': [w['strongest_channel'] for w in queue_data],
            'peak_times': [w['peak_time'] for w in queue_data],
            'time_intervals': np.diff([w['peak_time'] for w in queue_data]).tolist() if len(queue_data) > 1 else [],
            'first_id': queue_data[0]['parent_filename'] + f"_ch_{queue_data[0]['ch']}_..." if queue_data else None
        }
        
        return stats


    def get_all_queue_first_ids(self, queue_mapping_file='database_analysis/queue_mapping.json'):
        """
        Ottieni lista di tutti i first_id delle queues.
        
        Returns:
            list: Lista di first_id
        """
        import json
        
        with open(queue_mapping_file, 'r') as f:
            queue_mapping = json.load(f)
        
        return list(queue_mapping.keys())


    def play_whoops_close_in_time(self, first_ids, root_raw_audio_dir, queue_mapping_file='database_analysis/queue_mapping.json'):
        """
        Play audio segments per ogni queue dato il first_id.
        
        Args:
            first_ids: Lista di first_id delle queues da riprodurre
            root_raw_audio_dir: Directory con i file audio raw
            queue_mapping_file: Path al file JSON con il mapping
        """
        import json
        
        # Carica mapping
        with open(queue_mapping_file, 'r') as f:
            queue_mapping = json.load(f)
        
        for first_id in first_ids:
            if first_id not in queue_mapping:
                print(f"⚠️  {first_id} is not a queue first element, skipping...")
                continue
            
            queue_ids = queue_mapping[first_id]
            
            print(f"\n{'='*70}")
            print(f"🎵 QUEUE: {first_id}")
            print(f"   Contains {len(queue_ids)} whoops")
            print(f"{'='*70}")
            
            # Play ogni whoop della queue
            for idx, whoop_id in enumerate(queue_ids, 1):
                print(f"\n  [{idx}/{len(queue_ids)}] Loading: {whoop_id}")
                
                whoop_data = self.load_whoop_by_id(whoop_id)
                
                print(f"    → Ch {whoop_data['ch']} (strongest: {whoop_data['strongest_channel']}), "
                    f"Peak {whoop_data['peak_time']:.3f}s, F0 {whoop_data['f0_mean']:.1f}Hz")
                
                Utils.extract_and_play_audio_segment_from_raw_multichannel_audio(
                    f"{root_raw_audio_dir}/{whoop_data['parent_filename']}.wav",
                    whoop_data['start_peak'],
                    whoop_data['end_peak'],
                    whoop_data['strongest_channel'] - 1,
                    lenght_extension=1.0
                )
            
            # Play concatenated (dall'inizio del primo alla fine dell'ultimo)
            first_data = self.load_whoop_by_id(queue_ids[0])
            last_data = self.load_whoop_by_id(queue_ids[-1])
            
            print(f"\n  🔗 Playing CONCATENATED segment:")
            print(f"    → Window {first_data['start_peak']:.3f}s - {last_data['end_peak']:.3f}s "
                f"(duration: {last_data['end_peak'] - first_data['start_peak']:.2f}s)")
            
            Utils.extract_and_play_audio_segment_from_raw_multichannel_audio(
                f"{root_raw_audio_dir}/{first_data['parent_filename']}.wav",
                first_data['start_peak'],
                last_data['end_peak'],
                first_data['strongest_channel'] - 1,
                lenght_extension=1.0
            )

    def validate_cluster(self, input_csv_path, output_csv_path, root_raw_audio_dir):
        """
        Validazione manuale dei cluster di whoop vicini nel tempo.
        Legge un CSV con first_id, mostra i dati e chiede conferma all'utente.
        Salva risultati in un nuovo CSV con colonna 'is_valid'.
        
        Args:
            input_csv_path: Path al CSV con i first_id da validare
            output_csv_path: Path al CSV di output con risultati
            root_raw_audio_dir: Directory con i file audio raw
        """
        ids = pd.read_csv(input_csv_path).id.tolist()

        
        results = []
        
        for idx, id in enumerate(ids, 1):
            print(f"\n\nCandidate [{idx}/{len(ids)}]")
            
            data = self.load_whoop_by_id(id)
            print(f"📊 Analisi per candidato: {id}:")
            print(f"Fundamental frequency (F0): {data['f0_mean']:.1f} Hz")
            
            if data['strongest_channel'] is None or np.isnan(data['strongest_channel']):
                print("No strongest channel found in database for this whoop. The candidate is super weak and might be a false positive. Skip.")
                continue
            else:
                print(f"Playing strongest channel audio segment ({data['strongest_channel']})...")
                Utils.extract_and_play_audio_segment_from_raw_multichannel_audio(f"{root_raw_audio_dir}/{data['parent_filename']}.wav",
                                                                        data['start_peak'],
                                                                        data['end_peak'],
                                                                        data['strongest_channel'] - 1,
                                                                        lenght_extension=1.0)
            
            

            # Chiedi validazione all'utente
            while True:
                user_input = input("Press 1 to validate, 2 to discard and 3 to listen it again? (1/2/3): ").strip().lower()
                if user_input == '1':
                    print("Marked as VALID")
                    results.append({id})
                    break
                elif user_input == '2':
                    print("Marked as INVALID")
                    break
                elif user_input == '3':
                    print("Replaying audio segment...")
                    Utils.extract_and_play_audio_segment_from_raw_multichannel_audio(f"{root_raw_audio_dir}/{data['parent_filename']}.wav",
                                                                            data['start_peak'],
                                                                            data['end_peak'],
                                                                            data['strongest_channel'] - 1,
                                                                            lenght_extension=1.0)
                else:
                    print("Please enter '1' for validate, '2' to discard or '3' to listen it again.")

            
    
    
    
    
    
            # Salva i risultati in un CSV
            results_df = pd.DataFrame(results, columns=['id'])
            results_df.to_csv(output_csv_path, index=False)
            print(f"Results saved to {output_csv_path}")

    def extract_statistics_from_cluster(self, input_csv_path):
        
        ids = pd.read_csv(input_csv_path).id.tolist()

        print(f"TOT WHOOP IN THE CLUSTER: {len(ids)}")

        f0_values = []
        durations = []
        strong_channels = []
        peak_times_absolute = []
        num_channels_with_whoop = []

        for id in ids:
            data = self.load_whoop_by_id(id)
            f0_values.append(data['f0_mean'])
            durations.append(data['precise_duration'])
            num_channels_with_whoop.append(data['num_channels_with_whoop'])
            strong_channels.append(data['strongest_channel'])
            
            date = data['date']
            time = data['time']
            peak_time_relative = data['peak_time']
            peak_times_absolute.append(
                datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S.%f") + timedelta(seconds=peak_time_relative)
            )

        


        # Plots distribuzione
        self._plot_histogram_distribution(f0_values, "f0", bins="auto") # metti bins="auto" quando hai tanti valori
        self._plot_histogram_distribution(durations, "precise_duration", bins="auto")
        self._plot_histogram_distribution(num_channels_with_whoop, "num_channels_with_whoop", bins="auto", is_discrete=True)  # bins per valori discreti da 0 a 32

        start_time = datetime(2025, 9, 15, 0, 0, 0)   # 2025-09-15 00:00:00
        end_time = datetime(2025, 9, 21, 0, 0, 0)     # 2025-09-21 00:00:00
        self._plot_peak_time_distribution(peak_times_absolute, start_time, end_time, "Europe/Berlin")


        self._plot_peak_spatial_distribution(strong_channels, self.mics_coords, self.frame_boundaries)




    def make_collection_of_sounds_out_of_a_cluster(
        self,
        input_csv_path: str,
        output_dir: str,
        root_raw_audio_dir: str,
        extra_padding: float = 0.0,
    ) -> str:
        """
        Dato un CSV con una colonna 'id' che contiene gli HDF5 group_name dei whoop,
        estrae tutti i segmenti dal raw multicanale e li concatena in un unico file .wav.

        - input_csv_path: path al CSV del cluster (colonna 'id')
        - output_dir: cartella di destinazione per il file finale
        - root_raw_audio_dir: directory dove stanno i .wav grezzi multicanale
        - extra_padding: padding aggiuntivo (in secondi) prima e dopo ogni whoop
        Ritorna: percorso del file .wav creato.
        """

        print(f"Tutti i whoop del cluster: {input_csv_path} verrano concatenati in un unico file audio.")
        os.makedirs(output_dir, exist_ok=True)

        # leggi gli id dal csv
        ids = pd.read_csv(input_csv_path).id.tolist()

        all_segments = []
        sr_out = None

        for i, gid in enumerate(ids):
            print(f"Processing whoop {i+1}/{len(ids)}")
            data = self.load_whoop_by_id(gid, verbose=False)

            parent_filename = data["parent_filename"]
            ch  = int(data["strongest_channel"]) - 1
            sr  = int(data["sr"])

            raw_path = os.path.join(root_raw_audio_dir, f"{parent_filename}.wav")

            start_t = max(0.0, float(data["start_peak"]) - extra_padding)
            end_t   = float(data["end_peak"]) + extra_padding

            with sf.SoundFile(raw_path) as f:
                if f.samplerate != sr:
                    raise ValueError(f"Sample rate mismatch per {raw_path}: {f.samplerate} vs {sr}")

                start_sample = max(0, int(start_t * sr))
                end_sample   = min(f.frames, int(end_t * sr))  # ✅ f.frames invece di multich_audio.shape[0]

                f.seek(start_sample)
                block = f.read(end_sample - start_sample, dtype='float64')

            seg = block[:, ch]
            del block  # libera i 32 canali, tieni solo il mono
            # Normalizza per evitare clipping
            max_val = np.max(np.abs(seg))
            if max_val > 0:
                seg = seg / (max_val * 1.1)
            all_segments.append(seg)

            if sr_out is None:
                sr_out = sr


        if not all_segments:
            raise ValueError("Nessun whoop trovato nel cluster, niente da concatenare.")

        # concatena tutti i segmenti uno dopo l'altro
        concatenated = np.concatenate(all_segments, axis=0)

        # # normalizzazione soft per evitare clipping
        # max_val = np.max(np.abs(concatenated))
        # if max_val > 0:
        #     concatenated = concatenated / (max_val * 1.05)

        # nome file di output basato sul nome del cluster
        csv_base = os.path.splitext(os.path.basename(input_csv_path))[0]  # es. "CLUSTER_1_VALIDATED_ids"
        cluster_name = csv_base.split("_VALIDATED")[0]  # → "CLUSTER_1"
        out_path = os.path.join(output_dir, f"{cluster_name}_whoops_collection.wav")

        sf.write(out_path, concatenated, samplerate=sr_out)
        print(f"✓ Salvata collezione whoops cluster in: {out_path}")

        return out_path

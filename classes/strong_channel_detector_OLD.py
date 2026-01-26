"""Prende come input un whoop candidate da 30 sec da usare come esempio. Esegue ulteriormente HNR detection per localizzare temporalmente il whoop in modo più preciso"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
import warnings
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shapely.geometry import Polygon as ShapelyPolygon




# Assumiamo che WhoopDetector e PitchDetector siano disponibili
try:
    from classes.whoop_detector import WhoopDetector
    from classes.pitch_detector import PitchDetector
except ImportError:
    warnings.warn(
        "WhoopDetector o PitchDetector non trovati. "
        "Assicurati che i moduli siano disponibili nel path."
    )


@dataclass
class WhoopSearchWindow:
    """Definisce le finestre di ricerca temporale e frequenziale."""
    time_min: float
    time_max: float
    f0_min: float
    f0_max: float
    
    def time_range(self) -> str:
        return f"[{self.time_min:.3f}, {self.time_max:.3f}]s"
    
    def f0_range(self) -> str:
        return f"[{self.f0_min:.1f}, {self.f0_max:.1f}]Hz"


@dataclass
class ChannelAnalysisResult:
    """Risultato dell'analisi di un singolo canale."""
    channel_num: int
    contains_whoop: bool
    peak_time: Optional[float] = None
    f0: Optional[float] = None
    hnr_db: Optional[float] = None
    hnr_linear: Optional[float] = None
    
    def __repr__(self) -> str:
        if not self.contains_whoop:
            return f"Ch{self.channel_num:2d}: ✗ No whoop detected"
        return (
            f"Ch{self.channel_num:2d}: ✓ t={self.peak_time:.3f}s, "
            f"f0={self.f0:.1f}Hz, HNR={self.hnr_db:.1f}dB"
        )


class StrongChannelDetector:
    """
    Rileva i canali più forti contenenti un evento di whoop in un array multi-canale.
    
    Utilizza un segmento di riferimento per estrarre le finestre temporali e frequenziali
    di interesse, quindi ricerca l'evento in tutti gli altri canali.
    
    Attributes:
        signal_multichannel: Array audio multi-canale (samples, channels)
        sr: Sample rate [Hz]
        ref_channel: Indice del canale di riferimento
        start_time_ref: Tempo di inizio del riferimento [secondi]
        end_time_ref: Tempo di fine del riferimento [secondi]
        broken_channels: Lista di indici canali rotti
        num_channels: Numero totale di canali su cui operiamo l'analisi
        verbose: Abilita log dettagliati
        plot: Abilita visualizzazione grafici
    """

    def __init__(
        self,
        signal_multichannel: np.ndarray,
        sr: int,
        ref_channel: int,
        start_time_ref: float,
        end_time_ref: float,
        broken_channels: Optional[List[int]] = None,
        num_channels: Optional[int] = None,
        detector_config: Optional[Dict] = None,
        pitch_config: Optional[Dict] = None,
        verbose: bool = True,
        plot: bool = True
    ):
        """
        Inizializza il detector.
        
        Args:
            signal_multichannel: Array audio (samples, channels)
            sr: Sample rate in Hz
            ref_channel: Indice del canale di riferimento
            start_time_ref: Inizio della finestra di riferimento [secondi]
            end_time_ref: Fine della finestra di riferimento [secondi]
            broken_channels: Lista di canali da escludere (default: [])
            num_channels: Numero totale canali (auto-rilevato da shape se None)
            detector_config: Configurazione WhoopDetector
            pitch_config: Configurazione PitchDetector
            verbose: Abilita log
            plot: Abilita grafici
        """
        self.signal_multichannel = signal_multichannel
        self.sr = sr
        self.ref_channel = ref_channel
        self.start_time_ref = start_time_ref
        self.end_time_ref = end_time_ref
        self.broken_channels = set(broken_channels or [])
        self.num_channels = num_channels 
        self.verbose = verbose
        self.plot = plot
        
        # Conversione tempo → campioni per il segmento di riferimento
        self.start_sample = self._time_to_samples(self.start_time_ref)
        self.end_sample = self._time_to_samples(self.end_time_ref)
        
        # Stato
        self.search_window: Optional[WhoopSearchWindow] = None
        self.reference_peak_time: Optional[float] = None
        self.reference_f0: Optional[float] = None
        self.channel_results: List[ChannelAnalysisResult] = []
        
        # Configurazioni di default
        self.detector_config = detector_config or {
            'window_length_ms': 50,
            'hop_length_ms': 10,
            'f0_min': 250,
            'f0_max': 700,
            'window_type': 'hamming',
            'lowpass_cutoff': 15000,
            'highpass_cutoff': 2500,
            'normalize': True,
            'target_rms': 0.1
        }
        
        self.pitch_config = pitch_config or {
            'length_queue': 4,
            'hz_threshold': 25,
            'threshold_increment': 1.3,
            'padding_start_ms': 5,
            'padding_end_ms': 25,
            'freq_min': 200,
            'freq_max': 600
        }

    def _time_to_samples(self, time_sec: float) -> int:
        """Converte tempo in secondi a numero di campioni."""
        return int(time_sec * self.sr)

    def _log(self, message: str, level: str = "INFO") -> None:
        """Stampa un messaggio di log con livello."""
        if not self.verbose:
            return
        
        level_symbols = {
            "INFO": "ℹ",
            "OK": "✓",
            "WARN": "⚠",
            "ERROR": "✗"
        }
        symbol = level_symbols.get(level, "•")
        print(f"  {symbol} [{level}] {message}")

    def analyze_reference(
        self,
        peak_time_offset: float = 0.20,
        f0_offset: float = 30,
        detector_percentile: int = 90,
        detector_offset: int = 4,
        detector_window_sec: float = 0.7
    ) -> WhoopSearchWindow:
        """
        Analizza il segmento di riferimento per estrarre picco temporale e F0.
        
        Args:
            peak_time_offset: Margine temporale attorno al picco [secondi]
            f0_offset: Margine frequenziale attorno a F0 [Hz]
            detector_percentile: Percentile HNR per rilevazione picco
            detector_offset: Offset per fusione picchi
            detector_window_sec: Finestra di analisi intorno al picco
        
        Returns:
            WhoopSearchWindow con finestre di ricerca
            
        Raises:
            RuntimeError: Se non rilevato esattamente 1 picco nel riferimento
        """
        print(f"\n{'='*70}")
        print(f"ANALISI SEGMENTO DI RIFERIMENTO (Canale {self.ref_channel})")
        print(f"{'='*70}")

        # Step 0: Estrai segnale di riferimento
        ref_signal = self.signal_multichannel[
            self.start_sample:self.end_sample, self.ref_channel
        ]
        
        # Step 1: Rilevamento picco HNR
        self._log("Inizializzazione WhoopDetector...")
        detector = WhoopDetector(
            signal=ref_signal,
            sr=self.sr,
            **self.detector_config
        )
        
        detector.detect(
            percentile=detector_percentile,
            offset=detector_offset,
            window_sec=detector_window_sec,
            merge_overlaps=True
        )
        
        if self.plot:
            detector.plot_analysis(ch_num=self.ref_channel+1, figsize=(10, 5))
        
        if len(detector.peak_times_) == 0:
            raise RuntimeError("Nessun picco rilevato nel canale di riferimento")
        
        if len(detector.peak_times_) != 1:
            self._log(
                f"Attenzione: rilevati {len(detector.peak_times_)} picchi "
                f"(atteso 1)",
                level="WARN"
            )
        
        # anche se nel ref ci sono più picchi, prendiamo il primo (soluzione non geniale)
        self.reference_peak_time = detector.peak_times_[0]
        self._log(f"Picco temporale: {self.reference_peak_time:.3f}s", level="OK")
        
        # Step 2: Estrazione F0 attorno al picco
        time_window_start = self.reference_peak_time - peak_time_offset
        time_window_end = self.reference_peak_time + peak_time_offset
        time_window_start_sample = self._time_to_samples(time_window_start)
        time_window_end_sample = self._time_to_samples(time_window_end)
        
        window_segment = ref_signal[
            time_window_start_sample:time_window_end_sample
        ]
        
        self._log("Inizializzazione PitchDetector...")
        pitch_detector = PitchDetector(
            audio_segment=window_segment,
            sr=self.sr
        )
        
        self.reference_f0, *_ = pitch_detector.estimate_f0(
            plot=self.plot,
            **self.pitch_config
        )
        
        if self.reference_f0 is None:
            self._log("F0 non stimato nel canale di riferimento", level="WARN")
        else:
            self._log(f"Frequenza fondamentale: {self.reference_f0:.2f}Hz", level="OK")
        
        # Step 3: Costruzione finestre di ricerca
        self.search_window = WhoopSearchWindow(
            time_min=time_window_start,
            time_max=time_window_end,
            f0_min=self.reference_f0 - f0_offset if self.reference_f0 else 0,
            f0_max=self.reference_f0 + f0_offset if self.reference_f0 else 1000
        )
        
        print(f"\n{'='*70}")
        print(f"FINESTRE DI RICERCA ESTRATTE")
        print(f"{'='*70}")
        print(f"Finestra temporale: {self.search_window.time_range()}")
        print(f"Finestra frequenziale: {self.search_window.f0_range()}")
        print(f"{'='*70}\n")
        
        return self.search_window

    def _analyze_channel(
        self,
        channel_num: int,
        detector_percentile: int = 85,
        detector_offset: int = 4,
        detector_window_sec: float = 0.7
    ) -> ChannelAnalysisResult:
        """
        Analizza un singolo canale cercando l'evento all'interno
        delle finestre definite da analyze_reference().
        
        Args:
            channel_num: Indice del canale da analizzare
            detector_percentile: Percentile HNR per rilevazione
            detector_offset: Offset per fusione picchi
            detector_window_sec: Finestra di analisi intorno al picco
        
        Returns:
            ChannelAnalysisResult con esito della ricerca
        """
        result = ChannelAnalysisResult(channel_num=channel_num, contains_whoop=False)
        
        # Estrai il segmento del canale
        window_of_interest = self.signal_multichannel[
            self.start_sample:self.end_sample, channel_num
        ]
        
        # Rilevamento picchi HNR
        detector = WhoopDetector(
            signal=window_of_interest,
            sr=self.sr,
            **self.detector_config
        )
        
        detection_results = detector.detect(
            percentile=detector_percentile,
            offset=detector_offset,
            window_sec=detector_window_sec,
            merge_overlaps=True
        )
        
        peak_info = detector.get_peak_info()
        
        if self.plot:
            detector.plot_analysis(ch_num=channel_num+1, figsize=(5, 2.5))
        
        if len(peak_info) == 0:
            self._log(
                f"Canale {channel_num+1:2d}: Nessun picco HNR rilevato",
                level="WARN"
            )
            return result
        
        # Ricerca di picchi entro la finestra temporale
        hnr_values = detection_results['results']['hnr_smoothed']
        
        for peak_idx, peak in enumerate(peak_info):
            peak_time = peak['peak_time']
            
            # Verifica se peak_time è dentro la finestra temporale di riferimento
            if not (self.search_window.time_min <= peak_time <= self.search_window.time_max):
                self._log(
                    f"Canale {channel_num+1:2d}: Picco a {peak_time:.3f}s "
                    f"fuori finestra temporale [{self.search_window.time_min:.3f}s, "
                    f"{self.search_window.time_max:.3f}s]",
                    level="WARN"
                )
                continue
            
            self._log(
                f"Canale {channel_num+1:2d}: Picco HNR a {peak_time:.3f}s "
                f"entro finestra temporale",
                level="OK"
            )
            
            # Estrai segmento per stima F0 attorno al picco
            time_window_start_sample = self._time_to_samples(self.search_window.time_min)
            time_window_end_sample = self._time_to_samples(self.search_window.time_max)
            
            pitch_segment = window_of_interest[
                time_window_start_sample:time_window_end_sample
            ]
            
            pitch_detector = PitchDetector(
                audio_segment=pitch_segment,
                sr=self.sr
            )
            
            f0, *_ = pitch_detector.estimate_f0(
                plot=self.plot,
                **self.pitch_config
            )
            
            # Verifica se f0 è dentro la finestra frequenziale di riferimento
            if f0 is None:
                self._log(
                    f"Canale {channel_num+1:2d}: F0 non stimato",
                    level="WARN"
                )
                continue
            
            if not (self.search_window.f0_min <= f0 <= self.search_window.f0_max):
                self._log(
                    f"Canale {channel_num+1:2d}: F0={f0:.2f}Hz "
                    f"fuori finestra [{self.search_window.f0_min:.1f}Hz, "
                    f"{self.search_window.f0_max:.1f}Hz]",
                    level="WARN"
                )
                continue
            
            # Picco trovato! Estrai HNR e salva risultato
            self._log(
                f"Canale {channel_num+1:2d}: ✓ WHOOP RILEVATO "
                f"(f0={f0:.2f}Hz, t={peak_time:.3f}s)",
                level="OK"
            )
            
            peak_index = peak.get('peak_index', 0)
            hnr_db = hnr_values[peak_info[peak_idx]['peak_index']]
            hnr_linear = 10 ** (hnr_db / 10) if hnr_db is not None else None
            
            result.contains_whoop = True
            result.peak_time = peak_time
            result.f0 = f0
            result.hnr_db = hnr_db
            result.hnr_linear = hnr_linear
            break  # Accetta il primo picco matching
        
        return result

    def detect_strong_channels(
        self,
        detector_percentile: int = 85,
        detector_offset: int = 4,
        detector_window_sec: float = 0.7
    ) -> Dict:
        """
        Esegue la ricerca dell'evento in tutti i canali.
        
        Args:
            detector_percentile: Percentile HNR per rilevazione picco
            detector_offset: Offset per fusione picchi
            detector_window_sec: Finestra di analisi intorno al picco
        
        Returns:
            Dizionario con risultati aggregati
            
        Raises:
            RuntimeError: Se analyze_reference() non è stato eseguito prima
        """
        if self.search_window is None:
            raise RuntimeError(
                "Esegui prima analyze_reference() per definire le finestre di ricerca"
            )
        
        print(f"\n{'='*70}")
        print(f"ANALISI DI TUTTI I CANALI")
        print(f"{'='*70}\n")
        
        self.channel_results = []
        
        for ch in range(self.num_channels):
            if ch in self.broken_channels:
                self._log(f"Canale {ch+1:2d}: Canale rotto, saltato", level="WARN")
                result = ChannelAnalysisResult(channel_num=ch, contains_whoop=False)
                self.channel_results.append(result)
                continue
            
            result = self._analyze_channel(
                ch,
                detector_percentile=detector_percentile,
                detector_offset=detector_offset,
                detector_window_sec=detector_window_sec
            )
            self.channel_results.append(result)
        
        return self._aggregate_results()

    def _aggregate_results(self) -> Dict:
        """
        Aggrega i risultati in un formato strutturato.
        
        Returns:
            Dizionario con canali ordinati per HNR e metadati di ricerca
        """
        print(f"\n{'='*70}")
        print(f"RISULTATI AGGREGATI")
        print(f"{'='*70}\n")
        
        # Filtra canali con whoop rilevato
        channels_with_whoop = [
            r for r in self.channel_results if r.contains_whoop
        ]
        
        # Ranking per HNR (forza del segnale)
        channels_ranked = sorted(
            channels_with_whoop,
            key=lambda x: x.hnr_linear if x.hnr_linear is not None else -1,
            reverse=True
        )

        
        # Risultato principale
        strongest_channel = channels_ranked[0] if channels_ranked else None

        
        results = {
            'strongest_channel': strongest_channel,
            'channels_with_whoop': channels_with_whoop,
            'channels_ranked': channels_ranked,
            'num_channels_with_whoop': len(channels_with_whoop),
            'search_window': self.search_window,
            'reference_peak_time': self.reference_peak_time,
            'reference_f0': self.reference_f0,
            'all_results': self.channel_results
        }
        
        # Stampa risultati
        print(f"Canali con whoop rilevato: {results['num_channels_with_whoop']}")
        if channels_ranked:
            print(f"\nCanali (ordinati per HNR):")
            for i, result in enumerate(channels_ranked, 1):
                print(f"  {i}. Ch {result.channel_num + 1}: ✓ t={result.peak_time:.3f}s, f0={result.f0:.1f}Hz, HNR={result.hnr_db:.1f}dB")
        
        if strongest_channel:
            print(f"\n{'='*70}")
            print(f"CANALE PIÙ FORTE: {strongest_channel.channel_num + 1}")
            print(f"  HNR: {strongest_channel.hnr_db:.2f} dB "
                  f"({strongest_channel.hnr_linear:.2f} lineare)")
            print(f"  Peak time: {strongest_channel.peak_time:.3f}s")
            print(f"  F0: {strongest_channel.f0:.2f}Hz")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"NESSUN WHOOP RILEVATO IN ALCUN CANALE")
            print(f"{'='*70}\n")
        
        return results
    
    def get_channel_levels_array(self) -> np.ndarray:
        """
        Restituisce un array con livelli HNR lineari per ogni canale.
        Utile per plotting e comparazione con lo script originale.
        
        Returns:
            Array di shape (num_channels,) con valori HNR lineari
        """
        levels = np.zeros(self.num_channels)
        
        for result in self.channel_results:
            if result.contains_whoop and result.hnr_linear is not None:
                levels[result.channel_num] = result.hnr_linear
        
        return levels
    

    def _voronoi_finite_polygons_2d(self, vor, radius=None):
        """Ritorna regioni Voronoi finite (chiude quelle infinite)."""
        if vor.points.shape[1] != 2:
            raise ValueError("Richiede input 2D")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = np.ptp(vor.points).max() * 2


        # Mappa ridges per punto
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Ricostruisci regioni infinite
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                new_regions.append(vertices)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                t = vor.points[p2] - vor.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def plot_voronoi_2d(
        self,
        mic_positions: np.ndarray,
        boundaries: Optional[tuple] = None,
        use_db: bool = False,
        cmap: str = "hot",
        figsize: tuple = (12, 10),
        show_colorbar: bool = True,
        annotate: bool = False,
        title: Optional[str] = None,
    ) -> None:

        vor = Voronoi(mic_positions)
        fig, ax = plt.subplots(figsize=figsize)

        # Broken mask + HNR (IDENTICO)
        broken_mask = np.array([ch in self.broken_channels for ch in range(self.num_channels)])
        hnr_levels = self.get_channel_levels_array()
        if use_db:
            hnr_plot = np.where(broken_mask, -999, 
                                20 * np.log10(np.clip(hnr_levels, 1e-10, None)))
        else:
            hnr_plot = np.where(broken_mask, 0.0, hnr_levels)

        hnr_nonzero = hnr_plot[~broken_mask]
        if len(hnr_nonzero) > 0:
            p05, p95 = np.percentile(hnr_nonzero, [5, 95])
            norm_hnr = np.clip((hnr_plot - p05) / (p95 - p05), 0, 1)
        else:
            norm_hnr = np.zeros_like(hnr_plot)

        scalar_map = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
        colors = scalar_map.to_rgba(norm_hnr)[:, :3]

        # Boundaries + regioni (IDENTICO)
        if boundaries is not None:
            x_min, x_max, y_min, y_max = boundaries
            clip_box = ShapelyPolygon([(x_min, y_min), (x_min, y_max), 
                                        (x_max, y_max), (x_max, y_min)])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            clip_box = None

        regions, vertices = self._voronoi_finite_polygons_2d(vor)
        for i, region_idx in enumerate(regions):
            polygon = vertices[region_idx]
            if clip_box is not None:
                poly = ShapelyPolygon(polygon)
                poly = poly.intersection(clip_box)
                if poly.is_empty:
                    continue
                polygon = np.array(poly.exterior.coords[:-1])
            
            color_region = [0, 0, 0] if broken_mask[i] else colors[i]
            ax.fill(polygon[:, 0], polygon[:, 1], color=color_region, 
                    alpha=0.8 if broken_mask[i] else 0.75, 
                    edgecolor='black', linewidth=1.0)

        # Microfoni ROSSO/VERDE + NUMERO CANALE DENTRO
        mic_colors = ['red' if broken_mask[i] else 'limegreen' for i in range(len(mic_positions))]
        mic_sizes = [160 if broken_mask[i] else 120 for i in range(len(mic_positions))]

        scatter = ax.scatter(mic_positions[:, 0], mic_positions[:, 1], 
                            c=mic_colors, s=mic_sizes, edgecolor='black', linewidth=1, 
                            zorder=10)  # ← Bordo NERO

        # NUMERI CANALE DENTRO BOLLINI
        for i, (x, y) in enumerate(mic_positions):
            ax.text(x, y, str(i+1), ha='center', va='center', fontsize=7, 
                    fontweight='bold', color='black', zorder=11)

        # Colorbar
        if show_colorbar:
            cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.8)
            cbar.set_label("HNR Good Channels", fontsize=11)

        # LEGENDA SEMPLICE CON BORDO NERO + NUMERI
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                        markersize=12, label='Broken channel', 
                        markeredgecolor='black', markeredgewidth=3),  # ← Bordo NERO
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', 
                        markersize=12, label='Working channel', 
                        markeredgecolor='black', markeredgewidth=3)   # ← Bordo NERO
        ]

        # # AGGIUNGI NUMERI NELLA LEGENDA
        # from matplotlib.patches import Circle
        # red_patch = Circle((0,0), 0.1, facecolor='red', edgecolor='black', linewidth=3)
        # green_patch = Circle((0,0), 0.1, facecolor='limegreen', edgecolor='black', linewidth=3)
        # ax.add_patch(red_patch)
        # ax.add_patch(green_patch)

        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                    frameon=True, fancybox=True, shadow=True)

        ax.set_title(title or "Voronoi HNR Map", fontsize=14, fontweight='bold')
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()







        



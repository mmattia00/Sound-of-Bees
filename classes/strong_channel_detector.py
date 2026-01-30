"""Prende come input un whoop candidate da 0.5 sec da usare come esempio. è già abbastanza preciso quindi non si fa l'analisi della reference come nella versione precedente.
Prende direttamente come input inizio, fine peak time e f0 del whoop reference.
NB che questa classe per ora è scritta in modo da considerare i primi 16 canali in modo distinto dai secondi 16 perchè i dati che stiamo considerando per ora non sono allineati nel tempo.
Per ora semplicemente se ref_ch <16 allora analizza i canali 0-15, altrimenti 16-31 ho introdotto la variabile channels_of_interest e le funzioni che la usano sono:
    - detect_strong_channels: itera su channels_of_interest
    - get_channel_levels_array: ho piazzato un -16 nell'indice per allineare i canali 16-31 a 0-15 nell'array dei livelli
    - plot_voronoi_2d: Usa channel_of_interest in alcuni punti
PER ORA VA BENE COSI DA SISTEMARE IN FUTURO QUANDO CHANNELS_OF_INTEREST SARA 0:31 SEMPRE
"""
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
import warnings
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from shapely.geometry import Polygon as ShapelyPolygon
import sounddevice as sd
from matplotlib.patches import Patch





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
        channel_ref: int,
        start_time_ref: float,
        end_time_ref: float,
        f0_ref: float,
        f0_ref_median_frame_idx: Optional[int] = None,
        f0_offset: float = 30,
        broken_channels: Optional[List[int]] = None,
        num_channels: Optional[int] = None,
        detector_config: Optional[Dict] = None,
        pitch_config: Optional[Dict] = None,
        verbose: bool = False,
        plot: bool = False,
        listening_test: bool = False
    ):
        """
        Inizializza il detector.
        
        Args:
            signal_multichannel: Array audio (samples, channels)
            sr: Sample rate in Hz
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
        self.channel_ref = channel_ref
        self.start_time_ref = start_time_ref
        self.end_time_ref = end_time_ref
        self.f0_ref = f0_ref
        self.f0_ref_median_frame_idx = f0_ref_median_frame_idx
        self.f0_offset = f0_offset
        self.broken_channels = set(broken_channels or [])
        self.num_channels = num_channels 
        self.verbose = verbose
        self.plot = plot
        self.listening_test = listening_test
        
        # Conversione tempo → campioni per il segmento di riferimento
        self.start_sample = self._time_to_samples(self.start_time_ref)
        self.end_sample = self._time_to_samples(self.end_time_ref)

        if self.channel_ref - 1 < 16:
            self.channels_of_interest = list(range(16))
        else:
            self.channels_of_interest = list(range(16, 32))
        
        
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

        # definisci WhoopSearchWindow direttamente dai parametri di input

        self.search_window = WhoopSearchWindow(
            time_min=self.start_time_ref,
            time_max=self.end_time_ref,
            f0_min=self.f0_ref - self.f0_offset if self.f0_ref else 0,
            f0_max=self.f0_ref + self.f0_offset if self.f0_ref else 1000
        )

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


    def _analyze_channel(
        self,
        channel_num: int,
        detector_percentile: int = 85,
        detector_offset: int = 4,
        detector_window_sec: float = 0.5
    ) -> ChannelAnalysisResult:
        """
        Analizza un singolo canale cercando l'evento all'interno della finestra.
        
        Logica:
        1. HNR Detection: se no picco → return (canale non valido)
        2. Pitch Detection: se no f0 → return (canale non valido)
        3. Analisi code:
        - Itera da quella più lunga
        - Cerca coda con frame_idx di riferimento
        - Se trovata: verifica f0 in range, altrimenti continua con prossima coda
        4. Se nessuna coda valida: return (canale non valido)
        
        Args:
            channel_num: Indice del canale da analizzare
            detector_percentile: Percentile HNR per rilevazione
            detector_offset: Offset per fusione picchi
            detector_window_sec: Finestra di analisi intorno al picco
        
        Returns:
            ChannelAnalysisResult con esito della ricerca
        """
        result = ChannelAnalysisResult(channel_num=channel_num, contains_whoop=False)
        
        # ========== STEP 1: HNR DETECTION ==========
        window_of_interest = self.signal_multichannel[
            self.start_sample:self.end_sample, channel_num
        ]
        
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
        hnr_values = detection_results['results']['hnr_smoothed']

        if self.plot and self.listening_test:
            self.play_channel(channel_num)
        
        if self.plot:
            detector.plot_analysis(ch_num=channel_num+1, figsize=(5, 2.5))
        
        if len(peak_info) == 0:
            self._log(
                f"Canale {channel_num+1:2d}: Nessun picco HNR rilevato",
                level="WARN"
            )
            return result
        
        # ========== STEP 2: PITCH DETECTION ==========
        pitch_detector = PitchDetector(
            audio_segment=window_of_interest,
            sr=self.sr
        )
        
        f0_global, _, all_queues, all_queues_f0, whoop_pitch_info = pitch_detector.estimate_f0(
            **self.pitch_config
        )
        
        if f0_global is None:
            self._log(
                f"Canale {channel_num+1:2d}: F0 non stimato",
                level="WARN"
            )
            if self.plot:
                pitch_detector._plot_results(
                    np.asarray(pitch_detector.compute_fundamental_frequencies(
                        freq_min=self.pitch_config['freq_min'],
                        freq_max=self.pitch_config['freq_max']
                    )), None, all_queues, None
                )
            return result
        
        # ========== STEP 3: ANALISI DELLE CODE ==========
        # Ordina le code per lunghezza (dalla più lunga)
        queue_lengths = [len(q) for q in all_queues]
        queue_indices_sorted = sorted(range(len(all_queues)), key=lambda i: queue_lengths[i], reverse=True)
        
        # # Stampa info code
        # print(f"\n\n INFO CODE RILEVATE PER CANALE {channel_num+1}: \n\n")
        # for i, queue_idx in enumerate(queue_indices_sorted):
        #     print(f"    Queue {i}: lunghezza={len(all_queues[queue_idx])}, freq={all_queues_f0[queue_idx]:.2f}Hz, frames={all_queues[queue_idx]}")
        
        # Itera sulle code dalla più lunga
        valid_queue_idx = None
        valid_queue_freq = None
        valid_queue_frames = None
        
        for queue_idx in queue_indices_sorted:
            queue_freq = all_queues_f0[queue_idx]
            queue_frames = all_queues[queue_idx]

            if self.verbose:
                print(f"  • Analisi coda (Ch {channel_num+1:2d}): lunghezza={len(queue_frames)}, freq={queue_freq:.2f}Hz, frames={queue_frames}")
            
            # TEST 1: Controlla frame_idx di riferimento
            if self.f0_ref_median_frame_idx is not None:
                if self.f0_ref_median_frame_idx not in queue_frames:
                    self._log(
                        f"Canale {channel_num+1:2d}: Coda non contiene frame_idx {self.f0_ref_median_frame_idx}",
                        level="WARN"
                    )
                    continue  # Passa alla prossima coda
            
            # TEST 2: Verifica frequenza in range
            if not (self.search_window.f0_min <= queue_freq <= self.search_window.f0_max):
                self._log(
                    f"Canale {channel_num+1:2d}: F0={queue_freq:.2f}Hz "
                    f"fuori finestra [{self.search_window.f0_min:.1f}Hz, {self.search_window.f0_max:.1f}Hz]",
                    level="WARN"
                )
                continue  # Passa alla prossima coda
            
            # Coda valida trovata!
            valid_queue_idx = queue_idx
            valid_queue_freq = queue_freq
            valid_queue_frames = queue_frames
            break
        
        # ========== STEP 4: VERIFICA RISULTATO ==========
        if valid_queue_idx is None:
            self._log(
                f"Canale {channel_num+1:2d}: Nessuna queue valida trovata",
                level="WARN"
            )
            if self.plot:
                pitch_detector._plot_results(
                    np.asarray(pitch_detector.compute_fundamental_frequencies(
                        freq_min=self.pitch_config['freq_min'],
                        freq_max=self.pitch_config['freq_max']
                    )), None, all_queues, None
                )
            return result
        
        # ========== STEP 5: WHOOP RILEVATO - SALVA RISULTATI ==========
        peak_time = peak_info[0]['peak_time']
        peak_index = peak_info[0]['peak_index']
        hnr_db = hnr_values[peak_index]
        hnr_linear = 10 ** (hnr_db / 10) if hnr_db is not None else None
        
        self._log(
            f"Canale {channel_num+1:2d}: ✓ WHOOP RILEVATO "
            f"(f0={valid_queue_freq:.2f}Hz, t={peak_time:.3f}s)",
            level="OK"
        )
        
        # Plotta con coda evidenziata
        if self.plot:
            pitch_detector._plot_results(
                np.asarray(pitch_detector.compute_fundamental_frequencies(
                    freq_min=self.pitch_config['freq_min'],
                    freq_max=self.pitch_config['freq_max']
                )), valid_queue_frames, all_queues, valid_queue_freq
            )
        
        # Salva risultati
        result.contains_whoop = True
        result.peak_time = peak_time
        result.f0 = valid_queue_freq
        result.hnr_db = hnr_db
        result.hnr_linear = hnr_linear
        
        return result


    def detect_strong_channels(
        self,
        detector_percentile: int = 75,
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
        print(f"ANALISI SU TUTTI GLI ALTRI CANALI: in particolare {self.channels_of_interest}")
        print(f"{'='*70}\n")
        
        self.channel_results = []
        
        for ch in self.channels_of_interest:
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

        
        # Risultati principali
        strongest_channel = channels_ranked[0] if channels_ranked else None
        # FIX: Converti None → 0.0 (o np.nan) PRIMA di list comprehension
        list_hnr_levels = [
            r.hnr_linear if r.hnr_linear is not None else np.nan  # ← CAMBIA QUI!
            for r in self.channel_results
        ]
        
        results = {
            'strongest_channel': strongest_channel,
            'channels_with_whoop': channels_with_whoop,
            'channels_ranked': channels_ranked,
            'num_channels_with_whoop': len(channels_with_whoop),
            'search_window': self.search_window,
            'reference_peak_time': self.reference_peak_time,
            'reference_f0': self.reference_f0,
            'all_results': self.channel_results,
            'all_hnr_levels': list_hnr_levels
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
                levels[result.channel_num - 16] = result.hnr_linear
        
        return levels
    
    # def listening_test(self, pause_between=0.5, save_audios: Optional[bool] = False, folder: Optional[str] = None) -> None:
    #     """
    #     Estrae la finestra di interesse da ciascuno dei primi N canali
    #     e la riproduce in sequenza.
        
    #     Args:
    #         data (np.ndarray): Array audio (samples, channels)
    #         samplerate (int): Frequenza di campionamento
    #         start_time (float): Tempo di inizio in secondi
    #         end_time (float): Tempo di fine in secondi
    #         num_channels (int): Numero di canali da elaborare (default 16)
    #         pause_between (float): Pausa tra riproduzioni in secondi
    #         channel_broken (int or None): Canali rotti da escludere (default None)
    #     """

    #     # Converti i tempi in campioni
    #     test_start_peak = max(0, int((self.start_time_ref - 0.5) * self.sr))
    #     test_end_peak = min(self.signal_multichannel.shape[0], int((self.end_time_ref + 0.5) * self.sr))

    #     # Estrai il blocco usato per il listening test
    #     block = self.signal_multichannel[test_start_peak:test_end_peak, :]

    #     # Trova il massimo assoluto su tutti i canali e campioni
    #     global_max = np.max(np.abs(block))

    #     # Scegli un target (es. 0.8 per stare sotto al clipping)
    #     target_peak = 40

    #     if global_max > 0:
    #         gain = target_peak / global_max
    #     else:
    #         gain = 1.0

     

        
    #     print(f"\n{'='*60}")
    #     print(f"ESTRAZIONE E RIPRODUZIONE WHOOP SIGNAL")
    #     print(f"Canali considerati: {self.channels_of_interest}")
    #     print(f"{'='*60}\n")


    #     # Loop sui primi N canali
    #     for ch in self.channels_of_interest:
    #         if ch in self.broken_channels:
    #             print(f"▶ Canale {ch+1:2d}: ✗ Canale rotto, salto")
    #             continue
    #         else:
    #             # Estrai la finestra dal canale corrente
    #             tmp_audio = self.signal_multichannel[test_start_peak:test_end_peak, ch] * gain
                
                
    #             # Output
    #             print(f"▶ Canale {ch+1:2d}: ", end='', flush=True)
                
    #             # Riproduce
    #             try:
    #                 sd.play(tmp_audio, samplerate=self.sr, blocking=False)
    #                 # Attendi la fine della riproduzione
    #                 sd.wait()
    #                 print("✓")
    #             except Exception as e:
    #                 print(f"✗ Errore: {e}")
                
    #             # Pausa tra i canali (se non è l'ultimo)
    #             if ch < self.channels_of_interest[-1]:
    #                 time.sleep(pause_between)
        
    #     print(f"\n{'='*60}")
    #     print("✓ Riproduzione completata")
    #     print(f"{'='*60}\n")

    def play_channel(self, channel_num: int) -> None:
        """
        Estrae la finestra di interesse del canale specificato dal segnale multi-canale e la riproduce.
        
        """
        # Converti i tempi in campioni
        test_start_peak = max(0, int((self.start_time_ref - 0.5) * self.sr))
        test_end_peak = min(self.signal_multichannel.shape[0], int((self.end_time_ref + 0.5) * self.sr))

        # Estrai il blocco usato per il listening test
        block = self.signal_multichannel[test_start_peak:test_end_peak, self.channels_of_interest]

        # Trova il massimo assoluto su tutti i canali e campioni
        global_max = np.max(np.abs(block))

        # Scegli un target (es. 0.8 per stare sotto al clipping)
        target_peak = 5

        if global_max > 0:
            gain = target_peak / global_max
        else:
            gain = 1.0

     

        
        print(f"\n{'='*60}")
        print(f"ESTRAZIONE E RIPRODUZIONE WHOOP SIGNAL IN CANALE {channel_num+1}")
        print(f"{'='*60}\n")

        # Estrai la finestra dal canale corrente
        tmp_audio = self.signal_multichannel[test_start_peak:test_end_peak, channel_num] * gain
        
        
        # Riproduce
        try:
            sd.play(tmp_audio, samplerate=self.sr)
            # Attendi la fine della riproduzione
            sd.wait()
            print("✓")
        except Exception as e:
            print(f"✗ Errore: {e}")



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

        mic_positions = mic_positions[self.channels_of_interest]
        vor = Voronoi(mic_positions)
        fig, ax = plt.subplots(figsize=figsize)

        # Broken mask + HNR (IDENTICO)
        broken_mask = np.array([ch in self.broken_channels for ch in self.channels_of_interest])
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
            ax.text(x, y, str(self.channels_of_interest[i] + 1), ha='center', va='center', fontsize=7, 
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

    # Funzione per generare vertici esagono
    def _create_hexagon(self, center_x, center_y, radius):
            """Crea un esagono regolare centrato in (center_x, center_y)"""
            angles = np.linspace(0, 2*np.pi, 7)  # 7 punti per chiudere il poligono
            x = center_x + radius * np.cos(angles)
            y = center_y + radius * np.sin(angles)
            return np.column_stack([x, y])
        
    def plot_hexagon_hnr_map(
        self,
        mic_positions: np.ndarray,
        hexagon_radius: float = 0.5,
        use_db: bool = False,
        cmap: str = "hot",
        boundaries: List[float] = None,
        show_colorbar: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """
        Plotta esagoni regolari attorno ai microfoni, colorati in base ai livelli di HNR.
        Canali rotti evidenziati con pattern tratteggiato.
        
        Args:
            mic_positions: Coordinate dei microfoni
            hexagon_radius: Raggio dell'esagono (distanza dal centro al vertice)
            use_db: Se True, usa scala dB per HNR
            cmap: Colormap da usare
            boundaries: Limiti del grafico [xmin, xmax, ymin, ymax]
            show_colorbar: Se mostrare la colorbar
            title: Titolo del grafico
        """
        
        mic_positions = mic_positions[self.channels_of_interest]
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if boundaries is not None:
            ax.set_xlim(boundaries[0], boundaries[1])
            ax.set_ylim(boundaries[2], boundaries[3])

        

        # Broken mask + HNR
        broken_mask = np.array([ch in self.broken_channels for ch in self.channels_of_interest])
        hnr_levels = self.get_channel_levels_array()
        
        if use_db:
            hnr_plot = np.where(broken_mask, -999, 
                                20 * np.log10(np.clip(hnr_levels, 1e-10, None)))
        else:
            hnr_plot = np.where(broken_mask, 0.0, hnr_levels)
        
        # Normalizzazione HNR (escludendo canali rotti)
        hnr_nonzero = hnr_plot[~broken_mask]
        if len(hnr_nonzero) > 0:
            p05, p95 = np.percentile(hnr_nonzero, [5, 95])
            norm_hnr = np.clip((hnr_plot - p05) / (p95 - p05), 0, 1)
        else:
            norm_hnr = np.zeros_like(hnr_plot)
        
        # ColorMap
        scalar_map = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
        colors = scalar_map.to_rgba(norm_hnr)[:, :3]
        
        # Disegna esagoni con hatching per canali rotti
        for i, (x, y) in enumerate(mic_positions):
            hexagon = self._create_hexagon(x, y, hexagon_radius)
            
            if broken_mask[i]:
                # CANALI ROTTI: colore grigio scuro + hatching + bordo rosso
                color_hexagon = 'darkgray'
                alpha_val = 0.9
                hatch_pattern = '///'
                edgecolor_hex = 'red'
                linewidth_hex = 2.5
            else:
                # CANALI FUNZIONANTI: colore HNR + bordo nero
                color_hexagon = colors[i]
                alpha_val = 0.75
                hatch_pattern = None
                edgecolor_hex = 'black'
                linewidth_hex = 1.5
            
            ax.fill(hexagon[:, 0], hexagon[:, 1], 
                    color=color_hexagon, 
                    alpha=alpha_val, 
                    hatch=hatch_pattern,
                    edgecolor=edgecolor_hex,
                    linewidth=linewidth_hex,
                    zorder=1)
        
        # Disegna i microfoni (puntini + numeri)
        mic_colors = ['red' if broken_mask[i] else 'limegreen' for i in range(len(mic_positions))]
        mic_sizes = [160 if broken_mask[i] else 120 for i in range(len(mic_positions))]
        
        scatter = ax.scatter(mic_positions[:, 0], mic_positions[:, 1], 
                            c=mic_colors, s=mic_sizes, edgecolor='black', linewidth=1, 
                            zorder=10)
        
        # Numeri canale dentro ai puntini
        for i, (x, y) in enumerate(mic_positions):
            ax.text(x, y, str(self.channels_of_interest[i] + 1), 
                    ha='center', va='center', fontsize=7, 
                    fontweight='bold', color='black', zorder=11)
        
        # Colorbar
        if show_colorbar:
            cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.8)
            if use_db:
                cbar.set_label("HNR (dB)", fontsize=11)
            else:
                cbar.set_label("HNR", fontsize=11)
        
        # Legenda con hatching per broken channels
        legend_elements = [
            Patch(facecolor='darkgray', alpha=0.9, hatch='///', 
                edgecolor='red', linewidth=2, label='Broken channel'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', 
                    markersize=12, label='Working channel', 
                    markeredgecolor='black', markeredgewidth=1.5)
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                frameon=True, fancybox=True, shadow=True)
        
        # Impostazioni grafiche
        ax.set_title(title or "Hexagon HNR Map", fontsize=14, fontweight='bold')
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()








        



"""
SRP Localizator - Steered Response Power usando GCC-PHAT

Implementazione di un localizzatore di sorgenti sonore basato su
Steered Response Power (SRP) e GCC-PHAT (Generalized Cross-Correlation
with Phase Transform).
"""

import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import warnings
import matplotlib.pyplot as plt
from classes.pitch_detector import PitchDetector


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MicrophoneArray:
    """Rappresenta un array di microfoni con posizioni 2D."""
    positions: np.ndarray
    sample_rate: int
    validated_channels: List[int] | None = None
    boundaries: Tuple[float, float, float, float] | None = None  # (x_min, x_max, y_min, y_max)
    margin: float = 0.05 # margine extra per la griglia di ricerca

    def __post_init__(self):
        if self.validated_channels is None:
            self.validated_channels = []
        else:
            self.validated_channels = list(self.validated_channels)

    @property
    def n_mics(self) -> int:
        """Numero di microfoni."""
        return self.positions.shape[0]
    
    def add_margin(self) -> None:
        """Aggiunge margine alle boundaries."""
        if self.boundaries is not None:
            x_min, x_max, y_min, y_max = self.boundaries
            self.boundaries = (x_min - self.margin, x_max + self.margin,
                               y_min - self.margin, y_max + self.margin)
        else:
            warnings.warn("Boundaries non definite; impossibile aggiungere margine.")

    def get_mic_pair(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """Restituisce le posizioni di due microfoni."""
        return self.positions[i], self.positions[j]

    def get_all_pairs(self) -> List[Tuple[int, int]]:
        """
        Restituisce gli indici di tutte le coppie di microfoni
        che coinvolgono SOLO i canali validati.
        
        Se validated_channels = [0, 1, 3, 5], restituisce tutte le coppie
        composte esclusivamente da questi canali:
        [(0, 1), (0, 3), (0, 5), (1, 3), (1, 5), (3, 5)]
        
        I canali NON nella lista validated_channels vengono completamente ignorati.
        
        Returns:
            Lista di tuple (i, j) dove i < j e ENTRAMBI i e j sono in validated_channels
        """
        # Se non ci sono canali validati, restituisci lista vuota
        if not self.validated_channels:
            return []
        
        # Converti in set per ricerca O(1)
        valid_set = set(self.validated_channels)
        
        pairs: List[Tuple[int, int]] = []
        
        # Itera su tutti i canali validati
        for i in self.validated_channels:
            # Per ogni canale validato, cercane altri con indice più alto
            for j in self.validated_channels:
                if i < j:
                    pairs.append((i, j))
        
        return pairs

    def get_pairs_from_reference_channel(self, reference_channel: int) -> List[Tuple[int, int]]:
        """
        Restituisce le coppie di microfoni che coinvolgono un canale di riferimento specifico.
        
        Utile per localizzazione focalizzata quando hai già identificato
        il canale da cui il segnale è più forte. Riduce il rumore usando
        solo le coppie correlate al canale di interesse.
        
        Args:
            reference_channel: indice (0-based) del canale di riferimento
        
        Returns:
            Lista di tuple (i, j) dove almeno uno tra i e j è reference_channel,
            escludendo canali rotti
        
        Raises:
            ValueError: se reference_channel non è valido
        
        Example:
            >>> pairs = mic_array.get_pairs_from_reference_channel(3)
            >>> # Restituisce: [(3, 0), (3, 1), (3, 2), (3, 4), (3, 5), ..., (3, 15)]
        """
        if reference_channel < 0 or reference_channel >= self.n_mics:
            raise ValueError(
                f"reference_channel {reference_channel} non valido. "
                f"Microfoni disponibili: 0-{self.n_mics - 1}"
            )
        
        broken = set(self.channel_broken or [])
        
        if reference_channel in broken:
            warnings.warn(
                f"reference_channel {reference_channel} è marcato come rotto. "
                "Continuo comunque, ma i risultati potrebbero essere inaffidabili."
            )
        
        pairs: List[Tuple[int, int]] = []
        
        for i in range(self.n_mics):
            if i == reference_channel or i in broken:
                continue
            
            pairs.append((reference_channel, i))
        
        return pairs


@dataclass
class SearchGrid:
    """Definisce la griglia di ricerca per la localizzazione."""
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    resolution: float
    
    def generate_grid(self) -> np.ndarray:
        """Genera i punti della griglia."""
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        
        x = np.arange(x_min, x_max + self.resolution, self.resolution)
        y = np.arange(y_min, y_max + self.resolution, self.resolution)
        
        xx, yy = np.meshgrid(x, y)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        return grid_points
        


@dataclass
class LocalizationResult:
    """Risultato della localizzazione."""
    estimated_position: np.ndarray
    power_map: np.ndarray
    grid_points: np.ndarray
    max_power: float
    
    def __str__(self):
        x, y = self.estimated_position
        return (f"Estimated position: ({x:.4f}, {y:.4f})\n"
                f"Max power: {self.max_power:.6f}")


# ============================================================================
# GCC-PHAT CORRELATOR
# ============================================================================

class GCCPHATCorrelator:
    """
    Calcola la correlazione generalizzata con trasformata di fase (GCC-PHAT).
    
    La GCC-PHAT è robusta perché usa solo la fase e ignora le differenze
    di ampiezza tra i segnali.
    """
    
    def __init__(self, sr: int):
        """Inizializza il correlatore."""
        self.sr = sr
    
    def compute(self, sig1: np.ndarray, sig2: np.ndarray, 
                max_tau_ms: float = 100, plot: bool = False) -> Dict:
        """Computa GCC-PHAT tra due segnali."""
        # Lunghezza di FFT (padding per evitare aliasing)
        fft_len = 2 * max(len(sig1), len(sig2))
        
        # FFT dei due segnali
        X1 = np.fft.rfft(sig1, n=fft_len)
        X2 = np.fft.rfft(sig2, n=fft_len)
        
        # Cross-spettro
        Gxy = X1 * np.conj(X2)
        
        # GCC-PHAT: normalizzazione per la magnitude (whitening spettrale)
        mag = np.abs(Gxy)
        mag[mag < 1e-10] = 1e-10
        Gxy_phat = Gxy / mag
        
        # Inverse FFT per ottenere la correlazione generalizzata
        correlation = np.fft.irfft(Gxy_phat, n=fft_len)
        correlation = np.fft.fftshift(correlation)
        
        # Taglia il range di delay da cercare
        max_samples = int((max_tau_ms / 1000) * self.sr)
        center = len(correlation) // 2
        start_idx = center - max_samples
        end_idx = center + max_samples + 1
        correlation_windowed = correlation[start_idx:end_idx]
        
        # Trova il picco della correlazione
        peak_idx = np.argmax(correlation_windowed)
        delay_samples = peak_idx - max_samples
        
        # Converti in unità diverse
        delay_ms = (delay_samples / self.sr) * 1000
        delay_sec = delay_samples / self.sr
        
        # Lags array (per plotting)
        lags = (np.arange(-max_samples, max_samples + 1) / self.sr) * 1000  # in ms
        

        if plot:
            # plot del risultato 
            plt.figure(figsize=(6, 3))
            plt.plot(lags, correlation_windowed)
            plt.xlabel("Lag (ms)")
            plt.ylabel("Normalized Correlation")
            plt.grid()
            plt.show()

    
        return {
            'delay_samples': delay_samples,
            'delay_ms': delay_ms,
            'delay_sec': delay_sec,
            'correlation': correlation_windowed,
            'lags': lags,
            'peak_value': correlation_windowed[delay_samples + max_samples],
            'max_samples': max_samples # per indicizzare con TDOA esterna
        }
    
    def get_correlation_value(self, correlation: np.ndarray, 
                             lag_samples: float) -> float:
        """
        Restituisce il valore di correlazione per un lag (possibilmente frazionario).
        Usa interpolazione lineare per lag non-interi.
        """
        if lag_samples < 0 or lag_samples >= len(correlation):
            return 0.0
        
        lag_int = int(np.round(lag_samples))
        
        if abs(lag_samples - lag_int) < 1e-6:
            if 0 <= lag_int < len(correlation):
                return correlation[lag_int]
            else:
                return 0.0
        
        lag_floor = int(np.floor(lag_samples))
        lag_ceil = int(np.ceil(lag_samples))
        
        if lag_floor < 0 or lag_ceil >= len(correlation):
            if 0 <= lag_floor < len(correlation):
                return correlation[lag_floor]
            else:
                return 0.0
        
        frac = lag_samples - lag_floor

        


        return ((1 - frac) * correlation[lag_floor] + 
                frac * correlation[lag_ceil])


# ============================================================================
# SRP LOCALIZER
# ============================================================================

class SRPLocalizator:
    """
    Steered Response Power (SRP) Localizator usando GCC-PHAT.
    """
    
    def __init__(self, mic_array: MicrophoneArray, c: float = 343.0, reference_channel: Optional[int] = None):
        """
        Inizializza il localizzatore SRP.
        
        Args:
            mic_array: MicrophoneArray con posizioni e sample rate
            c: velocità del suono in m/s (default: 343 m/s a 20°C)
        """
        self.mic_array = mic_array
        self.c = c
        self.sr = mic_array.sample_rate
        self.correlator = GCCPHATCorrelator(self.sr)
        self.reference_channel = reference_channel
        
        self._pairwise_correlations = {}
        self._pitches = []

        # add margin to mic_array boundaries if defined
        if self.mic_array.boundaries is not None:
            self.mic_array.add_margin()

    
    def create_search_grid_full(self, resolution: float = 0.02) -> SearchGrid:
        """Crea una search grid che copre l'intero array di microfoni più margine."""
        
        x_range = (self.mic_array.boundaries[0], self.mic_array.boundaries[1]) # x_min to x_max
        y_range = (self.mic_array.boundaries[2], self.mic_array.boundaries[3]) # y_min to y_max
        
        return SearchGrid(x_range=x_range, y_range=y_range, resolution=resolution)
    
    def create_search_grid_centered_on_channel(self, 
                                              x_width: float = 0.2,
                                              y_height: float = 0.2,
                                              resolution: float = 0.02,
                                              reference_channel: Optional[int] = None) -> SearchGrid:
        """Crea una search grid centrata sulla posizione di un canale specifico."""
        
        if self.reference_channel is not None:
            reference_channel = self.reference_channel
        
        if reference_channel < 0 or reference_channel >= self.mic_array.n_mics:
            raise ValueError(
                f"reference_channel {reference_channel} non valido. "
                f"Microfoni disponibili: 0-{self.mic_array.n_mics - 1}"
            )
        
        ref_pos = self.mic_array.positions[reference_channel]
        ref_x, ref_y = ref_pos[0], ref_pos[1]
        
        x_range = (ref_x - x_width / 2, ref_x + x_width / 2)
        y_range = (ref_y - y_height / 2, ref_y + y_height / 2)

        # crop within boundaries
        x_range = (max(x_range[0], self.mic_array.boundaries[0]), 
                   min(x_range[1], self.mic_array.boundaries[1]))
        y_range = (max(y_range[0], self.mic_array.boundaries[2]), 
                   min(y_range[1], self.mic_array.boundaries[3]))
        
        return SearchGrid(x_range=x_range, y_range=y_range, resolution=resolution)
    
    
    def precompute_correlations(self, signals: np.ndarray, max_tau_ms: float = 100):
        """
        Precomputa GCC-PHAT per le coppie di microfoni.
        
        Se reference_channel è impostato, usa solo le coppie con quel canale.
        Altrimenti usa tutte le coppie.
        """
        print("Precomputing GCC-PHAT correlations...")
        
        if self.reference_channel is not None:
            pairs = self.mic_array.get_pairs_from_reference_channel(self.reference_channel)
            print(f"  Using reference channel {self.reference_channel}: {len(pairs)} pairs")
        else:
            pairs = self.mic_array.get_all_pairs()
            print(f"  Using all pairs: {len(pairs)} pairs")
        
        for i, j in pairs:
            sig_i = signals[:, i]
            sig_j = signals[:, j]
            
            # Corr sia (i,j) che (j,i)
            corr_result = self.correlator.compute(sig_i, sig_j, max_tau_ms, plot=False)
            
            self._pairwise_correlations[(i, j)] = corr_result
            
            
    
    def compute_theoretical_tdoa(self, point: np.ndarray, 
                                 mic_i: np.ndarray, mic_j: np.ndarray) -> float:
        """Calcola il ritardo temporale teorico (TDOA) tra due microfoni."""
        dist_i = np.linalg.norm(point - mic_i)
        dist_j = np.linalg.norm(point - mic_j)
        
        distance_diff = -(dist_j - dist_i)
        tdoa_sec = distance_diff / self.c
        
        return tdoa_sec
    
    
    def compute_tdoa_in_samples(self, point: np.ndarray, 
                               mic_i: np.ndarray, mic_j: np.ndarray) -> float:
        """Calcola il TDOA in campioni di segnale."""
        tdoa_sec = self.compute_theoretical_tdoa(point, mic_i, mic_j)
        tdoa_samples = tdoa_sec * self.sr
        # print(f"    TDOA samples: {tdoa_samples:.4f}")
        return tdoa_samples
    
    
    def evaluate_point(self, point):
        
        srp_power = 0.0

        for (i, j), corr_data in self._pairwise_correlations.items():
            mic_i, mic_j = self.mic_array.get_mic_pair(i, j)

            tdoa_samples = self.compute_tdoa_in_samples(point, mic_i, mic_j)

            # print(f"source coordinates: {point} mic i coordinates: {mic_i} mic j coordinates: {mic_j} delay in samples btw i and j: {tdoa_samples:.4f}")
            
            correlation = corr_data['correlation']

            # per indicizzare correttamente, shiftando l'indice
            max_samples = corr_data['max_samples']

            if -max_samples <= tdoa_samples <= max_samples:
                idx = tdoa_samples + max_samples
                # plot correlation con linea verticale in corrispondenza di lag_samples
                # plt.figure(figsize=(6, 3))
                # plt.plot(correlation, label='Correlation')
                # plt.axvline(x=idx, color='red', linestyle='--', label='Requested Lag')
                # plt.xlabel("Lag Samples")
                # plt.ylabel("Correlation Value")
                # plt.legend()
                # plt.grid()
                # plt.show()


                coherence = self.correlator.get_correlation_value(correlation, idx)
            else:
                coherence = np.nan  # fuori range
            
            srp_power += coherence
        
        return srp_power
    
    
    def localize(self, signals: np.ndarray, search_grid: SearchGrid, 
                 max_tau_ms: float = 100) -> LocalizationResult:
        """
        Esegue la localizzazione della sorgente sonora.
        
        Args:
            signals: array audio (n_samples, n_mics)
            search_grid: SearchGrid che definisce l'area di ricerca
            max_tau_ms: massimo ritardo da cercare in ms
            reference_channel: (opzionale) indice 0-based del canale da cui il whoop
                              è più forte. Se specificato, usa solo le coppie che
                              coinvolgono questo canale, riducendo il rumore.
        
        Returns:
            LocalizationResult con posizione stimata e mappa di potenza
        
        Esempio (localizzazione standard, tutte le coppie):
            >>> localizer = SRPLocalizator(mic_array)
            >>> grid = localizer.create_search_grid_full()
            >>> result = localizer.localize(signals, grid)
        
        Esempio (localizzazione focalizzata sul canale 4):
            >>> localizer = SRPLocalizator(mic_array)
            >>> grid = localizer.create_search_grid_centered_on_channel(4)
            >>> result = localizer.localize(signals, grid, reference_channel=4)
        """
        
        self.precompute_correlations(signals, max_tau_ms)
        
        grid_points = search_grid.generate_grid()
        n_points = len(grid_points)
        
        print(f"\nSearching grid ({n_points} points)...")
        power_map = np.zeros(n_points)
        
        for idx, point in enumerate(grid_points):
            power_map[idx] = self.evaluate_point(point)
            
            if (idx + 1) % max(1, n_points // 10) == 0:
                print(f"  Progress: {idx + 1}/{n_points}")
        
        max_idx = np.argmax(power_map)
        estimated_position = grid_points[max_idx]
        max_power = power_map[max_idx]
        
        return LocalizationResult(
            estimated_position=estimated_position,
            power_map=power_map,
            grid_points=grid_points,
            max_power=max_power
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def reshape_power_map(result: LocalizationResult, 
                     search_grid: SearchGrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshapa la power map da 1D a 2D per visualizzazione."""
    x_min, x_max = search_grid.x_range
    y_min, y_max = search_grid.y_range
    
    x = np.arange(x_min, x_max + search_grid.resolution, search_grid.resolution)
    y = np.arange(y_min, y_max + search_grid.resolution, search_grid.resolution)
    
    X, Y = np.meshgrid(x, y)
    Z = result.power_map.reshape(X.shape)
    
    return X, Y, Z


def plot_power_map_2d(result: LocalizationResult, 
                      mic_array: MicrophoneArray,
                      search_grid: SearchGrid,
                      figsize: Tuple[int, int] = (12, 8),
                      cmap: str = 'viridis',
                      show_colorbar: bool = True, 
                      ground_truth: Tuple[float, float] = None) -> None:
    """
    Visualizza la power map 2D con i microfoni sovrapposti.
    
    Mostra la mappa di potenza SRP come heatmap 2D, con i microfoni come
    pallini rossi alle loro coordinate reali e la posizione stimata della
    sorgente come stella gialla.
    
    Args:
        result: LocalizationResult della localizzazione
        mic_array: MicrophoneArray con le posizioni dei microfoni
        search_grid: SearchGrid usata nella localizzazione
        figsize: dimensioni della figura (larghezza, altezza)
        cmap: colormap per la heatmap (default: 'viridis')
        show_colorbar: se True, mostra la barra di colori
    
    Example:
        >>> localizer = SRPLocalizator(mic_array)
        >>> grid = localizer.create_search_grid_full()
        >>> result = localizer.localize(signals, grid, reference_channel=4)
        >>> plot_power_map_2d(result, mic_array, grid)
        >>> plt.show()
    """
    # Reshapa la power map per visualizzazione
    X, Y, Z = reshape_power_map(result, search_grid)
    
    # Crea la figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot della heatmap
    contour = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
    
    # Aggiungi contour lines per migliore leggibilità
    contour_lines = ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8)
    
    # Plotta i microfoni come pallini rossi
    mic_positions = mic_array.positions
    for mic_pos in mic_positions:
        ax.scatter(mic_pos[0], mic_pos[1], 
            color='red', s=100, marker='o', edgecolors='darkred', 
            linewidths=2, label='Microphones', zorder=5)

    # Annotazioni dei numeri dei canali (1-based per user-friendliness)
    for idx, pos in enumerate(mic_positions):
        if pos[0] == 0.0 and pos[1] == 0.0:
            continue  # Salta microfoni rotti
        else:
            ax.annotate(f'{idx+1}', xy=pos, xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, 
                   color='darkred', fontweight='bold')
    
    # Plotta la posizione stimata come stella gialla
    est_pos = result.estimated_position
    ax.scatter(est_pos[0], est_pos[1], 
               color='yellow', s=300, marker='*', edgecolors='gold', 
               linewidths=2, label='Estimated Position', zorder=6)
    
    # Plotta ground truth se disponibile
    if ground_truth is not None:
        ax.scatter(ground_truth[0], ground_truth[1],
                   color='cyan', s=200, marker='X', edgecolors='deepskyblue',
                   linewidths=2, label='Ground Truth', zorder=6)
        
    # Aggiunge la barra di colori
    if show_colorbar:
        cbar = plt.colorbar(contour, ax=ax, label='SRP Power')
    
    # Etichette e titolo
    ax.set_xlabel('X [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
    ax.set_title(f'SRP Power Map - Estimated Position: ({est_pos[0]:.4f}, {est_pos[1]:.4f})\nMax Power: {result.max_power:.6f}', 
                fontsize=14, fontweight='bold')
    
    # Griglia e aspetto
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    # ax.legend(loc='upper left', bbox_to_anchor=(-0.5, 1), fontsize=10)
    
    # Layout ottimizzato
    plt.tight_layout()

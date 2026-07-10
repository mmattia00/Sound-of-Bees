"""
SRP localizer based on GCC-PHAT.

This module implements a two-dimensional sound-source localizer based on
Steered Response Power (SRP) and GCC-PHAT (Generalized Cross-Correlation
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
    """Represent a microphone array with 2D positions."""
    positions: np.ndarray
    sample_rate: int
    validated_channels: List[int] | None = None
    boundaries: Tuple[float, float, float, float] | None = None  # (x_min, x_max, y_min, y_max)
    margin: float = 0.05 # extra margin for the search grid

    def __post_init__(self):
        if self.validated_channels is None:
            self.validated_channels = []
        else:
            self.validated_channels = list(self.validated_channels)

    @property
    def n_mics(self) -> int:
        """Number of microphones."""
        return self.positions.shape[0]
    
    def add_margin(self) -> None:
        """Add margin to the search boundaries."""
        if self.boundaries is not None:
            x_min, x_max, y_min, y_max = self.boundaries
            self.boundaries = (x_min - self.margin, x_max + self.margin,
                               y_min - self.margin, y_max + self.margin)
        else:
            warnings.warn("Boundaries are not defined; cannot add margin.")

    def get_mic_pair(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the positions of two microphones."""
        return self.positions[i], self.positions[j]

    def get_all_pairs(self) -> List[Tuple[int, int]]:
        """
        Return the indices of all microphone pairs that involve ONLY
        validated channels.
        
        If validated_channels = [0, 1, 3, 5], return all pairs composed
        exclusively of these channels:
        [(0, 1), (0, 3), (0, 5), (1, 3), (1, 5), (3, 5)]
        
        Channels NOT in validated_channels are completely ignored.
        
        Returns:
            List of tuples (i, j) where i < j and BOTH i and j are in validated_channels
        """
        # If there are no validated channels, return an empty list.
        if not self.validated_channels:
            return []
        
        # Convert to a set for O(1) membership checks.
        valid_set = set(self.validated_channels)
        
        pairs: List[Tuple[int, int]] = []
        
        # Iterate over all validated channels.
        for i in self.validated_channels:
            # For each validated channel, pair it with channels of higher index.
            for j in self.validated_channels:
                if i < j:
                    pairs.append((i, j))
        
        return pairs

    def get_pairs_from_reference_channel(self, reference_channel: int) -> List[Tuple[int, int]]:
        """
        Return microphone pairs that involve a specific reference channel.
        
        Useful for focused localization when the strongest channel has already
        been identified. It reduces noise by using only the pairs related to
        the channel of interest.
        
        Args:
            reference_channel: 0-based index of the reference channel
        
        Returns:
            List of tuples (i, j) where at least one of i or j is reference_channel,
            excluding broken channels
        
        Raises:
            ValueError: if reference_channel is invalid
        
        Example:
            >>> pairs = mic_array.get_pairs_from_reference_channel(3)
            >>> # Returns: [(3, 0), (3, 1), (3, 2), (3, 4), (3, 5), ..., (3, 15)]
        """
        if reference_channel < 0 or reference_channel >= self.n_mics:
            raise ValueError(
                f"reference_channel {reference_channel} is invalid. "
                f"Available microphones: 0-{self.n_mics - 1}"
            )
        
        broken = set(self.channel_broken or [])
        
        if reference_channel in broken:
            warnings.warn(
                f"reference_channel {reference_channel} is marked as broken. "
                "Continuing anyway, but the results may be unreliable."
            )
        
        pairs: List[Tuple[int, int]] = []
        
        for i in range(self.n_mics):
            if i == reference_channel or i in broken:
                continue
            
            pairs.append((reference_channel, i))
        
        return pairs
    
    def get_only_cross_audio_interface_pairs(self) -> List[Tuple[int, int]]:
        pairs = self.get_all_pairs()
        print("Number of pairs:", len(pairs))
        print("First 20 pairs:", pairs[:20])

        # Count cross-board pairs.
        cross_pairs = [(i, j) for (i, j) in pairs if (i < 16 and j >= 16) or (i >= 16 and j < 16)]
        print("Cross-board pairs:", len(cross_pairs))
        return cross_pairs


@dataclass
class SearchGrid:
    """Define the search grid for localization."""
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    resolution: float
    
    def generate_grid(self) -> np.ndarray:
        """Generate the grid points."""
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        
        x = np.arange(x_min, x_max + self.resolution, self.resolution)
        y = np.arange(y_min, y_max + self.resolution, self.resolution)
        
        xx, yy = np.meshgrid(x, y)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        return grid_points
        


@dataclass
class LocalizationResult:
    """Localization result."""
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
    Compute the GCC-PHAT (Generalized Cross-Correlation with Phase Transform).
    
    GCC-PHAT is robust because it uses phase only and ignores amplitude
    differences between signals.
    """
    
    def __init__(self, sr: int):
        """Initialize the correlator."""
        self.sr = sr
    
    def compute(self, sig1: np.ndarray, sig2: np.ndarray, 
                max_tau_ms: float = 100, plot: bool = False) -> Dict:
        """Compute GCC-PHAT between two signals."""
        # FFT length with zero-padding to reduce aliasing.
        fft_len = 2 * max(len(sig1), len(sig2))
        
        # FFT of both signals.
        X1 = np.fft.rfft(sig1, n=fft_len)
        X2 = np.fft.rfft(sig2, n=fft_len)
        
        # Cross-spectrum.
        Gxy = X1 * np.conj(X2)
        
        # GCC-PHAT normalization by magnitude (spectral whitening).
        mag = np.abs(Gxy)
        mag[mag < 1e-10] = 1e-10
        Gxy_phat = Gxy / mag
        
        # Inverse FFT to obtain the generalized correlation.
        correlation = np.fft.irfft(Gxy_phat, n=fft_len)
        correlation = np.fft.fftshift(correlation)
        
        # Limit the delay range to the search interval.
        max_samples = int((max_tau_ms / 1000) * self.sr)
        center = len(correlation) // 2
        start_idx = center - max_samples
        end_idx = center + max_samples + 1
        correlation_windowed = correlation[start_idx:end_idx]
        
        # Find the correlation peak.
        peak_idx = np.argmax(correlation_windowed)
        delay_samples = peak_idx - max_samples
        
        # Convert to different units.
        delay_ms = (delay_samples / self.sr) * 1000
        delay_sec = delay_samples / self.sr
        
        # Lag array for plotting.
        lags = (np.arange(-max_samples, max_samples + 1) / self.sr) * 1000  # in ms
        

        if plot:
            # Plot the result.
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
            'max_samples': max_samples # for indexing with an external TDOA
        }
    
    def get_correlation_value(self, correlation: np.ndarray, 
                             lag_samples: float) -> float:
        """
        Return the correlation value for a lag (possibly fractional).
        Use linear interpolation for non-integer lags.
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
    Steered Response Power (SRP) localizer using GCC-PHAT.
    """
    
    def __init__(self, mic_array: MicrophoneArray, c: float = 343.0, reference_channel: Optional[int] = None):
        """
        Initialize the SRP localizer.
        
        Args:
            mic_array: MicrophoneArray with positions and sample rate
            c: speed of sound in m/s (default: 343 m/s at 20°C)
        """
        self.mic_array = mic_array
        self.c = c
        self.sr = mic_array.sample_rate
        self.correlator = GCCPHATCorrelator(self.sr)
        self.reference_channel = reference_channel
        
        self._pairwise_correlations = {}
        self._pitches = []

        # Add margin to the microphone-array boundaries if defined.
        if self.mic_array.boundaries is not None:
            self.mic_array.add_margin()

    
    def create_search_grid_full(self, resolution: float = 0.02) -> SearchGrid:
        """Create a search grid that covers the full microphone array plus margin."""
        
        x_range = (self.mic_array.boundaries[0], self.mic_array.boundaries[1]) # x_min to x_max
        y_range = (self.mic_array.boundaries[2], self.mic_array.boundaries[3]) # y_min to y_max
        
        return SearchGrid(x_range=x_range, y_range=y_range, resolution=resolution)
    
    def create_search_grid_centered_on_channel(self, 
                                              x_width: float = 0.2,
                                              y_height: float = 0.2,
                                              resolution: float = 0.02,
                                              reference_channel: Optional[int] = None) -> SearchGrid:
        """Create a search grid centered on the position of a specific channel."""
        
        if self.reference_channel is not None:
            reference_channel = self.reference_channel
        
        if reference_channel < 0 or reference_channel >= self.mic_array.n_mics:
            raise ValueError(
                f"reference_channel {reference_channel} is invalid. "
                f"Available microphones: 0-{self.mic_array.n_mics - 1}"
            )
        
        ref_pos = self.mic_array.positions[reference_channel]
        ref_x, ref_y = ref_pos[0], ref_pos[1]
        
        x_range = (ref_x - x_width / 2, ref_x + x_width / 2)
        y_range = (ref_y - y_height / 2, ref_y + y_height / 2)

        # Crop to the configured search boundaries.
        x_range = (max(x_range[0], self.mic_array.boundaries[0]), 
                   min(x_range[1], self.mic_array.boundaries[1]))
        y_range = (max(y_range[0], self.mic_array.boundaries[2]), 
                   min(y_range[1], self.mic_array.boundaries[3]))
        
        return SearchGrid(x_range=x_range, y_range=y_range, resolution=resolution)
    
    
    def precompute_correlations(self, signals: np.ndarray, max_tau_ms: float = 100):
        """
        Precompute GCC-PHAT for the microphone pairs.
        
        If reference_channel is set, use only the pairs with that channel.
        Otherwise use all pairs.
        """
        print("Precomputing GCC-PHAT correlations...")
        
        if self.reference_channel is not None:
            pairs = self.mic_array.get_pairs_from_reference_channel(self.reference_channel)
            print(f"  Using reference channel {self.reference_channel}: {len(pairs)} pairs")
        else:
            pairs = self.mic_array.get_all_pairs()
            # pairs = self.mic_array.get_only_cross_audio_interface_pairs()
            print(f"  Using all pairs: {len(pairs)} pairs")
        
        for i, j in pairs:
            sig_i = signals[:, i]
            sig_j = signals[:, j]
            
            # Store the correlation for the (i, j) pair.
            corr_result = self.correlator.compute(sig_i, sig_j, max_tau_ms, plot=False)
            
            self._pairwise_correlations[(i, j)] = corr_result
            
            
    
    def compute_theoretical_tdoa(self, point: np.ndarray, 
                                 mic_i: np.ndarray, mic_j: np.ndarray) -> float:
        """Compute the theoretical time delay (TDOA) between two microphones."""
        dist_i = np.linalg.norm(point - mic_i)
        dist_j = np.linalg.norm(point - mic_j)
        
        distance_diff = -(dist_j - dist_i)
        tdoa_sec = distance_diff / self.c
        
        return tdoa_sec
    
    
    def compute_tdoa_in_samples(self, point: np.ndarray, 
                               mic_i: np.ndarray, mic_j: np.ndarray) -> float:
        """Compute the TDOA in signal samples."""
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

            # Shift the index so the center of the correlation is zero-delay.
            max_samples = corr_data['max_samples']

            if -max_samples <= tdoa_samples <= max_samples:
                idx = tdoa_samples + max_samples
                # Plot the correlation with a vertical line at the requested lag.
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
                coherence = np.nan  # out of range
            
            srp_power += coherence
        
        return srp_power
    
    
    def localize(self, signals: np.ndarray, search_grid: SearchGrid, 
                 max_tau_ms: float = 100) -> LocalizationResult:
        """
        Run sound-source localization.
        
        Args:
            signals: audio array (n_samples, n_mics)
            search_grid: SearchGrid defining the search area
            max_tau_ms: maximum delay to search in ms
            reference_channel: (optional) 0-based index of the channel where the whoop
                              is strongest. If specified, use only pairs involving
                              this channel to reduce noise.
        
        Returns:
            LocalizationResult with estimated position and power map
        
        Example (standard localization, all pairs):
            >>> localizer = SRPLocalizator(mic_array)
            >>> grid = localizer.create_search_grid_full()
            >>> result = localizer.localize(signals, grid)
        
        Example (focused localization on channel 4):
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
    """Reshape the power map from 1D to 2D for visualization."""
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
                      cmap: str = 'viridis',
                      show_colorbar: bool = True, 
                      ground_truth: Tuple[float, float] = None,
                      **figures_characteristics) -> None:
    """
    Visualize the 2D power map with microphones overlaid.
    
    Show the SRP power map as a 2D heatmap, with the microphones as red
    dots at their real coordinates and the estimated source position as a
    yellow star.
    
    Args:
        result: LocalizationResult from localization
        mic_array: MicrophoneArray with microphone positions
        search_grid: SearchGrid used in localization
        figsize: figure size (width, height)
        cmap: colormap per la heatmap (default: 'viridis')
        show_colorbar: if True, show the color bar
    
    Example:
        >>> localizer = SRPLocalizator(mic_array)
        >>> grid = localizer.create_search_grid_full()
        >>> result = localizer.localize(signals, grid, reference_channel=4)
        >>> plot_power_map_2d(result, mic_array, grid)
        >>> plt.show()
    """
    # Reshape the power map for visualization.
    X, Y, Z = reshape_power_map(result, search_grid)
    
    # Create the figure.
    fig, ax = plt.subplots(figsize=figures_characteristics.get('fig_size', (10, 6)))
    
    # Plot the heatmap.
    contour = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
    
    # Add contour lines for better readability.
    contour_lines = ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8)
    
    # Plot microphones as red dots.
    mic_positions = mic_array.positions
    for mic_pos in mic_positions:
        ax.scatter(mic_pos[0], mic_pos[1], 
            color='red', s=100, marker='o', edgecolors='darkred', 
            linewidths=2, label='Microphones', zorder=5)

    # Annotate channel numbers (1-based for user-friendliness).
    for idx, pos in enumerate(mic_positions):
        if pos[0] == 0.0 and pos[1] == 0.0:
            continue  # Skip broken microphones.
        else:
            ax.annotate(f'{idx+1}', xy=pos, xytext=(5, 5), 
                   textcoords='offset points', fontsize=8, 
                   color='darkred', fontweight='bold')
    
    # Plot the estimated position as a yellow star.
    est_pos = result.estimated_position
    ax.scatter(est_pos[0], est_pos[1], 
               color='yellow', s=300, marker='*', edgecolors='gold', 
               linewidths=2, label='Estimated Position', zorder=6)
    
    # Plot the ground truth if available.
    if ground_truth is not None:
        ax.scatter(ground_truth[0], ground_truth[1],
                   color='cyan', s=200, marker='X', edgecolors='deepskyblue',
                   linewidths=2, label='Ground Truth', zorder=6)
        
    # Add the color bar.
    if show_colorbar:
        cbar = plt.colorbar(contour, ax=ax, label='SRP Power')
        cbar.set_label(
                'SRP Power', 
                size=figures_characteristics.get('colorbar_labelsize', 12),
                weight='bold',
                labelpad=15  # label spacing (points)
            )

        # Set the color-bar tick font size.
        cbar.ax.tick_params(labelsize=figures_characteristics.get('colorbar_ticksize', 12))
    
    # Labels and title.
    ax.set_xlabel('X (m)', fontsize=figures_characteristics.get('label_fontsize', 12))
    ax.set_ylabel('Y (m)', fontsize=figures_characteristics.get('label_fontsize', 12))
    ax.set_title(f'SRP Power Map\nEstimated Position: ({est_pos[0]:.4f}, {est_pos[1]:.4f}) m', 
                fontsize=figures_characteristics.get('title_fontsize', 14), fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=figures_characteristics.get('ticks_fontsize', 12))

    # Temporary: limit the x-axis to 0.21 m to focus on the first 16 channels.
    # ax.set_xlim(left=0.0, right=0.21)
    # Grid and aspect ratio.
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Optimized layout.
    plt.tight_layout()

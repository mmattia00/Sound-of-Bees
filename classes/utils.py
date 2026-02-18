import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Optional
import sounddevice as sd
import soundfile as sf

DEFAULT_FIGURE_CHARACTERISTICS = {
    'fig_size': (12, 5),
    'label_fontsize': 14,
    'title_fontsize': 16,
    'legend_fontsize': 12,
    'tick_fontsize': 14,
    'colorbar_labelsize': 14,
    'colorbar_ticksize': 12,
}

    # Canali rotti
BROKEN_CHANNELS = [2, 3, 8, 13, 21, 25, 27, 28, 31]
BROKEN_CHANNELS_ZERO_BASED = [x - 1 for x in BROKEN_CHANNELS]  # zero-based index

# funzione helper privata
def _create_hexagon(center_x: float, center_y: float, radius: float) -> np.ndarray:
    """Crea un esagono regolare centrato in (center_x, center_y)"""
    angles = np.linspace(0, 2*np.pi, 7)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    return np.column_stack([x, y])


class Utils:

    _create_hexagon = staticmethod(_create_hexagon)  

    @staticmethod
    def load_coordinates_with_labels(filename: str):
        """
        Legge un file di testo nel formato:
        x, y, # commento

        La prima riga contiene i boundaries nel commento, del tipo:
        0.0, 0.205, 0.0, 0.32 # Boundaries: (x_min, x_max, y_min, y_max)

        Ritorna:
        - coords: array numpy strutturato con campi 'x', 'y', 'label'
        - boundaries: [x_max, x_min, y_max, y_min]
        """
        coords = []
        labels = []
        boundaries = None

        with open(filename, "r", encoding="utf-8") as f:
            # Leggi e parse della prima riga (boundaries)
            first_line = f.readline()
            second_line = f.readline()
            if "#" in second_line and "Boundaries" in second_line:
                data_part, comment_part = second_line.split("#", 1)
                data_part = data_part.strip()
                parts = [p.strip() for p in data_part.split(",")]

                if len(parts) == 4:
                    x_min, x_max, y_min, y_max = map(float, parts)
                    boundaries = [x_min, x_max, y_min, y_max]

            # Parse delle righe successive (canali)
            for line in f:
                if "#" in line:
                    data_part, comment_part = line.split("#", 1)
                    label = comment_part.strip()
                else:
                    data_part = line
                    label = ""

                data_part = data_part.strip()
                if not data_part:
                    continue

                parts = [p.strip() for p in data_part.split(",")]
                if len(parts) < 2:
                    continue

                x = float(parts[0])
                y = float(parts[1])
                coords.append((x, y))
                labels.append(label)

        coords = np.array(coords)
        labels = np.array(labels, dtype="U64")
        return coords, boundaries, labels
    


    @staticmethod
    def plot_hexagon_hnr_map(
        hnr_levels: np.ndarray,
        mic_positions: np.ndarray,
        channels_of_interest: List[int],
        broken_channels: Optional[List[int]] = None,
        hexagon_radius: float = 0.025,
        use_db: bool = False,
        cmap: str = "hot",
        boundaries: Optional[List[float]] = None,
        show_colorbar: bool = True,
        precise_localization: Optional[tuple] = None,  # ← (x, y) coordinate
        **figures_characteristics
    ) -> None:
        """
        Plotta esagoni regolari attorno ai microfoni, colorati in base ai livelli di HNR.
        
        Args:
            hnr_levels: Array di valori HNR per canale (shape: len(channels_of_interest))
            mic_positions: Coordinate COMPLETE dei microfoni (shape: [N, 2])
            channels_of_interest: Lista indici canali da plottare
            broken_channels: Lista indici canali rotti
            hexagon_radius: Raggio dell'esagono
            use_db: Se True, usa scala dB per HNR
            cmap: Colormap da usare
            boundaries: Limiti del grafico [xmin, xmax, ymin, ymax]
            show_colorbar: Se mostrare la colorbar
            precise_localization: Coordinate (x, y) per plottare posizione precisa come stella
            **figures_characteristics: Configurazione font/dimensioni
        """
        hnr_levels = np.nan_to_num(hnr_levels, nan=0.0)

        if broken_channels is None:
            broken_channels = []
        
        # Filtra posizioni microfoni
        mic_positions_filtered = mic_positions[channels_of_interest]
        
        fig, ax = plt.subplots(figsize=figures_characteristics.get('fig_size', (12, 10)))
        
        if boundaries is not None:
            ax.set_xlim(boundaries[0], boundaries[1])
            ax.set_ylim(boundaries[2], boundaries[3])
        
        # Broken mask + HNR processing
        broken_mask = np.array([ch in broken_channels for ch in channels_of_interest])
        
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
        
        # Disegna esagoni
        for i, (x, y) in enumerate(mic_positions_filtered):
            hexagon = _create_hexagon(x, y, hexagon_radius)
            
            if broken_mask[i]:
                color_hexagon = 'darkgray'
                alpha_val = 0.9
                hatch_pattern = '///'
                edgecolor_hex = 'red'
                linewidth_hex = 2.5
            else:
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
        mic_colors = ['red' if broken_mask[i] else 'limegreen' 
                    for i in range(len(mic_positions_filtered))]
        mic_sizes = [160 if broken_mask[i] else 120 
                    for i in range(len(mic_positions_filtered))]
        
        ax.scatter(mic_positions_filtered[:, 0], mic_positions_filtered[:, 1], 
                c=mic_colors, s=mic_sizes, edgecolor='black', linewidth=1, 
                zorder=10)
        
        # Numeri canale dentro ai puntini
        for i, (x, y) in enumerate(mic_positions_filtered):
            ax.text(x, y, str(channels_of_interest[i] + 1), 
                    ha='center', va='center', fontsize=7, 
                    fontweight='bold', color='black', zorder=11)
        
        # ========== STELLA PER LOCALIZZAZIONE PRECISA ==========
        if precise_localization is not None:
            x_loc, y_loc = precise_localization
            ax.scatter(x_loc, y_loc, 
                    marker='*',  # Stella
                    s=500,  # Dimensione grande
                    c='gold',  # Colore oro
                    edgecolor='black',  # Bordo nero
                    linewidth=2,  # Spessore bordo
                    zorder=15,  # Sopra tutto
                    label='Precise localization')
        
        # Colorbar
        if show_colorbar:
            cbar = fig.colorbar(scalar_map, ax=ax, shrink=0.8)
            label = 'HNR (dB)' if use_db else 'HNR'
            cbar.set_label(
                label, 
                size=figures_characteristics.get('colorbar_labelsize', 12),
                weight='bold',
                labelpad=15
            )
            cbar.ax.tick_params(labelsize=figures_characteristics.get('colorbar_ticksize', 12))
        
        # Legenda
        legend_elements = [
            Patch(facecolor='darkgray', alpha=0.9, hatch='///', 
                edgecolor='red', linewidth=2, label='Broken channel'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', 
                    markersize=12, label='Working channel', 
                    markeredgecolor='black', markeredgewidth=1.5)
        ]
        
        # Aggiungi stella alla legenda solo se presente
        if precise_localization is not None:
            legend_elements.append(
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                        markersize=20, label='Precise localization',
                        markeredgecolor='black', markeredgewidth=2)
            )
        
        ax.legend(handles=legend_elements, 
          loc='upper left',  # ← Ancoraggio della legenda
          bbox_to_anchor=(-0.6, 1),  # ← (x, y) relativo all'asse
          fontsize=figures_characteristics.get('legend_fontsize', 11),
          frameon=True, 
          fancybox=True, 
          shadow=True)
        
        # Impostazioni grafiche
        ax.set_title("Strong Channel Detection Map", 
                    fontsize=figures_characteristics.get('title_fontsize', 14), 
                    fontweight='bold')
        ax.tick_params(axis='both', which='major', 
                    labelsize=figures_characteristics.get('tick_fontsize', 14))
        ax.set_ylabel('Y (m)', fontsize=figures_characteristics.get('label_fontsize', 12))
        ax.set_xlabel('X (m)', fontsize=figures_characteristics.get('label_fontsize', 12))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()


    @staticmethod
    def extract_and_play_audio_segment_from_raw_multichannel_audio(raw_audio_path: str, start_time:float, end_time:float, channel:int, lenght_extension: Optional[float] = 1.0) -> None:
        """
        Estrae un segmento audio da un file raw e lo riproduce.
        
        Args:
            raw_audio_path: Percorso al file audio raw
            start_time: Tempo di inizio in secondi
            end_time: Tempo di fine in secondi
        """

        multichannel_audio, sr = sf.read(raw_audio_path)
        start_sample = max(0, int((start_time - lenght_extension/2) * sr))
        end_sample = min(multichannel_audio.shape[0], int((end_time + lenght_extension/2) * sr))

        # # ========== DEBUG ==========
        # duration_sec = (end_sample - start_sample) / sr
        # print(f"   - Finestra: [{start_time - 0.5:.3f}s, {end_time + 0.5:.3f}s]")
        # print(f"   - Samples: [{start_sample}, {end_sample}]")
        # print(f"   - Durata: {duration_sec:.3f}s ({end_sample - start_sample} samples)")
        # print(f"   - Canale: {channel} (zero-based index)")
        # print(f"   - Sample rate: {sr} Hz")
        # print(f"file path: {raw_audio_path}")
        # # ===========================

        audio_segment = multichannel_audio[start_sample:end_sample, channel]
        
        # Normalizza per evitare clipping
        max_val = np.max(np.abs(audio_segment))
        if max_val > 0:
            audio_segment = audio_segment / (max_val * 1.1)

        sd.play(audio_segment, sr)
        sd.wait()
        sd.sleep(500)  # Piccola pausa dopo la riproduzione

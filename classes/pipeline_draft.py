from os import times
from unittest import result
from scipy import signal
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from classes.pitch_detector import PitchDetector
from classes.harmonics_analyzer import HarmonicsAnalyzer
from classes.strong_channel_detector import StrongChannelDetector
from classes.whoop_localizator import MicrophoneArray, SRPLocalizator, plot_power_map_2d
from classes.utils import Utils
import numpy as np
import sounddevice as sd
import h5py
import re


@dataclass
class WhoopFeatures:
    """Contenitore risultati feature extraction."""
    ch: int = 0 # canale (1-based index)
    peak_time: float = 0.0 # sec
    start_peak: float = 0.0 # sec
    end_peak: float = 0.0 # sec
    rough_duration: float = 0.0 # sec
    spectrogram_dB: Optional[np.ndarray] = None
    spec_frequencies: Optional[np.ndarray] = None  # frequenze dello spettrogramma
    spec_times: Optional[np.ndarray] = None  # tempi dello spettrogramma
    date: Optional[str] = None  # "YYYY-MM-DD"
    time: Optional[str] = None  # "HH:MM:SS.ssssss"
    sr: Optional[int] = None  # sample rate

    # Pitch analysis
    f0_mean: Optional[float] = None # Hz
    precise_start_peak: Optional[float] = None # sec
    precise_end_peak: Optional[float] = None # sec
    precise_duration: Optional[float] = None # sec

    # Harmonics analysis
    weighted_shr: Optional[float] = None  # Weighted Harmonic Acoustic Ratio (WHAR)
    max_aligned_peaks: Optional[int] = None  # numero massimo di picchi armonici allineati

    # Rough localization analysis
    strongest_channel: Optional[int] = None  # canale più forte rilevato attorno a ch (spesso è ch stesso) (1-based index)
    hnr_levels: Optional[np.array] = None  # array dei livelli HNR di tutti i canali analizzati attorno a quello che ha catturato il candidato whoop (per ora 0:15 o 16:31 perchè i dati non sono allineati successivamente 0:32)
    num_channels_with_whoop: Optional[int] = None  # numero di canali con whoop rilevato attorno a ch

    # Precise localization analysis
    precise_localization: Optional[np.ndarray] = None  # [x, y] in metri

class PipelineDraft:
    def __init__(self, whoop_candidate_filename: str, parent_filename: str, root_raw_audio_dir: str, mics_coords: np.ndarray = None, frame_boundaries: list = None, channel_broken: list = []):
        
        self.whoop_candidate_filename = whoop_candidate_filename # es. audio_recording_2025-09-15T06_34_43.397145Z_ch_2_peak_12.345_start_11.845_end_12.845_hnr_15.2.wav
        self.parent_filename = parent_filename # cartella che contiene il candidato whoop es. audio_recording_2025-09-15T06_34_43.397145Z
        self.root_raw_audio_dir = root_raw_audio_dir  # es. E:/soundofbees
        
        # Parametri estratti (lazy)
        self._params = None

        # Audio (lazy loaded)
        self.whoop_audio = None
        self.multichannel_audio = None
        self.sr = None

        self.mics_coords = mics_coords # coordinate dei 32 microfoni
        self.frame_boundaries = frame_boundaries # boundaries del frame alveare (xmin, xmax, ymin, ymax)
        self.channel_broken = channel_broken  # lista canali rotti da ignorare (0-based index)
        
        # Risultati feature (lazy computed)
        self.features = WhoopFeatures()
        

    @property
    def params(self):
        """Lazy parse filename → attributi ch, start_peak, ecc."""
        if self._params is None:
            parts = self.whoop_candidate_filename.replace('.wav', '').split('_')

            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', self.parent_filename)
            time_match = re.search(r'T(\d{2})_(\d{2})_(\d{2})\.(\d+)', self.parent_filename)

            date = date_match.group(1)  # "2025-09-15"
            time = f"{time_match.group(1)}:{time_match.group(2)}:{time_match.group(3)}.{time_match.group(4)}"  # "06:40:43.498109"

            self._params = {
                'ch': int(parts[1]),
                'peak_time': float(parts[3]),
                'start_peak': float(parts[5]),
                'end_peak': float(parts[7]),
                'hnr_db': float(parts[9]),
                'date': date,
                'time': time
            }
        return self._params
    
    
    def store_feaures_from_params(self):
        self.features.ch = self.params['ch']
        self.features.peak_time = self.params['peak_time']
        self.features.start_peak = self.params['start_peak']
        self.features.end_peak = self.params['end_peak']
        self.features.date = self.params['date']
        self.features.time = self.params['time']
    def load_audios(self):
            
        raw_path = f"{self.root_raw_audio_dir}/{self.parent_filename}.wav"
        self.multichannel_audio, self.sr = sf.read(raw_path)  # (samples, ch)
        
        start_sample = int(self.params['start_peak'] * self.sr)
        end_sample = int(self.params['end_peak'] * self.sr)
        
        ch_idx = self.params['ch'] - 1
        self.whoop_audio = self.multichannel_audio[start_sample:end_sample, ch_idx]  # 1D array
        
        self.features.rough_duration = len(self.whoop_audio) / self.sr
        self.features.sr = self.sr  # salva sample rate nei features per uso futuro

    def compute_spectrogram(self, plot: bool = False, title: str = "", **figures_characteristics):
        """Calcola lo spettrogramma una sola volta"""
        nperseg = 1024
        noverlap = nperseg // 2
        nfft = nperseg * 4
        window = 'hann'
        
        frequencies, times, spectrogram_data = signal.spectrogram(
            self.whoop_audio, 
            fs=self.sr,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window=window,
            scaling='density'
        )

        self.features.spectrogram_dB = 20 * np.log10(spectrogram_data + 1e-10)
        self.features.spec_frequencies = frequencies
        self.features.spec_times = times
        
        # Converti in dB con guadagno di 20
        spectrogram_db = 20 * np.log10(spectrogram_data + 1e-10)

        if plot:
            plt.figure(figsize=figures_characteristics.get('fig_size', (10, 6)))

            vmin = np.nanmax(spectrogram_db) - 80
            vmax = np.nanmax(spectrogram_db)

            im = plt.pcolormesh(times, frequencies, spectrogram_db,
                                shading='gouraud', cmap='hot', vmin=vmin, vmax=vmax)

            plt.ylabel('Frequency (Hz)', fontsize=figures_characteristics.get('label_fontsize', 12))
            plt.xlabel('Time (s)', fontsize=figures_characteristics.get('label_fontsize', 12))
            plt.title(title, fontsize=figures_characteristics.get('title_fontsize', 14), fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=figures_characteristics.get('ticks_fontsize', 12))
            cbar = plt.colorbar(im, label='Power (dB)')

            # Imposta fontsize label colorbar
            # CORRETTO: labelpad invece di pad
            cbar.set_label(
                'Power (dB)', 
                size=figures_characteristics.get('colorbar_labelsize', 12),
                weight='bold',
                labelpad=15  # ✅ Spaziatura label (punti)
            )

            # Imposta fontsize numeri colorbar
            cbar.ax.tick_params(labelsize=figures_characteristics.get('colorbar_ticksize', 12))


            # qui aumenti i numerini degli assi
            plt.ylim([0, 20000])
            plt.xlim([times[0], times[-1]])
            plt.show()

        

    def extract_pitch_features(self, plot: bool = False, verbose: bool = False, **figures_characteristics) -> Dict[str, Any]:
        """PitchDetector → features."""
        detector = PitchDetector(self.whoop_audio, self.sr)
        f0, best_queue_dic, all_queues, all_queues_f0, whoop_pitch_info = detector.estimate_f0(
            length_queue=5, hz_threshold=25, threshold_increment=1.3,
            padding_start_ms=5, padding_end_ms=25,
            freq_min=200, freq_max=600
        )

        


        if whoop_pitch_info is not None:
            if verbose:
                detector.print_whoop_info()
                # for queue_idx in range(len(all_queues)):
                #     print(f"Queue length: {len(all_queues[queue_idx])}, idxs: {all_queues[queue_idx]}, median F0: {all_queues_f0[queue_idx]:.2f} Hz")

            # per calcolare le harmoniche meglio usare il peak time o il sample mediano del segmento rilevato con pitch detector???????
            # print(f"peak sample: {(self.features.peak_time - self.features.start_peak) * self.sr}")
            # print(f"f0 median sample: {whoop_pitch_info['f0_median_sample']}")

            relative_start_peak = whoop_pitch_info['start_sample_padded']/self.sr
            relative_end_peak = whoop_pitch_info['end_sample_padded']/self.sr

            self.features.precise_start_peak = self.features.start_peak + relative_start_peak
            self.features.precise_end_peak = self.features.start_peak + relative_end_peak
            self.features.precise_duration = (self.features.precise_end_peak - self.features.precise_start_peak)
            self.features.f0_mean = f0

        if plot:
            detector._plot_results(np.asarray(detector.compute_fundamental_frequencies(freq_min=200, freq_max=600)), best_queue_dic['frame_indices'], all_queues, f0, **figures_characteristics) 

        return best_queue_dic, whoop_pitch_info

    def extract_harmonics_features(self, best_queue_dic: dict, verbose: bool = False, plot_core: bool = False, plot_verbose: bool = False, **figures_characteristics):
        """Placeholder per future estrazione di feature armoniche."""

        harmonic_analyzer = HarmonicsAnalyzer(sr=self.sr, nfft=8192)

        # Calcola WHAR
        result = harmonic_analyzer.compute_weighted_shr(
            whoop_segment=self.whoop_audio,
            best_queue_dic=best_queue_dic,
            window_duration_ms=10,
            num_harmonics=10,
            bandwidth_hz=100,
            prominence_threshold_ratio=0.07,
            plot_core=plot_core,
            plot_verbose=plot_verbose,
            verbose=verbose,
            **figures_characteristics
        )


        self.features.weighted_shr = result['whar']
        self.features.max_aligned_peaks = result['alignment_max_peaks']

        

    def strongest_channel_detection(self, f0_median_frame_idx: Optional[int] = None, plot_core: bool = False, plot_verbose: bool = False, verbose: bool = False, listening_test: bool = False, **figures_characteristics):
        """Placeholder per future estrazione di feature del canale più forte."""
        # Crea detector e analizza
        detector = StrongChannelDetector(signal_multichannel=self.multichannel_audio, 
                                         sr=self.sr, 
                                         channel_ref=self.features.ch, 
                                         start_time_ref=self.features.start_peak,
                                         end_time_ref=self.features.end_peak,
                                         f0_ref=self.features.f0_mean,
                                         f0_ref_median_frame_idx=f0_median_frame_idx,
                                         f0_offset=30,
                                         broken_channels=self.channel_broken,
                                         num_channels=16,
                                         verbose=verbose,
                                         plot=plot_verbose,
                                         listening_test=listening_test)

        results = detector.detect_strong_channels(detector_percentile=50)

        # Accedi risultati
        if results['strongest_channel'] is not None:
            if verbose:
                print(f"Canale più forte: {results['strongest_channel'].channel_num + 1}")
                print(f"HNR: {results['strongest_channel'].hnr_db:.2f} dB")
                print(f"Numero canali con evento: {results['num_channels_with_whoop']}")
                print(f"Canali con whoop rilevato: {[ch.channel_num + 1 for ch in results['channels_with_whoop']]}")

            if plot_core:
                # detector.plot_voronoi_2d(
                #     self.mics_coords, 
                #     boundaries=self.frame_boundaries,  # ← QUI!
                #     use_db=False, 
                #     cmap="hot" 
                # )

      

                Utils.plot_hexagon_hnr_map(hnr_levels=results['all_hnr_levels'],
                                            mic_positions = self.mics_coords,
                                            channels_of_interest=detector.channels_of_interest,
                                            broken_channels=detector.broken_channels,
                                            hexagon_radius=0.025, 
                                            use_db = False,
                                            cmap = "hot",
                                            boundaries = self.frame_boundaries,
                                            **figures_characteristics) 
            
            self.features.strongest_channel = results['strongest_channel'].channel_num + 1
            self.features.hnr_levels = np.array(results['all_hnr_levels'])
            self.features.num_channels_with_whoop = results['num_channels_with_whoop']
        else:
            print("Nessun canale con whoop rilevato.")

    def precise_whoop_localization(self, validated_channels: List[int], verbose: bool = False, plot: bool = False, **figures_characteristics):
        mic_array = MicrophoneArray(
                            positions=self.mics_coords,
                            sample_rate=self.sr,
                            validated_channels=validated_channels,  # -> zero-based
                            boundaries=self.frame_boundaries,
                            margin=0.00
                        )
                
        localizer = SRPLocalizator(mic_array, c=343.0)

        # grid = localizer.create_search_grid_full(resolution=0.005)
        grid = localizer.create_search_grid_centered_on_channel(x_width=0.15, y_height=0.175, resolution=0.005, reference_channel=self.features.strongest_channel - 1)

        # print(f"Using search window: {results['search_window'].time_min:.3f}s - {results['search_window'].time_max:.3f}s")

        # Estrai i segnali ritagliati attorno al whoop    
        signals_cropped = self.multichannel_audio[int(self.features.start_peak*self.sr):int(self.features.end_peak*self.sr), :]
        # 4. Localizza!
        result = localizer.localize(signals_cropped, grid, max_tau_ms=100)
        
        if verbose:
            print(f"Whoop position: {result.estimated_position}")
            print(f"Power: {result.max_power}")

        if plot:
            # Visualizza il risultato
            plot_power_map_2d(result, mic_array, grid, ground_truth=None, **figures_characteristics)
            plt.show()

        self.features.precise_localization = np.array(result.estimated_position)
        



   
    
    def save_features_to_database(self, hdf5_path: str = 'whoop_database.h5'):
        """Salva TUTTE le features, compressione SOLO su spectrogram."""
        
        group_name = f"{self.parent_filename}_{self.whoop_candidate_filename.replace('.wav', '')}"
        
        with h5py.File(hdf5_path, 'a') as f:
            if group_name in f:
                print(f"⚠️  {group_name} esiste già, skip")
                return
            
            grp = f.create_group(group_name)

            # STRINGHE
            grp.attrs['date'] = self.features.date if self.features.date is not None else ""
            grp.attrs['time'] = self.features.time if self.features.time is not None else ""    
            
            # ===== SCALARI (senza compressione) =====
            grp.create_dataset('ch', data=np.array(self.features.ch))
            grp.create_dataset('peak_time', data=np.array(self.features.peak_time))
            grp.create_dataset('start_peak', data=np.array(self.features.start_peak))
            grp.create_dataset('end_peak', data=np.array(self.features.end_peak))
            grp.create_dataset('rough_duration', data=np.array(self.features.rough_duration))
            grp.create_dataset('sr', data=np.array(self.features.sr))
            
            
            # Pitch (NaN se None)
            grp.create_dataset('f0_mean', data=np.array(self.features.f0_mean) if self.features.f0_mean is not None else np.array(np.nan))
            grp.create_dataset('precise_start_peak', data=np.array(self.features.precise_start_peak) if self.features.precise_start_peak is not None else np.array(np.nan))
            grp.create_dataset('precise_end_peak', data=np.array(self.features.precise_end_peak) if self.features.precise_end_peak is not None else np.array(np.nan))
            grp.create_dataset('precise_duration', data=np.array(self.features.precise_duration) if self.features.precise_duration is not None else np.array(np.nan))

            # Harmonic analysis (NaN se None)
            grp.create_dataset('weighted_shr', data=np.array(self.features.weighted_shr) if self.features.weighted_shr is not None else np.array(np.nan))
            grp.create_dataset('max_aligned_peaks', data=np.array(self.features.max_aligned_peaks) if self.features.max_aligned_peaks is not None else np.array(np.nan))

            # Strong channel ( NaN se None)
            grp.create_dataset('strongest_channel', data=np.array(self.features.strongest_channel) if self.features.strongest_channel is not None else np.array(np.nan))
            grp.create_dataset('num_channels_with_whoop', data=np.array(self.features.num_channels_with_whoop) if self.features.num_channels_with_whoop is not None else np.array(np.nan))
            
            # ===== ARRAY PICCOLI (senza compressione) ===== ritrona array vuoto se None
                
            if self.features.hnr_levels is not None:
                grp.create_dataset('hnr_levels', data=np.array(self.features.hnr_levels))  # ~16 valori
            else:
                grp.create_dataset('hnr_levels', data=np.array([]))  # ~16 valori
                
            if self.features.precise_localization is not None:
                grp.create_dataset('precise_localization', data=np.array(self.features.precise_localization))  # [x,y]
            else:
                grp.create_dataset('precise_localization', data=np.array([]))  # [x,y]
            
            # ===== ARRAY GRANDE (SOLO spectrogram con compressione) ===== ritorna array vuoto se None
            if self.features.spectrogram_dB is not None:
                grp.create_dataset('spectrogram_dB', data=np.array(self.features.spectrogram_dB), compression='gzip', compression_opts=4)  # gzip livello 4
            else:
                grp.create_dataset('spectrogram_dB', data=np.array([]), compression='gzip', compression_opts=4)  # gzip livello 4

            if self.features.spec_frequencies is not None:
                grp.create_dataset('spec_frequencies', data=np.array(self.features.spec_frequencies), compression='gzip', compression_opts=4)  # gzip livello 4
            else:
                grp.create_dataset('spec_frequencies', data=np.array([]), compression='gzip', compression_opts=4)  # gzip livello 4

            if self.features.spec_times is not None:
                grp.create_dataset('spec_times', data=np.array(self.features.spec_times), compression='gzip', compression_opts=4)  # gzip livello 4
            else:
                grp.create_dataset('spec_times', data=np.array([]), compression='gzip', compression_opts=4)  # gzip livello 4

            print(f"✓ {group_name}")

    def run_full_pipeline(self, plot_core: bool = False, plot_verbose: bool = False, verbose: bool = False, listening_test: bool = False, 
                          save_to_database: bool = False, **figures_characteristics: Optional[dict] ) -> WhoopFeatures:
            """Esegui tutto → ritorna features."""
            print(f"Pipeline: {self.whoop_candidate_filename}")
            
            # carica raw signal multichannel e whoop segment
            self.load_audios()

            # estrai feature da filename del whoop come channel, start_peak, ecc.              
            self.store_feaures_from_params()

            if listening_test:
                

                Utils.extract_and_play_audio_segment_from_raw_multichannel_audio(
                    raw_audio_path=f"{self.root_raw_audio_dir}/{self.parent_filename}.wav",
                    start_time=self.features.start_peak,
                    end_time=self.features.end_peak,
                    channel=self.features.ch - 1,  # zero-based index
                    lenght_extension=1.0  # durata totale del segmento da riprodurre (in secondi)
                )



            # calcola spettrogramma
            self.compute_spectrogram(plot=plot_core, title=f"Spectrogram of Whoop Candidate Ch{self.features.ch}", **figures_characteristics)      

            # estrai feature di pitch
            best_queue_dic, whoop_pitch_info = self.extract_pitch_features(plot=plot_core, verbose=verbose, **figures_characteristics)

            # se il pitch è stato rilevato, prosegui con feature extraction
            if whoop_pitch_info is not None:

                # estrai feature armoniche
                self.extract_harmonics_features(
                    best_queue_dic=best_queue_dic,
                    verbose=verbose,
                    plot_core=plot_core,
                    plot_verbose=plot_verbose,
                    **figures_characteristics
                )

                
                # strongest channel detection
                self.strongest_channel_detection(whoop_pitch_info['f0_median_frame_idx'],plot_core=plot_core, plot_verbose=plot_verbose, verbose=verbose, listening_test=listening_test, **figures_characteristics)

                # se ho almeno 6 canali con whoop rilevato, posso fare la localizzazione precisa
                if self.features.num_channels_with_whoop is not None and self.features.num_channels_with_whoop >= 6:
                    print("Posso fare localizzazione precisa (>=6 canali con whoop rilevato).")
                    
                    # estrai indice dei 6 canali con whoop rilevato più forti
                    arr_np = np.array(self.features.hnr_levels, dtype=float)  # None → NaN
                    valid_mask = ~np.isnan(arr_np)
                    valid_values = arr_np[valid_mask]
                    top6_valid_idx = np.argsort(-valid_values)[:6]  # decrescente
                    validated_channels = np.where(valid_mask)[0][top6_valid_idx].tolist()
                    # ordina di nuovo dal canale più piccolo al più grande
                    validated_channels.sort()
                    print(f"Canali usati per localizzazione precisa: {[ch + 1 for ch in validated_channels]}")
                    
                    self.precise_whoop_localization(validated_channels, verbose=verbose, plot=plot_core, **figures_characteristics)
                else:
                    print("Non posso fare localizzazione precisa (<6 canali con whoop rilevato). Passa al prossimo candidate whoop")

            
            if save_to_database:
                self.save_features_to_database(hdf5_path='whoop_database.h5')

            
            
            return self.features

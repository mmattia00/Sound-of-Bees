import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
import os
import soundfile as sf

"""
Whoop detection utilities for mono audio recordings.

This module analyzes short-time HNR trajectories to detect whooping-like
acoustic events, merges overlapping candidate windows, and exposes helpers
used by the downstream pitch and post-processing pipelines.

"""


class WhoopDetector:
    """
    Whoop detection for mono audio recordings.
    
    It uses HNR (Harmonic-to-Noise Ratio) analysis over sliding windows
    to identify harmonic acoustic events characteristic of bee whooping.
    
    Parameters
    ----------
    signal : np.ndarray
        Mono audio signal
    sr : int
        Audio sample rate
    window_length_ms : float, default=50
        Analysis window length in milliseconds
    hop_length_ms : float, default=10
        Hop length between consecutive windows in milliseconds
    f0_min : float, default=250
        Minimum expected fundamental frequency (Hz)
    f0_max : float, default=700
        Maximum expected fundamental frequency (Hz)
    window_type : str, default='hamming'
        Window type ('hamming', 'hann', 'blackman')
    lowpass_cutoff : float, default=15000
        Low-pass filter cutoff frequency (Hz)
    highpass_cutoff : float, default=2500
        High-pass filter cutoff frequency (Hz)
    normalize : bool, default=True
        If True, normalize the signal to the target RMS
    target_rms : float, default=0.1
        Target RMS value for normalization
    
    Attributes
    ----------
    results_ : dict
        Results of the latest HNR analysis
    peaks_ : np.ndarray
        Indices of detected peaks
    peak_times_ : np.ndarray
        Peak center times (seconds)
    peak_windows_ : list of tuple
        List of tuples (start_time, end_time) for each peak window
    threshold_ : float
        HNR threshold used for detection
    """
    
    def __init__(self, signal, sr, window_length_ms=50, hop_length_ms=10,
                 f0_min=250, f0_max=700, window_type='hamming',
                 lowpass_cutoff=15000, highpass_cutoff=2500,
                 normalize=True, target_rms=0.1):
        
        self.signal = signal
        self.sr = sr
        self.window_length_ms = window_length_ms
        self.hop_length_ms = hop_length_ms
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.window_type = window_type
        self.lowpass_cutoff = lowpass_cutoff
        self.highpass_cutoff = highpass_cutoff
        self.normalize = normalize
        self.target_rms = target_rms
        
        # Attributi per i risultati
        self.results_ = None
        self.peaks_ = None
        self.peak_times_ = None
        self.peak_windows_ = None
        self.threshold_ = None
    
    # ==================== METODI STATICI DI UTILITÀ ====================
    # ==================== UTILITY STATIC METHODS ====================
    
    @staticmethod
    def autocorrelation(signal):
        """Compute the signal autocorrelation."""
        n = len(signal)
        signal = signal - np.mean(signal)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / np.arange(n, 0, -1)
        return autocorr
    
    @staticmethod
    def estimate_f0_autocorr(signal, sr, f0_min=75, f0_max=400):
        """Estimate the fundamental frequency via autocorrelation."""
        autocorr = WhoopDetector.autocorrelation(signal)
        lag_min = int(sr / f0_max)
        lag_max = int(sr / f0_min)
        if lag_max >= len(autocorr):
            lag_max = len(autocorr) - 1
        autocorr_search = autocorr[lag_min:lag_max]
        if len(autocorr_search) == 0:
            return None, autocorr, 0
        max_lag_local = np.argmax(autocorr_search)
        max_lag = max_lag_local + lag_min
        f0 = sr / max_lag
        return f0, autocorr, max_lag
    
    @staticmethod
    def calculate_hnr(signal, sr, f0_min=75, f0_max=400):
        """Compute the signal HNR (Harmonic-to-Noise Ratio)."""
        f0, autocorr, T0 = WhoopDetector.estimate_f0_autocorr(signal, sr, f0_min, f0_max)
        if f0 is None or T0 == 0:
            return None, None
        R0 = autocorr[0]
        RT0 = autocorr[T0]
        noise_energy = R0 - RT0
        if noise_energy <= 0 or RT0 <= 0:
            return None, f0
        hnr = RT0 / noise_energy
        hnr_db = 10 * np.log10(hnr)
        return hnr_db, f0
    
    @staticmethod
    def apply_window(signal, window_type='hamming'):
        """Apply a window function to the signal."""
        if window_type == 'hamming':
            window = np.hamming(len(signal))
        elif window_type == 'hann':
            window = np.hanning(len(signal))
        elif window_type == 'blackman':
            window = np.blackman(len(signal))
        else:
            window = np.ones(len(signal))
        return signal * window
    
    @staticmethod
    def apply_lowpass_filter(signal, sr, cutoff_hz=15000):
        """Apply a Butterworth low-pass filter."""
        nyquist = sr / 2
        normalized_cutoff = cutoff_hz / nyquist
        if normalized_cutoff >= 1:
            normalized_cutoff = 0.99
        b, a = butter(4, normalized_cutoff, btype='low')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    @staticmethod
    def apply_highpass_filter(signal, sr, cutoff_hz=2500):
        """Apply a Butterworth high-pass filter."""
        nyquist = sr / 2
        normalized_cutoff = cutoff_hz / nyquist
        if normalized_cutoff >= 1:
            normalized_cutoff = 0.99
        b, a = butter(4, normalized_cutoff, btype='high')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    @staticmethod
    def savitzky_golay_filter(data, window_size=11, polyorder=2):
        """Apply a Savitzky-Golay filter for smoothing."""
        if window_size % 2 == 0:
            window_size += 1
        if polyorder >= window_size:
            polyorder = window_size - 2
        filtered = savgol_filter(data, window_size, polyorder)
        return filtered
    
    @staticmethod
    def normalize_signal_rms(signal, target_rms=0.1):
        """Normalize the signal to a target RMS."""
        current_rms = np.sqrt(np.mean(signal**2))
        if current_rms < 1e-6:
            return signal
        scaling_factor = target_rms / current_rms
        return signal * scaling_factor
    
    @staticmethod
    def merge_overlapping_windows(peak_windows, peaks, peaks_times):
        """Merge overlapping time windows."""
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()
        if isinstance(peaks_times, np.ndarray):
            peaks_times = peaks_times.tolist()
        
        i = 0
        while i < len(peak_windows) - 1:
            start_i, end_i = peak_windows[i]
            start_next, end_next = peak_windows[i + 1]
            
            if end_i >= start_next:
                merged_window = (start_i, max(end_i, end_next))
                peak_windows[i] = merged_window
                del peak_windows[i + 1]
                del peaks[i + 1]
                del peaks_times[i + 1]
            else:
                i += 1
        
        return peak_windows, peaks, peaks_times
    
    # ==================== METODI PRINCIPALI ====================
    # ==================== MAIN METHODS ====================
    
    def preprocess_signal(self, signal):
        """
        Apply preprocessing to the signal: normalization and filters.
        
        Parameters
        ----------
        signal : np.ndarray
            Mono audio signal
        
        Returns
        -------
        np.ndarray
            Preprocessed signal
        """
        # Normalize first so the filter stages operate on a consistent level.
        if self.normalize:
            signal = self.normalize_signal_rms(signal, self.target_rms)
        
        # Apply the band-limiting filters used by the detector.
        signal = self.apply_lowpass_filter(signal, self.sr, self.lowpass_cutoff)
        signal = self.apply_highpass_filter(signal, self.sr, self.highpass_cutoff)
        
        return signal
    
    def analyze_hnr(self, signal):
        """
        Analyze the HNR of the signal over time windows.
        
        Parameters
        ----------
        signal : np.ndarray
            Preprocessed audio signal
        
        Returns
        -------
        dict
            Dictionary with keys: 'hnr_values', 'f0_values', 'time_centers',
            'valid_frames', 'total_frames'
        """
        window_length = int(self.window_length_ms * self.sr / 1000)
        hop_length = int(self.hop_length_ms * self.sr / 1000)
        
        num_frames = int((len(signal) - window_length) / hop_length) + 1
        hnr_values = []
        f0_values = []
        time_centers = []
        valid_frames = 0
        
        for i in range(num_frames):
            start = i * hop_length
            end = start + window_length
            if end > len(signal):
                break
            
            frame = signal[start:end]
            frame_windowed = self.apply_window(frame, self.window_type)
            hnr_db, f0 = self.calculate_hnr(frame_windowed, self.sr, 
                                           self.f0_min, self.f0_max)
            time_center = (start + window_length/2) / self.sr
            
            hnr_values.append(hnr_db)
            f0_values.append(f0)
            time_centers.append(time_center)
            
            if hnr_db is not None:
                valid_frames += 1
        
        return {
            'hnr_values': np.array(hnr_values),
            'f0_values': np.array(f0_values),
            'time_centers': np.array(time_centers),
            'valid_frames': valid_frames,
            'total_frames': len(hnr_values)
        }
    
    def postprocess_results(self, results, smoothing_window=5):
        """
        Postprocess the analysis results: fill NaNs and smooth trajectories.
        
        Parameters
        ----------
        results : dict
            HNR analysis results
        smoothing_window : int, default=5
            Window size for the Savitzky-Golay filter
        
        Returns
        -------
        dict
            Results with additional 'hnr_smoothed' and 'f0_smoothed' keys
        """
        results_processed = results.copy()
        hnr_values = results['hnr_values'].astype(float)
        f0_values = results['f0_values'].astype(float)
        
        hnr_valid = hnr_values[~np.isnan(hnr_values)]
        f0_valid = f0_values[~np.isnan(f0_values)]
        
        if len(hnr_valid) == 0 or len(f0_valid) == 0:
            return results_processed
        
        hnr_filled = hnr_values.copy()
        f0_filled = f0_values.copy()
        
        hnr_filled[np.isnan(hnr_values)] = np.mean(hnr_valid)
        f0_filled[np.isnan(f0_values)] = np.mean(f0_valid)
        
        results_processed['hnr_smoothed'] = self.savitzky_golay_filter(
            hnr_filled, smoothing_window)
        results_processed['f0_smoothed'] = self.savitzky_golay_filter(
            f0_filled, smoothing_window)
        
        return results_processed
    
    def find_peaks(self, hnr_values, time_centers, percentile=99, offset=2):
        """
        Detect HNR peaks using a percentile-based thresholding approach.
        
        Parameters
        ----------
        hnr_values : np.ndarray
            Smoothed HNR values
        time_centers : np.ndarray
            Window center times
        percentile : float, default=99
            Distribution percentile used to define the threshold
        offset : float, default=2
            dB offset added to the percentile value
        
        Returns
        -------
        tuple
            (peaks, peak_times, threshold) where:
            - peaks: indici dei picchi
            - peak_times: peak times (seconds)
            - threshold: threshold value used (dB)
        """
        hnr = np.array(hnr_values, dtype=float)
        valid = hnr[~np.isnan(hnr)]
        
        percentile_value = np.percentile(valid, percentile)
        threshold = percentile_value + offset
        
        peaks, _ = find_peaks(hnr, height=threshold)
        peak_times = time_centers[peaks]
        
        return peaks, peak_times, threshold
    
    def create_peak_windows(self, peak_times, signal_duration, window_sec=3.0):
        """
        Create time windows centered on detected peaks.
        
        Parameters
        ----------
        peak_times : np.ndarray
            Peak center times (seconds)
        signal_duration : float
            Total signal duration (seconds)
        window_sec : float, default=3.0
            Window duration around the peak (seconds)
        
        Returns
        -------
        list of tuple
            List of tuples (start_time, end_time) for each window
        """
        peak_windows = []
        for peak_time in peak_times:
            start_time = max(0, peak_time - window_sec / 2)
            end_time = min(signal_duration, peak_time + window_sec / 2)
            peak_windows.append((start_time, end_time))
        return peak_windows
    
    def detect(self, percentile=99.5, offset=4, window_sec=3.0, 
               merge_overlaps=True, smoothing_window=5):
        """
        Main whoop detection method.
        
        Parameters
        ----------
        percentile : float, default=99.5
            Peak detection percentile
        offset : float, default=4
            dB offset above the percentile
        window_sec : float, default=3.0
            Window duration around peaks (seconds)
        merge_overlaps : bool, default=True
            If True, merge overlapping windows
        smoothing_window : int, default=5
            Window size for Savitzky-Golay smoothing
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'peaks': indici dei picchi
            - 'peak_times': peak times (seconds)
            - 'peak_windows': list of tuples (start_time, end_time)
            - 'threshold': HNR threshold (dB)
            - 'results': full HNR analysis results
        """
        # Step 1: preprocess the raw audio before analysis.
        signal_processed = self.preprocess_signal(self.signal)
        
        # Step 2: compute the HNR trajectory.
        results = self.analyze_hnr(signal_processed)
        
        # Step 3: smooth and clean up the analysis results.
        results_smoothed = self.postprocess_results(results, smoothing_window)
        
        # Step 4: detect peaks on the smoothed HNR curve.
        peaks, peak_times, threshold = self.find_peaks(
            results_smoothed['hnr_smoothed'],
            results_smoothed['time_centers'],
            percentile=percentile,
            offset=offset
        )
        
        # Step 5: convert peak times into candidate event windows.
        signal_duration = len(self.signal) / self.sr
        peak_windows = self.create_peak_windows(peak_times, signal_duration, window_sec)
        
        # Merge overlapping candidate windows to avoid duplicates.
        if merge_overlaps and len(peak_windows) > 0:
            peak_windows, peaks, peak_times = self.merge_overlapping_windows(
                peak_windows, peaks, peak_times
            )
        
        # Store the latest detection results on the instance for downstream use.
        self.results_ = results_smoothed
        self.peaks_ = np.array(peaks)
        self.peak_times_ = np.array(peak_times)
        self.peak_windows_ = peak_windows
        self.threshold_ = threshold
        
        return {
            'peaks': self.peaks_,
            'peak_times': self.peak_times_,
            'peak_windows': self.peak_windows_,
            'threshold': self.threshold_,
            'results': self.results_
        }
    
    def save_segments(self, audio_filename, channel, output_dir="whooping_candidates"):
        """
        Save each candidate whooping segment as a WAV file.
        
        Parameters:
        -----------
        audio_filename : str
            Original audio file name (with path)
        channel : int
            Channel number
        output_dir : str
            Output root directory
        """

        # Extract the file name without path or extension.
        base_filename = os.path.splitext(os.path.basename(audio_filename))[0]

        # Create the output folder structure: whooping_candidates/original_file_name/
        output_folder = os.path.join(output_dir, base_filename)
        os.makedirs(output_folder, exist_ok=True)

        peaks_info = self.get_peak_info()

        for i in range(len(peaks_info)):
            start_time, end_time = peaks_info[i]['window_start'], peaks_info[i]['window_end']
            peak_time = peaks_info[i]['peak_time']
            segment = self.signal[int(start_time * self.sr):int(end_time * self.sr)]

            # Build a descriptive file name for the saved candidate segment.
            hnr_value = peaks_info[i]['peak_hnr_value']
            filename = f"ch_{channel:02d}_peaktime_{peak_time:.3f}_windowstart_{start_time:.3f}_windowend_{end_time:.3f}_hnrvalue_{hnr_value:.2f}.wav"
            output_path = os.path.join(output_folder, filename)

            # Write the WAV file.
            sf.write(output_path, segment, self.sr)

            print(f"Saved: {output_path}")
    
    
    def extract_segments(self, peak_windows=None):
        """
        Extract audio segments corresponding to peak windows.
        
        Parameters
        ----------
        signal : np.ndarray
            Original audio signal
        peak_windows : list of tuple, optional
            List of windows (start_time, end_time). If None, use self.peak_windows_
        
        Returns
        -------
        list of np.ndarray
            List of audio segments
        """
        if peak_windows is None:
            peak_windows = self.peak_windows_
        
        if peak_windows is None:
            raise ValueError("No windows available. Run detect() first.")
        
        segments = []
        for start_time, end_time in peak_windows:
            start_sample = int(start_time * self.sr)
            end_sample = int(end_time * self.sr)
            segment = self.signal[start_sample:end_sample]
            segments.append(segment)
        
        return segments
    
    def plot_analysis(self, ch_num=1, figsize=(10, 5), dpi=150):
        """
        Plot the HNR analysis results together with the detected peaks.
        
        Parameters
        ----------
        ch_num : int, default=1
            Channel number (used only in the title)
        figsize : tuple, default=(10, 5)
            Figure size
        dpi : int, default=150
            Figure resolution
        """
        if self.results_ is None:
            raise ValueError("No results available. Run detect() first.")
        
        time_centers = self.results_['time_centers']
        hnr = self.results_['hnr_smoothed']
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(time_centers, hnr, color='b', linewidth=2.2, label='HNR')
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Very harmonic (>20 dB)')
        ax.axhline(y=15, color='orange', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Moderate (>15 dB)')
        ax.axhline(y=13, color='red', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label='Low (>13 dB)')
        ax.fill_between(time_centers, 15, 25, color='yellow', alpha=0.07)
        
        if self.threshold_ is not None:
            ax.axhline(y=self.threshold_, color='purple', linestyle='-', 
                      linewidth=2.5, alpha=0.8, 
                      label=f'Threshold ({self.threshold_:.2f} dB)')
        
        if self.peak_windows_ is not None and len(self.peak_windows_) > 0:
            for i, (start, end) in enumerate(self.peak_windows_):
                label = 'Peak window' if i == 0 else ''
                ax.axvspan(start, end, alpha=0.15, color='red', label=label)
        
        if self.peaks_ is not None and len(self.peaks_) > 0:
            ax.scatter(time_centers[self.peaks_], hnr[self.peaks_], 
                      color='red', s=100, marker='o', 
                      edgecolors='darkred', linewidths=2,
                      label=f'Detected peaks ({len(self.peaks_)})', 
                      zorder=5, alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax.set_ylabel('HNR (dB)', fontsize=10, fontweight='bold')
        ax.set_title(f'HNR over time - channel {ch_num}', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='best', fontsize=8, framealpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def get_peak_info(self):
        """
        Return the detected peak information in a readable format.
        
        Returns
        -------
        list of dict
            List of dictionaries with information for each peak
        """
        if self.peaks_ is None or self.peak_times_ is None or self.peak_windows_ is None:
            return []
        
        peak_info = []
        for i, (peak_idx, peak_time, window) in enumerate(
            zip(self.peaks_, self.peak_times_, self.peak_windows_)):
            info = {
                'index': i,
                'peak_index': int(peak_idx),
                'peak_time': float(peak_time),
                'window_start': float(window[0]),
                'window_end': float(window[1]),
                'window_duration': float(window[1] - window[0]),
                'peak_hnr_value': float(self.results_['hnr_smoothed'][peak_idx])
            }
            peak_info.append(info)
        
        return peak_info


import sounddevice as sd
# # Main block for testing the class with a mono audio file
# if __name__ == "__main__":
#     # audio_file = "reduced_whooping_raw.wav"
#     audio_file = "sounds/whoop_examples/whooping_collection.wav"

#     print("\n" + "="*80)
#     print("WHOOPING SIGNALS - HNR-BASED ANALYSIS START")
#     print("="*80)

#     # Load audio file.
#     print(f"\nLoading: {audio_file}")
#     signal, sr = sf.read(audio_file)


#     # Limit to 10 seconds for a quick test.
#     if len(signal) > sr * 10:
#         signal = signal[:sr*10]

#     # Keep only the first channel if the file is stereo.
#     if len(signal.shape) > 1:
#         signal = signal[:, 0]


  
#     print(f'ANALYSIS START')

#     # init whoop detector
#     detector = WhoopDetector(
#         signal=signal,
#         sr=sr,
#         window_length_ms=40,
#         hop_length_ms=7,
#         f0_min=250,
#         f0_max=700,
#         window_type='hamming',
#         lowpass_cutoff=15000,
#         highpass_cutoff=2500,
#         normalize=True,
#         target_rms=0.1
#     )

#     # Run the detection.
#     detection_results = detector.detect(
#         percentile=62,
#         offset=4,
#         window_sec=0.5, # analysis window length around the peak
#         merge_overlaps=True
#     )

#     # Inspect the results.
#     print(f"Detected peaks: {len(detector.peak_times_)}")
#     print(f"HNR threshold: {detector.threshold_:.2f} dB")

#     # Print detailed peak information.
#     peak_info = detector.get_peak_info()
#     for info in peak_info:
#         print(f"Peak {info['index']}: t={info['peak_time']:.3f}s, "
#             f"window=[{info['window_start']:.3f}, {info['window_end']:.3f}]s")
        
    

#     # Direct access to the arrays.
#     # detector.peaks_         -> peak indices in the HNR array
#     # detector.peak_times_    -> peak center times (seconds)
#     # detector.peak_windows_  -> list of tuples (start_time, end_time)
    
#     # Optional playback
#     # Extract the audio segments.
#     segments = detector.extract_segments()
#     for segment in segments:
#         print(f"   - Playing segment of {len(segment)/sr:.3f} seconds")
#         sd.play(segment, sr)
#         sd.wait()
#         # wait half a second before continuing
#         sd.sleep(500)

#     # Visualize the results.
#     detector.plot_analysis(ch_num=1)


#     print("="*80)
#     print(f"ANALYSIS COMPLETED!")
#     print("="*80)


# Main block to test the class with multichannel audio files.
if __name__ == "__main__":
    # Folder containing the audio files.
    # raw_audio_folder = "E:\soundofbees"
    # raw_audio_folder = "/media/uni-konstanz/My Passport/soundofbees" # for testing in the lab
    raw_audio_folder = "sounds/whoop_examples/testing_pipeline" # for testing on example files
    # candidates_folder = "sounds\whoop_candidates_splitted" # new folder structure where each .wav is shorter and centered around the peak and hopefully contains exactly one whoop
    # candidates_folder = "/home/uni-konstanz/Desktop/Sound_of_bees/Sound-of-Bees/sounds/whoop_candidates_splitted" # for testing in the lab
    candidates_folder = "sounds/whoop_examples/whoop_candidates_splitted" # for testing on example files
    
    
    # starting_audiofile_name = "audio_recording_2025-09-15T00_00_43.625108Z.wav" # first file among those on the local disk
    starting_audiofile_name = "audio_recording_2025-09-16T01_35_48.799810Z.wav" # last file analyzed before interrupting the previous run
    # starting_audiofile_name = "audio_recording_2025-09-15T07_09_45.544480Z.wav" # a meaningful file for testing

    
    # Broken channels
    channel_broken = [2, 3, 8, 13, 21, 25, 27, 28, 31]
    channel_broken = [x - 1 for x in channel_broken]  # zero-based index
    
    print("\n" + "="*80)
    print("WHOOPING SIGNALS - BATCH ANALYSIS HNR BASED")
    print("="*80)
    
    # Find all .wav files in the folder.
    audio_files = sorted([f for f in os.listdir(raw_audio_folder) if f.endswith('.wav')])
    
    # Find the index of the starting audio file.
    try:
        start_index = audio_files.index(starting_audiofile_name)
    except ValueError:
        start_index = -1

    # Keep all files after the starting point.
    if start_index != -1:
        audio_files = audio_files[start_index:]
    


    if not audio_files:
        print("No .wav files found in " + raw_audio_folder)
        exit(1)
    
    print("Found " + str(len(audio_files)) + " audio files")
    print("="*80 + "\n")
    
    # Process each file.
    for file_idx, audio_filename in enumerate(audio_files, 1):
        audio_file = os.path.join(raw_audio_folder, audio_filename)
        
        print("\n[" + str(file_idx) + "/" + str(len(audio_files)) + "] Processing: " + audio_filename)
        print("-" * 80)
        
        try:
            # Load the audio file.
            signal_multichannel, sr = sf.read(audio_file)
            
            print("Loaded: " + str(sr) + " Hz, " + str(len(signal_multichannel)/sr) + "s")
            
        except Exception as e:
            print("Error loading " + audio_filename + ": " + str(e))
            continue
        
        # Process each channel.
        for j in range(signal_multichannel.shape[1]):
            
            if j in channel_broken:
                continue
            
            print("  CHANNEL " + str(j+1).zfill(2), end=" ")
            
            try:
                signal = signal_multichannel[:, j]

                # Initialize the whoop detector.
                detector = WhoopDetector(
                    signal=signal,
                    sr=sr,
                    window_length_ms=50,
                    hop_length_ms=10,
                    f0_min=250,
                    f0_max=700,
                    window_type='hamming',
                    lowpass_cutoff=15000,
                    highpass_cutoff=2500,
                    normalize=True,
                    target_rms=0.1
                )

                # Run the detection.
                detection_results = detector.detect(
                    percentile=95,
                    offset=4,
                    window_sec=0.5, # analysis window length around the peak
                    merge_overlaps=True
                )

                # Inspect the results.
                print(f"Detected peaks: {len(detector.peak_times_)}")
                print(f"HNR threshold: {detector.threshold_:.2f} dB")

                # Print detailed peak information.
                peak_info = detector.get_peak_info()
                for info in peak_info:
                    print(f"Peak {info['index']}: t={info['peak_time']:.3f}s, "
                        f"window=[{info['window_start']:.3f}, {info['window_end']:.3f}]s, "
                        f"HNR={info['peak_hnr_value']:.2f} dB")
                    

                # Direct access to the arrays.
                # detector.peaks_         -> peak indices in the HNR array
                # detector.peak_times_    -> peak center times (seconds)
                # detector.peak_windows_  -> list of tuples (start_time, end_time)
                
                # # Optional playback
                # # Extract the audio segments.
                # segments = detector.extract_segments()
                # for segment in segments:
                #     print(f"   - Playing segment of {len(segment)/sr:.3f} seconds")
                #     sd.play(segment, sr)
                #     sd.wait()
                #     # wait half a second before continuing
                #     sd.sleep(500)
                
                # Save the detected segments.
                detector.save_segments(audio_file, j+1, output_dir=candidates_folder)
                
                # # Optional: visualize the results.
                # detector.plot_analysis(ch_num=j+1)
                
            except Exception as e:
                print("Error: " + str(e))
                continue
        
        print("\n" + "="*80)
    
    print("\nBATCH ANALYSIS COMPLETED!")
    print("Results saved in: whooping_candidates/")
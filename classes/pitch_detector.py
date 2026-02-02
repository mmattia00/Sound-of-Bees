import numpy as np
import matplotlib.pyplot as plt


class PitchDetector:
    def __init__(self, audio_segment, sr):
        self.audio_segment = audio_segment
        self.sr = sr
        self.window_duration = 0.050
        self.hop_duration = 0.010
        self.nfft = 4096
        
        self.window_length = int(self.window_duration * sr)
        self.hop_length = int(self.hop_duration * sr)
        self.num_frames = int((len(self.audio_segment) - self.window_length) / self.hop_length) + 1
        self.whoop_info = None

    def compute_fft(self, frame):
        spectrum = np.fft.rfft(frame, n=self.nfft)
        magnitude = np.abs(spectrum)
        return spectrum, magnitude
    
    def compute_fundamental_frequency(self, magnitude, f0_min=200, f0_max=600):
        log_spectrum = np.log(magnitude + 1e-10)
        cepstrum = np.fft.irfft(log_spectrum, n=self.nfft)
        
        quefrency = np.arange(len(cepstrum)) / self.sr
        quefrency_min = 1.0 / f0_max
        quefrency_max = 1.0 / f0_min
        
        valid_range = (quefrency >= quefrency_min) & (quefrency <= quefrency_max)
        
        if not np.any(valid_range):
            return None, cepstrum, quefrency
        
        cepstrum_valid = np.abs(cepstrum[valid_range])
        quefrency_valid = quefrency[valid_range]
        
        peak_idx = np.argmax(cepstrum_valid)
        peak_quefrency = quefrency_valid[peak_idx]
        
        f0 = 1.0 / peak_quefrency
        
        return f0, cepstrum, quefrency

    def compute_fundamental_frequencies(self, freq_min=200, freq_max=600):
        fundamental_frequencies = []

        for i in range(self.num_frames):
            start = i * self.hop_length
            end = start + self.window_length
            frame = self.audio_segment[start:end]

            frame_windowed = frame * np.hamming(len(frame))
            spectrum, magnitude = self.compute_fft(frame_windowed)
                                                            
            f0_estimated, _, _ = self.compute_fundamental_frequency(magnitude, f0_min=freq_min, f0_max=freq_max)
            fundamental_frequencies.append(f0_estimated)

        return fundamental_frequencies
    
    def _plot_results(self, freq_array, best_queue_indices, all_queues, f0, **figures_characteristics):
        plt.figure(figsize=figures_characteristics.get('fig_size', (12, 6)))
        
        if self.hop_length is not None and self.sr is not None:
            x_axis = np.arange(len(freq_array)) * self.hop_length / self.sr
            x_label = 'Time (s)'
            title_suffix = f' - Hop: {self.hop_length/self.sr*1000:.1f}ms'
        else:
            x_axis = np.arange(len(freq_array))
            x_label = 'Frame Index'
            title_suffix = ''
        
        plt.plot(x_axis, freq_array, 'o-', color='lightgray', 
                linewidth=1, markersize=6, alpha=0.6, label='Original array')
        
        if all_queues:
            for i, queue in enumerate(all_queues):
                if queue != best_queue_indices:
                    plt.plot(x_axis[queue], freq_array[queue], 'o-', 
                            color='orange', alpha=0.3, linewidth=1, 
                            markersize=4, label='Candidate queues' if i == 0 else '')
        
        if best_queue_indices is not None and len(best_queue_indices) > 0:
            plt.plot(x_axis[best_queue_indices], freq_array[best_queue_indices], 'o-', 
                    color='red', linewidth=2.5, markersize=8, 
                    label=f'Best queue (n={len(best_queue_indices)})', zorder=5)
            
            if f0 is not None:
                plt.axhline(y=f0, color='red', linestyle='--', 
                        linewidth=1.5, alpha=0.7, 
                        label=f'f0 = {f0:.2f} Hz')
                
            if self.hop_length is not None and self.sr is not None:
                start_time = x_axis[best_queue_indices[0]]
                end_time = x_axis[best_queue_indices[-1]]
                plt.axvspan(start_time, end_time, alpha=0.2, color='red', label='Whoop region')
        else:
            plt.text(0.5, 0.5, 'No valid queue found', 
                    transform=plt.gca().transAxes, 
                    fontsize=figures_characteristics.get('label_fontsize', 14), color='red', ha='center', va='center')
        
        plt.xlabel(x_label, fontsize=figures_characteristics.get('label_fontsize', 12))
        plt.ylabel('Frequency (Hz)', fontsize=figures_characteristics.get('label_fontsize', 12))
        plt.title(f'Fundamental Frequency Estimation', 
                fontsize=figures_characteristics.get('title_fontsize', 14), fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=figures_characteristics.get('tick_fontsize', 10))
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best', fontsize=figures_characteristics.get('legend_fontsize', 12))
        plt.tight_layout()
        plt.show()


    def estimate_f0(self, length_queue=5, hz_threshold=10, threshold_increment=1.5, 
                padding_start_ms=20, padding_end_ms=20, 
                freq_min=200, freq_max=600):
    
        freq_array = np.asarray(self.compute_fundamental_frequencies(freq_min=freq_min, freq_max=freq_max))
        
        valid_mask = np.isfinite(freq_array) & (freq_array > 0)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2:
            return None, {'frame_indices': [], 'samples': [], 'f0_values': []}, [], [], None
        
        all_queues = []
        
        for start_idx in valid_indices:
            current_queue = [start_idx]
            current_freq = freq_array[start_idx]
            
            for next_idx in valid_indices[valid_indices > start_idx]:
                next_freq = freq_array[next_idx]
                
                is_valid = True
                
                for queue_position, prev_idx in enumerate(reversed(current_queue)):
                    prev_freq = freq_array[prev_idx]
                    
                    adjusted_threshold = hz_threshold * (threshold_increment ** queue_position)
                    diff = abs(next_freq - prev_freq)
                    
                    if diff > adjusted_threshold:
                        is_valid = False
                        break
                
                if is_valid:
                    current_queue.append(next_idx)
                    current_freq = next_freq
                else:
                    break
            
            if len(current_queue) >= length_queue:
                all_queues.append(current_queue)
        
        if not all_queues:
            return None, {'frame_indices': [], 'samples': [], 'f0_values': []}, [], [], None
        
        best_queue_indices = max(all_queues, key=len)
        f0 = np.median(freq_array[best_queue_indices])

        freq_values_in_queue = freq_array[best_queue_indices]
        differences = np.abs(freq_values_in_queue - f0)
        median_position_in_queue = np.argmin(differences)
        f0_median_frame_idx = best_queue_indices[median_position_in_queue]
        
        # ============= NUOVO: Crea il dizionario best_queue con 3 liste =============
        best_queue = {
            'frame_indices': best_queue_indices,
            'samples': [frame_idx * self.hop_length for frame_idx in best_queue_indices],
            'f0_values': freq_array[best_queue_indices].tolist()
        }
        # =========================================================================
        
        if self.hop_length is not None and self.sr is not None:
            start_frame_idx = best_queue_indices[0]
            end_frame_idx = best_queue_indices[-1]
            
            start_sample = start_frame_idx * self.hop_length
            end_sample = end_frame_idx * self.hop_length
            
            start_time = start_sample / self.sr
            end_time = end_sample / self.sr
            duration_ms = (end_time - start_time) * 1000

            f0_median_sample_in_segment = f0_median_frame_idx * self.hop_length
            
            padding_start_samples = int(padding_start_ms * self.sr / 1000)
            padding_end_samples = int(padding_end_ms * self.sr / 1000)
            
            start_sample_padded = max(0, start_sample - padding_start_samples)
            end_sample_padded = min(len(self.audio_segment), end_sample + padding_end_samples)
            
            duration_padded_ms = (end_sample_padded - start_sample_padded) / self.sr * 1000
            
            self.whoop_info = {
                'start_time': start_time,
                'end_time': end_time,
                'duration_ms': duration_ms,
                'start_sample': start_sample,
                'end_sample': end_sample,
                'start_sample_padded': start_sample_padded,
                'end_sample_padded': end_sample_padded,
                'duration_padded_ms': duration_padded_ms,
                'f0_median_frame_idx': f0_median_frame_idx,
                'f0_median_sample': f0_median_sample_in_segment,
                'frame_indices': best_queue_indices,
                'num_frames': len(best_queue_indices)
            }

        # Calcola F0 mediana per ogni queue
        all_queues_f0 = []
        for queue in all_queues:
            f0_queue = np.median(freq_array[queue])
            all_queues_f0.append(f0_queue)
        
        return f0, best_queue, all_queues, all_queues_f0, self.whoop_info

    
    def print_whoop_info(self):
        """
        Docstring for print_whoop_info
        Nota che start_sample, end_sample, start_sample_padded, end_sample_padded e f0_median_sample sono relativi al segmento audio ù
        analizzato. Se questo segmento è una porzione di un audio più grande, bisogna sommare l'offset iniziale del segmento per ottenere i valori
        relativi all'audio completo.
        :param self: Description
        """
        # Stampa TUTTE le informazioni sul whoop
        if self.whoop_info is not None:
            print(f"\n{'='*60}")
            print(f"WHOOP RILEVATO")
            print(f"{'='*60}")
            print(f"Start time: {self.whoop_info['start_time']:.3f} s")
            print(f"End time: {self.whoop_info['end_time']:.3f} s")
            print(f"Duration ms: {self.whoop_info['duration_ms']:.2f} ms")  
            print(f"Start sample (originale): {self.whoop_info['start_sample']}")
            print(f"End sample (originale): {self.whoop_info['end_sample']}")
            print(f"Start sample (padded): {self.whoop_info['start_sample_padded']}")
            print(f"End sample (padded): {self.whoop_info['end_sample_padded']}")
            print(f"Duration padded ms: {self.whoop_info['duration_padded_ms']:.2f} ms")
            print(f"f0 median frame idx: {self.whoop_info['f0_median_frame_idx']}")
            print(f"f0 median sample: {self.whoop_info['f0_median_sample']}")
            print(f"Numero di frames: {self.whoop_info['num_frames']}")

            print(f"{'='*60}\n")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

class AudioSynchronizer:
    def __init__(self, delay_samples = None, audio_source_1=None, audio_source_2=None, sr=None, verbose=True, plot=False):
        self.audio_source_1 = audio_source_1 # viene dal mic della prima scheda audio
        self.audio_source_2 = audio_source_2 # viene dal mic della seconda scheda audio
        self.sr = sr
        self.verbose = verbose
        self.plot = plot
        self.delay_samples = delay_samples

    def gcc_phat(self, max_tau_ms=100):
        """
        Questo metodo va chiamato SOLAMENTE in fase di calibrazione la prima volta, utilizzato su due canali a caso collegati a due schede audio
        diverse, per calcolare il delay tra le due schede audio. A quel punto diamo per assunto che per il sistema di acquisizione il delay rimanga costante.
        Diventa quindi un parametro da passare alla classe AudioSynchronizer per poter poi sincronizzare i canali in seguito.
        Questa classe diventa quindi importante nell'uso comune per fare il load dell'audio multicanale sincronizzando i primi 16 canali con i secondi 16
    
        
        Args:
            sig1 (np.ndarray): Primo segnale audio (reference)
            sig2 (np.ndarray): Secondo segnale audio (neighbor)
            sr (int): Sample rate in Hz
            max_tau_ms (float): Massimo delay da cercare in millisecondi
        
        Returns:
            dict: Contiene delay_samples, delay_ms, delay_sec, correlation, lags, peak_value
        """
        # Lunghezza di FFT (padding per evitare aliasing)
        fft_len = 2 * max(len(self.audio_source_1), len(self.audio_source_2))
        
        # FFT dei due segnali
        X1 = np.fft.rfft(self.audio_source_1, n=fft_len)
        X2 = np.fft.rfft(self.audio_source_2, n=fft_len)
        
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
        self.delay_samples = peak_idx - max_samples
        
        # Converti in unità diverse
        delay_ms = (self.delay_samples / self.sr) * 1000
        delay_sec = self.delay_samples / self.sr
        
        # Lags array (per plotting)
        lags = (np.arange(-max_samples, max_samples + 1) / self.sr) * 1000  # in ms
        # plot del risultato 
        if self.plot:
            plt.figure(figsize=(6, 3))
            plt.plot(lags, correlation_windowed)
            plt.xlabel("Lag (ms)")
            plt.ylabel("Normalized Correlation")
            plt.grid()
            plt.show()

        if self.verbose:
            print(f"Estimated delay: {self.delay_samples} samples, {delay_ms:.2f} ms, {delay_sec:.4f} sec")
            if delay_ms > 0:
                print(f"The second audio source lags behind the first. To synchronize, delay the second source of {delay_ms:.2f} ms.")
            else:
                print(f"The first audio source lags behind the second. To synchronize, delay the first source of {abs(delay_ms):.2f} ms.")
        return {
            'delay_samples': self.delay_samples,
            'delay_ms': delay_ms,
            'delay_sec': delay_sec,
            'correlation': correlation_windowed,
            'lags': lags,
            'peak_value': correlation_windowed[self.delay_samples + max_samples]
        }
    
    def load_synchronized_multichannel_audio(self, filepath):
        """
        Carica un file audio multicanale e sincronizza i canali della seconda scheda audio.
        
        Args:
            filepath (str): Percorso del file audio multicanale
        
        Returns:
            synced_audio (np.ndarray): Audio sincronizzato (samples, channels)
            sr (int): Sample rate
        """
        if self.delay_samples is None:
            raise ValueError("delay_samples non è stato calcolato. Esegui prima gcc_phat().")
        
        # Carica il file audio
        unsynced_audio, sr = sf.read(filepath)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SINCRONIZZAZIONE AUDIO MULTICANALE")
            print(f"{'='*60}")
            print(f"File: {filepath}")
            print(f"Shape originale: {unsynced_audio.shape}")
            print(f"Sample rate: {sr} Hz")
            print(f"Delay misurato: {self.delay_samples} campioni ({self.delay_samples/sr*1000:.2f} ms)")
        
        # Crea copia per sincronizzazione
        synced_audio = unsynced_audio.copy()
        
        if self.delay_samples > 0:
            # Seconda scheda (canali 16-31) è in ANTICIPO → ritarda (shift a destra)
            if self.verbose:
                print(f"→ Ritardo canali 16-31 di {self.delay_samples} campioni")
            
            for ch in range(16, 32):
                if ch < unsynced_audio.shape[1]:  # Controlla che il canale esista
                    # Zero-padding all'inizio + dati shiftati
                    synced_audio[:, ch] = np.concatenate([
                        np.zeros(self.delay_samples),
                        unsynced_audio[:-self.delay_samples, ch]
                    ])
        
        elif self.delay_samples < 0:
            # Prima scheda (canali 0-15) è in ANTICIPO → ritarda (caso raro)
            if self.verbose:
                print(f"→ Ritardo canali 0-15 di {-self.delay_samples} campioni")
            
            for ch in range(16):
                if ch < unsynced_audio.shape[1]:
                    synced_audio[:, ch] = np.concatenate([
                        np.zeros(-self.delay_samples),
                        unsynced_audio[:self.delay_samples, ch]
                    ])
        
        else:
            if self.verbose:
                print("→ Nessun ritardo rilevato, audio già sincronizzato")
        
        if self.verbose:
            print(f"Shape sincronizzata: {synced_audio.shape}")
            print(f"{'='*60}\n")
        
        return synced_audio, sr


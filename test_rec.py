import sounddevice as sd
import soundfile as sf

fs = 48000
channels = 16
device = 65

duration = 3.0  # secondi

print(sd.query_devices()[device])

data = sd.rec(int(duration*fs), samplerate=fs, channels=channels, device=device)
sd.wait()

sf.write("test_16ch.wav", data, fs)

import sounddevice as sd
import soundfile as sf

fs = 48000
channels = 8
device = 71

duration = 60.0  # secondi
# print all devices
print(sd.query_devices())
# print(sd.query_devices()[device])

data = sd.rec(int(duration*fs), samplerate=fs, channels=channels, device=device)
sd.wait()

sf.write("sounds/infrared_mics_interference_tests/infrared_on_front_facing.wav", data, fs)

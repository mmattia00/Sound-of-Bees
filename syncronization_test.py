from classes.audio_syncronizer import AudioSynchronizer
import sounddevice as sd
import soundfile as sf

def main():
    # Esempio di utilizzo
    ref_audio_path = "sounds/localization_test/sync_Point_a.wav"
    audio_multichannel, sr = sf.read(ref_audio_path)  # Carica il file audio multicanale


    audio_source_1 = audio_multichannel[:, 15]  # Primo canale (mic della prima scheda audio)
    audio_source_2 = audio_multichannel[:, 18]  # Secondo canale (mic della seconda scheda audio)
    
    synchronizer = AudioSynchronizer(audio_source_1=audio_source_1, audio_source_2=audio_source_2, sr=sr, verbose=True, plot=True)
    
    
    result = synchronizer.gcc_phat()


if __name__ == "__main__":
    main()
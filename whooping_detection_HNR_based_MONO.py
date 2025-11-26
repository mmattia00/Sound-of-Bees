from whoop_detector import WhoopDetector
import soundfile as sf
import sounddevice as sd


# unico scopo di questo script è mostrare quello che ho fatto all'inizio cioè sviluppare un analisi HNR based su file di esempio pieno di whoop e 
# di rilevarne la maggior parte possibile. Lo script completo per l'analisi batch è in whooping_detection_HNR_based_MULTICH.py ed esegue la 
# stessa analisi ma su file multicanale

if __name__ == "__main__":
    # audio_file = "reduced_whooping_raw.wav"
    audio_file = "sounds/whoop_examples/whooping_collection.wav"

    print("\n" + "="*80)
    print("WHOOPING SIGNALS - INIZIO ANALISI HNR BASED")
    print("="*80)

    # load audio file
    print(f"\nCaricamento: {audio_file}")
    signal, sr = sf.read(audio_file)


    # Limita a 10 secondi per test veloce
    if len(signal) > sr * 10:
        signal = signal[:sr*10]

    # tieni solo primo canale se stereo
    if len(signal.shape) > 1:
        signal = signal[:, 0]


  
    print(f'INIZIO ANALISI')

    # init whoop detector
    detector = WhoopDetector(
        signal=signal,
        sr=sr,
        window_length_ms=40,
        hop_length_ms=7,
        f0_min=250,
        f0_max=700,
        window_type='hamming',
        lowpass_cutoff=15000,
        highpass_cutoff=2500,
        normalize=True,
        target_rms=0.1
    )

    # Esegui la rilevazione
    detection_results = detector.detect(
        percentile=62,
        offset=4,
        window_sec=0.5, # lunghezza finestra di analisi intorno al picco
        merge_overlaps=True
    )

    # Accedi ai risultati
    print(f"Picchi rilevati: {len(detector.peak_times_)}")
    print(f"Threshold HNR: {detector.threshold_:.2f} dB")

    # Ottieni informazioni dettagliate sui picchi
    peak_info = detector.get_peak_info()
    for info in peak_info:
        print(f"Picco {info['index']}: t={info['peak_time']:.3f}s, "
            f"finestra=[{info['window_start']:.3f}, {info['window_end']:.3f}]s")
        
    

    # Accesso diretto agli array
    # detector.peaks_         -> indici dei picchi nell'array HNR
    # detector.peak_times_    -> tempi centrali dei picchi (secondi)
    # detector.peak_windows_  -> lista di tuple (start_time, end_time)
    
    # optional playback
    # Estrai i segmenti audio
    segments = detector.extract_segments()
    for segment in segments:
        print(f"   - Riproduzione segmento di {len(segment)/sr:.3f} secondi")
        sd.play(segment, sr)
        sd.wait()
        # aspetta mezzo secondo prima di continuare
        sd.sleep(500)

    # Visualizza i risultati
    detector.plot_analysis(ch_num=1)


    print("="*80)
    print(f"ANALISI COMPLETATA !")
    print("="*80)
    

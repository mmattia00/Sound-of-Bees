import sounddevice as sd
import soundfile as sf
from datetime import datetime, timezone, timedelta
import argparse
import sys
from pathlib import Path
import queue
import time
import threading
import numpy as np  # serve per concatenare blocchi


q1 = queue.Queue()
q2 = queue.Queue()
stop_event = threading.Event()


def audio_callback_1(indata, frames, time_info, status):
    if status:
        print("Master status:", status, file=sys.stderr)
    q1.put(indata.copy())


def audio_callback_2(indata, frames, time_info, status):
    if status:
        print("Slave status:", status, file=sys.stderr)
    q2.put(indata.copy())


def resolve_alsa_card_index(card_label):
    with open("/proc/asound/cards", "r") as f:
        lines = f.readlines()

    for line in lines:
        if f"[{card_label}" in line:
            return int(line.strip().split()[0])

    raise ValueError(f"ALSA card label '{card_label}' not found in /proc/asound/cards")


def compute_chunk_duration(now, base_duration, align_chunks):
    if not align_chunks:
        return base_duration

    # allinea al prossimo multiplo di base_duration sul minuto
    next_chunk_time = (now + timedelta(seconds=base_duration)).replace(second=0, microsecond=0)
    while next_chunk_time < now:
        next_chunk_time += timedelta(seconds=base_duration)

    duration_to_next_boundary = (next_chunk_time - now).total_seconds()
    return min(duration_to_next_boundary, base_duration)


def writer_thread_both(q1, q2, samplerate, channels, subtype, output_dir, duration, align_chunks):
    chunk_idx = 0

    while not stop_event.is_set():
        current_time = datetime.now()
        chunk_duration = compute_chunk_duration(current_time, duration, align_chunks)

        chunk_start_dt = datetime.now(timezone.utc)

        master_dir = Path(output_dir) / "master"
        slave_dir = Path(output_dir) / "slave"
        master_dir.mkdir(parents=True, exist_ok=True)
        slave_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{chunk_start_dt.strftime('%Y-%m-%dT%H_%M_%S.%f')}Z.wav"
        master_path = master_dir / filename
        slave_path = slave_dir / filename

        print(f"[writer] Recording chunk {chunk_idx} for {chunk_duration:.3f} s")
        print(f"         master -> {master_path}")
        print(f"         slave  -> {slave_path}")

        # buffer temporanei per accumulare i blocchi di questo chunk
        buffers1 = []
        buffers2 = []

        chunk_start_monotonic = time.time()

        while (time.time() - chunk_start_monotonic < chunk_duration) and not stop_event.is_set():
            # MASTER
            try:
                data1 = q1.get(timeout=0.1)
                buffers1.append(data1)
            except queue.Empty:
                pass

            # SLAVE
            try:
                data2 = q2.get(timeout=0.1)
                buffers2.append(data2)
            except queue.Empty:
                pass

        # concatena per avere un unico array per ciascun device
        if buffers1:
            data1_full = np.concatenate(buffers1, axis=0)
        else:
            data1_full = np.empty((0, channels))

        if buffers2:
            data2_full = np.concatenate(buffers2, axis=0)
        else:
            data2_full = np.empty((0, channels))

        # scrivi i due file
        if data1_full.shape[0] > 0:
            with sf.SoundFile(str(master_path), mode='w',
                              samplerate=samplerate,
                              channels=channels,
                              subtype=subtype,
                              format='WAV') as f_master:
                f_master.write(data1_full)

        if data2_full.shape[0] > 0:
            with sf.SoundFile(str(slave_path), mode='w',
                              samplerate=samplerate,
                              channels=channels,
                              subtype=subtype,
                              format='WAV') as f_slave:
                f_slave.write(data2_full)

        chunk_idx += 1


def main():
    parser = argparse.ArgumentParser(description="Continuous Audio Recording Script (2 devices, synchronized chunk writer)")
    parser.add_argument('--print-devices', action='store_true', help='Print list of audio devices and exit')
    parser.add_argument('--print-subtypes', action='store_true', help='Print list of available sound file subtypes and exit')

    args, remaining = parser.parse_known_args()

    if args.print_devices:
        print(sd.query_devices())
        sys.exit(0)

    if args.print_subtypes:
        print(sf.available_subtypes())
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Continuous Audio Recording Script (2 devices, synchronized chunk writer)")
    parser.add_argument('-d', '--duration', type=float, required=True,
                        help='Duration of each recording chunk in seconds')
    parser.add_argument('-t', '--total-duration', type=float,
                        help='Total duration of recording in seconds (default: runs indefinitely until cancelled)')
    parser.add_argument('-r', '--samplerate', type=int, required=True,
                        help='Sampling rate in Hz')
    parser.add_argument('-c', '--channels-per-device', type=int, default=10,
                        help='Number of input channels per device')
    parser.add_argument('--device1', type=str, required=True,
                        help='First device label/name substring for recording (e.g. BeeMaster)')
    parser.add_argument('--device2', type=str, required=True,
                        help='Second device label/name substring for recording (e.g. BeeSlave1)')
    parser.add_argument('--output-dir', type=str, default='test_audio_recordings',
                        help='Output directory for the recording files')
    parser.add_argument('--align-chunks', action='store_true',
                        help='Align recording chunks to specific time boundaries')
    parser.add_argument('--subtype', type=str, default='PCM_16',
                        help='Sound file subtype (e.g., PCM_16, PCM_24, FLOAT)')
    parser.add_argument('--blocksize', type=int, default=0,
                        help='Blocksize for sounddevice InputStream (0 = default)')
    parser.add_argument('--latency', type=str, default='high',
                        help="Latency setting for InputStream (e.g. 'low', 'high')")

    args = parser.parse_args(remaining)

    if args.total_duration is None:
        args.total_duration = float('inf')

    card1_index = resolve_alsa_card_index(args.device1)
    card2_index = resolve_alsa_card_index(args.device2)

    device1_query = f"hw:{card1_index},0"
    device2_query = f"hw:{card2_index},0"

    writer = threading.Thread(
        target=writer_thread_both,
        args=(q1, q2, args.samplerate, args.channels_per_device, args.subtype,
              args.output_dir, args.duration, args.align_chunks),
        daemon=True
    )

    try:
        with sd.InputStream(samplerate=args.samplerate,
                            device=device1_query,
                            channels=args.channels_per_device,
                            callback=audio_callback_1,
                            blocksize=args.blocksize,
                            latency=args.latency), \
             sd.InputStream(samplerate=args.samplerate,
                            device=device2_query,
                            channels=args.channels_per_device,
                            callback=audio_callback_2,
                            blocksize=args.blocksize,
                            latency=args.latency):

            print('#' * 80)
            print('Recording from TWO devices with SYNCHRONIZED chunk writer... Press Ctrl+C to stop.')
            print(f"Master: [{card1_index}] {args.device1}")
            print(f"Slave : [{card2_index}] {args.device2}")
            print(f"Samplerate: {args.samplerate} | Channels/device: {args.channels_per_device} | Subtype: {args.subtype}")
            print(f"Blocksize: {args.blocksize} | Latency: {args.latency}")
            print('#' * 80)

            writer.start()

            start_time = time.time()

            while True:
                if args.total_duration != float('inf'):
                    if time.time() - start_time >= args.total_duration:
                        print("Total recording duration reached. Exiting.")
                        break
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nRecording interrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        stop_event.set()
        writer.join(timeout=2)


if __name__ == "__main__":
    main()
import sounddevice as sd
import soundfile as sf
from datetime import datetime, timezone, timedelta
import argparse
import sys
from pathlib import Path
import queue
import time
import numpy as np  # <--- serve per concatenare i canali


def get_coord_filename(prefix, output_dir, point_name):
    filename = f"{prefix}_Point_{point_name}.wav"
    return output_dir / filename


# due queue, una per device
q1 = queue.Queue()
q2 = queue.Queue()


def audio_callback_1(indata, frames, time_info, status):
    if status:
        print("Device 1 status:", status, file=sys.stderr)
    q1.put(indata.copy())


def audio_callback_2(indata, frames, time_info, status):
    if status:
        print("Device 2 status:", status, file=sys.stderr)
    q2.put(indata.copy())


def main():
    parser = argparse.ArgumentParser(description="Continuous Audio Recording Script (2 devices)")
    parser.add_argument('--print-devices', action='store_true', help='Print list of audio devices and exit')
    parser.add_argument('--print-subtypes', action='store_true', help='Print list of available sound file subtypes and exit')

    # parse parziale per gestire subito le opzioni di sola stampa
    args, remaining = parser.parse_known_args()

    if args.print_devices:
        print(sd.query_devices())
        sys.exit(0)

    if args.print_subtypes:
        print(sf.available_subtypes())
        sys.exit(0)

    # ora definiamo il parser completo per la registrazione
    parser = argparse.ArgumentParser(description="Continuous Audio Recording Script (2 devices)")
    parser.add_argument('-d', '--duration', type=float, required=True,
                        help='Duration of each recording chunk in seconds')
    parser.add_argument('-t', '--total-duration', type=float,
                        help='Total duration of recording in seconds (default: runs indefinitely until cancelled)')
    parser.add_argument('-r', '--samplerate', type=int, required=True,
                        help='Sampling rate in Hz')
    parser.add_argument('-c', '--channels-per-device', type=int, default=16,
                        help='Number of input channels per device')
    parser.add_argument('--device1', type=int, required=True, help='First device index for recording')
    parser.add_argument('--device2', type=int, required=True, help='Second device index for recording')
    parser.add_argument('--prefix', type=str, default='audio_recording', help='Custom prefix for the filename')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for the recording files')
    parser.add_argument('--align-chunks', action='store_true', help='Align recording chunks to specific time boundaries')
    parser.add_argument('--subtype', type=str, default='PCM_16', help='Sound file subtype (e.g., PCM_16, PCM_24, FLOAT)')
    parser.add_argument('--point', type=str, required=True, help='Point identifier')

    args = parser.parse_args(remaining)


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # durata totale default = infinita
    if args.total_duration is None:
        args.total_duration = float('inf')

    chunk_idx = 0

    try:
        with sd.InputStream(samplerate=args.samplerate,
                            device=args.device1,
                            channels=args.channels_per_device,
                            callback=audio_callback_1), \
             sd.InputStream(samplerate=args.samplerate,
                            device=args.device2,
                            channels=args.channels_per_device,
                            callback=audio_callback_2):

            print('#' * 80)
            print('Recording from TWO devices... Press Ctrl+C to stop.')
            print('#' * 80)

            start_time = datetime.now()
            elapsed_time = 0.0

            while elapsed_time < args.total_duration:
                current_time = datetime.now()

                if args.align_chunks:
                    next_chunk_time = (current_time + timedelta(seconds=args.duration)).replace(second=0, microsecond=0)
                    while next_chunk_time < current_time:
                        next_chunk_time += timedelta(seconds=args.duration)
                    duration_to_next_boundary = (next_chunk_time - current_time).total_seconds()
                    chunk_duration = min(duration_to_next_boundary, args.duration)
                else:
                    chunk_duration = args.duration

                print(f"Recording audio chunk {chunk_idx} for {chunk_duration} seconds...")

                filename = get_coord_filename(args.prefix, output_dir, args.point)

                # apri file 32 canali (2x device)
                total_channels = args.channels_per_device * 2

                with sf.SoundFile(filename, mode='x',
                                  samplerate=args.samplerate,
                                  channels=total_channels,
                                  subtype=args.subtype) as file:
                    chunk_start_time = time.time()
                    while time.time() - chunk_start_time < chunk_duration:
                        # blocco da device 1
                        data1 = q1.get()
                        # blocco da device 2
                        data2 = q2.get()

                        # assicurati che abbiano stesso n. frame
                        min_frames = min(len(data1), len(data2))
                        if len(data1) != len(data2):
                            data1 = data1[:min_frames, :]
                            data2 = data2[:min_frames, :]

                        # concatena sui canali: (N, 32)
                        data32 = np.concatenate((data1, data2), axis=1)
                        file.write(data32)

                chunk_idx += 1
                elapsed_time = (datetime.now() - start_time).total_seconds()

            print("Total recording duration reached. Exiting.")

    except KeyboardInterrupt:
        print("\nRecording interrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

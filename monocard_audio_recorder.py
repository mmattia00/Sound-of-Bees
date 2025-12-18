from html import parser
import sounddevice as sd
import soundfile as sf
from datetime import datetime, timezone, timedelta
import argparse
import sys
from pathlib import Path
import queue
import time


# Function to convert datetime to formatted string
def dt_to_str(dt):
    """Converts a datetime object to a formatted string."""
    isoformat = "%Y-%m-%dT%H_%M_%S"
    dt_str = dt.strftime(isoformat)
    if dt.microsecond != 0:
        dt_str += ".{:06d}".format(dt.microsecond)
    if dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0:
        dt_str += "Z"
    return dt_str


# Function to generate timestamped filename
def get_timestamped_filename(prefix, output_dir, use_utc=False):
    if use_utc:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.now()
    timestamp_str = dt_to_str(now)
    filename = f"{prefix}_{timestamp_str}.wav"
    return output_dir / filename


def get_coord_filename(prefix, output_dir, x, y):
    # formattazione con 2 decimali; puoi cambiare la precisione
    x_str = f"{x:.2f}"
    y_str = f"{y:.2f}"
    filename = f"{prefix}_X_{x_str}_Y_{y_str}.wav"
    return output_dir / filename



# Callback function for streaming audio data
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


# Main function
def main():
    # 1) Parser minimale per le opzioni "solo stampa"
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--print-devices', action='store_true',
                             help='Print list of audio devices and exit')
    base_parser.add_argument('--print-subtypes', action='store_true',
                             help='Print list of available sound file subtypes and exit')

    # parse parziale: prende solo queste opzioni e lascia il resto in "remaining"
    args_base, remaining = base_parser.parse_known_args()

    if args_base.print_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        sys.exit(0)

    if args_base.print_subtypes:
        print("Available sound file subtypes:")
        print(sf.available_subtypes())
        sys.exit(0)

    # 2) Parser completo per la registrazione
    parser = argparse.ArgumentParser(
        description="Continuous Audio Recording Script",
        parents=[base_parser]  # così --print-devices resta documentato in -h
    )
    parser.add_argument('-d', '--duration', type=float,
                        help='Duration of each recording chunk in seconds')
    parser.add_argument('-t', '--total-duration', type=float,
                        help='Total duration of recording in seconds (default: runs indefinitely until cancelled)')
    parser.add_argument('-r', '--samplerate', type=int,
                        help='Sampling rate in Hz (default: maximum samplerate of the selected device if not provided)')
    parser.add_argument('-c', '--channels', type=int,
                        help='Number of audio channels (default: maximum number of input channels of the selected device if not provided)')
    parser.add_argument('--device', type=int,
                        help='Device index for recording')
    parser.add_argument('--prefix', type=str, default='audio_recording',
                        help='Custom prefix for the filename')
    parser.add_argument('--use-utc', action='store_true',
                        help='Use UTC time for the filename timestamp (default is local time)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for the recording files')
    parser.add_argument('--align-chunks', action='store_true',
                        help='Align recording chunks to specific time boundaries')
    parser.add_argument('--subtype', type=str, default='PCM_16',
                        help='Sound file subtype (e.g., PCM_16, PCM_24, FLOAT)')
    parser.add_argument('--x', type=float, required=True,
                        help='Coordinata X')
    parser.add_argument('--y', type=float, required=True,
                        help='Coordinata Y')

    args = parser.parse_args(remaining)


    if args.device is None:
        print("No recording device specified. Please choose a device from the available list below:")
        print(sd.query_devices())
        sys.exit(1)
   
    if args.print_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        sys.exit(0)


    if args.print_subtypes:
        print("Available sound file subtypes:")
        print(sf.available_subtypes())
        sys.exit(0)


    # Set default samplerate and channels if not provided
    if args.samplerate is None or args.channels is None:
        device_info = sd.query_devices(args.device, 'input')
        if args.samplerate is None:
            # soundfile expects an int, sounddevice provides a float:
            args.samplerate = int(device_info['default_samplerate'])
        if args.channels is None:
            args.channels = device_info['max_input_channels']


    # Determine chunk duration and total duration based on input
    if args.total_duration and not args.duration:
        args.duration = args.total_duration
    elif args.duration and not args.total_duration:
        args.total_duration = args.duration


    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


    global q
    q = queue.Queue()


    try:
        with sd.InputStream(samplerate=args.samplerate, device=1,
                            channels=16, callback=audio_callback):
            print('#' * 80)
            print('Recording... Press Ctrl+C to stop the recording.')
            print('#' * 80)


            start_time = datetime.now()
            elapsed_time = 0


            while True:
                current_time = datetime.now()
                if args.align_chunks:
                    # Calculate the next alignment point based on the duration argument
                    next_chunk_time = (current_time + timedelta(seconds=args.duration)).replace(second=0, microsecond=0)
                    while next_chunk_time < current_time:
                        next_chunk_time += timedelta(seconds=args.duration)


                    duration_to_next_boundary = (next_chunk_time - current_time).total_seconds()
                    chunk_duration = min(duration_to_next_boundary, args.duration)
                else:
                    chunk_duration = args.duration


                print(f"Recording audio chunk for {chunk_duration} seconds...")
                if args.total_duration and elapsed_time >= args.total_duration:
                    print("Total recording duration reached. Exiting.")
                    break


                # Create a new chunk file
                # filename = get_timestamped_filename(args.prefix, output_dir, use_utc=args.use_utc)
                filename = get_coord_filename(args.prefix, output_dir, args.x, args.y)



                # Write chunks for the specified duration
                with sf.SoundFile(filename, mode='x', samplerate=args.samplerate,
                                  channels=args.channels, subtype=args.subtype) as file:
                    chunk_start_time = time.time()
                    while time.time() - chunk_start_time < chunk_duration:
                        file.write(q.get())


                elapsed_time = (datetime.now() - start_time).total_seconds()


    except KeyboardInterrupt:
        print("\nRecording interrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

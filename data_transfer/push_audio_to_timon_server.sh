#!/bin/bash
# Batch uploader for daily audio folders.
# It scans the input directory in chronological order, skips folders that were already sent,
# and uploads each ready folder to the timon remote with rclone.
#
# Usage: push_audio_to_timon_server.sh [--verbose] <path_to_audio_folder> [rclone_remote]

set -u

# Parse --verbose flag if present
VERBOSE="${VERBOSE:-FALSE}"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE="TRUE"
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [--verbose] <path_to_audio_folder> [rclone_remote]"
    exit 1
fi

log() { echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') - $*"; }

vlog() {
    if [ "$VERBOSE" = "TRUE" ]; then
        echo "[VERBOSE] $(date -u +'%Y-%m-%dT%H:%M:%SZ') - $*" >&2
    fi
}

AUDIO_FOLDER="$1"
RCLONE_REMOTE="${2:-timon:/audio}"
CHECK_INTERVAL_SEC=300  # default scan interval (5 minutes)
LAST_SENT_FILE="$AUDIO_FOLDER/.last_sent_folder"

if [ ! -d "$AUDIO_FOLDER" ]; then
    echo "Error: Directory '$AUDIO_FOLDER' does not exist."
    exit 1
fi

vlog "Script started with VERBOSE=$VERBOSE"
vlog "AUDIO_FOLDER=$AUDIO_FOLDER"
vlog "RCLONE_REMOTE=$RCLONE_REMOTE"
vlog "LAST_SENT_FILE=$LAST_SENT_FILE"

read_last_sent() {
    if [ -f "$LAST_SENT_FILE" ]; then
        cat "$LAST_SENT_FILE"
    else
        echo ""
    fi
}

write_last_sent() {
    echo "$1" > "$LAST_SENT_FILE"
}

# Compare folder names lexicographically (works for ISO-like YYYY-MM-DDTHH).
is_less_or_equal() {
    [ "$1" = "" ] && return 0
    [ "$1" = "$2" ] && return 0
    if [[ "$1" < "$2" ]]; then
        return 0
    else
        return 1
    fi
}

while true; do
    vlog "=== SCAN CYCLE START ==="
    now_hour_utc=$(date -u +"%Y-%m-%dT%H")
    vlog "Current UTC hour: $now_hour_utc"
    last_sent="$(read_last_sent)"
    vlog "Last sent folder: ${last_sent:-[none]}"
    next_sleep=$CHECK_INTERVAL_SEC
    vlog "Initial next_sleep: $next_sleep seconds"

    # Iterate over subfolders in chronological order.
    vlog "Listing directories in $AUDIO_FOLDER..."
    for dir in "$AUDIO_FOLDER"/*; do
        [ -d "$dir" ] || continue
        base=$(basename "$dir")
        vlog "  Processing directory: $base"

        # Skip folders that were already sent in a previous cycle.
        if [[ -n "$last_sent" ]] && ! [[ "$base" > "$last_sent" ]]; then
            vlog "    -> Skipping (already sent or older)"
            continue
        fi
        vlog "    -> Not yet sent, checking contents..."

        # Inspect the newest WAV file in the folder to decide if recording is complete.
        lastfile=$(ls -1 "$dir"/*.wav 2>/dev/null | sort || true)
        if [ -z "$lastfile" ]; then
            log "No wav files in $base, skipping."
            vlog "    -> No .wav files found"
            continue
        fi
        lastfile=$(echo "$lastfile" | tail -n1)
        fname=$(basename "$lastfile")
        vlog "    -> Last file: $fname"

        # extract hour, minute, second from filename like 2025-09-10T13_47_42.199913Z.wav
        if [[ "$fname" =~ T([0-9]{2})_([0-9]{2})_([0-9]{2})\.([0-9]+)Z\.wav$ ]]; then
            file_hour=${BASH_REMATCH[1]}
            file_min=${BASH_REMATCH[2]}
            file_sec=${BASH_REMATCH[3]}
            vlog "    -> Extracted: hour=$file_hour min=$file_min sec=$file_sec"
        else
            log "Filename $fname does not match expected pattern, marking folder $base for manual check. Skipping."
            vlog "    -> Filename does not match pattern, skipping"
            continue
        fi

        send_folder=false

        # If the folder is older than the current hour, it is considered complete.
        if [[ "$base" < "$now_hour_utc" ]]; then
            vlog "    -> Folder is from past hour, ready to send"
            send_folder=true
        else
            # Otherwise, wait until the current hour looks complete enough to transfer.
            vlog "    -> Folder is from current hour, checking if complete..."
            if [ "$file_min" -eq 59 ]; then
                vlog "    -> Last file is at minute 59, waiting 2 minutes to ensure recording is complete..."
                sleep 120
                vlog "    -> Folder ready to send"
                send_folder=true
            else
                # Compute the next wake-up delay: remaining time in the hour plus a safety buffer.
                remaining_seconds=$(( (59 - 10#$file_min)*60 + 5*60 ))
                vlog "    -> Folder incomplete (minute=$file_min, not 59). Estimated wait: $remaining_seconds seconds"

                vlog "    -> Updating next_sleep from $next_sleep to $remaining_seconds"
                next_sleep=$remaining_seconds

                log "Folder $base incomplete (last file $fname). Will recheck after ~$remaining_seconds seconds."
                send_folder=false
            fi
        fi

        if [ "$send_folder" = true ]; then
            # Once the folder is considered ready, push it to the remote destination.
            log "Sending folder $base to $RCLONE_REMOTE/$base"
            vlog "    -> [ACTION] Starting rclone copy..."
            # Capture stderr to a temporary file and append failures to RCLONE_ERROR_LOG.
            RCLONE_ERROR_LOG="$AUDIO_FOLDER/rclone_errors.log"
            tmp_err_file=$(mktemp)
            vlog "    -> Temp error file: $tmp_err_file"
            vlog "    -> Executing: rclone copy \"$dir\" \"$RCLONE_REMOTE/$base\" -v -P --transfer 24 --create-empty-src-dirs"
            if rclone copy "$dir" "$RCLONE_REMOTE/$base" -v -P --transfers 24 --create-empty-src-dirs --timeout 30s --contimeout 15s --retries 3 --low-level-retries 3 --retries-sleep 5s 2>"$tmp_err_file"; then                log "Successfully sent $base"
                vlog "    -> [SUCCESS] rclone copy completed"
                write_last_sent "$base"
                last_sent="$base"
                vlog "    -> Wrote last_sent: $base"
                rm -f "$tmp_err_file"
            else
                rc=$?
                log "rclone failed for $base (rc=$rc) — will retry later. Details appended to $RCLONE_ERROR_LOG"
                vlog "    -> [FAILURE] rclone returned $rc, logging error details"
                echo "---- $(date -u +'%Y-%m-%dT%H:%M:%SZ') ERROR sending $base rc=$rc ----" >> "$RCLONE_ERROR_LOG"
                tail -n 200 "$tmp_err_file" >> "$RCLONE_ERROR_LOG" 2>/dev/null || cat "$tmp_err_file" >> "$RCLONE_ERROR_LOG"
                echo "" >> "$RCLONE_ERROR_LOG"
                vlog "    -> Error details appended to $RCLONE_ERROR_LOG"
                rm -f "$tmp_err_file"
            fi
        fi
    done
    vlog "=== SCAN CYCLE END ==="

    # Sleep for the chosen interval before starting the next scan cycle.
    log "Sleeping for $next_sleep seconds before next scan."
    vlog "About to sleep for $next_sleep seconds..."
    sleep "$next_sleep"
done



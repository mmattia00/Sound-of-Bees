#!/bin/bash
# Continuous acquisition orchestrator for daily recording sessions.
# It runs audio recording during the daytime, transfers audio during the night,
# and keeps the video transfer tmux session alive for the full duration.
#
# Usage: ./start_data_acquisition.sh

set -u

log() { echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') - $*"; }

AUDIO_PID=""
TRANSFER_PID=""
TMUX_SESSION="txfr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_FOLDER="/tank/recordings2026"
RCLONE_REMOTE="/mnt/A26-03-0300/recordings2026"
PUSH_SCRIPT="$SCRIPT_DIR/push_recordings_to_timon_server.py"
LOG_FILE="$SCRIPT_DIR/daily_session.log"

TIMON_HOST="timon.cascb.uni-konstanz.de"
CONNECTIVITY_RETRY=60
RECORD_START_HOUR=5

[[ -f "$PUSH_SCRIPT" ]] || { log "ERROR: $PUSH_SCRIPT not found"; exit 1; }
[[ -f "/home/valerie/bb2026/start_tmux_txfr.sh" ]] || { log "ERROR: start_tmux_txfr.sh not found"; exit 1; }


# ── Handlers ─────────────────────────────────────────────────────────────────

cleanup() {
    log "Interrupt received, stopping everything..."

    if [[ -n "$AUDIO_PID" ]] && kill -0 "$AUDIO_PID" 2>/dev/null; then
        kill "$AUDIO_PID"
        wait "$AUDIO_PID" 2>/dev/null || true
        log "Audio recorder stopped (PID $AUDIO_PID)"
    fi

    if [[ -n "$TRANSFER_PID" ]] && kill -0 "$TRANSFER_PID" 2>/dev/null; then
        kill -TERM -"$TRANSFER_PID" 2>/dev/null || true
        sleep 2
        kill -KILL -"$TRANSFER_PID" 2>/dev/null || true
        wait "$TRANSFER_PID" 2>/dev/null || true
        log "Audio transfer stopped (PID $TRANSFER_PID)"
    fi

    # Close the tmux session used for the video transfer.
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        log "Closing tmux session '$TMUX_SESSION'..."
        tmux kill-session -t "$TMUX_SESSION"
        log "Tmux session closed"
    fi

    sudo /home/valerie/bb2026/stop_all_service.sh
    log "All processes stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM


# ── Connettività ─────────────────────────────────────────────────────────────
# Connectivity checks are used to decide when the audio transfer can proceed.

timon_reachable() {
    ping -c 1 -W 3 "$TIMON_HOST" &>/dev/null
}

wait_for_connectivity() {
    while ! timon_reachable; do
        if [[ "10#$(date +'%H')" -ge "$RECORD_START_HOUR" ]]; then
            log "WARN: Timon is unreachable and the deadline has been reached"
            return 1
        fi
        log "Timon is unreachable, retrying in ${CONNECTIVITY_RETRY}s..."
        sleep "$CONNECTIVITY_RETRY"
    done
    log "Timon is reachable"
    return 0
}


# ── Partial cleanup ──────────────────────────────────────────────────────────
# Remove leftover partial files on the remote side before starting a transfer.

cleanup_partials() {
    log "Cleaning up leftover .partial files on Timon..."
    if ! timon_reachable; then
        log "Timon is unreachable, skipping .partial cleanup"
        return 0
    fi
    rclone delete "$RCLONE_REMOTE/audio" \
        --include "*.partial" \
        --min-age 1h \
        --log-level INFO 2>&1 | tee -a "$LOG_FILE" || true
    log "Partial cleanup completed"
}


# ── Transfer video (tmux) ────────────────────────────────────────────────────
# The video transfer is started once and then kept alive in a tmux session.

start_video_transfer() {
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        log "Tmux session '$TMUX_SESSION' is already active, skipping"
        return 0
    fi
    log "Starting video transfer via tmux (Jacob's script)..."
    bash "/home/valerie/bb2026/start_tmux_txfr.sh"
    log "Tmux session '$TMUX_SESSION' started"
}


# ── Transfer audio (python) ──────────────────────────────────────────────────
# Audio transfer is launched as a separate Python process and retried within the night window.

run_audio_transfer() {
    log "--- Starting audio transfer ---"

    wait_for_connectivity || return 2

    setsid bash -c "
        python3 '$PUSH_SCRIPT' '$BASE_FOLDER' '$RCLONE_REMOTE' --media audio
    " &
    TRANSFER_PID=$!
    log "Audio transfer PID: $TRANSFER_PID"

    while kill -0 "$TRANSFER_PID" 2>/dev/null; do
        if [[ "10#$(date +'%H')" -ge "$RECORD_START_HOUR" ]]; then
            log "WARN: deadline reached, stopping audio transfer..."
            kill -TERM -"$TRANSFER_PID" 2>/dev/null || true
            sleep 2
            kill -KILL -"$TRANSFER_PID" 2>/dev/null || true
            wait "$TRANSFER_PID" 2>/dev/null || true
            TRANSFER_PID=""
            return 2
        fi
        sleep 30
    done

    wait "$TRANSFER_PID"
    local rc=$?
    TRANSFER_PID=""

    if [[ $rc -ne 0 ]]; then
        log "WARN: audio transfer ended with error (rc=$rc)"
        return 1
    fi

    log "--- Audio transfer completed ---"
    return 0
}


# ── Recording ────────────────────────────────────────────────────────────────
# Audio recording runs during the daytime window.

start_recording() {
    log "=== START RECORDING ==="
    python3 "$SCRIPT_DIR/jack_multicard_audio_recorder.py" \
        -d 60 -o "$BASE_FOLDER/audio" >/dev/null 2>&1 &
    AUDIO_PID=$!
    log "Audio recorder PID: $AUDIO_PID"
    sudo /home/valerie/bb2026/start_all_service.sh
    log "=== RECORDING ACTIVE ==="
}

stop_recording() {
    log "=== STOP RECORDING ==="
    if [[ -n "$AUDIO_PID" ]] && kill -0 "$AUDIO_PID" 2>/dev/null; then
        kill "$AUDIO_PID"
        wait "$AUDIO_PID" 2>/dev/null || true
        log "Audio recorder stopped (PID $AUDIO_PID)"
    fi
    AUDIO_PID=""
    sudo /home/valerie/bb2026/stop_all_service.sh
    log "=== RECORDING STOPPED ==="
}

wait_until_hour() {
    local target=$1
    while [[ "10#$(date +'%H')" -lt "$target" ]]; do
        sleep 60
    done
}


# ── Loop principale ──────────────────────────────────────────────────────────
# The main loop alternates between the transfer window and the recording window.

log "=== SESSION LOOP START ==="

# Start the tmux-based video transfer immediately; it stays alive for the whole run.
start_video_transfer

while true; do
    current_hour=$(date +'%H')
    log "--- New cycle --- hour: ${current_hour}:xx"

    # ── TRANSFER window (00:00 – 04:59) ─────────────────────────────────────
    if [[ "10#$current_hour" -lt "$RECORD_START_HOUR" ]]; then
        log "Transfer window: starting audio transfer (video is already running via tmux)"

        cleanup_partials

        transfer_ok=false
        while [[ "10#$(date +'%H')" -lt "$RECORD_START_HOUR" ]]; do
            run_audio_transfer
            rc=$?
            if [[ $rc -eq 0 ]]; then
                transfer_ok=true
                log "Audio transfer completed, waiting for ${RECORD_START_HOUR}:00..."
                break
            elif [[ $rc -eq 2 ]]; then
                break
            else
                log "Audio transfer error (rc=$rc), retrying in 60s if still inside the window..."
                sleep 60
            fi
        done

        if ! $transfer_ok; then
            log "Transfer window exhausted -- Python will resume from the incomplete day tomorrow"
        fi

        wait_until_hour "$RECORD_START_HOUR"
    fi

    # ── RECORDING window (05:00 – 23:59) ─────────────────────────────────────
    log "Recording window: starting recording"
    start_recording

    log "Recording in progress, waiting for midnight..."
    while [[ "10#$(date +'%H')" -ne 0 ]]; do
        sleep 60
    done

    stop_recording
    log "--- Cycle completed, restarting ---"
done
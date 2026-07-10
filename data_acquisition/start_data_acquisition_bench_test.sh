#!/bin/bash
# Bench session wrapper for data acquisition and transfer.
# It records audio chunks, triggers the transfer step, and repeats the cycle
# for a limited number of iterations so the pipeline can be tested safely.
# It's meant to test if the full acquisition pipeline (with audio recording and transfer) works correctly on the bench before running it on the field.
# is resilient to eventual network interruptions, or other unexpected issues.
#
# Usage: ./bench_session_audio.sh [record_duration] [transfer_window] [max_cycles]

set -u

log() { echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') - $*"; }

AUDIO_PID=""
TRANSFER_PID=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_FOLDER="/tank/recordings2026"
RCLONE_REMOTE="/mnt/A26-03-0300/recordings2026"
PUSH_SCRIPT="$SCRIPT_DIR/push_recordings_to_timon_server.py"
LOG_FILE="$SCRIPT_DIR/bench_session_audio.log"

TIMON_HOST="timon.cascb.uni-konstanz.de"
CONNECTIVITY_RETRY=10   # bench: 10s invece di 60s

RECORD_DURATION=${1:-30}
TRANSFER_WINDOW=${2:-60}
MAX_CYCLES=${3:-2}

log "=== BENCH CONFIG: record=${RECORD_DURATION}s | transfer_window=${TRANSFER_WINDOW}s | cycles=${MAX_CYCLES} ==="
log "=== Daily resume is handled by push_recordings_to_timon_server.py (.last_sent_state) ==="

TRANSFER_DEADLINE=0

set_transfer_deadline() {
    TRANSFER_DEADLINE=$(( $(date +%s) + TRANSFER_WINDOW ))
    log "Transfer deadline: $(date -d @"$TRANSFER_DEADLINE" -u +'%H:%M:%S UTC')"
}

transfer_deadline_reached() {
    [[ $(date +%s) -ge $TRANSFER_DEADLINE ]]
}


# ── Handlers ─────────────────────────────────────────────────────────────────

cleanup() {
    log "Interrupt received, stopping everything..."

    if [[ -n "$AUDIO_PID" ]] && kill -0 "$AUDIO_PID" 2>/dev/null; then
        kill "$AUDIO_PID"
        wait "$AUDIO_PID" 2>/dev/null || true
        log "Audio recorder stopped"
    fi

    if [[ -n "$TRANSFER_PID" ]] && kill -0 "$TRANSFER_PID" 2>/dev/null; then
        kill -TERM -"$TRANSFER_PID" 2>/dev/null || true
        sleep 2
        kill -KILL -"$TRANSFER_PID" 2>/dev/null || true
        wait "$TRANSFER_PID" 2>/dev/null || true
        log "Transfer stopped"
    fi

    sudo /home/valerie/bb2026/stop_all_service.sh 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM


# ── Connettività ─────────────────────────────────────────────────────────────

timon_reachable() {
    ping -c 1 -W 3 "$TIMON_HOST" &>/dev/null
}

wait_for_connectivity() {
    while ! timon_reachable; do
        if transfer_deadline_reached; then
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

cleanup_partials() {
    log "Cleaning up .partial files on Timon..."
    if ! timon_reachable; then
        log "Timon is unreachable, skipping"
        return 0
    fi
    rclone delete "$RCLONE_REMOTE/audio" \
        --include "*.partial" --min-age 1h \
        --log-level INFO 2>&1 | tee -a "$LOG_FILE" || true
}


# ── Transfer audio ────────────────────────────────────────────────────────────

run_audio_transfer() {
    log "--- Starting audio transfer ---"

    wait_for_connectivity || return 2

    setsid bash -c "
        python3 '$PUSH_SCRIPT' '$BASE_FOLDER' '$RCLONE_REMOTE' --media audio
    " &
    TRANSFER_PID=$!
    log "Transfer PID: $TRANSFER_PID"

    while kill -0 "$TRANSFER_PID" 2>/dev/null; do
        if transfer_deadline_reached; then
            log "WARN: deadline reached, stopping transfer..."
            kill -TERM -"$TRANSFER_PID" 2>/dev/null || true
            sleep 2
            kill -KILL -"$TRANSFER_PID" 2>/dev/null || true
            wait "$TRANSFER_PID" 2>/dev/null || true
            TRANSFER_PID=""
            return 2
        fi
        sleep 5   # bench: poll every 5 seconds
    done

    wait "$TRANSFER_PID"
    local rc=$?
    TRANSFER_PID=""

    [[ $rc -ne 0 ]] && { log "WARN: transfer ended with error (rc=$rc)"; return 1; }
    log "--- Audio transfer completed ---"
    return 0
}


# ── Recording ────────────────────────────────────────────────────────────────

start_recording() {
    log "=== START RECORDING ==="
    python3 "$SCRIPT_DIR/jack_multicard_audio_recorder.py" \
        -d 60 -o "$BASE_FOLDER/audio" >/dev/null 2>&1 &
    AUDIO_PID=$!
    log "Audio recorder PID: $AUDIO_PID"
    sudo /home/valerie/bb2026/start_all_service.sh 2>/dev/null || true
}

stop_recording() {
    log "=== STOP RECORDING ==="
    if [[ -n "$AUDIO_PID" ]] && kill -0 "$AUDIO_PID" 2>/dev/null; then
        kill "$AUDIO_PID"
        wait "$AUDIO_PID" 2>/dev/null || true
    fi
    AUDIO_PID=""
    sudo /home/valerie/bb2026/stop_all_service.sh 2>/dev/null || true
}


# ── Loop bench ────────────────────────────────────────────────────────────────

log "=== BENCH SESSION LOOP START ==="

# Show the Python transfer state at startup.
state_file="$BASE_FOLDER/audio/.last_sent_state"
if [[ -f "$state_file" ]]; then
    log "Current Python state: $(cat "$state_file")"
else
    log "No Python state found -- the scan will start from the first available day"
fi

for (( cycle=1; cycle<=MAX_CYCLES; cycle++ )); do
    log ""
    log "━━━ CYCLE $cycle / $MAX_CYCLES ━━━"

    log "Recording for ${RECORD_DURATION}s..."
    start_recording
    sleep "$RECORD_DURATION"
    stop_recording

    set_transfer_deadline
    cleanup_partials

    transfer_ok=false
    while ! transfer_deadline_reached; do
        run_audio_transfer
        rc=$?
        if [[ $rc -eq 0 ]]; then
            transfer_ok=true
            log "✓ Transfer completed in cycle $cycle"
            break
        elif [[ $rc -eq 2 ]]; then
            break
        else
            log "Error (rc=$rc), retrying in 15s..."
            sleep 15
        fi
    done

    $transfer_ok || log "WARN: cycle $cycle -- transfer not completed within the window"

    # Show the Python transfer state after each cycle.
    [[ -f "$state_file" ]] && log "Python state after cycle $cycle: $(cat "$state_file")"

    log "━━━ END OF CYCLE $cycle ━━━"
done

log ""
log "=== BENCH COMPLETED ==="
[[ -f "$state_file" ]] && log "Final Python state: $(cat "$state_file")"
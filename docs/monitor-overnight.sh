#!/bin/bash
# Monitor the overnight TQ4/TQ3 PPL run
# Checks every 5 minutes, logs status, alerts on completion or failure

LOGFILE="/Users/hchome/Lean_llama.cpp/docs/metal-tq4-tq3-results.txt"
MONITOR_LOG="/Users/hchome/Lean_llama.cpp/docs/monitor.log"
PROCESS_NAME="llama-perplexity"

echo "$(date): Monitor started" > "$MONITOR_LOG"

while true; do
    PID=$(pgrep -f "$PROCESS_NAME" 2>/dev/null)
    
    if [ -z "$PID" ]; then
        # Process is gone — did it finish or crash?
        if [ -f "$LOGFILE" ] && grep -q "Both configs complete" "$LOGFILE" 2>/dev/null; then
            echo "$(date): SUCCESS — both configs completed" >> "$MONITOR_LOG"
            # Show final PPL lines
            grep -E "Final estimate" "$LOGFILE" >> "$MONITOR_LOG"
            osascript -e 'display notification "Both TQ4/TQ3 160-chunk runs completed!" with title "Overnight PPL Done" sound name "Glass"' 2>/dev/null
            break
        elif [ -f "$LOGFILE" ] && grep -q "TQ4_0" "$LOGFILE" && ! grep -q "TQ3_0" "$LOGFILE" 2>/dev/null; then
            # TQ4 done but crashed before TQ3
            echo "$(date): PARTIAL — TQ4 completed but TQ3 may have crashed" >> "$MONITOR_LOG"
            grep -E "Final estimate" "$LOGFILE" >> "$MONITOR_LOG" 2>/dev/null
            osascript -e 'display notification "TQ3 run may have crashed! Check monitor.log" with title "Overnight PPL PARTIAL" sound name "Basso"' 2>/dev/null
            break
        else
            echo "$(date): CRASH — process died before completing" >> "$MONITOR_LOG"
            tail -5 "$LOGFILE" >> "$MONITOR_LOG" 2>/dev/null
            osascript -e 'display notification "Overnight PPL run crashed! Check monitor.log" with title "Overnight PPL FAILED" sound name "Basso"' 2>/dev/null
            break
        fi
    else
        # Still running — log progress
        LAST_CHUNK=$(grep -oE '\[[0-9]+\]' "$LOGFILE" 2>/dev/null | tail -1 | tr -d '[]')
        CURRENT_CONFIG="unknown"
        if grep -q "TQ3_0" "$LOGFILE" 2>/dev/null; then
            CURRENT_CONFIG="TQ3_0"
        elif grep -q "TQ4_0" "$LOGFILE" 2>/dev/null; then
            CURRENT_CONFIG="TQ4_0"
        fi
        echo "$(date): Running — config=$CURRENT_CONFIG chunk=$LAST_CHUNK pid=$PID" >> "$MONITOR_LOG"
    fi
    
    sleep 300  # check every 5 minutes
done

echo "$(date): Monitor exiting" >> "$MONITOR_LOG"

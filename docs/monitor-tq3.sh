#!/bin/bash
LOGFILE="/Users/hchome/Lean_llama.cpp/docs/metal-tq3-rerun-results.txt"
MONITOR_LOG="/Users/hchome/Lean_llama.cpp/docs/monitor-tq3.log"

echo "$(date): TQ3 monitor started" > "$MONITOR_LOG"

while true; do
    PID=$(pgrep -f "llama-perplexity" 2>/dev/null)
    if [ -z "$PID" ]; then
        if [ -f "$LOGFILE" ] && grep -q "Complete:" "$LOGFILE" 2>/dev/null; then
            echo "$(date): SUCCESS" >> "$MONITOR_LOG"
            grep "Final estimate" "$LOGFILE" >> "$MONITOR_LOG"
            osascript -e 'display notification "TQ3 160-chunk run completed!" with title "TQ3 PPL Done" sound name "Glass"' 2>/dev/null
        else
            echo "$(date): CRASH" >> "$MONITOR_LOG"
            tail -5 "$LOGFILE" >> "$MONITOR_LOG" 2>/dev/null
            osascript -e 'display notification "TQ3 run crashed again!" with title "TQ3 FAILED" sound name "Basso"' 2>/dev/null
        fi
        break
    else
        LAST_CHUNK=$(grep -oE '\[[0-9]+\]' "$LOGFILE" 2>/dev/null | tail -1 | tr -d '[]')
        echo "$(date): Running — chunk=$LAST_CHUNK pid=$PID" >> "$MONITOR_LOG"
    fi
    sleep 300
done
echo "$(date): Monitor exiting" >> "$MONITOR_LOG"

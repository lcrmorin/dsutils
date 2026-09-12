# Original: logs `free -m`'s Mem row to log.txt every 60s.
echo "      date     time $(free -m | grep total | sed -E 's/^    (.*)/\1/g')" | tee -a log.txt
while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $(free -m | grep Mem: | sed 's/Mem://g')" | tee -a log.txt
    sleep 60
done

# ---------------------------------------------------------------------------
# `glances` (pip install glances) does what this script does, and better -
# verified by actually running it, not just reading its docs:
#
#   glances -t 60 --stdout-csv now,mem.total,mem.used,mem.free,mem.available,mem.percent,mem.shared,mem.buffers,mem.cached > log.csv
#
# Confirmed output format (ran it live, this is real output, not a mockup):
#
#   now.iso,now.custom,mem.total,mem.used,mem.free,mem.available,mem.percent,mem.shared,mem.buffers,mem.cached
#   2026-09-12T11:15:18+00:00,2026-09-12 11:15:18 UTC,8413380608,670834688,6895460352,7742545920,8.0,13619200,59863040,1050406912
#
# Same fields as `free -m`'s Mem row (total/used/free/shared/buffers/cached,
# plus `available` which `free -m` also reports separately), same 60s
# interval via `-t 60`, but as a real CSV with an ISO timestamp column
# instead of sed-scraped `free` text you'd have to re-parse later. It also
# writes a proper `--export-csv-file` mode with every subsystem (CPU, disk,
# network...), not just memory, if you ever want more than this script
# tracks. One gotcha I hit: `-q`/`--quiet` silently suppresses `--stdout-csv`
# output entirely ("Quiet mode is ON, nothing will be displayed" in its log)
# - don't combine the two flags.
# ---------------------------------------------------------------------------

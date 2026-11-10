#!/bin/bash
# Cron entry for the nightly GPU pins: bench/nightly_pins.py has the whole story.
#
# Crontab line (installed by hand, after review; 02:30 every night):
#
#   30 2 * * * /export/home/tbuck/nightly/jaccpot/bench/nightly_pins.sh >> /export/scratch/tbuck/nightly/cron.log 2>&1
#
# The path is the nightly's own jaccpot worktree, which every run moves to
# origin/main, so the driver updates itself with a one-night lag. Arguments pass
# through (`--dry-run`, `--pins S1,DET1`, `--force`, ...). The verdicts land in
# /export/scratch/tbuck/nightly/<tag>/SUMMARY.md and one line per run in
# /export/scratch/tbuck/nightly/history.log; the exit status is 1 on any FAIL and
# 3 when another run holds the lock.
#
# The driver enforces --total-cap-h (10 h by default, at most 11); the `timeout`
# below is only a backstop. Its SIGTERM makes the driver kill its process group
# and write the summary; SIGKILL follows five minutes later.

main() {
    export PATH=/usr/local/bin:/usr/bin:/bin
    umask 022
    local here
    here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || exit 2
    exec timeout --kill-after=300 12h /usr/bin/python3 "$here/nightly_pins.py" "$@"
}

# Wrapped in a function so bash has read the whole file before the driver moves
# this worktree to a new commit.
main "$@"

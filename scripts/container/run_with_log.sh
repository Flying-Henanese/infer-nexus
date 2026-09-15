#!/bin/sh

# Run one container command with stdout and stderr persisted in a bind-mounted
# host log file. `exec` keeps the wrapped process as PID 1 for Docker signals.
set -eu

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
    echo "usage: $0 LOG_FILE -- COMMAND [ARG...]" >&2
    exit 64
fi

log_file=$1
shift 2

mkdir -p "$(dirname "$log_file")"
exec "$@" >>"$log_file" 2>&1

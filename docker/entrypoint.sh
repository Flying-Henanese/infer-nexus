#!/bin/sh
# Ownership-fixing, privilege-dropping entrypoint shared by the CUDA and
# Ascend images. Named volumes mounted at /tmp/ray keep the ownership of
# their first population: image rebuilds that change APP_UID/APP_GID, or
# root debugging sessions, leave the volume unwritable for the app user,
# and `ray start` fails with "Permission denied: /tmp/ray/ray_current_cluster".
# When the container starts as root, re-align the volume and drop to the
# app user before exec-ing the command. Supplementary groups (device groups
# injected via compose group_add) are preserved through --keep-groups.
set -e

APP_USER="${APP_USER:-infer-nexus}"

if [ "$(id -u)" = "0" ]; then
    if ! command -v setpriv >/dev/null 2>&1; then
        echo "entrypoint: setpriv not found; install util-linux in the image" >&2
        exit 1
    fi
    chown -R "${APP_USER}:${APP_USER}" /tmp/ray
    exec setpriv --reuid="${APP_USER}" --regid="${APP_USER}" --keep-groups -- "$@"
fi

exec "$@"

#!/bin/sh
set -e
chown -R agent:agent /data /tmp/generated_cvs 2>/dev/null || true
exec gosu agent "$@"

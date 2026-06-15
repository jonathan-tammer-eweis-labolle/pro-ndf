#!/bin/bash
set -e

# Editable-install the project on every container start (fast after first run)
cd /workspace
[ -f pyproject.toml ] && pip install -e . -q

exec "$@"

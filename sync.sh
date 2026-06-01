#!/bin/bash
set -euo pipefail

cd /lustrehome/fonty/CliCIT2026/BrainrotLLMs
UV=/lustrehome/fonty/.local/bin/uv

# Remove any existing torch to avoid conflicts
$UV sync

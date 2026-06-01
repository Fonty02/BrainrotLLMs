#!/bin/bash
# Submit all generated HTCondor judge job files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

condor_submit "$SCRIPT_DIR/judge_jobs_1.htc"

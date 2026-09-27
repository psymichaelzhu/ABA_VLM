#!/bin/bash
set -e

mkdir -p checkpoints/slip

curl -L "<CLIP_CHECKPOINT_URL>" \
  -o checkpoints/slip/clip_vitb16_25ep.pt

curl -L "<SIMCLR_CHECKPOINT_URL>" \
  -o checkpoints/slip/simclr_vitb16_25ep.pt

curl -L "<SLIP_CHECKPOINT_URL>" \
  -o checkpoints/slip/slip_vitb16_25ep.pt
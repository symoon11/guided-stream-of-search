#!/bin/bash

cd src

ckpt="/home/seungyong/guided-stream-of-search/stream-of-search/outputs/sos-gpt2/checkpoint-50000"

python augment.py --ckpt "$ckpt" --seed 0 --mode last --depth 1 $@
python augment.py --ckpt "$ckpt" --seed 0 --mode last --depth 2 $@

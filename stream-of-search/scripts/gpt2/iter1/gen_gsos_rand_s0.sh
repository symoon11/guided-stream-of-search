#!/bin/bash

cd src

ckpt="/home/seungyong/train-countdown/stream-of-search/outputs/sos-gpt2/checkpoint-50000"

python augment.py --ckpt "$ckpt" --seed 0 --mode rand --depth 1 $@
python augment.py --ckpt "$ckpt" --seed 0 --mode rand --depth 2 $@

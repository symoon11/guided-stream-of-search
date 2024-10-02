#!/bin/bash

cd src

ckpt="/home/seungyong/train-countdown/stream-of-search/outputs/gsos1-last-s0-gpt2/checkpoint-20000"

python sample.py --ckpt "$ckpt" --seed 0 $@
python augment.py --ckpt "$ckpt" --seed 0 --mode last --depth 1 $@
python augment.py --ckpt "$ckpt" --seed 0 --mode last --depth 2 $@

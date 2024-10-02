#!/bin/bash

cd src

ckpt="/home/seungyong/train-countdown/stream-of-search/outputs/gsos2-rand-s2-gpt2/checkpoint-20000"

python sample.py --ckpt "$ckpt" --seed 2 $@
python augment.py --ckpt "$ckpt" --seed 2 --mode rand --depth 1 $@
python augment.py --ckpt "$ckpt" --seed 2 --mode rand --depth 2 $@

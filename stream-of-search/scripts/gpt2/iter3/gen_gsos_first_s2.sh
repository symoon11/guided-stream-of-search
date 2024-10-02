#!/bin/bash

cd src

ckpt="/home/seungyong/train-countdown/stream-of-search/outputs/gsos2-first-s2-gpt2/checkpoint-20000"

python sample.py --ckpt "$ckpt" --seed 2 $@
python augment.py --ckpt "$ckpt" --seed 2 --mode first --depth 1 $@
python augment.py --ckpt "$ckpt" --seed 2 --mode first --depth 2 $@

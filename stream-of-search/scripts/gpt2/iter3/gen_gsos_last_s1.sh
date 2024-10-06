#!/bin/bash

cd src

ckpt="/home/seungyong/guided-stream-of-search/stream-of-search/outputs/gsos2-last-s1-gpt2/checkpoint-20000"

python sample.py --ckpt "$ckpt" --seed 1 $@
python augment.py --ckpt "$ckpt" --seed 1 --mode last --depth 1 $@
python augment.py --ckpt "$ckpt" --seed 1 --mode last --depth 2 $@

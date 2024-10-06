#!/bin/bash

cd src

ckpt="/home/seungyong/guided-stream-of-search/stream-of-search/outputs/gsos2-last-s2-gpt2/checkpoint-20000"

python sample.py --ckpt "$ckpt" --seed 2 $@
python augment.py --ckpt "$ckpt" --seed 2 --mode last --depth 1 $@
python augment.py --ckpt "$ckpt" --seed 2 --mode last --depth 2 $@

#!/bin/bash

cd src

ckpt="/home/seungyong/guided-stream-of-search/stream-of-search/outputs/sos-gpt2/checkpoint-50000"

python augment.py --ckpt "$ckpt" --seed 2 --mode first --depth 1 $@
python augment.py --ckpt "$ckpt" --seed 2 --mode first --depth 2 $@

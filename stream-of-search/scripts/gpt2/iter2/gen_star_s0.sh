#!/bin/bash

cd src

ckpt="/home/seungyong/guided-stream-of-search/stream-of-search/outputs/star1-s0-gpt2/checkpoint-20000"

python sample.py --ckpt "$ckpt" --seed 0 $@

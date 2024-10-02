#!/bin/bash

cd src

ckpt="/home/seungyong/train-countdown/stream-of-search/outputs/star1-s2-gpt2/checkpoint-20000"

python sample.py --ckpt "$ckpt" --seed 2 $@

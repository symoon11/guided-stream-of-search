#!/bin/bash

cd src

ckpt="/home/seungyong/train-countdown/stream-of-search/outputs/sos-gpt2/checkpoint-50000"

python sample.py --ckpt "$ckpt" --seed 1 $@

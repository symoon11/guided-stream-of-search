#!/bin/bash

cd src

data_dir="/home/seungyong/train-countdown/stream-of-search/data/b4-rand-final"

python countdown_generate.py --seed 4 --data_dir "$data_dir" --num_test_samples 10000 $@

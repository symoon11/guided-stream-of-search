#!/bin/bash

cd src

data_dir="/home/seungyong/guided-stream-of-search/stream-of-search/data/b4-rand"

python countdown_generate.py --seed 4 --data_dir "$data_dir" $@

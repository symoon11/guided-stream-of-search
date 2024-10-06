#!/bin/bash

cd src

port=$(ruby -e 'require "socket"; puts Addrinfo.tcp("", 0).bind {|s| s.local_address.ip_port }')

ckpt="/home/seungyong/guided-stream-of-search/stream-of-search/outputs/sos-gpt2/checkpoint-50000"

accelerate launch --config_file ../configs/accelerate.yaml --main_process_port ${port} train.py --config ../configs/gpt2/iter2/gsos-rand-s0.conf --reset --ckpt "$ckpt" $@

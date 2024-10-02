#!/bin/bash

cd src

port=$(ruby -e 'require "socket"; puts Addrinfo.tcp("", 0).bind {|s| s.local_address.ip_port }')

accelerate launch --config_file ../configs/accelerate.yaml --main_process_port ${port} train.py --config ../configs/gpt2/sos.conf $@

#!/bin/bash

port=$(ruby -e 'require "socket"; puts Addrinfo.tcp("", 0).bind {|s| s.local_address.ip_port }')

accelerate launch --config_file accelerate_cfgs/deepspeed_config.yaml --main_process_port ${port} main.py task=countdown alg=ppo name=ppo-op $@

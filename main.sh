#!/usr/bin/env bash

ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
source "${ROOT}/util.sh"

INIT_GPU_NUM=$(util::get_gpu_num)
util::log_info "init GPU num: $INIT_GPU_NUM"

# get major number and minor number from a legal GPU
DEV=/dev/$(ls /dev | grep nvidia[0-9] | head -n 1)
DEV_NUMBER=$(printf "%d %d" $(stat --format "0x%t 0x%T" $DEV))

GPU_NO=0
while :
do
    # skip this no if device file already exists
    if [ -c "/dev/nvidia$GPU_NO" ]; then
        util::log_debug "/dev/nvidia$GPU_NO exists, skip"
        GPU_NO=`expr $GPU_NO + 1`
        continue
    fi

    CURRENT_GPU_NUM=$(util::get_gpu_num)

    # create specify device file to trick cgroup
    mknod -m 666 /dev/nvidia$GPU_NO c $DEV_NUMBER
    
    # break if have got all GPUs on the host
    if [ $(util::get_gpu_num) == "$CURRENT_GPU_NUM" ]; then
        util::log_debug "delete redundant /dev/nvidia$GPU_NO"
        rm /dev/nvidia$GPU_NO
        break
    fi

    util::log_debug "successfully get /dev/nvidia$GPU_NO"
    GPU_NO=`expr $GPU_NO + 1`
done

util::log_info "get extra $(expr $CURRENT_GPU_NUM - $INIT_GPU_NUM) GPU devices from host"
util::log_info "current GPU num: $CURRENT_GPU_NUM"
util::log_info "exec nvidia-smi:"
nvidia-smi
#!/usr/bin/env bash

function util::get_gpu_num() {
    echo "$(nvidia-smi -L | wc -l)"
}

function util::log_info() {
    echo "[INFO] $1"
}

function util::log_debug() {
    echo "[DEBUG] $1"
}

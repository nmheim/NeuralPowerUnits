#!/usr/bin/env sh

export JULIA_NUM_THREADS=16
echo JULIA_NUM_THREADS: $JULIA_NUM_THREADS

julia --project --color=yes addition_npu_l1_search.jl

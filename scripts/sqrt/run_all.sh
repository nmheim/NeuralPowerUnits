#!/usr/bin/env sh

export JULIA_NUM_THREADS=16
echo JULIA_NUM_THREADS: $JULIA_NUM_THREADS

julia --project --color=yes msel1_sqrt.jl
julia --project --color=yes msel2_sqrt.jl
julia --project --color=yes ard_sqrt.jl

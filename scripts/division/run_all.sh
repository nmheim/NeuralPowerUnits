#!/usr/bin/env sh

echo JULIA_NUM_THREADS: $JULIA_NUM_THREADS

julia --project ard_xovery.jl
julia --project msel1_xovery.jl
julia --project msel2_xovery.jl

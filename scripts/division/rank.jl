using DrWatson
@quickactivate "NIPS_2020_NMUX"

using DataFrames
using Statistics
using ValueHistories
using Flux
using NeuralArithmetic
using ConditionalDists
using GenerativeModels
using ProgressMeter
using LinearAlgebra
using Parameters

strdict2symdict(d) = Dict([Symbol(k)=>v for (k,v) in d]...)
strdict2symtuple(d) = (;strdict2symdict(d)...)

function readrows(dir::String)
    if isfile(dir) return Dict[] end

    pattern = basename(dir)
    files = readdir(dir, join=true)
    p = Progress(length(files), desc="$pattern: ")
    rows = map(files) do fn
        ps = strdict2symdict(parse_savename(fn)[2])
        ps[:id]  = reduce(*, [string(v) for v in values(ps)])

        @unpack model, history = load(fn)
        ls = hcat(get(history, :loss)[2]...)[:,end]
        ps[:loss] = ls[1]
        if occursin("ard", pattern)
            ps[:llh]  = ls[2]
        end
        ps[:L1]   = norm(params(model), 1)
        ps[:L2]   = norm(params(model), 2)
        next!(p)
        ps
    end
end

function readruns(dir::String, pattern::String)
    dirs = [d for d in readdir(dir, join=true) if occursin(pattern, d)]
    rows = vcat(map(readrows, dirs)...)
    runs = DataFrame(rows)
end

readruns(pattern::String) = readruns(datadir(), pattern)

function readruns(d::Dict)
    runs = readruns(d[:pattern])
    @dict(runs)
end

function aggregateruns(runs::DataFrame, pattern::String)
    if occursin("ard", pattern)
        mean_runs = by(runs, :id) do r
            (α0=first(r.α0),
             β0=first(r.β0),
             initnmu=first(r.initnmu),
             initnau=first(r.initnau),
             μL1=mean(r.L1),
             σL1=std(r.L1),
             μL2=mean(r.L2),
             σL2=std(r.L2),
             μllh=mean(r.llh),
             σllh=std(r.llh))
        end
        return mean_runs
    elseif occursin("mse", pattern)
        mean_runs = by(runs, :id) do r
            (initnmu=first(r.initnmu),
             initnau=first(r.initnau),
             μL1=mean(r.L1),
             σL1=std(r.L1),
             μL2=mean(r.L2),
             σL2=std(r.L2),
             μllh=mean(r.loss),
             σllh=std(r.loss))
        end
        return mean_runs
    else
        throw(ArgumentError("Unknown pattern: $pattern"))
    end
end

force   = false
pattern = "ard_xovery"

res, fname = produce_or_load(datadir(), @dict(pattern), readruns, force=force)
runs = res[:runs]
mean_runs = aggregateruns(runs, pattern)
sort!(mean_runs, :μllh)



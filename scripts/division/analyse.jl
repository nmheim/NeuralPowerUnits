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
using Distributions: Uniform

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "ard_utils.jl"))
x,y = generate(4,100,Uniform(0.1,5))

strdict2symdict(d) = Dict([Symbol(k)=>v for (k,v) in d]...)
strdict2symtuple(d) = (;strdict2symdict(d)...)

function readrows(dir::String)
    if isfile(dir) return Dict[] end

    pattern = basename(dir)
    files = readdir(dir, join=true)
    p = Progress(length(files), desc="$pattern: ")
    rows = map(files) do fn
        ps = strdict2symdict(parse_savename(fn)[2])
        ps[:name]  = basename(fn)

        @unpack model, history = load(fn)
        ls = hcat(get(history, :loss)[2]...)[:,end]
        ps[:loss] = ls[1]
        if occursin("ard", pattern)
            f(x) = Base.invokelatest(model.decoder.mapping.restructure,x)
            net = f(mean(model.encoder))
            ps[:mse]  = Flux.mse(net(x),y)
            ps[:L1]   = norm(params(net), 1)
            ps[:L2]   = norm(params(net), 2)
        else
            ps[:mse]  = Flux.mse(model(x),y)
            ps[:α0]   = NaN
            ps[:β0]   = NaN
            ps[:L1]   = norm(params(model), 1)
            ps[:L2]   = norm(params(model), 2)
        end
        next!(p)
        ps
    end
end

function readruns(dir::String, pattern::String)
    dirs = [d for d in readdir(dir, join=true) if occursin(pattern, d)]
    rows = vcat(map(readrows, dirs)...)
    #rows = Threads.@threads 
    runs = DataFrame(rows)
end

readruns(pattern::String) = readruns(datadir(), pattern)

function readruns(d::Dict)
    runs = readruns(d[:pattern])
    @dict(runs)
end

function aggregateruns(runs::DataFrame)
   mean_runs = by(runs, :name) do r
       (α0=first(r.α0),
        β0=first(r.β0),
        initnmu=first(r.initnmu),
        initnau=first(r.initnau),
        μmse=mean(r.mse),
        σmse=std(r.mse),
        μL1=mean(r.L1),
        σL1=std(r.L1),
        μL2=mean(r.L2),
        σL2=std(r.L2))
   end
end

force   = false

frames = Dict{Symbol,DataFrame}()
for pattern in ["ard_xovery", "msel1_xovery", "msel2_xovery"]
    res, fname = produce_or_load(
        datadir(), @dict(pattern), readruns, force=force)
    runs = res[:runs]
    mean_runs = aggregateruns(runs)
    sort!(mean_runs, :μmse)
    frames[Symbol(pattern)] = mean_runs
end

for (k,f) in frames
    @show k f[1,:name]
    display(first(f[:,2:end],5))
end

using DrWatson
@quickactivate "NIPS_2020_NMUX"

using DataFrames
using Statistics
using ValueHistories
using Flux
using NeuralArithmetic
using ConditionalDists
using GenerativeModels

#include(scriptsdir("division","ard_xovery.jl"))

strdict2symdict(d) = Dict([Symbol(k)=>v for (k,v) in d]...)
strdict2symtuple(d) = (;strdict2symdict(d)...)

function readrows(dir::String)
    files = readdir(dir, join=true)
    rows = map(files) do fn
        println(fn)
        ps = strdict2symdict(parse_savename(fn)[2])
        ps[:id]  = reduce(*, [string(v) for v in values(ps)])
        h  = load(fn)[:history]
        ls = hcat(get(h, :loss)[2]...)[:,end]
        ps[:loss] = ls[1]
        ps[:llh]  = ls[2]
        ps[:kld]  = ls[3]
        ps[:lpλ]  = ls[4]
        ps
    end
end

dirs = [dir for dir in readdir(datadir(), join=true) if occursin("xovery", dir)]
rows = vcat(map(readrows, dirs)...)

runs = DataFrame(rows)

mean_runs = by(runs, :id) do r
    (α0=first(r.α0),
     β0=first(r.β0),
     initnmu=first(r.initnmu),
     initnau=first(r.initnau),
     μllh=mean(r.llh),
     σllh=std(r.llh))
end

sort!(mean_runs, :μllh)

#runs = map()

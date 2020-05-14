using DrWatson
using ProgressMeter

using Statistics
using Flux
using NeuralArithmetic

# using ConditionalDists
# using GenerativeModels
# using LinearAlgebra

using DataFrames
using ValueHistories
using Parameters
# using Distributions: Uniform
# using GMExtensions

@with_kw struct MultL1SearchConfig
    batch::Int      = 128
    niters::Int     = 1e3
    lr::Real        = 5e-3

    βstart::Real    = 1f-4
    βend::Real      = 1f-2
    βgrowth::Real   = 10f0
    βstep::Int      = 10000

    lowlim::Real    = -1
    uplim::Real     = 1
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 100
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "gatednpu"

    run::Int        = 1
end



strdict2symdict(d) = Dict([Symbol(k)=>v for (k,v) in d]...)
strdict2symtuple(d) = (;strdict2symdict(d)...)

function readfiles(dir::String)
    if !isdir(dir) && error("not a directory") end

    files = readdir(dir, join=true)
    configs = map(f->strdict2symdict(parse_savename(f)[2]), files)

    p = Progress(length(files), desc="$(basename(dir)): ")
    rows = map(files) do fn
        @unpack history, model, c = load(fn)
        row = struct2dict(c)
        row[:name] = savename(row)
        row[:id] = hash(delete!(copy(row),:run))

        ls = reduce(hcat, get(history, :loss)[2])[:,end]
        row[:trn] = ls[1]
        row[:mse] = ls[2]
        row[:reg] = ls[3]
        row[:val] = ls[4]

        next!(p)
        return row
    end
    DataFrame(rows)
end


function aggregateruns(df::DataFrame)
    gdf = groupby(df, :id)
    cdf = combine(gdf,
                  :mse => mean,
                  :mse => std,
                  :reg => mean,
                  :reg => std,
                  :val => mean,
                  :val => std,
                  :fstinit => first,
                  :sndinit => first)
end

function find_best(id::UInt, df::DataFrame)
    fdf = filter(row->row[:id]==id, df)
    sort!(fdf, :mse)
    fdf[1,:name]
end


dir = "mult_npu_l1_search"
df = readfiles(datadir(dir))
display(df)
adf = aggregateruns(df)
sort!(adf,:mse_mean)
display(adf)
bestname = find_best(adf[1,:id], df)

res = load(datadir(dir, "$bestname.bson"))
@unpack model, history = res

using UnicodePlots
UnicodePlots.heatmap(model[1].W[end:-1:1,:])

#using Plots
#pyplot()
# p1 = plothistory(history)
# net = get_mapping(model)
# p2 = plot(
#     annotatedheatmap(net[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
#     annotatedheatmap(net[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
#     size=(600,300))
# # display(p1)
# # display(p2)
# wsave(plotsdir("mult_x12_x14_best", "$pattern-$(basename(name))-history.svg"), p1)
# wsave(plotsdir("mult_x12_x14_best", "$pattern-$(basename(name))-mapping.svg"), p2)


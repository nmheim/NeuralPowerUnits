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
    niters::Int     = 1e5
    lr::Real        = 5e-3

    βstart::Real    = 1f-5
    βend::Real      = 1f-3
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

function aggregateruns(df::DataFrame)
    gdf = groupby(df, :hash)
    combine(gdf) do df
        (μmse = mean(df.mse),
         σmse = std(df.mse),
         μreg = mean(df.reg),
         σreg = std(df.reg),
         μtrn = mean(df.trn),
         σtrn = std(df.trn),
         μval = mean(df.val),
         σval = std(df.val),
         fstinit = first(df.fstinit),
         sndinit = first(df.sndinit))
    end
end

function expand_config!(df::DataFrame)
    if !("config" in names(df)) && error("`config` not in dataframe") end
    for k in fieldnames(typeof(df.config[1]))
        df[!,k] = getfield.(df.config, k)
    end
end

function find_best(hash::UInt, df::DataFrame)
    fdf = filter(row->row[:hash]==hash, df)
    sort!(fdf, :mse)
    fdf[1,:path]
end


Base.last(h::MVHistory, k::Symbol) = get(h,k)[2][end]

folder = "mult_npu_l1_search"
df = collect_results!(datadir("$(folder)_results.bson"), datadir(folder), white_list=[],
                      special_list=[:trn => data -> last(data[:history], :loss)[1],
                                    :mse => data -> last(data[:history], :loss)[2],
                                    :reg => data -> last(data[:history], :loss)[3],
                                    :val => data -> last(data[:history], :loss)[4],
                                    :config => data -> data[:c],
                                    :hash => data -> hash(delete!(struct2dict(data[:c]),:run))])
expand_config!(df)
display(df)

adf = aggregateruns(df)
sort!(adf,:μmse)
display(adf)

bestrun = find_best(adf[1,:hash], df)
#bestrun = df[1,:path]
res = load(bestrun)
@unpack model, history = res

using UnicodePlots
UnicodePlots.heatmap(model[1].W[end:-1:1,:], height=100, width=100)

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


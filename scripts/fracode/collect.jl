using DrWatson
@quickactivate "NIPS_2020_NMUX"

using DataFrames
using Flux
using DiffEqFlux
using RecursiveArrayTools
using NeuralArithmetic
using ValueHistories

nrparams(p, thresh) = sum(abs.(p) .> thresh)
name(m::FastDense) = "FastDense"
name(m::FastGatedNPUX) = "FastGatedNPUX"
name(m::FastGatedNPU) = "FastGatedNPU"
name(m::FastChain) = name(m.layers[1])


ϵ = 1e-3
df = collect_results!(datadir("fracsir"), black_list=[:pred,:dudt,:ps,:nrps],
                      special_list=[:model => data -> name(data[:dudt]),
                                    :nrps  => data -> nrparams(data[:ps], ϵ)])

ff = filter(r->r.model=="FastGatedNPU", df)
sort!(ff,["nrps"])
display(ff[!,["mse","nrps","model","βim","βps","hdim","path"]])

sort!(df, "mse")
display(df[!,["mse","nrps","model","βim","βps","hdim","path"]])

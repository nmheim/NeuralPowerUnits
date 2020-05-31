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


ϵ = 1e-7
df = collect_results!(datadir("fracsir"), black_list=[:pred,:dudt,:ps,:nrps],
                      special_list=[:model => data -> name(data[:dudt]),
                                    :nrps  => data -> nrparams(data[:ps], ϵ)])

# df.model = map(df.dudt) do dudt
#     Base.typename(typeof(dudt.layers[1]))
# end

sort!(df,["mse","nrps"])
display(df[!,["mse","nrps","model","βim","βps","hdim"]])
error()


# df = collect_results!(datadir("fracosc"))
# 
# df.model = map(df.dudt) do dudt
#     Base.typename(typeof(dudt.layers[1]))
# end
# 
# ϵ = 1e-7
# df.nrps = map(p->nrparams(p,ϵ), df.ps)
# sort!(df,["mse","nrps"])
# display(df[!,["mse","nrps","model","βim","βps","hdim","lr"]])

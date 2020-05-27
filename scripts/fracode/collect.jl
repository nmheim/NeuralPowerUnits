using DrWatson
@quickactivate "NIPS_2020_NMUX"

using DataFrames
using Flux
using DiffEqFlux
using RecursiveArrayTools
using NeuralArithmetic

nrparams(p, thresh) = sum(abs.(p) .> thresh)

df = collect_results!(datadir("fracsir"))

df.model = map(df.dudt) do dudt
    Base.typename(typeof(dudt.layers[1]))
end


ϵ = 1e-5
df.nrps = map(p->nrparams(p,ϵ), df.ps)
sort!(df,["mse","nrps"])
display(df[!,["mse","nrps","model","βim","βps"]])

using DrWatson
@quickactivate "NIPS_2020_NMUX"

using DataFrames
using Plots
using LaTeXStrings
using OrdinaryDiffEq

include(joinpath(@__DIR__, "odedata.jl"))
pgfplotsx()
#pyplot()

#for loading the dataframe
using NeuralArithmetic
using RecursiveArrayTools
using Flux
using DiffEqFlux

function delete_from_savename(path,key)
    (dir,dict,_) = parse_savename(path)
    delete!(dict, key)
    joinpath(dir, savename(dict,digits=20))
end

function modeltype_to_typename!(df)
    conv = string ∘ Base.typename ∘ typeof
    df.model = map(df.dudt) do dudt
        conv(dudt.layers[1])
    end
end



@unpack df = load(datadir("results_fracosc.bson"))

modeltype_to_typename!(df)
df.hash = delete_from_savename.(df.path, "run")

p1 = plot(title="Pareto", legend=:outerleft)
gdf = filter(r->r.model=="FastGatedNPUX", df)
scatter!(p1, gdf.nrps, gdf.mse, ms=5, alpha=0.7, shape=:circ, label="GatedNPU")
gdf = filter(r->r.model=="FastGatedNPU", df)
scatter!(p1, gdf.nrps, gdf.mse, ms=5, alpha=0.7, shape=:diamond, label="GatedNPU (real)")
gdf = filter(r->r.model=="FastDense", df)
scatter!(p1, gdf.nrps, gdf.mse, ms=5, alpha=0.7, shape=:rect, label="Dense")

plot!(p1, yscale=:log10, xscale=:log10, xlabel="Nr. Paramters", ylabel="MSE")

x,_,t = fracosc_data()

p2 = plot(xlabel="Time", title="SIR Model",legend=:outerright)
plot!(p2, t, x[1,:], lw=4, c=:gray, alpha=0.8, label=L"True $S,I,R$")
plot!(p2, t, x[2,:], lw=4, c=:gray, alpha=0.8, label=false)
plot!(p2, t, x[3,:], lw=4, c=:gray, alpha=0.8, label=false)

sort!(df,"nrps")
x̂ = df[1,"pred"]
scatter!(p2, t, x̂[1,:], label=L"$\hat S$ smallest model", lw=2, c=1)
scatter!(p2, t, x̂[2,:], label=L"$\hat I$ smallest model", lw=2, c=2)
scatter!(p2, t, x̂[3,:], label=L"$\hat R$ smallest model", lw=2, c=3)

p = plot(p1,p2,size=(800,300))
savefig(p, plotsdir("pareto_sir.tikz"))
display(p)

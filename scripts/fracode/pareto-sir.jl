using DrWatson
@quickactivate "NIPS_2020_NPU"

using DataFrames
using Plots
using LaTeXStrings
using OrdinaryDiffEq
using QHull

include(joinpath(@__DIR__, "odedata.jl"))
include(srcdir("annotatedheatmap.jl"))

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

@unpack df = load(datadir("results_fracsir.bson"))
df.hash = delete_from_savename.(df.path, "run")

function Plots.rotate(x::Vector,n::Int)
    if n==0
        return x
    else
        rotate(vcat(x[2:end],x[1:1]), n-1)
    end
end

function hull(nrps::Vector, mse::Vector)
    ch = chull(Array{Float64}(hcat(nrps, mse)))
    return ch.vertices
end

gdf = filter(r->r.model=="FastGatedNPUX", df)
npux_nrps = Array{Float64}(gdf.nrps)
npux_mse  = Array{Float64}(gdf.mse)
npux_idx  = rotate(hull(gdf.nrps, min.(gdf.mse, 1e5)), 0)[1:end-3]

gdf = filter(r->r.model=="FastGatedNPU", df)
npu_nrps = gdf.nrps
npu_mse  = gdf.mse
npu_idx  = rotate(hull(gdf.nrps, min.(gdf.mse, 1e5)), 3)[1:end-1]

gdf = filter(r->r.model=="FastDense", df)
dense_nrps = gdf.nrps
dense_mse  = gdf.mse
dense_idx  = rotate(hull(gdf.nrps, gdf.mse), 1)[4:end]
 
# xbins = 0:7:200
# ybins = 10. .^(-2.2:0.3:3)
# 
# npux_hist = fit(Histogram, (npux_nrps,min.(npux_mse,ybins[end])), (xbins, ybins))
# npu_hist = fit(Histogram, (npu_nrps,min.(npu_mse,1e5)), (xbins, ybins))
# dense_hist = fit(Histogram, (dense_nrps,min.(dense_mse,1e5)), (xbins, ybins))
# 
# p1 = plot(npux_hist, yscale=:log10, c=:blues, alpha=0.5)
# plot!(p1, npu_hist, yscale=:log10, c=:reds, alpha=0.5)
# plot!(p1, dense_hist, yscale=:log10, c=:greens, alpha=0.5)
# 
# plot!(p1, npux_nrps[npux_idx], npux_mse[npux_idx], c=1, label=false, lw=2)
# plot!(p1, npu_nrps[npu_idx], npu_mse[npu_idx], c=2, label=false, lw=2)
# plot!(p1, dense_nrps[dense_idx], dense_mse[dense_idx], c=3, label=false, lw=2)
# 
# plot!(p1, ylim=(ybins[1], ybins[end]), xlim=(xbins[1], xbins[end]), colorbar=false)
# display(p1)
# error()

pgfplotsx()
#pyplot()

α = 0.25

p1 = plot(title="Pareto", legend=:topright)
cmap = palette(:tab10)

scatter!(p1, npux_nrps, npux_mse, c=cmap[3],
         ms=5, alpha=α, shape=:circ, label=false)
scatter!(p1, dense_nrps, dense_mse, c=cmap[1],
         ms=5, alpha=α, shape=:pentagon, label=false)
scatter!(p1, npu_nrps, npu_mse, c=cmap[2],
         ms=5, alpha=α, shape=:diamond, label=false)

plot!(p1, dense_nrps[dense_idx], dense_mse[dense_idx],
      c=cmap[1], label="Dense", lw=2, ls=:dash)
plot!(p1, npux_nrps[npux_idx], npux_mse[npux_idx],
      c=cmap[3], label="NPU", lw=2)
plot!(p1, npu_nrps[npu_idx], npu_mse[npu_idx],
      c=cmap[2], label="NPU (real)", lw=2, ls=:dashdot)

plot!(p1, yscale=:log10, xscale=:log10, xlabel="Nr. Paramters", ylabel="MSE",
      ylim=(5e-3,1e4))

x,_,t = fracsir_data()

p2 = plot(xlabel="Time", title="SIR Model",legend=:right)
plot!(p2, t, x[1,:], lw=4, c=:gray, alpha=0.8, label=L"True $S,I,R$")
plot!(p2, t, x[2,:], lw=4, c=:gray, alpha=0.8, label=false)
plot!(p2, t, x[3,:], lw=4, c=:gray, alpha=0.8, label=false)

sort!(df,"mse")
x̂ = load(datadir("fracsir", basename(df[1,"path"])))[:pred]
scatter!(p2, t, x̂[1,:], label=L"$\hat S$", lw=2, c=cmap[1])
scatter!(p2, t, x̂[2,:], label=L"$\hat I$", lw=2, c=cmap[2])
scatter!(p2, t, x̂[3,:], label=L"$\hat R$", lw=2, c=cmap[3])

p = plot(p1,p2,size=(900,300))
savefig(p, plotsdir("pareto_sir.tikz"))
display(p)

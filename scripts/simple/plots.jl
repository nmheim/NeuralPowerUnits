using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using LinearAlgebra
using NeuralArithmetic
using DataFrames
using Plots
using LaTeXStrings


include(srcdir("turbocmap.jl"))
include(joinpath(@__DIR__, "dataset.jl"))
#pgfplotsx()

nantozero(z) = isnan(z) ? 0 : z
function inftoextreme(z)
    if isinf(z)
        m = maxintfloat(typeof(z))
        return z<0 ? -m : m
    else
        return z
    end
end

folder = datadir("simple")
run = 2
npu = load(joinpath(folder, "pos-gatednpux_lr=0.005_niters=20000_run=$(run)_βl1=1e-5.bson"))[:model]
run = 4
nalu = load(joinpath(folder, "pos-nalu_lr=0.005_niters=20000_run=$(run).bson"))[:model]
run = 8
nmu = load(joinpath(folder, "pos-nmu_lr=0.005_niters=20000_run=$(run).bson"))[:model]
run = 11
dense = load(joinpath(folder, "pos-dense_lr=0.005_niters=20000_run=$(run).bson"))[:model]


posnegx = Float32.(collect(-4:0.1:4))
posnegy = Float32.(collect(-4:0.1:4))
posx = Float32.(collect(0.1:0.1:4))
posy = Float32.(collect(0.1:0.1:4))

plotsize = (400,400)
cmap = :inferno
clim = (-3,2)
clip(x) = max(min(x,clim[2]),clim[1])
levels = range(clim..., length=10)
rminvalid = clip ∘ nantozero ∘ inftoextreme

pyplot()
@info "Plotting Addition..."
func(model,x,y) = rminvalid(log10(addloss(model,x,y)))
s1 = contour(posnegx, posnegy, (x,y)->func(npu,x,y),
             c=cmap, clim=clim, title="NPU Add", ylabel="y",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)

s2 = contour(posnegx, posnegy, (x,y)->func(nalu,x,y),
             c=cmap, clim=clim, title="NALU Add",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)

s3 = contour(posnegx, posnegy, (x,y)->func(nmu,x,y),
             c=cmap, clim=clim, title="NMU Add", ylabel="y", xlabel="x",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)

s4 = contour(posnegx, posnegy, (x,y)->func(dense,x,y),
             c=cmap, clim=clim, title="Dense Add", xlabel="x",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
p1 = plot(s1,s2,s3,s4,size=plotsize)
savefig(p1, papersdir("simple_err_add.pdf"))

@info "Plotting Multiplication..."
func(model,x,y) = rminvalid(log10(multloss(model,x,y)))
s1 = contour(posnegx, posnegy, (x,y)->func(npu,x,y),
             c=cmap, clim=clim, title="NPU Mult", ylabel="y",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
s2 = contour(posnegx, posnegy, (x,y)->func(nalu,x,y),
             c=cmap, clim=clim, title="NALU Mult",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
s3 = contour(posnegx, posnegy, (x,y)->func(nmu,x,y),
             c=cmap, clim=clim, title="NMU Mult", ylabel="y", xlabel="x",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
s4 = contour(posnegx, posnegy, (x,y)->func(dense,x,y),
             c=cmap, clim=clim, title="Dense Mult",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
p2 = plot(s1,s2,s3,s4,size=plotsize)
savefig(p2, papersdir("simple_err_mult.pdf"))


@info "Plotting Division..."
func(model,x,y) = rminvalid(log10(divloss(model,x,y)))
s1 = contour(posnegx, posnegy, (x,y)->func(npu,x,y),
             c=cmap, clim=clim, title="NPU Div", ylabel="y",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
s2 = contour(posnegx, posnegy, (x,y)->func(nalu,x,y),
             c=cmap, clim=clim, title="NALU Div",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
s3 = contour(posnegx, posnegy, (x,y)->func(nmu,x,y),
             c=cmap, clim=clim, title="NMU Div", ylabel="y", xlabel="x",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
s4 = contour(posnegx, posnegy, (x,y)->func(dense,x,y),
             c=cmap, clim=clim, title="Dense Div", xlabel="x",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
p3 = plot(s1,s2,s3,s4,size=plotsize)
savefig(p3, papersdir("simple_err_div.pdf"))

@info "Plotting Sqrt..."
clim = (-3,0)
clip(x) = max(min(x,clim[2]),clim[1])
levels = range(clim..., length=10)
rminvalid = clip ∘ nantozero ∘ inftoextreme

func(model,x,y) = rminvalid(log10(sqrtloss(model,x,y)))
s1 = contour(posx, posy, (x,y)->func(npu,x,y),
             c=cmap, clim=clim, title="NPU Sqrt", ylabel="y",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
s2 = contour(posx, posy, (x,y)->func(nalu,x,y),
             c=cmap, clim=clim, title="NALU Sqrt",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
s3 = contour(posx, posy, (x,y)->func(nmu,x,y),
             c=cmap, clim=clim, title="NMU Sqrt", ylabel="y", xlabel="x",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
s4 = contour(posx, posy, (x,y)->func(dense,x,y),
             c=cmap, clim=clim, title="Dense Sqrt", xlabel="x",
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
p4 = plot(s1,s2,s3,s4,size=plotsize)
savefig(p4, papersdir("simple_err_sqrt.pdf"))

display(plot(p1,p2,p3,p4))

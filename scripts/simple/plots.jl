using DrWatson
@quickactivate "NIPS_2020_NPU"

using Flux
using LinearAlgebra
using NeuralArithmetic
using DataFrames
using Plots
using LaTeXStrings

include(joinpath(@__DIR__, "dataset.jl"))


nantozero(z) = isnan(z) ? 0 : z
function inftoextreme(z)
    if isinf(z)
        m = maxintfloat(typeof(z))
        return z<0 ? -m : m
    else
        return z
    end
end

umin = 0.01
umax = 2
folder = datadir("simple_lr005_umin=$(umin)_umax=$(umax)")

run  = 2
file = "npu_lr=0.005_niters=20000_run=$(run)_umax=2_umin=0.01_βl1=0.bson"
npu = load(joinpath(folder, file))[:model]
run = 37
file = "realnpu_lr=0.005_niters=20000_run=$(run)_umax=2_umin=0.01_βl1=0.bson"
realnpu = load(joinpath(folder, file))[:model]
run = 15
file = "nalu_lr=0.005_niters=20000_run=$(run)_umax=2_umin=0.01.bson"
nalu = load(joinpath(folder, file))[:model]
run = 36
file = "inalu_lr=0.005_niters=20000_run=$(run)_t=20_umax=2_umin=0.01.bson"
inalu = load(joinpath(folder, file))[:model]
run = 8
file = "nmu_lr=0.005_niters=20000_run=$(run)_umax=2_umin=0.01.bson"
nmu = load(joinpath(folder, file))[:model]
run = 12
file = "dense_lr=0.005_niters=20000_run=$(run)_umax=2_umin=0.01.bson"
dense = load(joinpath(folder, file))[:model]


posnegx = Float32.(collect(-4.1:0.2:4))
posnegy = Float32.(collect(-4.1:0.2:4))
posx = Float32.(collect(0.1:0.1:5))
posy = Float32.(collect(0.1:0.1:5))

plotsize = (1000,200)
layout = grid(1,5)
cmap = cgrad(:inferno, rev=true)

clim = (-4,2)
clip(x) = max(min(x,clim[2]),clim[1])
levels = range(clim..., length=10)
rminvalid = clip ∘ nantozero ∘ inftoextreme

npu_addloss(x,y)    = rminvalid(log10(addloss(npu,x,y)))
npu_multloss(x,y)   = rminvalid(log10(multloss(npu,x,y)))
npu_divloss(x,y)    = rminvalid(log10(divloss(npu,x,y)))
nalu_addloss(x,y)   = rminvalid(log10(addloss(nalu,x,y)))
nalu_multloss(x,y)  = rminvalid(log10(multloss(nalu,x,y)))
nalu_divloss(x,y)   = rminvalid(log10(divloss(nalu,x,y)))
inalu_addloss(x,y)  = rminvalid(log10(addloss(inalu,x,y)))
inalu_multloss(x,y) = rminvalid(log10(multloss(inalu,x,y)))
inalu_divloss(x,y)  = rminvalid(log10(divloss(inalu,x,y)))
nmu_addloss(x,y)    = rminvalid(log10(addloss(nmu,x,y)))
nmu_multloss(x,y)   = rminvalid(log10(multloss(nmu,x,y)))
nmu_divloss(x,y)    = rminvalid(log10(divloss(nmu,x,y)))
dense_addloss(x,y)  = rminvalid(log10(addloss(dense,x,y)))
dense_multloss(x,y) = rminvalid(log10(multloss(dense,x,y)))
dense_divloss(x,y)  = rminvalid(log10(divloss(dense,x,y)))
npu_sqrtloss(x,y)   = rminvalid(log10(sqrtloss(npu,x,y)))
nalu_sqrtloss(x,y)  = rminvalid(log10(sqrtloss(nalu,x,y)))
inalu_sqrtloss(x,y) = rminvalid(log10(sqrtloss(inalu,x,y)))
nmu_sqrtloss(x,y)   = rminvalid(log10(sqrtloss(nmu,x,y)))
dense_sqrtloss(x,y) = rminvalid(log10(sqrtloss(dense,x,y)))

realnpu_addloss(x,y)    = rminvalid(log10(addloss(realnpu,x,y)))
realnpu_multloss(x,y)   = rminvalid(log10(multloss(realnpu,x,y)))
realnpu_divloss(x,y)    = rminvalid(log10(divloss(realnpu,x,y)))
realnpu_sqrtloss(x,y)   = rminvalid(log10(sqrtloss(realnpu,x,y)))


gr()
@info "Plotting NPU..."
npuadd = contour(posnegx, posnegy, (x,y)->npu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
npumult = contour(posnegx, posnegy, (x,y)->npu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
npudiv = contour(posnegx, posnegy, (x,y)->npu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
npusqrt = contour(posx, posy, (x,y)->npu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)

@info "Plotting RealNPU..."
realnpuadd = contour(posnegx, posnegy, (x,y)->realnpu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
realnpumult = contour(posnegx, posnegy, (x,y)->realnpu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
realnpudiv = contour(posnegx, posnegy, (x,y)->realnpu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
realnpusqrt = contour(posx, posy, (x,y)->realnpu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)

@info "Plotting NALU..."
naluadd = contour(posnegx, posnegy, (x,y)->nalu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
nalumult = contour(posnegx, posnegy, (x,y)->nalu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
naludiv = contour(posnegx, posnegy, (x,y)->nalu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
nalusqrt = contour(posx, posy, (x,y)->nalu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)

@info "Plotting iNALU..."
inaluadd = contour(posnegx, posnegy, (x,y)->inalu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
inalumult = contour(posnegx, posnegy, (x,y)->inalu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
inaludiv = contour(posnegx, posnegy, (x,y)->inalu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
inalusqrt = contour(posx, posy, (x,y)->inalu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)


@info "Plotting NMU..."
nmuadd = contour(posnegx, posnegy, (x,y)->nmu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
nmumult = contour(posnegx, posnegy, (x,y)->nmu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
nmudiv = contour(posnegx, posnegy, (x,y)->nmu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
nmusqrt = contour(posx, posy, (x,y)->nmu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)

@info "Plotting Dense..."
denseadd = contour(posnegx, posnegy, (x,y)->dense_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
densemult = contour(posnegx, posnegy, (x,y)->dense_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
densediv = contour(posnegx, posnegy, (x,y)->dense_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)
densesqrt = contour(posx, posy, (x,y)->dense_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false)

p = plot(
    plot!(denseadd, xticks=false, ylabel="Addition\ny", title="Dense"),
    plot!(naluadd,  xticks=false, yticks=false, title="NALU"),
    plot!(nmuadd,   xticks=false, yticks=false, title="NMU"),
    plot!(npuadd,   xticks=false, yticks=false, title="NPU",
        colorbar=true,colorbar_title="log(error)"),

    plot!(densemult, xticks=false, ylabel="Multiplication\ny"),
    plot!(nalumult,  xticks=false, yticks=false),
    plot!(nmumult,   xticks=false, yticks=false),
    plot!(npumult,   xticks=false, yticks=false,
        colorbar=true,colorbar_title="log(error)"),
    
    plot!(densediv, ylabel="Division\ny"),
    plot!(naludiv, yticks=false),
    plot!(nmudiv,  yticks=false),
    plot!(npudiv,  yticks=false,
        colorbar=true,colorbar_title="log(error)"),
    
    plot!(densesqrt, ylabel="Square root\ny", xlabel="x"),
    plot!(nalusqrt, yticks=false, xlabel="x"),
    plot!(nmusqrt,  yticks=false, xlabel="x"),
    plot!(npusqrt,  yticks=false, xlabel="x",
        colorbar=true,colorbar_title="log(error)"),

    layout = grid(4,4,widths=[0.21,0.21,0.21,0.36]),
    size = (800,700)
)
savefig(p,plotsdir("small_simple_err.pdf"))
display(p)

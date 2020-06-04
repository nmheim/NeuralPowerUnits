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

folder = datadir("simple")
run = 19
npu = load(joinpath(folder, "pos-gatednpux_lr=0.005_niters=20000_run=10_βl1=0.bson"))[:model]
run = 9
realnpu = load(joinpath(folder, "pos-realnpu_lr=0.005_niters=20000_run=$(run)_βl1=0.bson"))[:model]
run = 8
nalu = load(joinpath(folder, "pos-nalu_lr=0.005_niters=20000_run=$(run).bson"))[:model]
run = 4
inalu = load(joinpath(folder, "pos-inalu_lr=0.001_niters=20000_run=$(run).bson"))[:model]
run = 8
nmu = load(joinpath(folder, "pos-nmu_lr=0.005_niters=20000_run=$(run).bson"))[:model]
run = 1
dense = load(joinpath(folder, "pos-dense_lr=0.005_niters=20000_run=$(run).bson"))[:model]


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


pyplot()
@info "Plotting NPU..."
npuadd = contour(posnegx, posnegy, (x,y)->npu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
npumult = contour(posnegx, posnegy, (x,y)->npu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
npudiv = contour(posnegx, posnegy, (x,y)->npu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
npusqrt = contour(posx, posy, (x,y)->npu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)

@info "Plotting RealNPU..."
realnpuadd = contour(posnegx, posnegy, (x,y)->realnpu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
realnpumult = contour(posnegx, posnegy, (x,y)->realnpu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
realnpudiv = contour(posnegx, posnegy, (x,y)->realnpu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
realnpusqrt = contour(posx, posy, (x,y)->realnpu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)

@info "Plotting NALU..."
naluadd = contour(posnegx, posnegy, (x,y)->nalu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
nalumult = contour(posnegx, posnegy, (x,y)->nalu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
naludiv = contour(posnegx, posnegy, (x,y)->nalu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
nalusqrt = contour(posx, posy, (x,y)->nalu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)

@info "Plotting iNALU..."
inaluadd = contour(posnegx, posnegy, (x,y)->inalu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
inalumult = contour(posnegx, posnegy, (x,y)->inalu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
inaludiv = contour(posnegx, posnegy, (x,y)->inalu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
inalusqrt = contour(posx, posy, (x,y)->inalu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)


@info "Plotting NMU..."
nmuadd = contour(posnegx, posnegy, (x,y)->nmu_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
nmumult = contour(posnegx, posnegy, (x,y)->nmu_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
nmudiv = contour(posnegx, posnegy, (x,y)->nmu_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
nmusqrt = contour(posx, posy, (x,y)->nmu_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)

@info "Plotting Dense..."
denseadd = contour(posnegx, posnegy, (x,y)->dense_addloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
densemult = contour(posnegx, posnegy, (x,y)->dense_multloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
densediv = contour(posnegx, posnegy, (x,y)->dense_divloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)
densesqrt = contour(posx, posy, (x,y)->dense_sqrtloss(x,y),
             c=cmap, clim=clim,
             levels=levels, fill=true, colorbar=false, aspect_ratio=:equal)

row1 = plot(plot!(npuadd,title="NPU", ylabel="Addition\ny", xticks=false),
            plot!(realnpuadd,title="RealNPU", xticks=false, yticks=false),
            plot!(nmuadd,title="NMU", xticks=false, yticks=false),
            plot!(naluadd,title="NALU", xticks=false, yticks=false),
            plot!(inaluadd,title="iNALU", xticks=false, yticks=false),
            plot!(denseadd,title="Dense",colorbar=true, xticks=false, yticks=false,
                  colorbar_title=L"\log(|t_1-\hat{t}_1|)"),
            size=plotsize,layout=grid(1,6))

row2 = plot(plot!(npumult, ylabel="Multiplication\ny", xticks=false),
            plot!(realnpumult, xticks=false, yticks=false),
            plot!(nmumult, xticks=false, yticks=false),
            plot!(nalumult, xticks=false, yticks=false),
            plot!(inalumult, xticks=false, yticks=false),
            plot!(densemult,xticks=false, yticks=false,
                  colorbar=true,colorbar_title=L"\log(|t_2-\hat{t}_2|)"),
            size=plotsize,layout=grid(1,6))

row3 = plot(plot!(npudiv, ylabel="Division\ny"),
            plot!(realnpudiv, yticks=false),
            plot!(nmudiv, yticks=false),
            plot!(naludiv, yticks=false),
            plot!(inaludiv, yticks=false),
            plot!(densediv, yticks=false,
                  colorbar=true,colorbar_title=L"\log(|t_3-\hat{t}_3|)"),
            size=plotsize,layout=grid(1,6))

row4 = plot(plot!(npusqrt,ylabel="Square-root\ny",xlabel="x"),
            plot!(realnpusqrt,xlabel="x", yticks=false),
            plot!(nmusqrt,xlabel="x", yticks=false),
            plot!(nalusqrt,xlabel="x", yticks=false),
            plot!(inalusqrt,xlabel="x", yticks=false),
            plot!(densesqrt, yticks=false,
                  colorbar=true,xlabel="x",colorbar_title=L"\log(|t_4-\hat{t}_4|)"),
            size=plotsize,layout=grid(1,6))

p = plot(row1,row2,row3,row4,layout=(4,1),size=(800,600))
#savefig(p,plotsdir("simple_err.pdf"))
display(p)

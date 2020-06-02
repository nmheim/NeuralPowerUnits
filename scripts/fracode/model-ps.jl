using DrWatson
@quickactivate "NIPS_2020_NPU"

using Plots
using LaTeXStrings

using Flux
using NeuralArithmetic
using RecursiveArrayTools
using DiffEqFlux

include(srcdir("annotatedheatmap.jl"))

fname = "gatednpu_hdim=6_lr=0.005_niters=3000_run=4_αinit=0.2_βim=0_βps=1.bson"
# fname = "gatednpux_hdim=6_lr=0.005_niters=3000_run=4_αinit=0.2_βim=0.1_βps=0.1.bson"
# fname = "gatednpux_hdim=6_lr=0.005_niters=3000_run=3_αinit=0.2_βim=1_βps=1.bson"
# fname = "gatednpux_hdim=15_lr=0.005_niters=3000_run=5_αinit=0.2_βim=1_βps=1.bson"
# fname = "gatednpux_hdim=6_lr=0.005_niters=3000_run=4_αinit=0.2_βim=0.1_βps=0.1.bson"
# fname = "gatednpux_hdim=9_lr=0.005_niters=3000_run=4_αinit=0.2_βim=0.1_βps=0.1.bson"
res = load(datadir("fracsir",fname))

nonfast(m::FastNAU) = NAU(m.in,m.out)
nonfast(m::FastGatedNPU) = GatedNPU(m.in,m.out)
nonfast(m::FastGatedNPUX) = GatedNPUX(m.in,m.out)
nonfast(m::FastChain) = Chain(map(nonfast, m.layers)...)
name(m::NAU) = "NAU"
name(m::GatedNPU) = "NPU (real)"
name(m::GatedNPUX) = "NPU"

@unpack dudt, ps = res
m = Flux._restructure(nonfast(dudt),ps)


fontsize=11
pyplot()
cmap = :balance
p0 = annotatedheatmap(zeros(3,1), annotationtexts=[L"0.09I-"*"\n"*L"0.15R^{0.6}",
                                                   L"0.11r"*"\n"*L"-0.1R",
                                                   L"-0.12r"*"\n"*L"+0.13I"],
                      annotationargs=(:black,fontsize),
                      c=cmap, clim=(-1,1), aspect_ratio=:equal, title="Output")

p1 = annotatedheatmap(m[2].W[end:-1:1,:], title="$(name(m[2]))\nLayer 2",
                      aspect_ratio=:equal, c=cmap, clim=(-0.2,0.2), colobar=false,
                      annotationargs=(:white,fontsize))

p2 = annotatedheatmap(zeros(6,1), annotationtexts=["1.0",L"R^{0.64}","I","1.0","1.0",L"\,S^{0.62}"*"\n"*L"\times\,I^{0.57}"],
                      annotationargs=(:black, fontsize), aspect_ratio=:equal,
                      c=cmap, clim=(-1,1), title="Hidden")
annotate!(p2,0.25,6,L"r=")

p3 = annotatedheatmap(m[1].W[end:-1:1,:], title="$(name(m[1]))\nLayer 1",
                      aspect_ratio=:equal, c=cmap, clim=(-1.2,1.2), colorbar=false,
                      annotationargs=(:white,fontsize))

p4 = annotatedheatmap(zeros(3,1), annotationtexts=["R","I","S"], annotationargs=(:black,fontsize),
                      aspect_ratio=:equal, c=cmap, clim=(-1.0,1.0), title="Input")

β = 1/12
plt = plot(p0,p1,p2,p3,p4,layout=grid(1,5,widths=[β,6β,β,3β,β]),size=(1000,500),colorbar=false)
savefig(plt, plotsdir("sir_gatednpu_modelps.pdf"))
display(plt)

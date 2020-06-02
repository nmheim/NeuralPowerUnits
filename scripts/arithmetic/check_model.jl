using DrWatson
@quickactivate "NIPS_2020_NPU"

using Flux
using NeuralArithmetic
using ValueHistories
using UnicodePlots

include(srcdir("arithmetic_dataset.jl"))
include(srcdir("unicodeheat.jl"))
include(joinpath(@__DIR__, "sobolconfigs.jl"))

fname = "batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=gatednpux_niters=100000_overlap=0.25_run=1_sampler=sobol_sndinit=rand_subset=0.5_uplim=0.5_βend=1e-7_βgrowth=10_βstart=1e-9_βstep=10000.bson"
fname = "batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=nmu_niters=100000_overlap=0.25_run=1_sampler=sobol_sndinit=rand_subset=0.5_uplim=0.5_βend=1e-7_βgrowth=10_βstart=1e-9_βstep=10000.bson"
fname = datadir("invx_l1_runs", fname)

@unpack model, history, c, val = load(fname)
@info c

x = zeros(Float32,c.inlen)
x[1] = 0.01

@info "test" x model(x) invx(x,c.subset) val
display(heat(model))

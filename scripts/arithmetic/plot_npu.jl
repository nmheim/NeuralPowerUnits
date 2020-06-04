using DrWatson
@quickactivate "NIPS_2020_NPU"

using Flux
using NeuralArithmetic
using ValueHistories
using Plots
pyplot()

include(srcdir("arithmetic_dataset.jl"))
include(joinpath(@__DIR__, "sobolconfigs.jl"))

fname = "batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=gatednpux_niters=100000_overlap=0.25_run=1_sampler=sobol_sndinit=rand_subset=0.5_uplim=0.5_βend=1e-7_βgrowth=10_βstart=1e-9_βstep=10000.bson"
fname = datadir("invx_l1_runs", fname)

dir = "/home/niklas/repos/NIPS_2020_NPU/data"
realnpuadd    = load(joinpath(dir, "add_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.01_model=gatednpu_niters=100000_overlap=0.25_run=7_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]
npuadd        = load(joinpath(dir, "add_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.01_model=gatednpux_niters=100000_overlap=0.25_run=14_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]
naluadd       = load(joinpath(dir, "add_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.01_model=nalu_niters=100000_overlap=0.25_run=8_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]
nmuadd        = load(joinpath(dir, "add_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.01_model=nmu_niters=100000_overlap=0.25_run=4_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]
naivenpuadd   = load(joinpath(dir, "add_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.01_model=npux_niters=100000_overlap=0.25_run=1_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]

# realnpumult   = load(joinpath(dir, "mult_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.005_model=gatednpu_niters=100000_overlap=0.25_run=2_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]
# npumult       = load(joinpath(dir, "mult_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.005_model=gatednpux_niters=100000_overlap=0.25_run=2_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]
# nalumult      = load(joinpath(dir, "mult_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.005_model=nalu_niters=100000_overlap=0.25_run=5_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]
# nmumult       = load(joinpath(dir, "mult_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.005_model=nmu_niters=100000_overlap=0.25_run=4_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]
# naivenpumult  = load(joinpath(dir, "mult_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=-1_lr=0.005_model=npux_niters=100000_overlap=0.25_run=7_sampler=sobol_sndinit=rand_subset=0.5_uplim=1_βend=0.0001_βgrowth=10_βstart=1e-5_βstep=10000.bson"))[:model]
# 
# realnpuinvx   = load(joinpath(dir, "invx_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=gatednpu_niters=100000_overlap=0.25_run=2_sampler=sobol_sndinit=rand_subset=0.5_uplim=0.5_βend=1e-7_βgrowth=10_βstart=1e-9_βstep=10000.bson"))[:model]
# npuinvx       = load(joinpath(dir, "invx_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=gatednpux_niters=100000_overlap=0.25_run=9_sampler=sobol_sndinit=rand_subset=0.5_uplim=0.5_βend=1e-7_βgrowth=10_βstart=1e-9_βstep=10000.bson"))[:model]
# naluinvx      = load(joinpath(dir, "invx_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=nalu_niters=100000_overlap=0.25_run=4_sampler=sobol_sndinit=rand_subset=0.5_uplim=0.5_βend=1e-7_βgrowth=10_βstart=1e-9_βstep=10000.bson"))[:model]
# nmuinvx       = load(joinpath(dir, "invx_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=nmu_niters=100000_overlap=0.25_run=3_sampler=sobol_sndinit=rand_subset=0.5_uplim=0.5_βend=1e-7_βgrowth=10_βstart=1e-9_βstep=10000.bson"))[:model]
# naivenpuinvx  = load(joinpath(dir, "invx_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=npux_niters=100000_overlap=0.25_run=1_sampler=sobol_sndinit=rand_subset=0.5_uplim=0.5_βend=1e-7_βgrowth=10_βstart=1e-9_βstep=10000.bson"))[:model]
# 
# realnpusqrt   = load(joinpath(dir, "sqrt_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=gatednpu_niters=100000_overlap=0.25_run=3_sampler=sobol_sndinit=rand_subset=0.5_uplim=2_βend=0.0001_βgrowth=10_βstart=1e-6_βstep=10000.bson"))[:model]
# npusqrt       = load(joinpath(dir, "sqrt_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=gatednpux_niters=100000_overlap=0.25_run=6_sampler=sobol_sndinit=rand_subset=0.5_uplim=2_βend=0.0001_βgrowth=10_βstart=1e-6_βstep=10000.bson"))[:model]
# nalusqrt      = load(joinpath(dir, "sqrt_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=nalu_niters=100000_overlap=0.25_run=10_sampler=sobol_sndinit=rand_subset=0.5_uplim=2_βend=0.0001_βgrowth=10_βstart=1e-6_βstep=10000.bson"))[:model]
# nmusqrt       = load(joinpath(dir, "sqrt_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=nmu_niters=100000_overlap=0.25_run=8_sampler=sobol_sndinit=rand_subset=0.5_uplim=2_βend=0.0001_βgrowth=10_βstart=1e-6_βstep=10000.bson"))[:model]
# naivenpusqrt  = load(joinpath(dir, "sqrt_l1_runs/batch=128_fstinit=rand_inlen=100_lowlim=0_lr=0.005_model=npux_niters=100000_overlap=0.25_run=4_sampler=sobol_sndinit=rand_subset=0.5_uplim=2_βend=0.0001_βgrowth=10_βstart=1e-6_βstep=10000.bson"))[:model]

function concat(m::Chain{<:Tuple{<:NAU,<:GatedNPUX}})
  W  = m[1].W[:,end:-1:1]
  Re = m[2].Re[:,end:-1:1]
  Im = m[2].Im[:,end:-1:1]
  g  = reshape(m[2].g, 1, :)
  cat(W,Re',Im',g',dims=2)
end

function concat(m::Chain{<:Tuple{<:NAU,<:GatedNPU}})
  W  = m[1].W[:,end:-1:1]
  Re = m[2].W[:,end:-1:1]
  g  = reshape(m[2].g, 1, :)
  cat(W,Re',g',dims=2)
end

function concat(m::Chain{<:Tuple{<:NALU,<:NALU}})
  W1 = NeuralArithmetic.weights(m[1].nac)
  W2 = NeuralArithmetic.weights(m[2].nac)'
  G1 = m[1].G
  G2 = m[2].G'
  cat(W1,W2,G1,G2,dims=2)
end

function concat(m::Chain{<:Tuple{<:NAU,<:NMU}})
  W  = m[1].W[:,end:-1:1]
  Re = m[2].W[:,end:-1:1]
  cat(W,Re',dims=2)
end

function concat(m::Chain{<:Tuple{<:NAU,<:NPUX}})
  W  = m[1].W[:,end:-1:1]
  Re = m[2].Re[:,end:-1:1]
  Im = m[2].Im[:,end:-1:1]
  cat(W,Re',Im',dims=2)
end

cmap = :bluesreds
clim = (-2,2)
p1 = heatmap(concat(npuadd), c=cmap, clim=clim)
p2 = heatmap(concat(realnpuadd), c=cmap, clim=clim)
p3 = heatmap(concat(naivenpuadd), c=cmap, clim=clim)
p4 = heatmap(concat(nmuadd), c=cmap, clim=clim)
p5 = heatmap(concat(naluadd), c=cmap, clim=clim)
plot(p1,p2,p3,p4,p5,layout=(5,1))

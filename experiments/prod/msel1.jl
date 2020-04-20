using DrWatson
@quickactivate "arithmetic"

using Flux
using Plots
using NeuralArithmetic
using Distributions: Uniform
using Parameters
using ValueHistories
using LinearAlgebra
using GMExtensions

include(joinpath(@__DIR__, "utils.jl"))
include(srcdir("utils.jl"))
include(srcdir("colors.jl"))

pattern = "simple_prod_msel1"
outdir  = datadir("tests", pattern)

@with_kw struct MSEL1Config
    batch::Int      = 50
    inlen::Int      = 4
    outlen::Int     = 1
    niters::Int     = 50000
    lr::Real        = 0.01
    lowlim::Int     = -2
    uplim::Int      = 2
    βL1             = 1
    initnau::String = "rand"
    initnmu::String = "rand"
end

function task(x)
    x1 = x[1,:]
    x2 = x[2,:]
    x3 = x[3,:]
    x4 = x[4,:]
    y = (x1 .+ x2 .+ x3 .+ x4) .* (x1 .+ x2)
    reshape(y, 1, :)
end

function generate(inlen::Int, batch::Int, r::Uniform)
    x = Float32.(rand(r, inlen, batch))
    y = task(x)
    (x,y)
end

function run(config)
    @unpack initnau, initnmu, lowlim, uplim = config
    @unpack niters, batch, inlen, outlen, βL1, lr = config
    model   = mapping(inlen, outlen, initf(initnau), initf(initnmu))
    ps      = params(model)
    function loss(x,y)
        mse = Flux.mse(model(x),y)
        L1  = βL1 * norm(ps,1)
        (mse+L1), mse, L1
    end
    range   = Uniform(lowlim,uplim)
    data    = (generate(inlen,batch,range) for _ in 1:niters)
    opt     = RMSProp(lr)
    history = train!(loss, model, data, opt)
    return @dict(model, history)
end

#################### Single run with default params ############################

config = MSEL1Config()
res, fname = produce_or_load(outdir, config, run, force=false)

m = res[:model]
h = res[:history]

function Plots.plot(h::MVHistory)
    idx, ls = get(h, :loss)
    ls = reduce(hcat, ls)
    tot = ls[1,:]
    mse = ls[2,:]
    L1  = ls[3,:]

    p1 = plot(idx, tot, label="Loss", yscale=:log10, lw=2)
    plot!(p1, idx, mse, label="MSE", lw=2)
    plot!(p1, idx, L1,  label="L1", lw=2)

    ps = reduce(hcat, get(h, :μz)[2])'
    p2 = plot(idx, ps, legend=false, lw=2)
    plot(p1,p2,layout=(2,1))
end

pyplot()
p1 = plot(h)
ps = [annotatedheatmap(l.W[end:-1:1,:], c=:bluesreds, title=summary(l), clim=(-1,1)) for l in m]
p2 = plot(ps..., size=(600,300))
# display(p1)
# display(p2)
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.svg"), p1)
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.svg"), p2)

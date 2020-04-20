using DrWatson
@quickactivate "arithmetic"

using Flux
using Plots
using GenerativeModels
using GMExtensions
using NeuralArithmetic
using Distributions: Uniform
using Parameters
using ValueHistories
using LinearAlgebra

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "ard_utils.jl"))
include(srcdir("utils.jl"))

pattern = "simple_div_sqrt_ard"
outdir  = datadir("tests", pattern)

@with_kw struct ARDConfig
    batch::Int      = 50
    inlen::Int      = 4
    outlen::Int     = 1
    niters::Int     = 150000
    α0::Float32     = Float32(1e-1)
    β0::Float32     = Float32(1e-1)
    lr::Real        = 0.002
    lowlim::Real    = 0.1
    uplim::Real     = 3
    initnau::String = "rand"
    initnmu::String = "rand"
end

function task(x)
    x1 = x[1,:]
    x2 = x[2,:]
    x3 = x[3,:]
    x4 = x[4,:]
    y = sqrt.(x1 .+ x2) ./ (x3 .+ x4)
    reshape(y, 1, :)
end


function generate(inlen::Int, batch::Int, r::Uniform)
    x = Float32.(rand(r, inlen, batch))
    y = task(x)
    (x,y)
end

function run(config)
    @unpack initnau, initnmu, lowlim, uplim = config
    @unpack niters, batch, inlen, outlen, α0, β0, lr = config
    net     = mapping(inlen, outlen, initf(initnau), initf(initnmu))
    model   = ardnet(net, α0, β0, outlen)
    loss    = (x,y) -> notelbo(model,x,y,α0=α0,β0=β0)
    range   = Uniform(lowlim,uplim)
    data    = (generate(inlen,batch,range) for _ in 1:niters)
    opt     = RMSProp(lr)
    history = train!(loss, model, data, opt)
    return @dict(model, history)
end

#################### Single run with default params ############################

config = ARDConfig()
res, fname = produce_or_load(outdir, config, run, force=false)

m = res[:model]
h = res[:history]

function Plots.plot(h::MVHistory)
    kys = [k for k in keys(h)]
    idx, ls = get(h, :loss)
    ls = reduce(hcat, ls)
    tot = ls[1,:]
    llh = ls[2,:]
    kld = ls[3,:]
    lpλ = ls[3,:]

    p1 = plot(idx, tot, label="Loss", lw=2)
    plot!(p1, idx, llh, label="LLH", lw=2)
    plot!(p1, idx, kld, label="KLD", lw=2)
    plot!(p1, idx, lpλ, label="LPL", lw=2)

    kys = filter(x->x!=:loss, kys)

    plt = [p1]
    for k in kys
        ps = reduce(hcat, get(h, k)[2])'
        p = plot(idx, ps, legend=false, lw=2, ylabel=k)
        push!(plt, p)
    end
    plot(plt..., layout=(length(plt),1), size=(300,900))
end

pyplot()
p1 = plot(h)
net = get_mapping(m)
ps = [annotatedheatmap(l.W[end:-1:1,:], c=:bluesreds, title=summary(l), clim=(-1,1)) for l in net]
p2 = plot(ps..., size=(600,300))
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.svg"), p1)
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.svg"), p2)

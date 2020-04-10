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
include(srcdir("utils.jl"))

@with_kw struct Config
    batch::Int      = 50
    inlen::Int      = 4
    outlen::Int     = 1
    niters::Int     = 30000
    α0::Float32     = 1f0
    β0::Float32     = 1f0
    lr::Real        = 0.001
    lowlim::Int     = 1
    uplim::Int      = 3
    initnau::String = "diag"
    initnmu::String = "zero"
end

function init_diag(T, a::Int, b::Int)
    m = zeros(T,a,b)
    m[diagind(m,0)] .= 1f0
    return m
end

function initf(s::String)
    if s == "diag"
        return (a,b) -> init_diag(Float32,a,b)
    elseif s == "glorotuniform"
        return (a,b) -> Flux.glorot_uniform(Float32,a,b)
    elseif s == "rand"
        return (a,b) -> rand(Float32,a,b)
    elseif s == "randn"
        return (a,b) -> randn(Float32,a,b)
    elseif s == "zero"
        return (a,b) -> zeros(Float32,a,b)
    else
        throw(ArgumentError("Unkown init: $init"))
    end
end

function f(x::Array{T,2}) where T
    x1 = x[1,:]
    x2 = x[2,:]
    y = x1 ./ x2
    reshape(y, 1, :)
end

function generate(inlen::Int, batch::Int, r::Uniform)
    x = Float32.(rand(r, inlen, batch))
    y = f(x)
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


config = Config()
res, fname = produce_or_load(datadir("division_xovery"), config, run)

m = res[:model]
h = res[:history]

pyplot()
p1 = plothistory(h)
net = get_mapping(m)
p2 = plot(
    annotatedheatmap(net[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
    annotatedheatmap(net[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
    size=(600,300))
wsave(plotsdir("division_xovery", "$(basename(splitext(fname)[1]))-history.svg"), p1)
wsave(plotsdir("division_xovery", "$(basename(splitext(fname)[1]))-mapping.svg"), p2)

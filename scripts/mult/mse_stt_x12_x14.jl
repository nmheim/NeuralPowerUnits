using DrWatson
@quickactivate "arithmetic"

using Flux
using NeuralArithmetic
using Distributions: Uniform
using Parameters
using ValueHistories
using LinearAlgebra
using GMExtensions
using SpecialFunctions

include(joinpath(@__DIR__, "utils.jl"))

pattern = "mult_mse_stt_x12_x14"

@with_kw struct Config
    batch::Int      = 50
    inlen::Int      = 4
    outlen::Int     = 1
    niters::Int     = 500000
    lr::Real        = 0.002
    lowlim::Int     = -2
    uplim::Int      = 2
    v               = Float32(1e-2)
    σ               = Float32(1e-2)
    initnau::String = "rand"
    initnmu::String = "rand"
end

normalize_logSt(v::Real) = loggamma((v+1)/2) -loggamma(v/2) -log(π*v)/2
normalize_logSt(v::Real, σ::Real) = normalize_logSt(v) - log(σ)

function logSt(t::Real, v::Real)
    normalize_logSt(v) -(v+1)/2*log(1+t^2/v)
end

function logSt(t::Real, v::Real, σ::Real)
    normalize_logSt(v,σ) -(v+1)/2*log(1+t^2/(v*σ^2))
end

using Zygote: @adjoint, pull_block_vert
@adjoint function reduce(::typeof(vcat), As::AbstractVector{<:AbstractVecOrMat})
  cumsizes = cumsum(size.(As, 1))
  return reduce(vcat, As), Δ -> (nothing, map((sz, A) -> pull_block_vert(sz, Δ, A), cumsizes, As))
end

function logSt(t::Flux.Params, v::Real)
    n0 = normalize_logSt(v)
    ls = map(p -> -(v+1)/2 .* log.(1 .+ p.^2 ./v), t)
    sum(reduce(vcat, map(vec, ls)))
end

function logSt(t::Flux.Params, v::Real, σ::Real)
    n0 = normalize_logSt(v)
    ls = map(p -> -(v+1)/2 .* log.(1 .+ p.^2 ./(v*σ^2)), t)
    sum(reduce(vcat, map(vec, ls)))
end

# net = Chain(Dense(2,2), Dense(2,2))
# ps = params(net)
# display(logSt(ps, 1.0))
# gs = Flux.gradient(()->logSt(ps,1.0), ps)
# error()

function run(config)
    @unpack initnau, initnmu, lowlim, uplim = config
    @unpack niters, batch, inlen, outlen, v, σ, lr = config
    model   = mapping(inlen, outlen, initf(initnau), initf(initnmu))
    p0 = [v,σ]
    loss    = (x,y) -> Flux.mse(model(x),y) + logSt(params(model), abs(p0[1]), abs(p0[2]))
    range   = Uniform(lowlim,uplim)
    data    = (generate(inlen,batch,range) for _ in 1:niters)
    opt     = RMSProp(lr)
    history = train!(loss, model, data, opt, p0)
    return @dict(model, history)
end

#################### Single run with default params ############################

using Plots
include(srcdir("utils.jl"))
config = Config()
res, fname = produce_or_load(datadir("$(pattern)_run1"), config, run, force=true)

m = res[:model]
h = res[:history]

pyplot()
p1 = plothistory(h)
p2 = plot(
    annotatedheatmap(m[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
    annotatedheatmap(m[2].W[end:-1:1,:], c=:bluesreds, title="NPU", clim=(-1,1)),
    size=(600,300))
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.svg"), p1)
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.svg"), p2)
error()

################################################################################


# set up dict which will be permuted to yield all config combinations
config_dicts = Dict(
    :βL1 => 10f0 .^ (-1f0:2f0),
    :init => [("diag", "zero"), ("diag","one"), ("rand","rand"),
              ("glorotuniform", "glorotuniform"),
              ("randn","randn")])

# permute and flatten :init -> :initnau, initnmu
config_dicts = map(dict_list(config_dicts)) do config
    i = pop!(config,:init)
    d = Dict{Symbol,Any}(:initnau=>i[1], :initnmu=>i[2])
    for k in keys(config)
        d[k] = config[k]
    end
    d
end

Threads.@threads for d in config_dicts
    config = Config()
    config = reconstruct(config, d)
    for nr in 1:10
        res, fname = produce_or_load(
            datadir("$(pattern)_run$nr"),
            config, run)
    end
end

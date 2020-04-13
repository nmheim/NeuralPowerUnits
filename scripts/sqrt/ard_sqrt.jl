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

pattern = "mult_ard_sqrt"

@with_kw struct Config
    batch::Int      = 50
    inlen::Int      = 4
    outlen::Int     = 1
    niters::Int     = 30000
    α0::Float32     = 1f0
    β0::Float32     = 1f0
    lr::Real        = 0.001
    lowlim::Int     = 0
    uplim::Int      = 3
    initnau::String = "rand"
    initnmu::String = "rand"
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

# config = Config()
# res, fname = produce_or_load(datadir(pattern), config, run, force=true)
# 
# m = res[:model]
# h = res[:history]
# 
# pyplot()
# p1 = plothistory(h)
# net = get_mapping(m)
# p2 = plot(
#     annotatedheatmap(net[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
#     annotatedheatmap(net[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
#     size=(600,300))
# wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.svg"), p1)
# wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.svg"), p2)
# error()

################################################################################


# set up dict which will be permuted to yield all config combinations
config_dicts = Dict(
    :α0 => 10f0 .^ (-4f0:1f0),
    :β0 => 10f0 .^ (-4f0:1f0),
    :init => [("diag", "zero"), ("diag","one"),
              ("rand","rand"), ("glorotuniform", "glorotuniform"),
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

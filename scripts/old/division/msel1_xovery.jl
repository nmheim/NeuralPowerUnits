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

@with_kw struct Config
    batch::Int      = 50
    inlen::Int      = 4
    outlen::Int     = 1
    niters::Int     = 30000
    lr::Real        = 0.001
    lowlim::Int     = 1
    uplim::Int      = 3
    βL1             = 10
    initnau::String = "diag"
    initnmu::String = "zero"
end

function run(config)
    @unpack initnau, initnmu, lowlim, uplim, βL1 = config
    @unpack niters, batch, inlen, outlen, lr = config
    model   = mapping(inlen, outlen, initf(initnau), initf(initnmu))
    ps      = params(model)
    loss    = (x,y) -> Flux.mse(model(x),y) + βL1*norm(ps, 1)
    range   = Uniform(lowlim,uplim)
    data    = (generate(inlen,batch,range) for _ in 1:niters)
    opt     = RMSProp(lr)
    history = train!(loss, model, data, opt)
    return @dict(model, history)
end

#################### Single run with default params ############################

# pattern = "division_msel1_xovery_run1"
# config = Config()
# res, fname = produce_or_load(datadir(pattern), config, run, force=true)
# 
# m = res[:model]
# h = res[:history]
# 
# pyplot()
# p1 = plothistory(h)
# p2 = plot(
#     annotatedheatmap(m[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
#     annotatedheatmap(m[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
#     size=(600,300))
# wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.svg"), p1)
# wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.svg"), p2)
# error()


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
            datadir("division_msel1_xovery_run$nr"),
            config, run)
    end
end

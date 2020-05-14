using DrWatson
@quickactivate "NIP_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Parameters
using ValueHistories
using UnicodePlots

using Flux
using LinearAlgebra
using NeuralArithmetic
using GenerativeModels

include(srcdir("schedules.jl"))
include(srcdir("arithmetic_dataset.jl"))
include(srcdir("arithmetic_models.jl"))
include(srcdir("arithmetic_ard_models.jl"))

@with_kw struct AddARDConfig
    batch::Int      = 128
    niters::Int     = 100000
    lr::Real        = 1e-3

    α0::Real        = 1f-2
    β0::Real        = 1f2

    lowlim::Real    = -1
    uplim::Real     = 1
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 10
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "gatednpu"
end

function run(c::AddARDConfig)
    generate = arithmetic_dataset(+, c.inlen,
        d=Uniform(c.lowlim,c.uplim),
        subset=c.subset,
        overlap=c.overlap)
    test_generate = arithmetic_dataset(+, c.inlen,
        d=Uniform(c.lowlim-4,c.uplim+4),
        subset=c.subset,
        overlap=c.overlap)

    net   = get_model(c.model, c.inlen, c.fstinit, c.sndinit)
    model = ardnet_αβ(net, c.α0, c.β0, 1)
    loss  = (x,y) -> notelbo(model, x, y)

    data     = (generate(c.batch) for _ in 1:c.niters)
    val_data = test_generate(1000)

    opt      = RMSProp(c.lr)
    history  = train!(loss, model, data, val_data, opt)

    return @dict(model, history, c)
end

pattern = basename(splitext(@__FILE__)[1])
config = AddARDConfig()
outdir  = datadir("$(pattern)_run=1")
res, fname = produce_or_load(outdir, config, run, force=true)

m = res[:model]
h = res[:history]

using Plots
include(srcdir("plots.jl"))

pyplot()
if config.inlen < 30
    p1 = plot(h)
    wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.png"), p1)
end

ps = map(l->Plots.heatmap(l.W[end:-1:1,:], c=:bluesreds, 
                          title=summary(l), clim=(-2,2)), m)
p2 = plot(ps..., size=(600,300))

wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.png"), p2)

using DrWatson
@quickactivate "NIP_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Parameters
using ValueHistories
using UnicodePlots

using LinearAlgebra
using SpecialFunctions
using Flux
using NeuralArithmetic

include(srcdir("schedules.jl"))
include(srcdir("arithmetic_dataset.jl"))
include(srcdir("arithmetic_models.jl"))
include(srcdir("arithmetic_st_models.jl"))

@with_kw struct MultStConfig
    batch::Int      = 128
    niters::Int     = 200000
    lr::Real        = 1e-4

    v::Real         = 1f0
    σstart::Real    = 1f1
    σend::Real      = 1f-1
    σdecay::Real    = 1f-1
    σstep::Int      = 10000

    lowlim::Real    = -1
    uplim::Real     = 1
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 20
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "gatednpu"
end


function run(c::MultStConfig)
    generate = arithmetic_dataset(*, c.inlen,
        d=Uniform(c.lowlim,c.uplim),
        subset=c.subset,
        overlap=c.overlap)
    test_generate = arithmetic_dataset(*, c.inlen,
        d=Uniform(c.lowlim-4,c.uplim+4),
        subset=c.subset,
        overlap=c.overlap)

    data     = (generate(c.batch) for _ in 1:c.niters)
    val_data = test_generate(1000)
    opt      = RMSProp(c.lr)

    model = get_model(c.model, c.inlen, c.fstinit, c.sndinit)

    schedule = false

    if !schedule
        σ = [c.σstart]
        model = StModel(model, [c.v], σ)
        ps = params(model.m)

        loss = function (x,y)
            m = Flux.mse(model(x),y)
            s = -logSt(ps, c.v, σ[1])
            m+s, m, s
        end
        history  = train!(loss, model, data, val_data, opt)
    else
        loss = function (x,y,σ)
            m = Flux.mse(model(x), y)
            s = logSt(params(model), c.v, σ)
            m+s, m, s
        end
        ps = params(model)
        σdecay = ExpSchedule(c.σstart, c.σend, c.σdecay, c.σstep)
        history  = train!(loss, model, data, val_data, opt, σdecay)
    end
    
    return @dict(model, history, c)
end

pattern = basename(splitext(@__FILE__)[1])
config = MultStConfig()
outdir  = datadir("tests", pattern)
res, fname = produce_or_load(outdir, config, run, force=true)

m = get_mapping(res[:model])
h = res[:history]

using Plots
include(srcdir("plots.jl"))

pyplot()
if config.inlen < 30
    p1 = plot(h,logscale=false)
    wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.svg"), p1)
end

ps = map(l->Plots.heatmap(l.W[end:-1:1,:], c=:bluesreds,
                    title=summary(l), clim=(-1,1)), m)
p2 = plot(ps..., size=(600,300))

wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.svg"), p2)

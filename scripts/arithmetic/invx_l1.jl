using DrWatson
@quickactivate "NIPS_2020_NPU"

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

include(srcdir("schedules.jl"))
include(srcdir("arithmetic_dataset.jl"))
include(srcdir("arithmetic_models.jl"))

@with_kw struct DivL1Config
    batch::Int      = 128
    niters::Int     = 1e5
    lr::Real        = 5e-3

    βstart::Real    = 1f-9
    βend::Real      = 1f-6
    βgrowth::Real   = 10f0
    βstep::Int      = 10000

    lowlim::Real    = 0
    uplim::Real     = 0.5
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0
    sampler::String = "sobol"

    inlen::Int      = 100
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "gatednpux"

end


function run(c::DivL1Config)
    generate = arithmetic_invx_dataset(c.inlen, c.subset, c.lowlim, c.uplim,
        sampler=c.sampler)
    test_generate = arithmetic_invx_dataset(c.inlen, c.subset, c.lowlim-2, c.uplim+2,
        sampler=c.sampler)
    model = get_model(c.model, c.inlen, c.fstinit, c.sndinit)
    βgrowth = ExpSchedule(c.βstart, c.βend, c.βgrowth, c.βstep)
    ps = params(model)

    function loss(x,y,β)
        mse = Flux.mse(model(x),y)
        L1  = β * norm(ps,1)
        (mse+L1), mse, L1
    end
    
    data     = (generate(c.batch) for _ in 1:c.niters)
    val_data = test_generate(1000)

    opt      = RMSProp(c.lr)
    history  = train!(loss, model, data, val_data, opt, βgrowth)

    return @dict(model, history)
end

pattern = basename(splitext(@__FILE__)[1])
config = DivL1Config()
outdir  = datadir("tests", pattern)
res, fname = produce_or_load(outdir, config, run, force=true)

m = get_mapping(res[:model])
h = res[:history]





error()







using Plots
include(srcdir("plots.jl"))

pyplot()
if config.inlen < 30
    p1 = plot(h)
    wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.png"), p1)
end

ps = map(l->Plots.heatmap(l.W[end:-1:1,:], c=:bluesreds,
                    title=summary(l), clim=(-1,1)),
         m)
p2 = plot(ps..., size=(600,300))

wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.png"), p2)

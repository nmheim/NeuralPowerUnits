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

include(srcdir("schedules.jl"))
include(srcdir("arithmetic_dataset.jl"))
include(srcdir("arithmetic_models.jl"))

@with_kw struct MultL1Config
    batch::Int      = 128
    niters::Int     = 50000
    lr::Real        = 1e-3

    βstart::Real    = 1f-8
    βend::Real      = 1f0
    βgrowth::Real   = 10f0
    βstep::Int      = 1000

    lowlim::Real    = -2
    uplim::Real     = 2
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 20
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "npu"

end


function run(c::MultL1Config)
    generate = arithmetic_dataset(*, c.inlen,
        d=Uniform(c.lowlim,c.uplim),
        subset=c.subset,
        overlap=c.overlap)
    test_generate = arithmetic_dataset(*, c.inlen,
        d=Uniform(c.lowlim-4,c.uplim+4),
        subset=c.subset,
        overlap=c.overlap)

    model = get_model(c.model, c.inlen, c.fstinit, c.sndinit)
    βgrowth = ExpSchedule(c.βstart, c.βend, c.βgrowth, c.βstep)
    ps = params(model)

    function loss(x,y,β)
        mse = Flux.mse(model(x),y)
        L1  = β * norm(ps,1)
        (mse+L1), mse, L1
    end
    
    data     = (generate(c.batch) for _ in 1:c.niters)
    val_data = generate(1000)

    opt      = RMSProp(c.lr)
    history  = train!(loss, model, data, val_data, opt, βgrowth)

    return @dict(model, history)
end

pattern = basename(splitext(@__FILE__)[1])
config = MultL1Config()
outdir  = datadir("tests", pattern)
res, fname = produce_or_load(outdir, config, run, force=true)

m = get_mapping(res[:model])
h = res[:history]

using Plots
include(srcdir("plots.jl"))

pyplot()
p1 = plot(h)
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.svg"), p1)

ps = map(l->Plots.heatmap(l.W[end:-1:1,:], c=:bluesreds,
                    title=summary(l), clim=(-1,1)),
         m)
p2 = plot(ps..., size=(600,300))

wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.svg"), p2)

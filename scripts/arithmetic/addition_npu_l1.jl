using DrWatson
@quickactivate "NIP_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Parameters
using ValueHistories

using Flux
using LinearAlgebra
using NeuralArithmetic

include(srcdir("arithmetic_dataset.jl"))
include(srcdir("arithmetic_models.jl"))

@with_kw struct AddL1Config
    batch::Int      = 128
    inlen::Int      = 10
    niters::Int     = 50000
    lr::Real        = 0.01
    lowlim::Real    = 1
    uplim::Real     = 2
    βL1             = 1
    fstinit::String = "glorotuniform"
    sndinit::String = "glorotuniform"
    model::String   = "npu"
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0
end


function run(c::AddL1Config)
    generate = arithmetic_dataset(+, c.inlen,
        d=Uniform(c.lowlim,c.uplim),
        subset=c.subset,
        overlap=c.overlap)
    test_generate = arithmetic_dataset(+, c.inlen,
        d=Uniform(c.lowlim-4,c.uplim+4),
        subset=c.subset,
        overlap=c.overlap)

    model = get_model(c.model, c.inlen, c.fstinit, c.sndinit)
    ps = params(model)

    function loss(x,y)
        mse = Flux.mse(model(x),y)
        L1  = c.βL1 * norm(ps,1)
        (mse+L1), mse, L1
    end
    
    data     = (generate(c.batch) for _ in 1:c.niters)
    val_data = generate(1000)

    opt      = ADAM(c.lr)
    history  = train!(loss, model, data, val_data, opt)

    return @dict(model, history)
end

pattern = basename(splitext(@__FILE__)[1])
config = AddL1Config()
outdir  = datadir("tests", pattern)
res, fname = produce_or_load(outdir, config, run, force=true)

m = res[:model]
h = res[:history]

using Plots
using GMExtensions
include(srcdir("plots.jl"))

pyplot()
p1 = plot(h)
ps = [heatmap(l.W[end:-1:1,:], c=:bluesreds, title=summary(l), clim=(-1,1)) for l in m]
p2 = plot(ps..., size=(600,300))
# display(p1)
# display(p2)
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.svg"), p1)
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.svg"), p2)

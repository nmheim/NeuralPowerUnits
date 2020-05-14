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

@with_kw struct AddL1SearchConfig
    batch::Int      = 128
    niters::Int     = 5e5
    lr::Real        = 5e-3

    βstart::Real    = 1f-4
    βend::Real      = 1f-2
    βgrowth::Real   = 10f0
    βstep::Int      = 10000

    lowlim::Real    = -1
    uplim::Real     = 1
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 100
    fstinit::String = "rand"
    sndinit::String = "rand"
    model::String   = "gatednpu"

    run::Int        = 1
end


function run(c::AddL1SearchConfig)
    generate = arithmetic_dataset(+, c.inlen,
        d=Uniform(c.lowlim,c.uplim),
        subset=c.subset,
        overlap=c.overlap)
    test_generate = arithmetic_dataset(+, c.inlen,
        d=Uniform(c.lowlim-4,c.uplim+4),
        subset=c.subset,
        overlap=c.overlap)

    togpu = Flux.gpu
    #togpu = identity

    model = get_model(c.model, c.inlen, c.fstinit, c.sndinit) |> togpu
    βgrowth = ExpSchedule(c.βstart, c.βend, c.βgrowth, c.βstep)
    ps = params(model)

    function loss(x,y,β)
        mse = Flux.mse(model(x),y)
        L1  = β * norm(ps,1)
        (mse+L1), mse, L1
    end
    
    data     = (togpu(generate(c.batch)) for _ in 1:c.niters)
    val_data = test_generate(1000) |> togpu

    opt      = RMSProp(c.lr)
    history  = train!(loss, model, data, val_data, opt, βgrowth, log=false)

    return @dict(model, history, c)
end

# set up dict which will be permuted to yield all config combinations
config_dicts = Dict(:βend => 10f0 .^ (-3f0:-1f0),
                    :init => [("rand","rand"),
                              ("glorotuniform", "glorotuniform")],
                    :model => ["gatednpu", "gatednpux"])

# permute and flatten :init -> :initnau, initnmu
config_dicts = map(dict_list(config_dicts)) do config
    i = pop!(config,:init)
    d = Dict{Symbol,Any}(:fstinit=>i[1], :sndinit=>i[2])
    for k in keys(config)
        d[k] = config[k]
    end
    d
end

for d in config_dicts
    config = AddL1SearchConfig()
    for nr in 1:5
        d[:run] = nr
        config = reconstruct(config, d)
        res, fname = produce_or_load(
            datadir(basename(splitext(@__FILE__)[1])), config, run)
    end
end

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
include(srcdir("unicodeheat.jl"))
include(joinpath(@__DIR__, "sobolconfigs.jl"))

function run(c::MultL1SearchConfig)
    generate = arithmetic_dataset(mult, c.inlen, c.subset, c.overlap, c.lowlim, c.uplim,
        sampler=c.sampler)
    test_generate = arithmetic_dataset(mult, c.inlen, c.subset, c.overlap, c.lowlim-4, c.uplim+4,
        sampler=c.sampler)

    #togpu = Flux.gpu
    togpu = identity

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
    val = Flux.mse(model(val_data[1]), val_data[2])

    model = model |> cpu
    return @dict(model, history, c, val)
end

################################################################################

config = MultL1SearchConfig()
@progress name="All runs: " for i in 1:20
    @info config
    for m in ["gatednpux","nalu","nmu","npux","inalu"]
        config = if m == "nmu"
            MultL1SearchConfig(run=i, model=m, βstart=1f-9, niters=20000)
        else
            MultL1SearchConfig(run=i, model=m)
        end
        res, fname = produce_or_load(datadir(basename(splitext(@__FILE__)[1])),
                                     config, run, digits=10)
        @info "Validation error run #$i: $(res[:val]) ($m)"
        display(heat(res[:model]))
    end
end



################################################################################
# this code performs grid search
################################################################################

# set up dict which will be permuted to yield all config combinations
# config_dicts = Dict(:βend => 10f0 .^ (-6f0:-5f0),
#                     :init => [("rand","rand"),
#                               ("glorotuniform", "glorotuniform")],
#                     :model => ["gatednpu", "gatednpux", "nmu", "nalu"])
# 
# # permute and flatten :init -> :initnau, initnmu
# config_dicts = map(dict_list(config_dicts)) do config
#     i = pop!(config,:init)
#     d = Dict{Symbol,Any}(:fstinit=>i[1], :sndinit=>i[2])
#     for k in keys(config)
#         d[k] = config[k]
#     end
#     d
# end
# 
# @progress name="Mult Search: " for d in config_dicts
#     config = MultL1SearchConfig()
#     for nr in 1:5
#         d[:run] = nr
#         config = reconstruct(config, d)
#         res, fname = produce_or_load(
#             datadir(basename(splitext(@__FILE__)[1])), config, run, digits=10)
#     end
# end

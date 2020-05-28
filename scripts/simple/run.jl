using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using LinearAlgebra
using Statistics
using NeuralArithmetic

include(joinpath(@__DIR__, "dataset.jl"))

train_range = "pos"
nr_runs = 20


function generate()
    if train_range == "pos-neg"
        return generate_pos_neg()
    elseif train_range == "pos"
        return generate_pos()
    else
        error("Unknown train range: $train_range")
    end
end

function result_dict(model::Chain, config::Dict)
    res = Dict{Symbol,Any}([(k,v) for (k,v) in config])
    res[:model] = model

    (x,y) = generate()
    res[:mse] = Flux.mse(model(x),y)

    (xt,yt) = test_generate()
    res[:val] = Flux.mse(model(xt),yt)

    # training error
    x = 0.1f0:0.1f0:2
    y = 0.1f0:0.1f0:2
    xy = reduce(hcat, map(t->[t...], Iterators.product(x,y)))
    t = model(xy)
    res[:add_trn]  = mean(abs, t[1,:] - vec(f1(xy)))
    res[:mult_trn] = mean(abs, t[2,:] - vec(f2(xy)))
    res[:div_trn]  = mean(abs, t[3,:] - vec(f3(xy)))
    res[:sqrt_trn] = mean(abs, t[4,:] - vec(f4(xy)))


    # validation error
    x = -4.1f0:0.2f0:4f0
    y = -4.1f0:0.2f0:4f0
    xy = reduce(hcat, map(t->[t...], Iterators.product(x,y)))
    t = model(xy)
    res[:add_val]  = mean(abs, t[1,:] - vec(f1(xy)))
    res[:mult_val] = mean(abs, t[2,:] - vec(f2(xy)))
    res[:div_val]  = mean(abs, t[3,:] - vec(f3(xy)))

    x = 0.1f0:0.1f0:4
    y = 0.1f0:0.1f0:4
    xy = reduce(hcat, map(t->[t...], Iterators.product(x,y)))
    t = model(xy)
    res[:sqrt_val] = mean(abs, t[4,:] - vec(f4(xy)))
    display(res)

    return res
end

function run_npu(c::Dict)
    hdim = 6
    model = Chain(GatedNPUX(2,hdim),NAU(hdim,4))
    ps = params(model)
    opt = ADAM(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y) + c[:βl1]*norm(ps, 1) #+ 0.1norm(model.Im,1)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end

function run_nmu(c::Dict)
    hdim = 6
    model = Chain(NMU(2,hdim),NAU(hdim,4))
    ps = params(model)
    opt = ADAM(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end

function run_nalu(c::Dict)
    hdim = 6
    model = Chain(NALU(2,hdim),NALU(hdim,4))
    ps = params(model)
    opt = RMSProp(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end

function run_inalu(c::Dict)
    hdim = 6
    model = Chain(iNALU(2,hdim),iNALU(hdim,4))
    ps = params(model)
    opt = RMSProp(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end

function run_dense(c::Dict)
    model = Chain(Dense(2,10,σ),Dense(10,10,σ),Dense(10,4))
    ps = params(model)
    opt = ADAM(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end


@progress for run in 1:nr_runs
    res, _ = produce_or_load(datadir("simple"),
                             Dict(:niters=>20000, :βl1=>1e-5, :lr=>0.005, :run=>run),
                             run_npu,
                             prefix="$train_range-gatednpux",
                             force=false, digits=6)
    res, _ = produce_or_load(datadir("simple"),
                             Dict(:niters=>20000, :lr=>0.005, :run=>run),
                             run_nalu,
                             prefix="$train_range-nalu", force=false, digits=6)
    res, _ = produce_or_load(datadir("simple"),
                             Dict(:niters=>20000, :lr=>0.005, :run=>run),
                             run_nmu,
                             prefix="$train_range-nmu", force=false, digits=6)
    res, _ = produce_or_load(datadir("simple"),
                             Dict(:niters=>20000, :lr=>0.005, :run=>run),
                             run_dense,
                             prefix="$train_range-dense", force=false, digits=6)
    res, _ = produce_or_load(datadir("simple"),
                             Dict(:niters=>20000, :lr=>0.001, :run=>run),
                             run_inalu,
                             prefix="$train_range-inalu", force=false, digits=6)
end

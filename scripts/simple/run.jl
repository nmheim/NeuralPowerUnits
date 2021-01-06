using DrWatson
@quickactivate

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using LinearAlgebra
using Statistics
using NeuralArithmetic

include(joinpath(@__DIR__, "dataset.jl"))

function result_dict(model::Chain, config::Dict)
    @unpack umin, umax = config
    res = Dict{Symbol,Any}([(k,v) for (k,v) in config])
    res[:model] = model

    (x,y) = generate_range(umin=umin, umax=umax)
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
    @unpack umin, umax = c
    hdim = 6
    #model = Chain(NPU(2,hdim),NAU(hdim,4))
    model = Chain(NPU(2,hdim),NAU(hdim,4))
    ps = params(model)
    opt = ADAM(c[:lr])
    data = (generate_range(umin=umin,umax=umax) for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y) + c[:βl1]*norm(ps, 1) #+ 0.1norm(model.Im,1)
    (x,y) = generate_range(umin=umin,umax=umax)
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end

function run_realnpu(c::Dict)
    @unpack umin, umax = c
    hdim = 6
    model = Chain(RealNPU(2,hdim),NAU(hdim,4))
    ps = params(model)
    opt = ADAM(c[:lr])
    data = (generate_range(umin=umin,umax=umax) for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y) + c[:βl1]*norm(ps, 1) #+ 0.1norm(model.Im,1)
    (x,y) = generate_range(umin=umin,umax=umax)
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end

function run_nmu(c::Dict)
    @unpack umin, umax = c
    hdim = 6
    model = Chain(NMU(2,hdim),NAU(hdim,4))
    ps = params(model)
    opt = ADAM(c[:lr])
    data = (generate_range(umin=umin,umax=umax) for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate_range(umin=umin,umax=umax)
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end

function run_nalu(c::Dict)
    @unpack umin, umax = c
    hdim = 6
    model = Chain(NALU(2,hdim),NALU(hdim,4))
    ps = params(model)
    opt = RMSProp(c[:lr])
    data = (generate_range(umin=umin,umax=umax) for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate_range(umin=umin,umax=umax)
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end

function inalu_reg(p::Array, t::Real)
    r = min.(-p,p) .+ t
    sum(max.(r, 0)) ./ t
end

inalu_reg(ps::Flux.Params,t::Real) = sum(p->inalu_reg(p,t), ps)

function run_inalu(c::Dict)
    @unpack umin, umax = c
    hdim = 6
    model = Chain(iNALU(2,hdim),iNALU(hdim,4))
    ps = params(model)
    opt = RMSProp(c[:lr])


    data = (generate_range(umin=umin,umax=umax) for _ in 1:(c[:niters]/2))
    (x,y) = generate_range(umin=umin,umax=umax)
    cb = Flux.throttle(() -> (@info loss(x,y) Flux.mse(model(x),y) inalu_reg(ps,c[:t])), 0.1)

    loss(x,y) = Flux.mse(model(x),y)
    Flux.train!(loss, ps, data, opt, cb=cb)

    regloss(x,y) = Flux.mse(model(x),y) + 1e-5*inalu_reg(ps,c[:t])
    data = (generate_range(umin=umin,umax=umax) for _ in 1:(c[:niters]/2))
    Flux.train!(regloss, ps, data, opt, cb=cb)

    return result_dict(model,c)
end

function run_dense(c::Dict)
    @unpack umin, umax = c
    hdim = 50
    model = Chain(Dense(2,hdim,σ),Dense(hdim,hdim,σ),Dense(hdim,4))
    #model = Chain(Dense(2,hdim,σ),Dense(hdim,4))
    ps = params(model)
    opt = ADAM(c[:lr])
    data = (generate_range(umin=umin,umax=umax) for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate_range(umin=umin,umax=umax)
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return result_dict(model,c)
end

#train_ranges = [(0.01f0,1), (0.1f0, 0.2f0), (1,2), (1.1f0,1.2f0), (10,20)]
train_ranges = [(0.01f0,2)]
nr_runs = 20

for (umin,umax) in train_ranges
    @info "Running training range: ($umin,$umax)"
    data_directory = datadir("simple_lr005_umin=$(umin)_umax=$(umax)")

    @progress for run in 1:nr_runs
        res, _ = produce_or_load(data_directory,
                                 Dict(:niters=>20000, :βl1=>0, :lr=>0.005, :run=>run, :umin=>umin, :umax=>umax),
                                 run_npu,
                                 prefix="npu",
                                 force=false, digits=6)
        res, _ = produce_or_load(data_directory,
                                 Dict(:niters=>20000, :βl1=>0, :lr=>0.005, :run=>run, :umin=>umin, :umax=>umax),
                                 run_realnpu,
                                 prefix="realnpu",
                                 force=false, digits=6)
        res, _ = produce_or_load(data_directory,
                                 Dict(:niters=>20000, :lr=>0.005, :run=>run, :umin=>umin, :umax=>umax),
                                 run_nalu,
                                 prefix="nalu", force=false, digits=6)
        res, _ = produce_or_load(data_directory,
                                 Dict(:niters=>20000, :lr=>0.005, :run=>run, :umin=>umin, :umax=>umax),
                                 run_nmu,
                                 prefix="nmu", force=false, digits=6)
        res, _ = produce_or_load(data_directory,
                                 Dict(:niters=>20000, :lr=>0.005, :run=>run, :umin=>umin, :umax=>umax),
                                 run_dense,
                                 prefix="dense", force=false, digits=6)
        res, _ = produce_or_load(data_directory,
                                 Dict(:niters=>20000, :lr=>0.005, :run=>run, :t=>20, :umin=>umin, :umax=>umax),
                                 run_inalu,
                                 prefix="inalu", force=false, digits=6)
    end
end

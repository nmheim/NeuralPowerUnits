using Distributions
using DrWatson
using Flux
using NeuralArithmetic

function dense_abs(config)
    @unpack hdim = config

    task(x) = abs(x)
    xs = rand(Uniform(-1,1), 1, 100, 10000)
    data = [(xs[:,:,i],task.(xs[:,:,i])) for i in 1:size(xs,3)]

    model = Chain(Dense(1,hdim,σ), Dense(hdim,hdim,σ), Dense(hdim,1))
    trainmodel(model, data, config)
end
function dense_square(config)
    @unpack hdim = config

    task(x) = x^2
    xs = rand(Uniform(-1,1), 1, 100, 10000)
    data = [(xs[:,:,i],task.(xs[:,:,i])) for i in 1:size(xs,3)]

    model = Chain(Dense(1,hdim,σ), Dense(hdim,1))
    trainmodel(model, data, config)
end
function dense_sin(config)
    @unpack hdim = config

    task(x) = sin(x)
    xs = rand(Uniform(-3,3), 1, 100, 10000)
    data = [(xs[:,:,i],task.(xs[:,:,i])) for i in 1:size(xs,3)]

    model = Chain(Dense(1,hdim,σ), Dense(hdim,1))
    trainmodel(model, data, config)
end



function npu_abs(config)
    @unpack hdim = config

    task(x) = abs(x)
    xs = Float32.(rand(Uniform(-1,1), 1, 100, 10000))
    data = [(xs[:,:,i],task.(xs[:,:,i])) for i in 1:size(xs,3)]

    model = Chain(NPU(1,hdim), NAU(hdim,1))
    trainmodel(model, data, config)
end
function npu_square(config)
    @unpack hdim = config

    task(x) = x^2
    xs = Float32.(rand(Uniform(-1,1), 1, 100, 10000))
    data = [(xs[:,:,i],task.(xs[:,:,i])) for i in 1:size(xs,3)]

    model = Chain(NPU(1,hdim), NAU(hdim,1))
    trainmodel(model, data, config)
end
function npu_sin(config)
    @unpack hdim = config

    task(x) = sin(x)
    xs = Float32.(rand(Uniform(-3,3), 1, 100, 10000))
    data = [(xs[:,:,i],task.(xs[:,:,i])) for i in 1:size(xs,3)]

    model = Chain(NPU(1,hdim), NAU(hdim,1))
    trainmodel(model, data, config)
end




function trainmodel(model, data, config)
    @unpack lr = config

    opt = ADAM(lr)
    ps = Flux.params(model)
    l1(ps) = sum(x->sum(abs,x), ps)
    loss(x,y) = Flux.mse(model(x), y)
    
    logcb() = @info loss(data[1]...)
    Flux.train!(loss, ps, data, opt, cb=logcb)
   
    @dict(model)
end

predict(m,x::Real) = m([x])[1]

config = Dict("lr" => 1e-1, "hdim" => 5)
res, _ = produce_or_load(datadir(), config, dense_abs, prefix = "dense-abs")
denseabs = res[:model]

config = Dict("lr" => 1e-1, "hdim" => 5)
res, _ = produce_or_load(datadir(), config, dense_square, prefix = "dense-square")
dense_sq = res[:model]

config = Dict("lr" => 1e-1, "hdim" => 5)
res, _ = produce_or_load(datadir(), config, dense_sin, prefix = "dense-sin", force=false)
densesin = res[:model]


config = Dict("lr" => 1e-2, "hdim" => 5)
res, _ = produce_or_load(datadir(), config, npu_abs, prefix = "npu-abs")
npuabs = res[:model]

config = Dict("lr" => 1e-1, "hdim" => 5)
res, _ = produce_or_load(datadir(), config, npu_square, prefix = "npu-square")
npu_sq = res[:model]

config = Dict("lr" => 1e-2, "hdim" => 5)
res, _ = produce_or_load(datadir(), config, npu_sin, prefix = "npu-sin", force=false)
npusin = res[:model]


using Plots
using LaTeXStrings
pgfplotsx()
#pyplot()
theme(:default, lw=2, legend=false)

function plotlines(f, dense, npu, x; plotnpu=true, vlines=[-1,1])
    plt = plot(x, f, label=L"f(x)", color="black") 
    vline!(plt, vlines, color="black", lw=1, alpha=0.5, label="Train Range")
    plot!(plt, x, x->predict(dense,x), label="Dense", ls=:dash, color=2)
    if plotnpu
        plot!(plt, x, x->predict(npu,x), label="NPU", ls=:dashdot, color=3)
    end
    plt
end

# WITHOUT NPU
x = -2:0.1f0:2
p1 = plotlines(x->x^2, dense_sq, npu_sq, x, plotnpu=false)
plot!(p1, title=L"f(x) = x^2")

p2 = plotlines(abs, denseabs, npuabs, x, plotnpu=false)
plot!(p2, title=L"f(x) = |x|")

x = -8:0.1f0:8
p3 = plotlines(sin, densesin, npusin, x, plotnpu=false, vlines=[-3,3])
p3 = plot!(p3, title=L"f(x) = \sin(x)", legend=true, ylims=(-1.5,1.5))

p = plot(p1,p2,p3,layout=(1,3),size=(600,200))
savefig(p, "1d-extrapolation-nonpu.tikz")


# WITH NPU
x = -2:0.1f0:2
p1 = plotlines(x->x^2, dense_sq, npu_sq, x)
plot!(p1, title=L"f(x) = x^2")

p2 = plotlines(abs, denseabs, npuabs, x)
plot!(p2, title=L"f(x) = |x|")

x = -8:0.1f0:8
p3 = plotlines(sin, densesin, npusin, x, vlines=[-3,3])
p3 = plot!(p3, title=L"f(x) = \sin(x)", legend=true, ylims=(-1.5,1.5))

p = plot(p1,p2,p3,layout=(1,3),size=(600,200))
savefig(p, "1d-extrapolation.tikz")
plot!(p3, size = (250,200))
savefig(p3, "1d-extrapolation-sin.tikz")
display(p3)

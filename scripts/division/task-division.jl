using DrWatson
@quickactivate "arithmetic"

using Flux
using Plots
using GenerativeModels
using GMExtensions
using NeuralArithmetic
using Distributions: Uniform
using Parameters
using ValueHistories
using LinearAlgebra

include(joinpath(@__DIR__, "utils.jl"))
include(srcdir("utils.jl"))

#@with_kw struct Config
    T           = Float32
    batch       = 50
    inlen       = 4
    outlen      = 1
    niters      = 30000
    α0          = T(1)
    β0          = T(1)
    lr          = 0.001
    esamples    = 1
    lowlim = 1
    uplim = 3
    initnau = "diag"
    initnmu = "zero"
#end

function init_diag(T, a::Int, b::Int)
    m = zeros(T,a,b)
    m[diagind(m,0)] .= T(1)
    return m
end

function initf(s::String)
    if s == "diag"
        return (a,b) -> init_diag(T,a,b)
    elseif s == "glorotuniform"
        return (a,b) -> Flux.glorot_uniform(T,a,b)
    elseif s == "rand"
        return (a,b) -> rand(T,a,b)
    elseif s == "randn"
        return (a,b) -> randn(T,a,b)
    elseif s == "zero"
        return (a,b) -> zeros(T,a,b)
    else
        throw(ArgumentError("Unkown init: $init"))
    end
end


setup = @dict(T, batch, inlen, outlen, niters, α0, β0, lr,
              esamples)

function f(x::Array{T,2}) where T
    x1 = x[1,:]
    x2 = x[2,:]
    y = x1 ./ x2
    reshape(y, 1, :)
end

function generate(inlen::Int, batch::Int, r::Uniform)
    x = T.(rand(r, inlen, batch))
    y = f(x)
    (x,y)
end

function run(config)
    @unpack niters, batch, inlen, outlen, α0, β0, lr = config
    initnau = "diag"
    initnmu = "zero"
    net     = mapping(inlen, outlen, initf(initnau), initf(initnmu))
    model   = ardnet(net, α0, β0, outlen)
    loss    = (x,y) -> notelbo(model,x,y,α0=α0,β0=β0)
    range   = Uniform(lowlim,uplim)
    data    = (generate(inlen,batch,range) for _ in 1:niters)
    opt     = RMSProp(lr)
    history = train!(loss, model, data, opt)
    return @dict(model, history)
end


file = produce_or_load(
    datadir("10-param-func"),
    setup,
    run,
    force=true
)

pyplot()

m = file[1][:model]
h = file[1][:history]
z = get(h, :μz)[2]
p1 = plothistory(h)

nmu = m.decoder.mapping.restructure(mean(m.encoder))
(x,y) = generate()
display(nmu(x))
display(f(x))
display(task)

p2 = plot(
    annotatedheatmap(nmu[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
    annotatedheatmap(nmu[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
    size=(600,300)
)
display(p1)

# savefig(p1, plotsdir("10-param-func-bayes-$task-history.pdf"))
# savefig(p2, plotsdir("10-param-func-bayes-$task.pdf"))

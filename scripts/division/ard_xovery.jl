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

function init_diag(T, a::Int, b::Int)
    m = zeros(T,a,b)
    m[diagind(m,0)] .= T(1)
    return m
end

function run(setup)
    @unpack T, initnau, initnmu, inlen, outlen, α0, β0, lr, niters, lowlim, uplim = setup

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

    net     = mapping(inlen, outlen, initf(initnau), initf(initnmu))
    model   = ardnet(net, α0, β0, outlen)
    loss    = (x,y) -> notelbo(model,x,y,α0=α0,β0=β0)
    range   = Uniform(lowlim, uplim)
    data    = (generate(inlen, outlen, range) for _ in 1:niters)
    opt     = RMSProp(lr)
    history = train!(loss, model, data, opt)
    return @dict model history
end



T           = Float32
batch       = 50
inlen       = 4
outlen      = 1
niters      = 30000
α0          = T(1)
β0          = T(1)
lr          = 0.0001
lowlim      = 1
uplim       = 3
initnau     = "rand"
initnmu     = "rand"


setup = @dict(T, batch, inlen, outlen, niters, α0, β0, lr,
              initnau, initnmu, uplim, lowlim)

run(setup)
error("yeeeeeeah")

res, fname = produce_or_load(datadir("division_xovery"), setup, run, force=true)

m = res[:model]
h = res[:history]

pyplot()
p1 = plothistory(h)
net = get_mapping(m)
p2 = plot(
    annotatedheatmap(net[1].W[end:-1:1,:], c=:bluesreds, title="NAU", clim=(-1,1)),
    annotatedheatmap(net[2].W[end:-1:1,:], c=:bluesreds, title="ReNMUX", clim=(-1,1)),
    size=(600,300))
wsave(plotsdir("division_xovery", "$(basename(splitext(fname)[1]))-history.svg"), p1)
wsave(plotsdir("division_xovery", "$(basename(splitext(fname)[1]))-mapping.svg"), p2)

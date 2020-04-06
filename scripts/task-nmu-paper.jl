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

include(joinpath(@__DIR__, "setup.jl"))

task = "task-nmu-paper"
T           = Float32
batch       = 50
inlen       = 4
outlen      = 1
niters      = 30000
α0          = T(1)
β0          = T(1)
lr          = 0.004
esamples    = 1
r           = Uniform(-2,2)

initf(s...) = rand(T, s...)

setup = @dict(T, batch, inlen, outlen, niters, α0, β0, lr,
              esamples, r, initf)

net = Chain(
            NAU(inlen, inlen, init=initf),
            ReNMUX(inlen, outlen, init=initf))

zlen = length(Flux.destructure(net)[1])
μz = Flux.destructure(net)[1]
σz = ones(T,zlen)/2
λz = ones(T,zlen)
σx = ones(T,1)

function f(x)
    x1 = x[1,:]
    x2 = x[2,:]
    x3 = x[3,:]
    x4 = x[4,:]
    y = (x1 .+ x2) .* (x1 .+ x2 .+ x3 .+ x4)
    reshape(y, 1, :)
end

rand_uniform(s...) = rand(r, s...)
Flux.@nograd rand_uniform

function generate()
    x = T.(rand_uniform(inlen, batch))
    y = f(x)
    (x,y)
end

file = produce_or_load(
    datadir("10-param-func"),
    setup,
    prefix="bayes-$task",
    run,
    # force=true
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

# savefig(p1, plotsdir("10-param-func-bayes-$task-history.pdf"))
# savefig(p2, plotsdir("10-param-func-bayes-$task.pdf"))

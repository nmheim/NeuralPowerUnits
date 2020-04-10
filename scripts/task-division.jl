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

function perfect_nau(T::Type, a::Int, b::Int)
    if a == b
        m = zeros(T, a, b)
        m[1,1] = T(1)
        m[2,2] = T(1)
        return m
    else
        error("not sure how to create perfect NAU")
    end
end

function perfect_nmu(T::Type, a::Int, b::Int)
    if a == 1
        m = zeros(T, a, b)
        m[1] = T(1)
        m[2] = T(-1)
        return m
    else
        error("not sure how to create perfect NMU")
    end
end

task = "task-division"
T           = Float32
batch       = 50
inlen       = 4
outlen      = 1
niters      = 30000
α0          = T(1)
β0          = T(1)
lr          = 0.001
esamples    = 1
r           = Uniform(1,3)

function init_nau(T, a::Int, b::Int)
    m = zeros(T,a,b)
    m[diagind(m,0)] .= T(1)
    return m
end

init_nmu(T, a::Int, b::Int) = zeros(T, a, b)

setup = @dict(T, batch, inlen, outlen, niters, α0, β0, lr,
              esamples, r)
 
net = Chain(
            NAU(inlen, inlen, init=(s...)->init_nau(T,s...)),
            ReNMUX(inlen, outlen, init=(s...)->init_nmu(T,s...))
           )
# net = Chain(
#             NAU(inlen, inlen),
#             ReNMUX(inlen, outlen)
#            )


zlen = length(Flux.destructure(net)[1])
μz = Flux.destructure(net)[1]
λz = ones(T,zlen)/10
σz = ones(T,zlen)/10
σx = ones(T,1)

function f(x)
    x1 = x[1,:]
    x2 = x[2,:]
    y = x1 ./ x2
    reshape(y, 1, :)
end

function generate()
    x = T.(rand(r, inlen, batch))
    y = f(x)
    (x,y)
end

(x,y) = generate()

file = produce_or_load(
    datadir("10-param-func"),
    setup,
    prefix="bayes-$task",
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

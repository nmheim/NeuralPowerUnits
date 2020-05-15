using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Parameters
using ValueHistories

using Flux
using LinearAlgebra
using NeuralArithmetic
using Distributions: Uniform
using UnicodePlots

include(srcdir("schedules.jl"))
include(srcdir("arithmetic_dataset.jl"))
include(srcdir("arithmetic_models.jl"))
include(srcdir("arithmetic_st_models.jl"))

struct GatedNPU
    W::AbstractMatrix
    g::AbstractVector
end

GatedNPU(in::Int, out::Int; init=Flux.glorot_uniform) =
    GatedNPU(init(out,in), Flux.ones(in)/2)

Flux.@functor GatedNPU

function mult(W::AbstractMatrix{T}, g::AbstractVector{T}, x::AbstractArray{T}) where T
    g = min.(max.(g, 0), 1)
    #g = Flux.σ.(g)
    #g = tanh.(g)
    r = abs.(x) .+ eps(T)
    r = g .* r .+ (1 .- g) .* T(1)
    k = map(i -> T(i < 0 ? pi : 0.0), x)
    z = exp.(W * log.(r)) .* cos.(W*k)
    #g .* z + (1 .- g) .* T(1)
end

(l::GatedNPU)(x) = mult(l.W, l.g, x)

@with_kw struct Config
    batch::Int      = 128
    niters::Int     = 5000
    lr::Real        = 5e-3

    lowlim::Real    = 0
    uplim::Real     = 3
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0

    inlen::Int      = 30
end


c = Config()
d = c.inlen
init(a,b) = rand(Float32,a,b)/10
init(a,b) = Flux.glorot_uniform(a,b)
model = Chain(NAU(d,d,init=init), GatedNPU(d,1,init=init))
#model = Chain(NAU(d,d,init=init), NPU(d,1,init=init))

generate = arithmetic_dataset(/, c.inlen,
    d=Uniform(c.lowlim,c.uplim),
    subset=c.subset,
    overlap=c.overlap)
test_generate = arithmetic_dataset(/, c.inlen,
    d=Uniform(c.lowlim-4,c.uplim+4),
    subset=c.subset,
    overlap=c.overlap)

ps = Flux.params(model)

function loss(x,y)
    mse = Flux.mse(model(x),y)
    l1  = norm(ps, 1)
    mse + 0.0001l1
    # st  = -logSt(ps, 0.5f0, 1f0)
    # mse + st
    #mse
end

data     = (generate(c.batch) for _ in 1:c.niters)
val_data = test_generate(1000)

opt      = RMSProp(c.lr)
(x,y) = generate(1000)
(tx,ty) = test_generate(1000)
display(model(x))
function plotting(model::Chain{<:Tuple{<:NAU,<:NPU}})
    p1 = UnicodePlots.heatmap(model[1].W[end:-1:1,:], height=d, width=d);
    p2 = UnicodePlots.heatmap(model[2].W[end:-1:1,:], height=d, width=d);
    display(p1);display(p2);
end
function plotting(model::Chain{<:Tuple{<:NAU,<:GatedNPU}})
    p1 = UnicodePlots.heatmap(model[1].W[end:-1:1,:], height=d, width=d);
    p2 = UnicodePlots.heatmap(model[2].W[end:-1:1,:], height=d, width=d);
    display(p1);display(p2);
    p3 = UnicodePlots.heatmap(model[2].g', height=d, width=d);
    display(p3);
end

callback = [Flux.throttle(()->(@info(loss(x,y), Flux.mse(model(x),y), Flux.mse(model(tx),ty));
                               plotting(model);
                              ),1)]
Flux.train!(loss, ps, data, opt, cb=callback)

m = get_mapping(res[:model])
h = res[:history]

using Plots
include(srcdir("plots.jl"))

pyplot()
p1 = plot(h)
wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.svg"), p1)

ps = map(l->Plots.heatmap(l.W[end:-1:1,:], c=:bluesreds,
                    title=summary(l), clim=(-1,1)),
         m)
p2 = plot(ps..., size=(600,300))

wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.svg"), p2)

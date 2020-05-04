using DrWatson
@quickactivate "NIP_2020_NMUX"

using Parameters
using ValueHistories
using Flux
using NeuralArithmetic
using UnicodePlots

include(srcdir("arithmetic_dataset.jl"))

@with_kw struct Config
    batch::Int      = 1000
    inlen::Int      = 10
    niters::Int     = 10000
    lr::Real        = 1e-3
    lowlim::Real    = -2
    uplim::Real     = 2
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0
end

const ComplexMatrix = AbstractArray{Complex{Float32},2}

struct NPUX
    W::ComplexMatrix
end

NPUX(in::Int, out::Int, init=Flux.glorot_uniform) = NPUX(init(out,in) .+ 0im)

(m::NPUX)(x::AbstractArray{<:Complex}) = exp.(m.W * log.(x))
(m::NPUX)(x::AbstractArray) = m(x .+ 0im)

function run(c::Config)
    generate = arithmetic_dataset(*, c.inlen,
        d=Uniform(c.lowlim,c.uplim),
        subset=c.subset,
        overlap=c.overlap)
    test_generate = arithmetic_dataset(*, c.inlen,
        d=Uniform(c.lowlim-4,c.uplim+4),
        subset=c.subset,
        overlap=c.overlap)

    model = Chain(NAU(Flux.glorot_uniform(c.inlen,c.inlen) .+ 0im), NPUX(c.inlen,1))
    model = Chain(NAU(c.inlen,c.inlen), NPU(c.inlen,1))

    function loss(x,y)
        ŷ = model(x)
        sum(abs.(ŷ .- y))
    end

    data     = (generate(c.batch) for _ in 1:c.niters)
    opt      = RMSProp(c.lr)
    ps       = params(model)

    # display(model)
    # display(loss(x,y))
    # gs = Flux.gradient(()->loss(x,y), ps)
    # display(gs[model[1].W])
    # error()
    
    (x,y) = generate(100)
    (tx,ty) = test_generate(1000)
    callbacks = [
        Flux.throttle(() -> (@info loss(x,y) loss(tx,ty)/1000),1)
    ]

    history  = Flux.train!(loss, ps, data, opt, cb=callbacks)

    return @dict(model, history)
end

config = Config()
res = run(config)

m = res[:model]
h = res[:history]

h,w = config.inlen, config.inlen
display(heatmap(real.(m[1].W[end:-1:1,:]), title=summary(m[1]), height=h, width=w))
display(heatmap(real.(m[2].W[end:-1:1,:]), title=summary(m[2]), height=h, width=w))

display(heatmap(abs.(m[1].W[end:-1:1,:]), title=summary(m[1]), height=h, width=w))
display(heatmap(abs.(m[2].W[end:-1:1,:]), title=summary(m[2]), height=h, width=w))

# using Plots
# using GMExtensions
# include(srcdir("plots.jl"))
# 
# pyplot()
# p1 = plot(h,logscale=false)
# #p1 = plothistory(h)
# net = get_mapping(m)
# ps = [Plots.heatmap(l.W[end:-1:1,:], c=:bluesreds, title=summary(l), clim=(-1,1)) for l in net]
# p2 = plot(ps..., size=(600,300))
# # display(p1)
# # display(p2)
# wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-mapping.png"), p2)
# # wsave(plotsdir(pattern, "$(basename(splitext(fname)[1]))-history.png"), p1)

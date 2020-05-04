
using DrWatson
@quickactivate "NIP_2020_NMUX"

using Parameters
using ValueHistories
using LinearAlgebra
using Flux
using NeuralArithmetic
using Plots
pyplot()
include(srcdir("arithmetic_dataset.jl"))


const ComplexMatrix = AbstractArray{Complex{Float32},2}

struct NPUX
    W::ComplexMatrix
end

NPUX(in::Int, out::Int, init=Flux.glorot_uniform) = NPUX(init(out,in) .+ 0im)

(m::NPUX)(x::AbstractArray{<:Complex}) = exp.(m.W * log.(x))
(m::NPUX)(x::AbstractArray) = m(x .+ 0im)


function loss(x,y)
    ŷ = model(x)
    sum(abs.(ŷ .- y))
end


@with_kw struct Config
    batch::Int      = 1000
    inlen::Int      = 10
    niters::Int     = 3000
    lr::Real        = 5e-3
    lowlim::Real    = -2
    uplim::Real     = 2
    subset::Real    = 0.5f0
    overlap::Real   = 0.25f0
end;


function im_norm(t::Flux.Params)
    ls = map(p -> norm(imag.(p), 1), t)
    sum(ls)
end


function run(c::Config, model)
    generate = arithmetic_dataset(*, c.inlen,
        d=Uniform(c.lowlim,c.uplim),
        subset=c.subset,
        overlap=c.overlap)
    test_generate = arithmetic_dataset(*, c.inlen,
        d=Uniform(c.lowlim-4,c.uplim+4),
        subset=c.subset,
        overlap=c.overlap)

    function loss(x,y)
        ŷ = model(x)
        #TODO: try with abs . real:
        sum(abs2, real.(ŷ .- y)) + im_norm(params(model))/10
        #sum(abs2, ŷ .- y) + im_norm(params(model))
    end

    data     = (generate(c.batch) for _ in 1:c.niters)
    opt      = RMSProp(c.lr)
    ps       = params(model)

    (x,y) = generate(100)
    (tx,ty) = test_generate(1000)
    callbacks = [
        Flux.throttle(() -> (
        train_loss = loss(x,y)/100;
        valid_loss = loss(tx,ty)/1000;
        @info("Rerun `weave` to get rid of logs", train_loss, valid_loss)), 1)
    ]

    history  = Flux.train!(loss, ps, data, opt, cb=callbacks)

    return @dict(model, history)
end;


config = Config()
nau = Flux.fmap(ComplexMatrix, NAU(config.inlen,config.inlen))
model = Chain(nau, NPUX(config.inlen,1))
(res,_) = produce_or_load(
    prefix="npux",
    datadir("jmds"),
    config,
    c -> run(c, model),
    force=true
)

m = res[:model]
h = res[:history]

h,w = config.inlen, config.inlen
display(heatmap(real.(m[1].W[end:-1:1,:]), title=summary(m[1]), c=:bluesreds, clim=(-1,1)))
display(heatmap(real.(m[2].W[end:-1:1,:]), title=summary(m[2]), c=:bluesreds, clim=(-1,1)))


config = Config()
model = Chain(NAU(config.inlen,config.inlen), NPU(config.inlen,1))
(res,_) = produce_or_load(
    prefix="npu",
    datadir("jmds"),
    config,
    c -> run(c, model)
)

m = res[:model]
h = res[:history]

h,w = config.inlen, config.inlen
display(heatmap(real.(m[1].W[end:-1:1,:]), title=summary(m[1]), c=:bluesreds, clim=(-1,1)))
display(heatmap(real.(m[2].W[end:-1:1,:]), title=summary(m[2]), c=:bluesreds, clim=(-1,1)))


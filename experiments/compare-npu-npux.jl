
using DrWatson
@quickactivate "NIP_2020_NMUX"

using Parameters
using ValueHistories
using LinearAlgebra
using Flux
using NeuralArithmetic
using UnicodePlots

include(srcdir("arithmetic_dataset.jl"))
include(srcdir("arithmetic_st_models.jl"))


const ComplexMatrix = AbstractArray{Complex{Float32},2}

struct NPU_X1
    W::Matrix
end

NPU_X1(in::Int, out::Int, init=Flux.glorot_uniform) = NPU_X1(init(out,in))
(m::NPU_X1)(x::AbstractArray) = exp.(m.W * real.(log.(x .+ 0im)))
Flux.@functor NPU_X1



struct NPU_X2
    W::Matrix
    b::Vector{<:Complex}
end

NPU_X2(in::Int, out::Int, init=Flux.glorot_uniform) = NPU_X2(init(out,in), zeros(ComplexF32,in))
(m::NPU_X2)(x::AbstractArray) = exp.(m.W * real.(log.(x .+ m.b)))
Flux.@functor NPU_X2


struct NPUX
    W::ComplexMatrix
end

NPUX(in::Int, out::Int, init=Flux.glorot_uniform) = NPUX(init(out,in) .+ 0im)

(m::NPUX)(x::AbstractArray{<:Complex}) = exp.(m.W * log.(x))
(m::NPUX)(x::AbstractArray) = m(x .+ 0im)


@with_kw struct Config
    batch::Int      = 1000
    inlen::Int      = 10
    niters::Int     = 100000
    lr::Real        = 5e-3
    lowlim::Real    = 1
    uplim::Real     = 3
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
        d = ŷ .- y

        mse = sum(abs2, real.(d))
        st  = sum(logSt(params(model), 0.5, 2f0))
        mse - st
        # l1i = 0sum(abs, imag.(d))
        # l1p = 100im_norm(params(model))
        # mse + l1i + l1p
        #+ norm(imag.(d), 1)# + im_norm(params(model))
        #sum(abs2, ŷ .- y) + im_norm(params(model))
        # sum(abs2, real.(d))
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
        display(heatmap(real.(model[1].W[end:-1:1,:]), title="real"));
        display(heatmap(real.(model[2].W[end:-1:1,:]), height=config.inlen, width=config.inlen));
        #display(heatmap(reshape(real.(model[2].b),1,:), height=config.inlen, width=config.inlen));
        #display(heatmap(imag.(model[1].W[end:-1:1,:]), title="imag"));
        @info("Rerun `weave` to get rid of logs", train_loss, valid_loss)), 1)
    ]

    history  = Flux.train!(loss, ps, data, opt, cb=callbacks)

    return @dict(model, history)
end;


config = Config()
#nau = Flux.fmap(ComplexMatrix, NAU(config.inlen,config.inlen))
# nau = NAU(Flux.glorot_uniform(config.inlen,config.inlen) +
#           Flux.glorot_uniform(config.inlen,config.inlen)*1im)
nau = NAU(config.inlen,config.inlen)
npu = NPU(config.inlen,1)
model = Chain(nau, npu)
(res,_) = produce_or_load(
    prefix="npux",
    datadir("jmds"),
    config,
    c -> run(c, model),
    force=true
)

m = res[:model]
h = res[:history]

# h,w = config.inlen, config.inlen
# p1 = heatmap(real.(m[1].W[end:-1:1,:]), title=summary(m[1]), c=:bluesreds, clim=(-1,1))
# p2 = heatmap(real.(m[2].W[end:-1:1,:]), title=summary(m[2]), c=:bluesreds, clim=(-1,1))
# display(plot(p1,p2,title="real values", size=(600,300)))
# p1 = heatmap(imag.(m[1].W[end:-1:1,:]), title=summary(m[1]), c=:bluesreds, clim=(-1,1))
# p2 = heatmap(imag.(m[2].W[end:-1:1,:]), title=summary(m[2]), c=:bluesreds, clim=(-1,1))
# display(plot(p1,p2,title="imag values", size=(600,300)))
# 
# 
# config = Config()
# model = Chain(NAU(config.inlen,config.inlen), NPU(config.inlen,1))
# (res,_) = produce_or_load(
#     prefix="npu",
#     datadir("jmds"),
#     config,
#     c -> run(c, model)
# )
# 
# m = res[:model]
# h = res[:history]
# 
# h,w = config.inlen, config.inlen
# display(heatmap(real.(m[1].W[end:-1:1,:]), title=summary(m[1]), c=:bluesreds, clim=(-1,1)))
# display(heatmap(real.(m[2].W[end:-1:1,:]), title=summary(m[2]), c=:bluesreds, clim=(-1,1)))


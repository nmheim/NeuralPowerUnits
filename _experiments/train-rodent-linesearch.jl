using DrWatson
@quickactivate "NIPS_2020_NMUX"

using ValueHistories
using Parameters

using Plots
using StatsPlots
using LaTeXStrings

using Flux
using OrdinaryDiffEq
using Random: randn!

using Revise
using ConditionalDists
using IPMeasures
using GenerativeModels
using GMExtensions
using NeuralArithmetic

includet(srcdir("harmonic.jl"))
includet(srcdir("construct.jl"))
unicodeplots()

force_run = true

batchsize = 10
mintlen   = 5
T         = Float32
dt        = T(0.1)
slen      = 2
tlen      = 30
dt        = 0.2f0
noise     = 0.02f0
initf     = (s...) -> Flux.glorot_uniform(s...)
#initf = (s...) -> rand(T,s...)/100

generate(ω,tlen) = generate_harmonic(ω, batchsize; ω0=0.5, noise=noise,
                                     dt=dt, steps=tlen)[1]

save_name = datadir("train-rodent-vanderpol.bson")

# training curriculum
curriculum = [
    Dict(:ω=>0.6, :niter=>1000,  :lr=>0.001),
    Dict(:ω=>0.8, :niter=>1500,  :lr=>0.001),
    Dict(:ω=>1.0, :niter=>1500,  :lr=>0.001),
    Dict(:ω=>2.0, :niter=>1500,  :lr=>0.001),
    Dict(:ω=>3.0, :niter=>1500,  :lr=>0.001),
]

ode = Chain(
    ReNMUX(slen, slen),
    NAU(slen, slen),
   )

zlen = length(Flux.destructure(ode)[1]) + slen
enc = conv_encoder(1, zlen, mintlen,
                   init_conv=initf, init_dense=initf)
H(sol) = reshape(hcat(sol.u...)[1:1,:], :)
rec(x) = mean(model.decoder, mean(model.encoder, x))

loss(x) = -elbo(model, x)
model = Rodent(slen, tlen, dt, enc, ode=ode, observe=H, olen=tlen)
history = MVHistory()

for c in curriculum
    @unpack ω, niter, lr = c
    train_data = [generate(ω,tlen) for _ in 1:niter]
    opt = Descent(lr)

    for d in train_data
        gs = Flux.gradient(ps) do
            train_loss = loss(d...)
            @info ii train_loss
            train_loss
        end

        update!(opt, ps, gs)
    end
end

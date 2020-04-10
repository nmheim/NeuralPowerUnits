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

function GenerativeModels.elbo(m::Rodent, x::AbstractArray{T,3}; β=1) where T
    xf = reshape(x, :, size(x,3))

    μz = mean(m.encoder, x)
    σ2z = var(m.encoder, xf)
    rz = randn!(similar(μz))
    z = μz .+ sqrt.(σ2z) .* rz
    #display(z[1:4,:])

    llh = sum(logpdf(m.decoder, xf, z))
    kld = sum(IPMeasures._kld_gaussian(μz,σ2z,mean_var(m.prior)...))
    lpλ = sum(logpdf(m.hyperprior, var(m.prior)))

    llh - β*(kld - lpλ)
end

loss(x) = -elbo(model, x)
model = Rodent(slen, tlen, dt, enc, ode=ode, observe=H, olen=tlen)
history = MVHistory()

curr_iter = zeros(Int)

if isfile(save_name) && !force_run
    model, history = load_checkpoint(save_name)
else
    for c in curriculum
        global history
        @unpack niter, lr, ω = c
    
        train_data = [(generate(ω,tlen),) for _ in 1:niter]
        opt = Descent(lr)
    
        cb = [
            () -> (curr_iter[1] += 1),
            Flux.throttle(() -> (x = train_data[1][1][1,:,1];
                r  = reshape(x, 1, tlen);
                l  = loss(train_data[1][1]);
                p1 = plot(r', title="$curr_iter Loss: $l");
                z  = mean(model.encoder,train_data[1][1]);
                p1 = plot!(p1, reshape(mean(model.decoder,z[:,1]), 1, tlen)');
                p2 = latentboxplot(z,1:zlen);
                display(plot(p1,p2))), 1),
            #Flux.throttle(mvhistory_callback(history, model, loss, train_data[1]...), 0.2),
            Flux.throttle(()->(save_checkpoint(save_name, model, history)), 20),
        ]
    
        # because we are varying tlen ...
        global model = reconstruct(model, tlen, dt, tlen)
        ps = params(model)
    
        Flux.train!(loss, ps, train_data, opt, cb=cb)
        save_checkpoint(save_name, model, history)
    end
end

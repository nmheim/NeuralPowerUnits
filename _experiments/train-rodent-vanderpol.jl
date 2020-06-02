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

includet(srcdir("vanderpol.jl"))
includet(srcdir("construct.jl"))
unicodeplots()

force_run = true

batchsize = 10
mintlen   = 5
slen      = 2
T         = Float32
dt        = T(0.1)
#initf     = (s...) -> randn(T, s...)
initf     = Flux.glorot_uniform

save_name = datadir("train-rodent-vanderpol.bson")

# training curriculum
curriculum = [
    Dict(:tlen=>40, :niter=>2000, :lr=> 0.001),
    Dict(:tlen=>40, :niter=>200, :lr=> 0.001),
    Dict(:tlen=>40, :niter=>200, :lr=> 0.001),
]

ode = Chain(
    ReNMUX(slen,   2*slen),
    NAU(2*slen, 2*slen),

    ReNMUX(2*slen, 2*slen),
    NAU(2*slen, 2*slen),

    ReNMUX(2*slen, 2*slen),
    NAU(2*slen, slen),
   )

zlen = length(Flux.destructure(ode)[1]) + slen
tlen = curriculum[1][:tlen]
enc = conv_encoder(slen, zlen, mintlen,
                   init_conv=initf, init_dense=initf)
H(sol) = reshape(hcat(sol.u...)[1:end,:], :)
rec(x) = mean(model.decoder, mean(model.encoder, x))

function GenerativeModels.elbo(m::Rodent, x::AbstractArray{T,3}; β=1) where T
    xf = reshape(x, :, size(x,3))

    μz = mean(m.encoder, x)
    σ2z = var(m.encoder, xf)
    rz = randn!(similar(μz))
    z = μz .+ sqrt.(σ2z) .* rz

    llh = sum(logpdf(m.decoder, xf, z))
    kld = sum(IPMeasures._kld_gaussian(μz,σ2z,mean_var(m.prior)...))
    lpλ = sum(logpdf(m.hyperprior, var(m.prior)))

    llh - β*(kld - lpλ)
end

loss(x) = -elbo(model, x)
model = Rodent(slen, tlen, dt, enc, ode=ode, observe=H, olen=2*tlen)
history = MVHistory()

data = vanderpol_dataset()[:u]
# TODO: find a better solution for reshaping to vec because of convolutions in encoder #
# TODO: add gamma prior to rodent #
# TODO: get rid of callbacks in train loop #
# TODO: are the logpdfs correctly implemented? see nc deck.
# TODO: fix double computation of mean_var in kld/sampling
# TODO: create conditional dists issue to remind about DistributionsAD rewrite.
# think about how to handle batches. it would be nice to return vectors when
# vectors are passed in and matrices when matrices are passed in... #
function vanderpol_sample(batchsize::Int, tlen::Int)
    start = rand(1:(size(data,2)-tlen), batchsize)    
    batch = map((s)->data[:,s:s+tlen-1], start)
    cat(batch..., dims=3)
end

curr_iter = zeros(Int)

if isfile(save_name) && !force_run
    model, history = load_checkpoint(save_name)
else
    for c in curriculum
        global history
        @unpack tlen, niter, lr = c
    
        train_data = [(vanderpol_sample(batchsize,tlen),) for _ in 1:niter]
        opt = RMSProp(lr)
    
        cb = [
            () -> (curr_iter[1] += 1),
            Flux.throttle(() -> (x = train_data[1][1][:,:,1];
                r  = reshape(x, 2, tlen);
                l  = loss(train_data[1][1]);
                p1 = plot(r', title="$curr_iter Loss: $l");
                z  = mean(model.encoder,train_data[1][1]);
                p1 = plot!(p1, reshape(mean(model.decoder,z[:,1]), 2, tlen)');
                p2 = latentboxplot(z,1:zlen);
                display(plot(p1,p2))), 1),
            #Flux.throttle(mvhistory_callback(history, model, loss, train_data[1]...), 0.2),
            Flux.throttle(()->(save_checkpoint(save_name, model, history)), 20),
        ]
    
        # because we are varying tlen ...
        global model = reconstruct(model, tlen, dt, tlen*2)
        ps = params(model)
    
        Flux.train!(loss, ps, train_data, opt, cb=cb)
        save_checkpoint(save_name, model, history)
    end
end

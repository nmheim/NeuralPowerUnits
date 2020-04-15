using DrWatson
@quickactivate "NIPS_2020_NMUX"

using ValueHistories
using Parameters
using Suppressor

using Plots
using StatsPlots
using LaTeXStrings

using LinearAlgebra
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
include(srcdir("construct.jl"))
unicodeplots()

force_run = true

batchsize = 20
mintlen   = 5
T         = Float32
dt        = T(0.1)
slen      = 2
tlen      = 30
dt        = 0.2f0
noise     = 0.02f0

#initnpu = (s...) -> init_diag(T,s...)
initnpu = (s...) -> rand(T,s...)

initnau = (s...) -> randn(T,s...) / 10
#initnau = (s...) -> rand(T,s...)
#initnau = Flux.glorot_uniform

initenc = Flux.glorot_uniform
#initenc = (s...) -> randn(T,s...)/10

ode = Chain(
    NPU(slen, slen, init=initnpu),
    NAU(slen, slen, init=initnau))

#initode = (s...) -> rand(T,s...) / 10
#initode = (s...) -> init_diag(T,s...)

ω = 1.0
niter = 10000
lr = 0.01
opt = Descent(lr)




generate(ω,tlen) = generate_harmonic_fullstate(ω, batchsize; ω0=ω, noise=noise,
                                     dt=dt, steps=tlen)[1]

function shared_encoder(slen, zlen, mintlen)
    odenet(x) = ones(T,1,size(x,3)) .* Flux.destructure(ode)[1]

    act = tanh
    conv_zlen = zlen-slen
    dense_zlen = slen

    #densenet = Chain(
    #    x -> reshape(x[:,1:mintlen,:], :, size(x,3)),
    #    Dense(mintlen, 20, act, initW=initenc, initb=initenc),
    #    Dense(20, dense_zlen, initW=initenc, initb=initenc))
    densenet = x -> x[:,1,:]
    CatLayer(odenet, densenet)
end

function rodent(slen::Int, tlen::Int, dt::T, encoder;
                ode=Dense(slen,slen),
                observe=sol->reshape(hcat(sol.u...), :),
                olen=slen*tlen) where T
    zlen = length(Flux.destructure(ode)[1]) + slen

    # hyperprior
    hyperprior = InverseGamma(T(1), T(1), zlen, true)

    # prior
    μpz = NoGradArray(zeros(T, zlen))
    λ2z = ones(T, zlen) #/ 10
    prior = Gaussian(μpz, λ2z)

    # encoder
    σ2z = ones(T, zlen) #/ 10
    enc_dist = CMeanGaussian{DiagVar}(encoder, σ2z)

    # decoder
    σ2x = ones(T, 1)
    decoder = FluxODEDecoder(slen, tlen, dt, ode, observe)
    dec_dist = CMeanGaussian{ScalarVar}(decoder, σ2x, olen)

    Rodent(hyperprior, prior, enc_dist, dec_dist)
end


zlen = length(Flux.destructure(ode)[1]) + slen
enc = shared_encoder(slen, zlen, mintlen)
H(sol) = reshape(hcat(sol.u...), :)
#H(sol) = hcat(sol.u...)

model = rodent(slen, tlen, dt, enc, ode=ode, observe=H, olen=tlen)
ps = params(model.encoder.mapping, ode)
rec(x) = reshape(mean(model.decoder, mean(model.encoder, x)), 2,tlen,batchsize)
loss(x) = Flux.mse(x, rec(x)) #+ norm(params(ode),1)
history = MVHistory()

train_data = [(generate(ω,tlen),) for _ in 1:niter]

x = generate(0.5,tlen)
#Flux.train!(loss, ps, train_data, opt, cb=cb)


function train!(loss, ode, model, data, opt, history=MVHistory())
    train_loss = 0f0
    ps = params(ode)

    plotprogress = Flux.throttle(() -> (x = train_data[1][1][:,:,1];
        r  = reshape(x, 2, tlen);
        l  = loss(train_data[1][1]);
        p1 = plot(r', title="Step $i: $lr $train_loss");
        z  = mean(model.encoder,train_data[1][1]);
        p1 = plot!(p1, reshape(mean(model.decoder,z[:,1]), 2, tlen)');
        p2 = latentboxplot(z,1:zlen);
        display(plot(p1,p2))), 1)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1

    try 
        for d in data
            gs = gradient(ps) do
                train_loss = loss(d...)
                return train_loss
            end
            plotprogress()
            Flux.Optimise.update!(opt, ps, gs)
            i += 1
        end
    catch e
        println("Error during training!")
        for (exc,bt) in Base.catch_stack()
            showerror(stdout,exc,bt)
            println()
        end
    end

    history
end


function linesearch!(loss, ode, model, data, opt, history=MVHistory())
    train_loss = 0f0
    ps = params(ode)
    prev_ps = params(f32(ode))
    lr = copy(opt.eta)

    plotprogress = Flux.throttle(() -> (x = train_data[1][1][:,:,1];
        r  = reshape(x, 2, tlen);
        l  = loss(train_data[1][1]);
        p1 = plot(r', title="Step $i: $lr $train_loss");
        z  = mean(model.encoder,train_data[1][1]);
        p1 = plot!(p1, reshape(mean(model.decoder,z[:,1]), 2, tlen)');
        p2 = latentboxplot(z,1:zlen);
        display(plot(p1,p2))), 1)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1

    try 
        for d in data
            success = false
            gs = nothing
            while !success
                try
                    #println("$i reconstruct: $opt")
                    gs = gradient(()->loss(d...), ps)
                    success = true
                    opt = reconstruct(opt, eta=lr)
                    prev_ps = params(f32(ode))
                    #println(ps)
                    #println(prev_ps)
                catch e
                    if i == 1
                        throw(ValueError("First iteration has to work!"))
                    end
                    println(ps)
                    println(prev_ps)
                    ps = prev_ps
                    opt = reconstruct(opt, eta=opt.eta/2)
                    println("reduce: $opt")
                    error()
                end
                
                # warnings = @capture_err begin 
                #    gs = gradient(()->loss(d...), ps)
                # end
                # println(warnings)
                # warnings = ""
                # if isempty(warnings)
                #     success = true
                #     opt = reconstruct(opt, eta=lr)
                #     prev_ps = params(f32(ode))
                #     println("reconstruct: $opt")
                # else
                #     if i == 1
                #         throw(ValueError("First iteration has to work!"))
                #     end
                #     ps = prev_ps
                #     opt = reconstruct(opt, eta=opt.eta/2)
                #     println("reduce: $opt")
                # end
            end
            plotprogress()
            Flux.Optimise.update!(opt, ps, gs)
            i += 1
        end
    catch e
        println("Error during training!")
        for (exc,bt) in Base.catch_stack()
            showerror(stdout,exc,bt)
            println()
        end
    end
end


#train!(loss, ode, model, train_data, opt)
linesearch!(loss, ode, model, train_data, opt)

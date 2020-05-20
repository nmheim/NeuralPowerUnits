using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using Parameters
using LinearAlgebra
using NeuralArithmetic
using ValueHistories
using Plots
unicodeplots()

_sine_taylor(x,n) = (-1)^n * x^(2n+1) / factorial(2n+1)
_sum_taylor(f,x,n;n0=0) = sum(map(i->f(x,i),n0:n))
sine_taylor(x,n) = _sum_taylor(_sine_taylor,x,n)

_exp_taylor(x,n) = x^n/factorial(n)
exp_taylor(x,n) = _sum_taylor(_exp_taylor,x,n)

_log_taylor(x,n) = (-1)^(n+1) * x^n/n
log_taylor(x,n) = _sum_taylor(_log_taylor,x,n,n0=1)



function run(c::Dict, f::Function)
    #@unpack dim, lowlim, uplim, niter, lr = c
    @unpack dim, lowlim, uplim, niter, lr = c
    x = Float32.(reshape(lowlim:0.1:uplim, 1, :))
    y = f.(x)
    xt = Float32.(reshape((lowlim*2):0.2:(uplim*1.5),1,:))
    yt = f.(xt)

    h = MVHistory()
    data = Iterators.repeated((x,y), niter)
    opt = ADAM(lr)
    model = Chain(NMU(1,dim), NAU(dim,1))
    ps = params(model)
    mse(x,y) = sum(abs2, model(x) .- y)
    loss(x,y) = mse(x,y)

    cb = [Flux.throttle(()->(
               # p1 = plot(vec(xt), vec(yt), yscale=:log10);
               # plot!(p1, vec(xt), abs.(vec(model(xt))));
               p1 = plot(vec(xt), vec(yt));
               plot!(p1, vec(xt), vec(model(xt)));
               display(p1);
               @info loss(x,y) loss(xt,yt)
              ), 1),
          Flux.throttle(() -> (push!(h, :μz, Flux.destructure(model)[1]);
                               push!(h, :mse, mse(xt,yt))), 0.1)
         ]
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c, :history=>h)
end

expres, expfile = produce_or_load(datadir("exp_log_sin"),
                         Dict(:lowlim=>-5, :uplim=>5, :dim=>10, :niter=>1000, :lr=>1e-4),
                         c -> run(c, exp),
                         prefix="nmu_exp",
                         force=false, digits=8)

logres, logfile = produce_or_load(datadir("exp_log_sin"),
                         Dict(:lowlim=>0.1, :uplim=>2, :dim=>10, :niter=>1000, :lr=>1e-3),
                         c -> run(c, log),
                         prefix="nmu_log",
                         force=false)

sinres, sinfile = produce_or_load(datadir("exp_log_sin"),
                         Dict(:lowlim=>-5, :uplim=>5, :dim=>4, :niter=>10000, :lr=>1e-3),
                         c -> run(c, sin),
                         prefix="nmu_sin",
                         force=false)

pgfplotsx()
expmodel = expres[:model]
c = expres[:config]
@unpack dim, lowlim, uplim, niter = c

xt = Float32.(collect((lowlim*2):0.5:(uplim*2)))
yt = exp.(xt)

p1 = plot(xt, yt, label="Exp", lw=2, ls=:dash, yscale=:log10, ylim=[1e-4,1e4])
plot!(p1, xt, abs.(vec(expmodel(reshape(xt,1,:)))),
      lw=2, label="NMU", size=(300,200), legend=:bottomright)
plot!(p1, [lowlim, lowlim], [1e-5, 1e5], c=:gray, label=false)
plot!(p1, [uplim, uplim], [1e-5, 1e5], c=:gray, label="Train range")
savefig(p1, plotsdir("exp_log_sin", "nmu_exp.tikz"))
display(p1)

h = expres[:history]
z = reduce(hcat, get(h, :μz)[2])
p2 = plot(z')
display(p2)


logmodel = logres[:model]
c = logres[:config]
@unpack dim, lowlim, uplim, niter = c
xt = Float32.(collect((lowlim*2):0.1:(uplim*4)))
yt = log.(xt)

p2 = plot(xt, yt, label="Log", lw=2, ls=:dash)
plot!(p2, xt, vec(logmodel(reshape(xt,1,:))), lw=2, label="NMU", size=(300,200))
vline!(p2, [lowlim, uplim], lw=2, c=:gray, label="Train range")
savefig(p2, plotsdir("exp_log_sin", "nmu_log.tikz"))
display(p2)

sinmodel = sinres[:model]
c = sinres[:config]
@unpack dim, lowlim, uplim, niter = c
xt = Float32.(collect((lowlim*2):0.1:(uplim*2)))
yt = sin.(xt)

p3 = plot(xt, yt, label="Sin", lw=2, ls=:dash)
plot!(p3, xt, vec(sinmodel(reshape(xt,1,:))), lw=2, label="NMU", size=(300,200))
plot!(p3, xt, sine_taylor.(xt,3), ylim=(-5,5))
plot!(p3, xt, sine_taylor.(xt,4), ylim=(-5,5))
vline!(p3, [lowlim, uplim], lw=2, c=:gray, label="Train range")
savefig(p3, plotsdir("exp_log_sin", "nmu_sin.tikz"))
display(p3)

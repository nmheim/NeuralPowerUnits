using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using Parameters
using LinearAlgebra
using NeuralArithmetic
using Plots
pyplot()


function run(c::Dict, f::Function)
    @unpack dim, lowlim, uplim, niter = c
    x = Float32.(reshape(lowlim:0.1:uplim, 1, :))
    y = f.(x)

    data = Iterators.repeated((x,y), niter)
    opt = ADAM(1e-3)
    model = Chain(GatedNPUX(1,dim), NAU(dim,1))
    ps = params(model)
    loss(x,y) = sum(abs2, model(x) .- y) #+ βl1*norm(model[1].Im)

    cb = [Flux.throttle(()->(
               xt = Float32.(reshape((lowlim*2):0.5:(uplim*1.5),1,:));
               yt = f.(xt);
               p1 = plot(vec(xt), vec(yt), yscale=:log10);
               plot!(p1, vec(xt), abs.(vec(model(xt))));
               display(p1);
               @info loss(x,y) loss(xt,yt)
              ), 1)
         ]
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c)
end

expres, _ = produce_or_load(datadir("exp_log_sin"),
                         Dict(:lowlim=>-5, :uplim=>5, :dim=>10, :niter=>50000),
                         c -> run(c, exp),
                         prefix="gatednpux_exp",
                         force=false)

logres, _ = produce_or_load(datadir("exp_log_sin"),
                         Dict(:lowlim=>0.1, :uplim=>2, :dim=>10, :niter=>50000),
                         c -> run(c, log),
                         prefix="gatednpux_log",
                         force=false)

pgfplotsx()
expmodel = expres[:model]
c = expres[:config]
@unpack dim, lowlim, uplim, niter = c

xt = Float32.(collect((lowlim*2):0.5:(uplim*2)))
yt = exp.(xt)

p1 = plot(xt, yt, label="Exp", yscale=:log10, lw=2, ls=:dash)
plot!(p1, xt, abs.(vec(expmodel(reshape(xt,1,:)))), lw=2, label="GatedNPUX", size=(300,200))
vline!(p1, [lowlim, uplim], lw=2, c=:gray, label="Train range")
savefig(p1, plotsdir("exp_log_sin", "gatednpux_exp.tikz"))
display(p1)


logmodel = logres[:model]
c = logres[:config]
@unpack dim, lowlim, uplim, niter = c
xt = Float32.(collect((lowlim*2):0.1:(uplim*4)))
yt = log.(xt)

p2 = plot(xt, yt, label="Log", lw=2, ls=:dash)
plot!(p2, xt, vec(logmodel(reshape(xt,1,:))), lw=2, label="GatedNPUX", size=(300,200))
vline!(p2, [lowlim, uplim], lw=2, c=:gray, label="Train range")
savefig(p2, plotsdir("exp_log_sin", "gatednpux_log.tikz"))
display(p2)

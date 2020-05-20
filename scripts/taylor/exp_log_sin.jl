using DrWatson
@quickactivate "NIPS_2020_NPUX"

using Flux
using Parameters
using NeuralArithmetic
using ValueHistories
using Plots

expconfig = Dict(:lowlim=>-5, :uplim=>5, :dim=>10, :niter=>100000, :lr=>1e-3, :βl1=>0.1)
logconfig = Dict(:lowlim=>0.1, :uplim=>2, :dim=>10, :niter=>50000, :lr=>1e-3, :βl1=>0.01)
sinconfig = Dict(:lowlim=>-5, :uplim=>5, :dim=>4, :niter=>50000, :lr=>1e-3, :βl1=>0.1)

function loadmodels(prefixes::Vector, config::Dict, problem::String)
    map(prefixes) do prefix
        igs = occursin("npux", prefix) ? [] : [:βl1]
        name = savename(config, ignores=igs, digits=9)
        res = load(datadir("exp_log_sin", "$(prefix)_$(problem)_$(name).bson"))
        res[:model]
    end
end

expmodels = loadmodels(["npu","npux","gatednpux","dense","nalu"], expconfig, "exp")
logmodels = loadmodels(["npu","npux","gatednpux","dense","nalu"], logconfig, "log")
sinmodels = loadmodels(["npu","npux","gatednpux","dense","nalu"], sinconfig, "sin")

name(l) = summary(l)
name(l::GatedNPUX) = "GatedNPUX"
name(l::Dense) = "Dense"
name(m::Chain) = name(m[1])

pgfplotsx()
linewidth = 1.4
linestyles = Iterators.cycle([:dash, :dashdot, :solid, :dot])

@unpack lowlim, uplim, niter = expconfig
xt = Float32.(collect((lowlim*2):0.5:(uplim*2)))
yt = exp.(xt)

p1 = plot(xt, yt, label="Exp", lw=linewidth, ls=:dash, yscale=:log10, ylim=[1e-4,1e4])
for (m, ls) in zip(expmodels, linestyles)
    plot!(p1, xt, abs.(vec(m(reshape(xt,1,:)))),
          lw=linewidth, label=name(m), size=(300,200), legend=false, ls=ls)
end
plot!(p1, [lowlim, lowlim], [1e-5, 1e5], c=:gray, label=false)
plot!(p1, [uplim, uplim], [1e-5, 1e5], c=:gray, label="Train range")
savefig(p1, plotsdir("exp_log_sin", "exp_comp.tikz"))
display(p1)


@unpack lowlim, uplim, niter = logconfig
xt = Float32.(collect((lowlim*2):0.1:(uplim*4)))
yt = log.(xt)

p2 = plot(xt, yt, label="Log", lw=linewidth, ls=:dash, legend=:bottomright)
for (m, ls) in zip(logmodels, linestyles)
    plot!(p2, xt, vec(m(reshape(xt,1,:))),
          lw=linewidth, label=name(m), size=(300,200),
          ls=ls)
end
vline!(p2, [lowlim, uplim], lw=linewidth, c=:gray, label="Train range")
savefig(p2, plotsdir("exp_log_sin", "log_comp.tikz"))
display(p2)

@unpack lowlim, uplim, niter = sinconfig
xt = Float32.(collect((lowlim*2):0.1:(uplim*2)))
yt = sin.(xt)

p3 = plot(xt, yt, label="Sin", lw=linewidth, ls=:dash, ylim=(-2,2), legend=false)
for (m, ls) in zip(sinmodels, linestyles)
    plot!(p3, xt, vec(m(reshape(xt,1,:))),
          lw=linewidth, label=name(m), size=(300,200), ls=ls)
end
vline!(p3, [lowlim, uplim], lw=linewidth, c=:gray, label="Train range")
savefig(p3, plotsdir("exp_log_sin", "sin_comp.tikz"))
display(p3)

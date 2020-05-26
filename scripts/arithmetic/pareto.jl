using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using NeuralArithmetic
using ValueHistories
using DataFrames
using Sobol

include(joinpath(@__DIR__, "sobolconfigs.jl"))
include(joinpath(@__DIR__, "collect.jl"))
include(srcdir("arithmetic_dataset.jl"))

function sobol_samples(c)
    s = SobolSeq(c.inlen)
    # discard first zero sample
    next!(s)
    x = reduce(hcat, [next!(s) for i = 1:10000])
    xs = c.uplim * 2
    xe = c.lowlim * 2
    Float32.(x .* (xs - xe) .+ xe)
end

nrparams(x::Array, thresh) = sum(abs.(x) .> thresh)
nrparams(m::NAU, thresh) = nrparams(m.W, thresh)
nrparams(m::NMU, thresh) = nrparams(m.W, thresh)
nrparams(m::NPUX, thresh) = sum(map(x->nrparams(x,thresh), [m.Re,m.Im]))
nrparams(m::GatedNPUX, thresh) = sum(map(x->nrparams(x,thresh), [m.Re,m.Im,m.g]))
nrparams(m::GatedNPU, thresh) = sum(map(x->nrparams(x,thresh), [m.W,m.g]))
nrparams(m::NAC, thresh) = sum(map(x->nrparams(x,thresh), [m.W,m.M]))
nrparams(m::NALU, thresh) = sum(map(x->nrparams(x,thresh), [m.nac,m.G,m.b]))
nrparams(m::Chain, thres) = sum(map(x->nrparams(x,thres), m))

task(x::Array,c::SqrtL1SearchConfig) = sqrt(x,c.subset)
task(x::Array,c::DivL1SearchConfig) = invx(x,c.subset)
task(x::Array,c::AddL1SearchConfig) = add(x,c.subset,c.overlap)
task(x::Array,c::MultL1SearchConfig) = mult(x,c.subset,c.overlap)

res = load(datadir("pareto","thresh=1e-5.bson"))
df = res[:df]

using Plots
pyplot()
ps = []
for dft in groupby(df, "task")
    s1 = plot(title=dft.task[1])
    for dfm in groupby(dft,"model")
        #if dfm.model[1] != "gatednpu"
            scatter!(s1, log10.(dfm.reg), log10.(dfm.mse),
            #scatter!(s1, dfm.val, dfm.reg,
                     ylabel="log(mse)", xlabel="log(nr params)",
                     ms=5, label=dfm.model[1], alpha=0.5, ylim=-(-2,10))
        #end
    end
    push!(ps, s1)
end
plot(ps...)

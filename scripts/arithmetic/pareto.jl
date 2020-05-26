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

function pareto(d::Dict)
    @unpack thresh = d
    df = collect_all_results!(["add_l1_runs",
                               "mult_l1_runs",
                               "invx_l1_runs",
                               "sqrt_l1_runs"])
    @progress for row in eachrow(df)
        m = load(row.path)[:model]
        x = sobol_samples(row.config)
        y = task(x,row.config)
        row.val = Flux.mse(m(x),y)
        row.reg = nrparams(m, thresh)
    end
    return @dict(df)
end

(res,fname) = produce_or_load(datadir("pareto"),
                          Dict(:thresh=>1e-5),
                          pareto,
                          digits=10,
                          force=false)
df = combine(groupby(res[:df], ["model","task"])) do gdf
    gdf[1:min(10,size(gdf,1)),:]
end

using Plots
using LaTeXStrings
pgfplotsx()

models = Dict("npux"=>"NPU",
              "gatednpux"=>"GatedNPU",
              "nalu"=>"NALU",
              "nmu"=>"NMU")
ms     = 4
alpha  = 0.7
xscale = :log10
yscale = :log10
plotmodels = ["gatednpux","nalu","nmu","npux"]
#plotmodels = ["gatednpux","nalu","nmu"]

s1 = plot(title="Addition +")
pdf  = filter(row->row.task=="add", df)
for m in plotmodels
    mdf = filter(row->row.model==m, pdf)
    scatter!(s1, mdf.reg, mdf.val,
             label=models[m], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             ylabel="Validation MSE", legend=false)
end

s2 = plot(title="Multiplication \$\\times\$")
pdf  = filter(row->row.task=="mult", df)
for m in plotmodels
    mdf = filter(row->row.model==m, pdf)
    scatter!(s2, mdf.reg, mdf.val,
             label=models[m], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms)
end

s3 = plot(title="Division \$\\div\$")
pdf  = filter(row->row.task=="invx", df)
for m in plotmodels
    mdf = filter(row->row.model==m, pdf)
    scatter!(s3, mdf.reg, mdf.val,
             label=models[m], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             ylabel="Validation MSE",
             xlabel="Nr. Parameters", legend=false)
end

s4 = plot(title="Square root \$\\sqrt\\cdot\$")
pdf  = filter(row->row.task=="sqrt", df)
for m in plotmodels
    mdf = filter(row->row.model==m, pdf)
    scatter!(s4, mdf.reg, mdf.val,
             label=models[m], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             xlabel="Nr. Parameters", legend=false)
end

p1 = plot(s1,s2,s3,s4,layout=(2,2))

s1 = plot(title="Addition +")
pdf  = filter(row->row.task=="add", df)
for m in plotmodels
    mdf = filter(row->row.model==m, pdf)
    scatter!(s1, mdf.reg, mdf.mse,
             label=models[m], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             ylabel="Traingin MSE", legend=false)
end

s2 = plot(title="Multiplication \$\\times\$")
pdf  = filter(row->row.task=="mult", df)
for m in plotmodels
    mdf = filter(row->row.model==m, pdf)
    scatter!(s2, mdf.reg, mdf.mse,
             label=models[m], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms)
end

s3 = plot(title="Division \$\\div\$")
pdf  = filter(row->row.task=="invx", df)
for m in plotmodels
    mdf = filter(row->row.model==m, pdf)
    scatter!(s3, mdf.reg, mdf.mse,
             label=models[m], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             ylabel="Traingin MSE",
             xlabel="Nr. Parameters", legend=false)
end

s4 = plot(title="Square root \$\\sqrt\\cdot\$")
pdf  = filter(row->row.task=="sqrt", df)
for m in plotmodels
    mdf = filter(row->row.model==m, pdf)
    scatter!(s4, mdf.reg, mdf.mse,
             label=models[m], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             xlabel="Nr. Parameters", legend=false, ylim=(1e-3,1e4))
end

p2 = plot(s1,s2,s3,s4,layout=(2,2))

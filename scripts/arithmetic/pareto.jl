using DrWatson
@quickactivate "NIPS_2020_NPU"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Plots
using LaTeXStrings
using DataFrames
using Flux
using NeuralArithmetic
pgfplotsx()
#pyplot()

include(joinpath(@__DIR__, "sobolconfigs.jl"))
include(srcdir("arithmetic_dataset.jl"))

#res = load(datadir("pareto","thresh=1e-5.bson"))
res = load("/home/niklas/repos/npu/data/pareto/thresh=1e-5.bson")
df = combine(groupby(res[:df], ["model","task"])) do gdf
    gdf = sort(DataFrame(gdf),"val")
    gdf[1:min(10,size(gdf,1)),:]
end
df.nrps = zeros(length(df.model))
ϵ = 1e-2
for row in eachrow(df)
    row.nrps = sum(abs.(Flux.destructure(row.modelps)[1]) .> ϵ)
end

models = Dict("npux"=>"NaiveNPU",
              "gatednpux"=>"NPU",
              "nalu"=>"NALU",
              "nmu"=>"NMU",
              "gatednpu"=>"RealNPU")
ms     = 4
alpha  = 0.7
xscale = :log10
yscale = :log10
plotmodels = [(:circle,"gatednpux"),(:pentagon,"nalu"),(:diamond,"nmu"),
              (:utriangle,"npux"),(:rect,"gatednpu")]
#plotmodels = ["gatednpux","nalu","nmu"]

s1 = plot(title="Addition +")
s2 = plot(title="Multiplication \$\\times\$")
s3 = plot(title="Division \$\\div\$")
s4 = plot(title="Square root \$\\sqrt\\cdot\$")
pdf  = filter(row->row.task=="add", df)
for (marker,model) in plotmodels
    mdf = filter(row->row.model==model, pdf)
    scatter!(s1, mdf.nrps, mdf.val, marker=marker,
             label=models[model], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             ylabel="Testing MSE", legend=false, xlabel="Nr. Parameters")
end

pdf  = filter(row->row.task=="mult", df)
for (marker,model) in plotmodels
    mdf = filter(row->row.model==model, pdf)
    scatter!(s2, mdf.nrps, mdf.val, marker=marker,
             label=models[model], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             legend=false, xlabel="Nr. Parameters", ylim=(1e-1,1e5))
end

pdf  = filter(row->row.task=="invx", df)
for (marker,model) in plotmodels
    mdf = filter(row->row.model==model, pdf)
    scatter!(s3, mdf.nrps, mdf.val, m=marker,
             label=models[model], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             legend=false, xlabel="Nr. Parameters")
end

pdf  = filter(row->row.task=="sqrt", df)
for (marker,model) in plotmodels
    mdf = filter(row->row.model==model, pdf)
    scatter!(s4, mdf.nrps, mdf.val, marker=marker,
             label=models[model], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             legend=:outerright, xlabel="Nr. Parameters", ylim=(1e-4,1e10))
end

p1 = plot(s1,s2,s3,s4,layout=(1,4), size=(900,200))

s1 = plot(title="Addition +")
s2 = plot(title="Multiplication \$\\times\$")
s3 = plot(title="Division \$\\div\$")
s4 = plot(title="Square root \$\\sqrt\\cdot\$")
pdf  = filter(row->row.task=="add", df)
for (marker,model) in plotmodels
    mdf = filter(row->row.model==model, pdf)
    scatter!(s1, mdf.nrps, mdf.mse, marker=marker,
             label=models[model], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             ylabel="Traingin MSE", legend=false)
end

pdf  = filter(row->row.task=="mult", df)
for (marker,model) in plotmodels
    mdf = filter(row->row.model==model, pdf)
    scatter!(s2, mdf.nrps, mdf.mse, marker=marker,
             label=models[model], yscale=yscale,
             legend=false, xscale=xscale, alpha=alpha, ms=ms)
end

pdf  = filter(row->row.task=="invx", df)
for (marker,model) in plotmodels
    mdf = filter(row->row.model==model, pdf)
    scatter!(s3, mdf.nrps, mdf.mse, marker=marker,
             label=models[model], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             legend=false)
end

pdf  = filter(row->row.task=="sqrt", df)
for (marker,model) in plotmodels
    mdf = filter(row->row.model==model, pdf)
    scatter!(s4, mdf.nrps, mdf.mse, marker=marker,
             label=models[model], yscale=yscale, xscale=xscale, alpha=alpha, ms=ms,
             legend=:outerright, ylim=(1e-3,1e4))
end

p2 = plot(s1,s2,s3,s4,layout=(1,4), size=(900,200))
#p = plot(p2,p1,layout=(2,1),size=(900,400))
savefig(p1, plotsdir("pareto.tikz"))
display(p1)

using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Plots
using LaTeXStrings
pgfplotsx()

include(joinpath(@__DIR__, "sobolconfigs.jl"))
include(srcdir("arithmetic_dataset.jl"))

res = load(datadir("pareto","thresh=1e-5.bson"))
df = combine(groupby(res[:df], ["model","task"])) do gdf
    gdf[1:min(10,size(gdf,1)),:]
end

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

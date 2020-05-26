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
using Statistics
using Measurements
using Sobol

include(joinpath(@__DIR__, "sobolconfigs.jl"))
include(joinpath(@__DIR__, "collect.jl"))
include(srcdir("arithmetic_dataset.jl"))

function sobol_samples(c)
    s = SobolSeq(c.inlen)
    # discard first zero sample
    next!(s)
    x = reduce(hcat, [next!(s) for i = 1:10000])
    xs = c.lowlim*2
    xe = c.uplim*2
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

"""
Creates table like this:
| task | npu | npux | ... |

with "key" values

expects dataframe with columns "task" and "model" (and "key")
"""
function create_table(df, key)
    table = DataFrame()
    table.task = unique(df.task)
    table.gatednpux = Vector{}(undef, 4)
    table.nalu = Vector{}(undef, 4)
    table.nmu = Vector{}(undef, 4)
    table.npux = Vector{}(undef, 4)
    
    for gdf in groupby(μdf, ["model","task"])
        row = gdf[1,:]
        table[findall(r->r.task==row.task, eachrow(table)), row.model] = row[key]
    end
    return table
end


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
df = res[:df]

# remove rows with infs
df = combine(groupby(df, ["model","task"])) do gdf
    gdf.val[findall(isinf, gdf.val)] .= 1e10
    gdf.mse[findall(isinf, gdf.mse)] .= 1e10
    gdf[1:min(10,size(gdf,1)),:]
end

# average runs
μdf = combine(groupby(df,["model","task"])) do gdf
    σmse = std(gdf.mse)
    σval = std(gdf.val)
    σmse = isinf(σmse) ? 1e20 : σmse
    σval = isinf(σval) ? 1e20 : σval
    (task = gdf.task[1],
     model = gdf.model[1],
     mse = measurement(mean(gdf.mse), σmse),
     val = measurement(mean(gdf.val), σval),
    )
end


table = create_table(μdf, "mse")
@pt table
table = create_table(μdf, "val")
@pt table

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
using PrettyTables

include(joinpath(@__DIR__, "sobolconfigs.jl"))
include(joinpath(@__DIR__, "collect.jl"))
include(srcdir("arithmetic_dataset.jl"))

function sobol_samples(xs,xe,dim)
    s = SobolSeq(dim)
    # discard first zero sample
    next!(s)
    x = reduce(hcat, [next!(s) for i = 1:10000])
    Float32.(x .* (xs - xe) .+ xe)
end

function sobol_samples(c)
    xs = c.lowlim*2
    xe = c.uplim*2
    sobol_samples(xs,xe,c.inlen)
end

function sobol_samples(c::DivL1SearchConfig)
    xs = -0.5
    xe = 0
    sobol_samples(xs,xe,c.inlen)
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
        table[findall(r->r.task==row.task, eachrow(table)), row.model] .= row[key]
    end
    return table
end

function print_table(df::DataFrame)
    f = (v,i,j) -> (v isa Real ? round(v,digits=7) : v)
    function high(data,i,j)
        if data[i,j] isa Real
            b = data[i,j] == minimum(filter(!ismissing, Array(df[i,2:end])))
            b isa Missing ? false : b
        else
            false
        end
    end
    h = Highlighter(high, bold=true, foreground=:yellow)
    pretty_table(df,names(df),formatters=f,highlighters=h)
end

function revalidate(d::Dict)
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

function latex_table(results::DataFrame)

    function latex_row(row)
        vals = convert(Vector, row[2:end])
        i  = findmin(map(x->isinf(x) ? 1e30 : x,vals))[2]
        ss = map(x->string(round(x,digits=7)), vals)
        ss = map(s->replace(s,"±"=>"\$\\pm\$"), ss)
        ss[i] = "\\textbf{$(ss[i])}"
        srow = DataFrame(Dict(zip(names(row[2:end]),ss)))
    end

    r1 = latex_row(results[1,:])
    r2 = latex_row(results[2,:])
    r3 = latex_row(results[3,:])
    r4 = latex_row(results[4,:])

    latex_str = (
raw"""
\begin{tabular}{lcccc}
\toprule
Task & NPU & NALU & NMU & NaiveNPU\\
\midrule
""" *
"Add  & $(r1[1,:gatednpux]) & $(r1[1,:nalu]) & $(r1[1,:nmu]) & $(r1[1,:npux]) \\\\\n" *
"Mult & $(r2[1,:gatednpux]) & $(r2[1,:nalu]) & $(r2[1,:nmu]) & $(r2[1,:npux]) \\\\\n" *
"Div  & $(r3[1,:gatednpux]) & $(r3[1,:nalu]) & $(r3[1,:nmu]) & $(r3[1,:npux]) \\\\\n" *
"Sqrt & $(r4[1,:gatednpux]) & $(r4[1,:nalu]) & $(r4[1,:nmu]) & $(r4[1,:npux]) \\\\\n" *
raw"""\bottomrule
\end{tabular}
"""
    )
   
end



(res,fname) = produce_or_load(datadir("pareto"),
                          Dict(:thresh=>1e-5),
                          revalidate,
                          digits=10,
                          force=false)
df = res[:df]

# t = filter(r->r.model=="nalu", df)
# t = filter(r->r.task=="mult",t)
# error()


df = combine(groupby(df, ["model","task"])) do gdf
    gdf = sort(DataFrame(gdf),"val")
    if gdf.model[1]=="nmu" && gdf.task[1]=="mult"
        display(gdf[!,["model","task","mse","val","βstart","run"]])
    end
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
print_table(table)

fname = plotsdir("arithmetic100_mse.tex")
open(fname, "w") do file
    @info "Writing dataframe to $fname"
    write(file, latex_table(table))
end

table = create_table(μdf, "val")
print_table(table)

fname = plotsdir("arithmetic100_val.tex")
open(fname, "w") do file
    @info "Writing dataframe to $fname"
    write(file, latex_table(table))
end

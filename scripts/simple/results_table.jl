using DrWatson
@quickactivate "NIPS_2020_NPUX"

using DataFrames
using Statistics
using Flux
using NeuralArithmetic
using PrettyTables
include(joinpath(@__DIR__, "dataset.jl"))

train_range = "pos"
function generate()
    if train_range == "pos-neg"
        return generate_pos_neg()
    elseif train_range == "pos"
        return generate_pos()
    else
        error("Unknown train range: $train_range")
    end
end

function aggregateruns(dataframe::DataFrame)
    gdf = groupby(dataframe, :hash)
    combine(gdf) do df
        (μmse  = mean(df.mse),
         μval  = mean(df.val),
         μadd  = mean(df.add_val),
         μmult = mean(df.mult_val),
         μdiv  = mean(df.div_val),
         μsqrt = mean(df.sqrt_val),
         σmse  = std(df.mse),
         σval  = std(df.val),
         σadd  = std(df.add_val),
         σmult = std(df.mult_val),
         σdiv  = std(df.div_val),
         σsqrt = std(df.sqrt_val),
         model = first(df.model))
    end
end

function find_best(df::DataFrame, key::String)
    combine(groupby(df,"model")) do sdf
        filter(r->r[key]==minimum(sdf[!,key]), sdf)
    end
end

function delete_from_savename(path,key)
    (dir,dict,_) = parse_savename(path)
    delete!(dict, key)
    joinpath(dir, savename(dict,digits=20))
end

name(m::GatedNPUX) = "GatedNPU"
name(m::Dense) = "Dense"
name(m::NALU) = "NALU"
name(m::NMU) = "NMU"
name(m::Chain) = name(m[1])

function collect_folder!(folder::String)
    sum_val(data) = sum([data[:add_val], data[:mult_val],
                         data[:div_val], data[:sqrt_val]])
    _df = collect_results!(datadir(folder), black_list=[:model,:val],
                           special_list=[:model=>data->name(data[:model]),
                                         :val  =>data->sum_val(data)])
    _df.hash = delete_from_savename.(_df.path, "run")
    return _df
end

"""
Creates table like this:
| task | npu | npux | ... |
"""
function table_models_tasks(df::DataFrame)
    result = DataFrame(Union{Measurement,Missing}, 4, length(df.model)+1)
    rename!(result, vcat(["task"], df.model))
    result[!,1] = ["Add", "Mult", "Div", "Sqrt"]

    for m in df.model
        mdf = filter(:model=>model->model==m, df)
        @assert size(mdf,1) == 1
        μcol = [mdf[1,k] for k in ["μadd","μmult","μdiv","μsqrt"]]
        σcol = [mdf[1,k] for k in ["σadd","σmult","σdiv","σsqrt"]]
        result[!,m] = measurement.(μcol,σcol)
    end
    return result
end

function print_table(df::DataFrame, bold=:col)
    f = (v,i,j) -> (v isa Real ? round(v,digits=5) : v)
    function high(data,i,j)
        if data[i,j] isa Real
            if bold == :col
                b = data[i,j] == minimum(filter(!ismissing, Array(df[!,j])))
            elseif bold == :row
                b = data[i,j] == minimum(filter(!ismissing, Array(df[i,2:end])))
            else
                error("bold must be either :col or :row")
            end

            b isa Missing ? false : b
        else
            false
        end
    end
    h = Highlighter(high, bold=true, foreground=:yellow)
    pretty_table(df,names(df),formatters=f,highlighters=h)
end

function filter_by_best_average(df::DataFrame, adf::DataFrame, measure::String)
    best = filter_best_model_task(adf,"μ$measure")
    all_best = [filter(r->r.hash==h, df) for h in best.hash]
    all_best = vcat(all_best...)
end

function latex_table(results::DataFrame)

    function latex_row(row)
        vals = convert(Vector, row[2:end])
        i  = findmin(vals)[2]
        ss = map(x->string(round(x,digits=3)), vals)
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
Task & GatedNPU & NALU & NMU & Dense\\
\midrule
""" *
"Add  & $(r1[1,:GatedNPU]) & $(r1[1,:NALU]) & $(r1[1,:NMU]) & $(r1[1,:Dense]) \\\\\n" *
"Mult & $(r2[1,:GatedNPU]) & $(r2[1,:NALU]) & $(r2[1,:NMU]) & $(r2[1,:Dense]) \\\\\n" *
"Div  & $(r3[1,:GatedNPU]) & $(r3[1,:NALU]) & $(r3[1,:NMU]) & $(r3[1,:Dense]) \\\\\n" *
"Sqrt & $(r4[1,:GatedNPU]) & $(r4[1,:NALU]) & $(r4[1,:NMU]) & $(r4[1,:Dense]) \\\\\n" *
raw"""\bottomrule
\end{tabular}
"""
    )
   
end

df = collect_folder!(datadir("simple"))
sort!(df,"model")
adf = aggregateruns(df)
key = "val"
best = find_best(df,key)
print_table(best[!,["model",key,"add_val","mult_val","div_val","sqrt_val","path"]])

r = table_models_tasks(adf)
print_table(r, :row)

fname = papersdir("simple_err.tex")
open(fname, "w") do file
    @info "Writing dataframe to $fname"
    write(file, latex_table(r))
end


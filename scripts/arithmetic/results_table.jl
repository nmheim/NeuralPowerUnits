using DrWatson
@quickactivate "NIPS_2020_NPUX"

using DataFrames
using Statistics
using Flux
using ValueHistories
using NeuralArithmetic
include(joinpath(@__DIR__, "configs.jl"))

function aggregateruns(dataframe::DataFrame)
    gdf = groupby(dataframe, :hash)
    combine(gdf) do df
        (μmse = mean(df.mse),
         σmse = std(df.mse),
         μreg = mean(df.reg),
         σreg = std(df.reg),
         μtrn = mean(df.trn),
         σtrn = std(df.trn),
         μval = mean(df.val),
         σval = std(df.val),
         fstinit = first(df.fstinit),
         sndinit = first(df.sndinit),
         βend = first(df.βend),
         model = first(df.model),
         task = first(df.task))
    end
end

function expand_config!(df::DataFrame)
    if "config" in names(df)
        for k in fieldnames(typeof(df.config[1]))
            df[!,k] = getfield.(df.config, k)
        end
    elseif "c" in names(df)
        for k in fieldnames(typeof(df.c[1]))
            df[!,k] = getfield.(df.c, k)
        end
    else
        error("neither `config` nor `c` in dataframe")
    end
end

function find_best(df::DataFrame, model::String, task::String, key::String)
    fdf = filter(r->r.model==model, df)
    fdf = filter(r->r.task==task, fdf)
    sort!(fdf, key)
    fdf[1,:]
end


Base.last(h::MVHistory, k::Symbol) = get(h,k)[2][end]

function delete_rows_with_nans!(df::DataFrame, cols=[:trn,:mse,:reg,:val])
     idxs = map(cols) do col
        findall(isnan, df[!,col])
    end
    idx = unique(vcat(idxs...))
    if length(idx) != 0
        @info "Deleting rows with nans: $(idx)"
        delete!(df, idx)
    end
end

function delete_from_savename(path,key)
    (dir,dict,_) = parse_savename(path)
    delete!(dict, key)
    joinpath(dir, savename(dict,digits=20))
end

function collect_folder!(folder::String)
    _df = collect_results!(datadir(folder), white_list=[],
                          special_list=[:trn => data -> last(data[:history], :loss)[1],
                                        :mse => data -> last(data[:history], :loss)[2],
                                        :reg => data -> last(data[:history], :loss)[3],
                                        :val => data -> last(data[:history], :loss)[4],
                                        :config => data -> data[:c],
                                        :task => data -> split(basename(folder),"_")[1],
                                       ],
                         )
    _df.hash = delete_from_savename.(_df.path, "run")
    delete_rows_with_nans!(_df)
    expand_config!(_df)
    return _df
end

collect_all_results!(folders::Vector{String}) = vcat(map(collect_folder!, folders)...)

"""
Creates table like this:
| task | npu | npux | ... |
"""
function best_models_for_tasks(df::DataFrame, key::String)
    best = combine(groupby(df,"model")) do modeldf
        combine(groupby(modeldf, "task")) do taskdf
            tdf = sort!(DataFrame(taskdf), key)
            tdf[1,:]
        end
    end

    best = select(best, "model", "task", key)
    tasks = unique(best, "task").task
    models = unique(best, "model").model

    result = DataFrame(Union{Float64,Missing}, length(tasks), length(models)+1)
    #DataFrame([Vector{t}(undef, nrows) for i = 1:ncols])
    rename!(result, vcat(["task"], models))
    result[!,1] = tasks

    for m in models
        mdf = filter(:model=>model->model==m, best)
        for (i,t) in zip(1:length(tasks), tasks)
            mtdf = filter(:task=>task->task==t, mdf)
            if size(mtdf) == (1,3)
                v = mtdf[1,key]
                result[i,m] = v
            elseif size(mtdf) == (0,3)
                result[i,m] = missing
            else
                error("Expected single row no row.")
            end
        end
    end
    return result
end

function plot_result_folder(df::DataFrame, cols::Vector{String}, measure::String)
    ps = []
    gdf = groupby(df, "model")

    modelplot = plot(title="models", yscale=:log10, legend=false)

    for (name,_df) in zip(keys(gdf), gdf)
        name = name[1]
        m = _df[!,measure] 
        plot!(modelplot, [name for _ in 1:length(m)], m)
        plot!(modelplot, [name], [minimum(m)], marker="o")
        for col in cols
            v = _df[!,col]
            xscale = col=="βend" ? :log10 : :identity
            p = scatter(v,m, xlabel=col, ylabel=measure, title=name,
                        yscale=:log10, xscale=xscale)
            push!(ps, p)
        end

    end
    #plot(ps...,modelplot)
    plot(ps...,modelplot,layout=(:,3))
end

heat(m::Chain) = heatmap(cat(model[1].W[end:-1:1,:], model[2].W', dims=2))
heat(m::Chain{<:Tuple{<:NAU,<:GatedNPUX}}) =
    heatmap(cat(model[1].W[end:-1:1,:], model[2].Re', model[2].Im',dims=2))

function print_table(df::DataFrame)
    f = (v,i,j) -> (v isa Real ? round(v,digits=5) : v)
    function high(data,i,j)
        if data[i,j] isa Real
            b = data[i,j] == minimum(filter(!ismissing, Array(df[i,2:end])))
            b isa Missing ? false : b
        else
            false
        end
    end
    h = Highlighter(high, bold=true, foreground=:yellow)
    pretty_table(df,formatters=f,highlighters=h)
end

key = "mse"
folders = ["addition_npu_l1_search"
          ,"mult_npu_l1_search"
          ,"sqrt_npu_l1_search"
          ,"div_npu_l1_search"]
folders = map(datadir, folders)

# using Plots
# _df = collect_folder!(folders[1])
# pyplot()
# display(plot_result_folder(_df,["βend", "fstinit", "sndinit"], key))
# error()

df = collect_all_results!(folders)
adf = aggregateruns(df)

models = ["gatednpux","nmu","nalu"]
bestdf = best_models_for_tasks(df, key)
#bestdf = best_models_for_tasks(adf, "μ$key")
print_table(bestdf)

row = find_best(df,"gatednpu","sqrt",key)
model = load(row.path)[:model]
display(row.config)
heat(model)

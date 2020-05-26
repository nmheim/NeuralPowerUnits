using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using NeuralArithmetic
using ValueHistories
using DataFrames
include(joinpath(@__DIR__, "sobolconfigs.jl"))

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

Base.last(h::MVHistory, k::Symbol) = get(h,k)[2][end]

function delete_rows_with_nans!(df::DataFrame, cols=[:trn,:mse,:reg,:val])
    idxs = map(cols) do col
        findall(isnan, df[!,col])
    end
    idx = sort(unique(vcat(idxs...)))
    if length(idx) != 0
        @info "Deleting rows with nans:" df[idx,"path"]
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
                                        :modelps => data -> data[:model],
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

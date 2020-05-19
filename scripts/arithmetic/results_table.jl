using DrWatson
@quickactivate "NIPS_2020_NPUX"

using DataFrames
using Statistics
using Flux
using ValueHistories
using NeuralArithmetic
include(joinpath(@__DIR__, "configs.jl"))

function aggregateruns(df::DataFrame)
    gdf = groupby(df, :hash)
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
         sndinit = first(df.sndinit))
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

function find_best(hash::UInt, df::DataFrame, key::String)
    fdf = filter(row->row[:hash]==hash, df)
    sort!(fdf, key)
    fdf[1,:path]
end


Base.last(h::MVHistory, k::Symbol) = get(h,k)[2][end]


folders = ["addition_npu_l1_search"
          ,"mult_npu_l1_search"
          ,"sqrt_npu_l1_search"
          ,"div_npu_l1_search"]
folders = map(datadir, folders)

folder = folders[4]
df = collect_results!(datadir(folder), white_list=[],
                      special_list=[:trn => data -> last(data[:history], :loss)[1],
                                    :mse => data -> last(data[:history], :loss)[2],
                                    :reg => data -> last(data[:history], :loss)[3],
                                    :val => data -> last(data[:history], :loss)[4],
                                    :config => data -> data[:c],
                                    :hash => data -> hash(delete!(struct2dict(data[:c]),:run)),
                                    :task => data -> split(folder,"_")[1],
                                   ],
                     )
expand_config!(df)
display(df)

adf = aggregateruns(df)
key = "val"
sort!(adf,"μ$key")
display(adf)

# bestrun = find_best(adf[1,:hash], df, key)
bestrun = sort!(df,key)[1,:path]
res = load(bestrun)
@unpack model, history = res

using UnicodePlots
UnicodePlots.heatmap(model[1].W[end:-1:1,:], height=100, width=100)



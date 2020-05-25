using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using NeuralArithmetic
using ValueHistories
using DataFrames

include(joinpath(@__DIR__, "configs.jl"))

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
    idx = unique(vcat(idxs...))
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
    #delete_rows_with_nans!(_df)
    expand_config!(_df)
    return _df
end

collect_all_results!(folders::Vector{String}) = vcat(map(collect_folder!, folders)...)


nrparams(x::Array, thresh) = sum(abs.(x) .> thresh)
nrparams(m::NAU, thresh) = nrparams(m.W, thresh)
nrparams(m::NMU, thresh) = nrparams(m.W, thresh)
nrparams(m::GatedNPUX, thresh) = sum(map(x->nrparams(x,thresh), [m.Re,m.Im,m.g]))
nrparams(m::GatedNPU, thres) = sum(map(x->nrparams(x,thresh), [m.W,m.g]))
nrparams(m::NAC, thresh) = sum(map(x->nrparams(x,thresh), [m.W,m.M]))
nrparams(m::NALU, thresh) = sum(map(x->nrparams(x,thresh), [m.nac,m.G,m.b]))
nrparams(m::Chain, thres) = sum(map(x->nrparams(x,thres), m))

folders = ["addition_npu_l1_search"
          ,"mult_npu_l1_search"
          ,"sqrt_npu_l1_search"
          ,"div_npu_l1_search"]

df = collect_all_results!(folders)

function pareto(d::Dict)
    @unpack thresh = d
    #x = validation_samples(df[1,"config"],[4.5f0,2.5f0,1.5f0,0.3f0,0.2f0,0.1f0,1f0,2f0,3f0,10f0])
    @progress for row in eachrow(df)
        m = load(row.path)[:model]
        x = validation_samples(row.config)
        y = task(x,row.config)
        row.val = Flux.mse(m(x),y)
        row.reg = nrparams(m, thresh)
    end
    return @dict(df)
end

res = produce_or_load(datadir("pareto"),Dict(:thresh=>0.001), pareto)
df = res[:df]

p1 = scatterplot([0],[0],title="Pareto")
for g in groupby(df,"model")
    scatterplot!(p1, g.val, g.reg)
end

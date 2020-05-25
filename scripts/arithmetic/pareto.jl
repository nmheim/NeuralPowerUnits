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

#include(joinpath(@__DIR__, "configs.jl"))
include(joinpath(@__DIR__, "sobolconfigs.jl"))
include(srcdir("arithmetic_dataset.jl"))

function validation_samples(c,xs)
    samples = []
    (ii,jj) = ranges(c.inlen,c.subset,c.overlap)
    ii = ii.start:5:ii.stop
    jj = jj.start:5:jj.stop
    indices = Iterators.product(ii,jj,xs,xs)
    for (i,j,x,y) in indices
        z = zeros(Float32,c.inlen)
        z[i] = x
        z[j] = y
        push!(samples,z)
    end
    reduce(hcat, samples)
end

validation_samples(c::Union{MultL1SearchConfig,AddL1SearchConfig,DivL1SearchConfig}) =
    validation_samples(c,[-4.5f0,-2.5f0,-1.5f0,-0.3f0,-0.2f0,0.1f0,1f0,2f0,3f0,10f0])

validation_samples(c::SqrtL1SearchConfig) =
    validation_samples(c,[4.5f0,2.5f0,1.5f0,0.3f0,0.2f0,0.1f0,1f0,2f0,3f0,10f0,20f0])

function sobol_samples(c)
    s = SobolSeq(c.inlen)
    # discard first zero sample
    next!(s)
    x = reduce(hcat, [next!(s) for i = 1:10000])
    xs = c.uplim * 4
    xe = c.lowlim * 4
    Float32.(x .* (xs - xe) .+ xe)
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


nrparams(x::Array, thresh) = sum(abs.(x) .> thresh)
nrparams(m::NAU, thresh) = nrparams(m.W, thresh)
nrparams(m::NMU, thresh) = nrparams(m.W, thresh)
nrparams(m::GatedNPUX, thresh) = sum(map(x->nrparams(x,thresh), [m.Re,m.Im,m.g]))
nrparams(m::GatedNPU, thresh) = sum(map(x->nrparams(x,thresh), [m.W,m.g]))
nrparams(m::NAC, thresh) = sum(map(x->nrparams(x,thresh), [m.W,m.M]))
nrparams(m::NALU, thresh) = sum(map(x->nrparams(x,thresh), [m.nac,m.G,m.b]))
nrparams(m::Chain, thres) = sum(map(x->nrparams(x,thres), m))

folders = ["addition_npu_l1_search"
          ,"mult_npu_l1_search"
          ,"sqrt_npu_l1_search"
          ,"div_npu_l1_search"]

folders = ["add_l1_runs"
          ,"mult_l1_runs"
          ,"invx_l1_runs"
          ,"sqrt_l1_runs"]

df = collect_all_results!(folders)

task(x::Array,c::SqrtL1SearchConfig) = sqrt(x,c.subset)
task(x::Array,c::DivL1SearchConfig) = invx(x,c.subset)
task(x::Array,c::AddL1SearchConfig) = add(x,c.subset,c.overlap)
task(x::Array,c::MultL1SearchConfig) = mult(x,c.subset,c.overlap)

function pareto(d::Dict)
    @unpack thresh = d
    c = first(filter(r->r.task=="sqrt",df)).config
    x = validation_samples(c)
    x = sobol_samples(c)
    @progress for row in eachrow(df)
        m = load(row.path)[:model]
        #x = validation_samples(row.config)
        y = task(x,row.config)
        row.val = Flux.mse(m(x),y)
        row.reg = nrparams(m, thresh)
    end
    return @dict(df)
end

(res,_) = produce_or_load(datadir("pareto"),
                          prefix="sobol",
                          Dict(:thresh=>1e-3),
                          pareto,
                          digits=10,
                          force=false)
df = res[:df]

ps = []
for dft in groupby(df, "task")
    s1 = plot(title=dft.task[1])
    for dfm in groupby(dft,"model")
        scatter!(s1, log10.(dfm.val), log10.(dfm.reg),
                 ms=5, label=dfm.model[1], alpha=0.5)
    end
    push!(ps, s1)
end
plot(ps...)

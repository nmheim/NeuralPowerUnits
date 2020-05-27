using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

using Flux
using NeuralArithmetic
using ValueHistories
using UnicodePlots
using DataFrames
using PrettyTables
using Sobol

include(joinpath(@__DIR__, "configs.jl"))
include(srcdir("unicodeheat.jl"))
include(srcdir("arithmetic_dataset.jl"))

# function validation_samples(c::DivL1SearchConfig)
#     samples = []
#     len = round(Int,c.inlen*c.subset)
# 
#     # examples like [2,0...] , [0,2,0...]
#     for i in 1:len
#         for d in [-3f0,-2f0,-1f0,-0.1f0,0.1f0,1f0,2f0,3f0]
#             z = zeros(Float32,c.inlen)
#             z[i] = d
#             push!(samples, z)
#         end
#     end
# 
#     # examples like [0.x, 0.x, ...]
#     for i in 1:len
#         for s in [-0.2f0,-0.1f0,0.01f0, 0.1f0, 0.2f0]
#             push!(samples, ones(Float32,c.inlen) .* s)
#         end
#     end
# 
#     # random normal exapmles
#     # for i in 1:len
#     #     push!(samples, randn(Float32,c.inlen))
#     # end
# 
#     reduce(hcat, samples)
# end
# 
# function validation_samples(c::Union{MultL1SearchConfig,AddL1SearchConfig})
#     samples = []
#     (ii,jj) = ranges(c.inlen,c.subset,c.overlap)
#     ii = ii.start:5:ii.stop
#     jj = jj.start:5:jj.stop
#     xs = [1f0,2f0,3f0,4f0,0.1f0,0.2f0]
#     xs = vcat(xs, .-xs, [0f0])
#     indices = Iterators.product(ii,jj,xs,xs)
#     for (i,j,x,y) in indices
#         z = zeros(Float32,c.inlen)
#         z[i] = x
#         z[j] = y
#         push!(samples,z)
#     end
#     reduce(hcat, samples)
# end
# 
#     #xs = [1f0,2f0,3f0,4f0,0.1f0,0.2f0]

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

    # validation_samples(c,[ 1f0, 2f0, 3f0, 4f0, 0.1f0, 0.2f0,
    #                       -1f0,-2f0,-3f0,-4f0,-0.1f0,-0.2f0])

validation_samples(c::SqrtL1SearchConfig) =
    #validation_samples(c,[4.5f0,2.5f0,1.5f0,0.3f0,0.2f0,0.02f0,0.01f0,0.1f0,1f0,2f0,3f0,10f0])
    validation_samples(c,[0.001f0,0.01f0,0.02f0,0.1f0,1f0,2f0,3f0])

function sobol_samples(c)
    s = SobolSeq(c.inlen)
    # discard first zero sample
    next!(s)
    x = reduce(hcat, [next!(s) for i = 1:100000])
    xs = c.uplim * 4
    xe = c.lowlim * 4
    Float32.(x .* (xs - xe) .+ xe)
end

task(x::Array,c::SqrtL1SearchConfig) = sqrt(x,c.subset)
task(x::Array,c::DivL1SearchConfig) = invx(x,c.subset)
task(x::Array,c::AddL1SearchConfig) = add(x,c.subset,c.overlap)
task(x::Array,c::MultL1SearchConfig) = mult(x,c.subset,c.overlap)

function table_models_tasks(df::DataFrame, key::String)
    best = select(df, "model", "task", key)
    tasks = unique(best, "task").task
    models = unique(best, "model").model

    result = DataFrame(Union{Float64,Missing}, length(tasks), length(models)+1)
    #DataFrame([Vector{t}(undef, nrows) for i = 1:ncols])
    rename!(result, map(string, vcat(["task"], models)))
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
    pretty_table(df,names(df),formatters=f,highlighters=h)
end

@unpack bestmodels = load(datadir("arithmetic_best_models.bson"))

key = "val"
print_table(table_models_tasks(bestmodels,key))
@progress for row in eachrow(bestmodels)
    #x = sobol_samples(row.config)
    x = validation_samples(row.config)
    y = task(x,row.config)
    m = load(row.path)[:model]
    row.val = Flux.mse(m(x),y)
end
print_table(table_models_tasks(bestmodels,key))

error()

display(c)
X = validation_samples(c)
#t = invx(X,c.subset)
t = mult(X,c.subset,c.overlap)
y = model(X)

err = sum(abs, t-y)
display(model(X))
display(t)
display(err)

heat(model)

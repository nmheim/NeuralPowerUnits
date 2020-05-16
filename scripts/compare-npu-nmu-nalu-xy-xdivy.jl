using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using LinearAlgebra
using NeuralArithmetic

f(x::Array) = cat(reshape(x[1,:] .* x[2,:], 1, :),
                  reshape(x[1,:] ./ x[2,:], 1, :),
                  dims=1)


train_range = "pos-neg"

function generate_pos_neg()
    x = rand(Float32, 2, 100) .* 4 .- 2
    y = f(x)
    (x,y)
end

function generate_pos()
    x = rand(Float32, 2, 100) .* 2 .+ 0.1f0
    y = f(x)
    (x,y)
end

function test_generate()
    x = rand(Float32, 2, 10000) .* 8 .- 4
    y = f(x)
    (x,y)
end

function generate()
    if train_range == "pos-neg"
        return generate_pos_neg()
    elseif train_range == "pos"
        return generate_pos()
    else
        error("Unknown train range: $train_range")
    end
end

function run_npu(c::Dict)
    model = GatedNPUX(2,2,initRe=(s...)->rand(Float32,s...)/10)
    ps = params(model)
    opt = RMSProp(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y) + c[:βl1]*norm(model.Im, 1) #+ 0.1norm(model.Im,1)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c)
end

function run_nmu(c::Dict)
    model = NMU(2,2)
    ps = params(model)
    opt = RMSProp(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c)
end

function run_nalu(c::Dict)
    model = Chain(NALU(2,2))
    ps = params(model)
    opt = RMSProp(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c)
end

function run_dense(c::Dict)
    model = Chain(Dense(2,10,σ),Dense(10,10,σ),Dense(10,2))
    ps = params(model)
    opt = RMSProp(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c)
end


res, _ = produce_or_load(datadir("layercomparison"),
                         Dict(:niters=>40000, :βl1=>0.1, :lr=>0.002),
                         run_npu,
                         prefix="$train_range-gatednpux",
                         force=false, digits=6)
npu = res[:model]

res, _ = produce_or_load(datadir("layercomparison"),
                         Dict(:niters=>20000, :lr=>0.001),
                         run_nmu,
                         prefix="$train_range-nmu")
nmu = res[:model]

res, _ = produce_or_load(datadir("layercomparison"),
                         Dict(:niters=>40000, :lr=>0.001),
                         run_nalu,
                         prefix="$train_range-nalu", force=false)
nalu = res[:model]

res, _ = produce_or_load(datadir("layercomparison"),
                         Dict(:niters=>40000, :lr=>0.001),
                         run_dense,
                         prefix="$train_range-dense", force=false)
dense = res[:model]



using DataFrames

multloss(model,x::Real,y::Real) = abs(model([x,y])[1] - f([x,y])[1])
multloss(model,x::Array,y::Array) = Flux.mse(model(x)[1,:], y[1,:])
divloss(model,x::Real,y::Real) = abs(model([x,y])[2] - f([x,y])[2])
divloss(model,x::Array,y::Array) = Flux.mse(model(x)[2,:], y[2,:])

(x,y)   = generate()
(xt,yt) = test_generate()
df = DataFrame(model=String[],
               mult_trn=Float32[], div_trn=Float32[],
               mult_val=Float32[], div_val=Float32[])

push!(df, ("NPU", multloss(npu,x,y), divloss(npu,x,y),
           multloss(npu,xt,yt), divloss(npu,xt,yt)))

push!(df, ("NMU", multloss(nmu,x,y), divloss(nmu,x,y),
           multloss(nmu,xt,yt), divloss(nmu,xt,yt)))

push!(df, ("NALU", multloss(nalu,x,y), divloss(nalu,x,y),
           multloss(nalu,xt,yt), divloss(nalu,xt,yt)))

push!(df, ("Dense", multloss(dense,x,y), divloss(dense,x,y),
           multloss(dense,xt,yt), divloss(dense,xt,yt)))

for col in ["mult_trn", "div_trn", "mult_val", "div_val"]
    df[!,col] = round.(df[!,col], digits=4)
end

latex_str = (
raw"""
\begin{tabular}{lcccc}
\toprule
Model & MultTrain & DivTrain & MultTest & DivTest\\
\midrule
""" *
"NPU & $(df[1,:mult_trn]) & $(df[1,:div_trn]) & $(df[1,:mult_val]), & $(df[1,:div_val]) \\\\\n" *
"NMU & $(df[2,:mult_trn]) & $(df[2,:div_trn]) & $(df[2,:mult_val]), & $(df[2,:div_val]) \\\\\n" *
"NALU & $(df[3,:mult_trn]) & $(df[3,:div_trn]) & $(df[3,:mult_val]), & $(df[3,:div_val]) \\\\\n" *
"Dense & $(df[4,:mult_trn]) & $(df[4,:div_trn]) & $(df[4,:mult_val]), & $(df[4,:div_val]) \\\\\n" *
raw"""\bottomrule
\end{tabular}
"""
)

display(df)
fname = papersdir("table-x*y-xdivy-$train_range.tex")
open(fname, "w") do file
    @info "Writing dataframe to $fname"
    #latex_str = repr(MIME("text/latex"), df)
    write(file, latex_str)
end

using Plots
include(srcdir("turbocmap.jl"))
pyplot()

x = Float32.(collect(-4:0.1:4))
y = Float32.(collect(-4:0.1:4))

clim = (-3,2)
s1 = heatmap(x,y,(x,y)->log10(multloss(npu,x,y)), clim=clim, c=turbo_cgrad)
s2 = heatmap(x,y,(x,y)->log10(divloss(npu,x,y)), clim=clim, c=turbo_cgrad)
p1 = plot(s1,s2)

s1 = heatmap(x,y,(x,y)->log10(multloss(nmu,x,y)+eps()), clim=clim, c=turbo_cgrad)
s2 = heatmap(x,y,(x,y)->log10(divloss(nmu,x,y)), clim=clim, c=turbo_cgrad)
p2 = plot(s1,s2)

s1 = heatmap(x,y,(x,y)->log10(multloss(nalu,x,y)), clim=clim, c=turbo_cgrad)
s2 = heatmap(x,y,(x,y)->log10(divloss(nalu,x,y)), clim=clim, c=turbo_cgrad)
p3 = plot(s1,s2)

s1 = heatmap(x,y,(x,y)->log10(multloss(dense,x,y)), clim=clim, c=turbo_cgrad)
s2 = heatmap(x,y,(x,y)->log10(divloss(dense,x,y)), clim=clim, c=turbo_cgrad)
p4 = plot(s1,s2)

display(p1)

#model = GatedNPU(2,2, init=(s...)->rand(Float32,s...)/10)
#model = NALU(2,2)
#opt = ADAM(0.001)
# loss(x,y) = Flux.mse(model(x),y) + 0.1norm(ps, 1) #+ 0.1norm(model.Im,1)
# 
# display(f(randn(2,10)))

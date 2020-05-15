using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using LinearAlgebra
using NeuralArithmetic

f(x::Array) = cat(reshape(x[1,:] .* x[2,:], 1, :),
                  reshape(x[1,:] ./ x[2,:], 1, :),
                  dims=1)

function generate()
    x = rand(Float32, 2, 100) .* 4 .- 2
    y = f(x)
    (x,y)
end

function generate_pos()
    x = rand(Float32, 2, 100) .* 2
    y = f(x)
    (x,y)
end

function test_generate()
    x = rand(Float32, 2, 1000) .* 8 .- 4
    y = f(x)
    (x,y)
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
    model = NALU(2,2)
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
                         prefix="gatednpux",
                         force=false, digits=6)
npu = res[:model]

res, _ = produce_or_load(datadir("layercomparison"),
                         Dict(:niters=>20000, :lr=>0.001),
                         run_nmu,
                         prefix="nmu")
nmu = res[:model]

res, _ = produce_or_load(datadir("layercomparison"),
                         Dict(:niters=>20000, :lr=>0.001),
                         run_nalu,
                         prefix="nalu", force=false)
nalu = res[:model]


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

latex_str = L"""
\begin{tabular}{lcccc}
    \toprule
	Model & MultTrain & DivTrain & MultTest & DivTest\\
	\midrule
    NPU & $(df[1,:mult_trn]) & 0.000918221 & 0.000114184 & 1.24065 \\
	NMU & 0.0 & 10.0826 & 0.0 & 2569.56 \\
	NALU & 1.00368 & 11.1217 & 23.2803 & 2560.12 \\
    \bottomrule
\end{tabular}
"""

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
raw"""\bottomrule
\end{tabular}
"""
)

display(df)
fname = papersdir("table_x*y_xdivy.tex")
open(fname, "w") do file
    @info "Writing dataframe to $fname"
    #latex_str = repr(MIME("text/latex"), df)
    write(file, latex_str)
end

# using Plots
# include(srcdir("turbocmap.jl"))
# pyplot()
# 
# x = Float32.(collect(-4:0.1:4))
# y = Float32.(collect(-4:0.1:4))
# 
# s1 = heatmap(x,y,(x,y)->multloss(npu,x,y), clim=(0,0.1))
# s2 = heatmap(x,y,(x,y)->divloss(npu,x,y), clim=(0,0.1))
# p1 = plot(s1,s2)
# 
# s1 = heatmap(x,y,(x,y)->multloss(nmu,x,y), clim=(0,0.1))
# s2 = heatmap(x,y,(x,y)->divloss(nmu,x,y), clim=(0,0.1))
# p2 = plot(s1,s2)
# 
# display(p1)

#model = GatedNPU(2,2, init=(s...)->rand(Float32,s...)/10)
#model = NALU(2,2)
#opt = ADAM(0.001)
# loss(x,y) = Flux.mse(model(x),y) + 0.1norm(ps, 1) #+ 0.1norm(model.Im,1)
# 
# display(f(randn(2,10)))

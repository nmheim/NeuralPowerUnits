using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using LinearAlgebra
using Random
using DataFrames
using NeuralArithmetic

Random.seed!(0)

f1(x::Array) = reshape(x[1,:] .+ x[2,:], 1, :)
f1(x::Array) = reshape(x[1,:], 1, :)
f2(x::Array) = reshape(x[1,:] .* x[2,:], 1, :)
f3(x::Array) = reshape(x[1,:] ./ x[2,:], 1, :)
f4(x::Array) = reshape(sqrt.(x[1,:]), 1, :)
f(x::Array) = cat(f1(x),f2(x),f3(x),f4(x),dims=1)

train_range = "pos"

function generate_pos_neg()
    x = rand(Float32, 2, 100) .* 4 .- 2
    y = f(x)
    (x,y)
end

function generate_pos()
    x = rand(Float32, 2, 100) .* 2 .+ 0.01f0
    y = f(x)
    (x,y)
end

function test_generate()
    x = rand(Float32, 2, 10000) .* 6
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
    # model = Chain(NAU(2,4),
    #               #GatedNPUX(4,4,initRe=(s...)->rand(Float32,s...)/10))
    #               GatedNPUX(4,4))
    #model = GatedNPUX(2,4)
    hdim = 6
    model = Chain(GatedNPUX(2,hdim),NAU(hdim,4))
    #model = Chain(NAU(2,hdim),GatedNPUX(hdim,4))
    #model = GatedNPUX(2,4,initRe=(s...)->rand(Float32,s...)/10)
    #model = NPU(2,4)
    #model = GatedNPUX(2,4)
    ps = params(model)
    opt = ADAM(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y) #+ c[:βl1]*norm(ps, 1) #+ 0.1norm(model.Im,1)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c)
end

function run_nmu(c::Dict)
    model = NMU(2,4)
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
    #model = Chain(NALU(2,4))
    hdim = 6
    model = Chain(NALU(2,hdim),NALU(hdim,4))
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
    model = Chain(Dense(2,10,σ),Dense(10,10,σ),Dense(10,4))
    ps = params(model)
    opt = ADAM(c[:lr])
    data = (generate() for _ in 1:c[:niters])
    loss(x,y) = Flux.mse(model(x),y)
    (x,y) = generate()
    cb = Flux.throttle(() -> (@info loss(x,y)), 0.1)
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c)
end


res, _ = produce_or_load(datadir("simplefunctions"),
                         Dict(:niters=>20000, :βl1=>1e-5, :lr=>0.01),
                         run_npu,
                         prefix="$train_range-gatednpux",
                         force=true, digits=6)
npu = res[:model]

# res, _ = produce_or_load(datadir("simplefunctions"),
#                          Dict(:niters=>20000, :lr=>0.001),
#                          run_nmu,
#                          prefix="$train_range-nmu")
# nmu = res[:model]
# 
res, _ = produce_or_load(datadir("simplefunctions"),
                         Dict(:niters=>20000, :lr=>0.01),
                         run_nalu,
                         prefix="$train_range-nalu", force=true)
nalu = res[:model]

# res, _ = produce_or_load(datadir("simplefunctions"),
#                          Dict(:niters=>50000, :lr=>0.001),
#                          run_dense,
#                          prefix="$train_range-dense", force=false)
# dense = res[:model]




addloss(model,x::Real,y::Real)  = abs(model([x,y])[1] - f1([x,y])[1])
multloss(model,x::Real,y::Real) = abs(model([x,y])[2] - f2([x,y])[1])
divloss(model,x::Real,y::Real)  = abs(model([x,y])[3] - f3([x,y])[1])
sqrtloss(model,x::Real,y::Real) = abs(model([x,y])[4] - f4([x,y])[1])

# multloss(model,x::Array,y::Array) = Flux.mse(model(x)[1,:], y[1,:])
# divloss(model,x::Array,y::Array) = Flux.mse(model(x)[2,:], y[2,:])

(x,y)   = generate()
(xt,yt) = test_generate()
# df = DataFrame(model=String[],
#                mult_trn=Float32[], div_trn=Float32[],
#                mult_val=Float32[], div_val=Float32[])
# 
# push!(df, ("NPU", multloss(npu,x,y), divloss(npu,x,y),
#            multloss(npu,xt,yt), divloss(npu,xt,yt)))
# 
# push!(df, ("NMU", multloss(nmu,x,y), divloss(nmu,x,y),
#            multloss(nmu,xt,yt), divloss(nmu,xt,yt)))
# 
# push!(df, ("NALU", multloss(nalu,x,y), divloss(nalu,x,y),
#            multloss(nalu,xt,yt), divloss(nalu,xt,yt)))
# 
# push!(df, ("Dense", multloss(dense,x,y), divloss(dense,x,y),
#            multloss(dense,xt,yt), divloss(dense,xt,yt)))
# 
# for col in ["mult_trn", "div_trn", "mult_val", "div_val"]
#     df[!,col] = round.(df[!,col], digits=6)
# end
# 
# latex_str = (
# raw"""
# \begin{tabular}{lcccc}
# \toprule
# Model & MultTrain & DivTrain & MultTest & DivTest\\
# \midrule
# """ *
# "NPU & $(df[1,:mult_trn]) & $(df[1,:div_trn]) & $(df[1,:mult_val]), & $(df[1,:div_val]) \\\\\n" *
# "NMU & $(df[2,:mult_trn]) & $(df[2,:div_trn]) & $(df[2,:mult_val]), & $(df[2,:div_val]) \\\\\n" *
# "NALU & $(df[3,:mult_trn]) & $(df[3,:div_trn]) & $(df[3,:mult_val]), & $(df[3,:div_val]) \\\\\n" *
# "Dense & $(df[4,:mult_trn]) & $(df[4,:div_trn]) & $(df[4,:mult_val]), & $(df[4,:div_val]) \\\\\n" *
# raw"""\bottomrule
# \end{tabular}
# """
# )
# 
# display(df)
# fname = papersdir("table-x*y-xdivy-$train_range.tex")
# open(fname, "w") do file
#     @info "Writing dataframe to $fname"
#     write(file, latex_str)
# end

using Plots
using LaTeXStrings
include(srcdir("turbocmap.jl"))
#pgfplotsx()
pyplot()

clim = (-4,2)
nantozero(z) = isnan(z) ? 0 : z
function inftoextreme(z)
    if isinf(z)
        m = maxintfloat(typeof(z))
        return z<0 ? -m : m
    else
        return z
    end
end
rminvalid = nantozero ∘ inftoextreme

@info "Plotting NPU..."

x = Float32.(collect(-4:0.1:4))
y = Float32.(collect(-4:0.1:4))
func(x,y) = rminvalid(log10(addloss(npu,x,y)))
s1 = heatmap(x,y,func, c=turbo_cgrad, clim=clim, title="NPU "*L"+", ylabel="y")
func(x,y) = rminvalid(log10(multloss(npu,x,y)))
s2 = heatmap(x,y,func, c=turbo_cgrad, clim=clim, title="NPU "*L"\times", colorbar=false, ylabel="y")
func(x,y) = rminvalid(log10(divloss(npu,x,y)))
s3 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="NPU "*L"\div", colorbar=false, ylabel="y", xlabel="x")
x = Float32.(collect(0:0.1:4))
y = Float32.(collect(0:0.1:4))
func(x,y) = rminvalid(log10(sqrtloss(npu,x,y)))
s4 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="NPU sqrt", colorbar=false, ylabel="y", xlabel="x")
p1 = plot(s1,s2,s3,s4)
# display(npu[1].W)
# display(npu[2].Re)
# display(npu[2].Im)
# display(npu[2].g)
display(npu[1].Re)
display(npu[1].Im)
display(npu[1].g)
display(npu[2].W)


x = Float32.(collect(-4:0.1:4))
y = Float32.(collect(-4:0.1:4))
func(x,y) = rminvalid(log10(addloss(nalu,x,y)))
s1 = heatmap(x,y,func, c=turbo_cgrad, clim=clim, title="NALU "*L"+", ylabel="y")
func(x,y) = rminvalid(log10(multloss(nalu,x,y)))
s2 = heatmap(x,y,func, c=turbo_cgrad, clim=clim, title="NALU "*L"\times", colorbar=false, ylabel="y")
func(x,y) = rminvalid(log10(divloss(nalu,x,y)))
s3 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="NALU "*L"\div", colorbar=false, ylabel="y", xlabel="x")
x = Float32.(collect(0:0.1:4))
y = Float32.(collect(0:0.1:4))
func(x,y) = rminvalid(log10(sqrtloss(nalu,x,y)))
s4 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="NALU sqrt", colorbar=false, ylabel="y", xlabel="x")
p2 = plot(s1,s2,s3,s4)

display(plot(p1,p2))
error()


@info "Plotting NMU..."
func(x,y) = rminvalid(log10(multloss(nmu,x,y)))
s3 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="NMU "*L"\times", colorbar=false)
func(x,y) = rminvalid(log10(divloss(nmu,x,y)))
s4 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="NMU "*L"\div", colorbar=false, xlabel="x")
@info "Plotting NALU..."
func(x,y) = rminvalid(log10(multloss(nalu,x,y)))
s5 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="NALU "*L"\times", colorbar=false)
func(x,y) = rminvalid(log10(divloss(nalu,x,y)))
s6 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="NALU "*L"\div", colorbar=false, xlabel="x")
func(x,y) = rminvalid(log10(multloss(dense,x,y)))
@info "Plotting Dense..."
s7 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="Dense "*L"\times",
             colorbar_title=L"\log |\hat t-t|^2")
func(x,y) = rminvalid(log10(divloss(dense,x,y)))
s8 = heatmap(x,y,func, clim=clim, c=turbo_cgrad, title="Dense "*L"\div",
             colorbar_title=L"\log |\hat t-t|^2", xlabel="x")

fname = papersdir("compare-npu-nmu-nalu-xy-xdivy.png")
@info "Saving plot to $fname"
plt = plot(s1,s3,s5,s7, s2,s4,s6,s8, layout=(2,4), size=(1000,400))
savefig(plt, fname)
@info "Display plot..."
display(plt)

#model = GatedNPU(2,2, init=(s...)->rand(Float32,s...)/10)
#model = NALU(2,2)
#opt = ADAM(0.001)
# loss(x,y) = Flux.mse(model(x),y) + 0.1norm(ps, 1) #+ 0.1norm(model.Im,1)
# 
# display(f(randn(2,10)))

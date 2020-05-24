using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using Parameters
using LinearAlgebra
using NeuralArithmetic
using ValueHistories
using Plots
unicodeplots()

include(srcdir("unicodeheat.jl"))
#pyplot()

# f(x) = x^2
# f(x) = x^4/10 + x^3/10 - (13x^2)/10 - x/10 + 6/5

prefix = "complicated_rational"
f(x) = (x^3-2x)/(2(x^2-5))

#prefix = "rational"
#f(x) = 1/(x-2.15)

function validationplot(xt,yt,tt,lowlim,uplim)
    p1 = plot(vec(xt), vec(yt))
    plot!(p1, vec(xt), vec(tt))
    vline!([lowlim,uplim],c=:gray)
end


function generate(f,lowlim,uplim)
    x = Float32.(reshape(lowlim:0.1:uplim, 1, :))
    y = f.(x)
    xt = Float32.(reshape((lowlim*1.1):0.1:(uplim*1.1),1,:))
    yt = f.(xt)
    (x,y,xt,yt)
end

iml1(model::GatedNPUX) = norm(model.Im,1)
iml1(model::NAU) = 0
iml1(model::Dense) = 0
iml1(model::NALU) = 0
iml1(model::NMU) = 0
iml1(model::Chain) = sum(iml1, model)

function get_model(layer,dim)
    if layer == "nalu"
        return Chain(NALU(1,dim),
                     NALU(dim,dim),
                     NALU(dim,1))
    elseif layer == "gatednpux"
        return Chain(GatedNPUX(1,dim),
                     NAU(dim,dim),
                     GatedNPUX(dim,1))
    elseif layer == "dense"
        return Chain(Dense(1,dim,σ),
                     Dense(dim,dim,σ),
                     Dense(dim,1))
    elseif layer == "nmu"
        return Chain(NMU(1,dim), NAU(dim,dim), NMU(dim,1))
    else
        error("unknown layer: $layer")
    end
end

function run(c::Dict, f::Function)
    @unpack dim, lowlim, uplim, niter, lr, βpsl1, βiml1, layer = c
    h = MVHistory()
    (x,y,xt,yt) = generate(f,lowlim,uplim)
    data = Iterators.repeated((x,y), niter)
    opt = ADAM(lr)

    model = get_model(layer, dim)
    ps = params(model)

    mse(x,y) = sum(abs2, model(x) .- y)
    loss(x,y) = mse(x,y) + βiml1*iml1(model) + βpsl1*norm(ps,1)

    niter = 1
    cb = [Flux.throttle(()->(
               p1 = validationplot(xt,yt,model(xt),lowlim,uplim);
               display(p1);
               println();
               display(heat(model));
               println();
               @info niter mse(x,y) loss(x,y) loss(xt,yt)
              ), 1),
          Flux.throttle(() -> (push!(h, :μz, Flux.destructure(model)[1]);
                               push!(h, :mse, mse(xt,yt))), 0.1),
          () -> (niter += 1)
         ]
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c, :history=>h)
end


npures, _ = produce_or_load(datadir("polynomial"),
                         Dict(:lowlim =>  -5,
                              :uplim  =>  5,
                              :dim    =>  10,
                              :niter  =>  100000,
                              :lr     =>  1e-4,
                              :layer  => "gatednpux",
                              :βiml1  =>  0.0,
                              :βpsl1  =>  0.0),
                         c -> run(c, f), prefix=prefix,
                         force=false, digits=8)
npu = npures[:model]
nalures, _ = produce_or_load(datadir("polynomial"),
                         Dict(:lowlim =>  -5,
                              :uplim  =>  5,
                              :dim    =>  10,
                              :niter  =>  100000,
                              :lr     =>  1e-4,
                              :layer  => "nalu",
                              :βiml1  =>  0.0,
                              :βpsl1  =>  0.0),
                         c -> run(c, f), prefix=prefix,
                         force=false, digits=8)
nalu = nalures[:model]
nmures, _ = produce_or_load(datadir("polynomial"),
                         Dict(:lowlim =>  -5,
                              :uplim  =>  5,
                              :dim    =>  10,
                              :niter  =>  10000,
                              :lr     =>  1e-4,
                              :layer  => "nmu",
                              :βiml1  =>  0.0,
                              :βpsl1  =>  0.0),
                         c -> run(c, f), prefix=prefix,
                         force=false, digits=8)
nmu = nmures[:model]
denseres, _ = produce_or_load(datadir("polynomial"),
                         Dict(:lowlim =>  -5,
                              :uplim  =>  5,
                              :dim    =>  10,
                              :niter  =>  50000,
                              :lr     =>  1e-3,
                              :layer  => "dense",
                              :βiml1  =>  0.0,
                              :βpsl1  =>  0.0),
                         c -> run(c, f), prefix=prefix,
                         force=false, digits=8)
dense = denseres[:model]



xt = Float32.(reshape(-5:0.05:5,1,:))
yt = f.(xt)

pyplot()
p1 = plot(vec(xt), vec(yt),label="Truth", ylim=(-6,6))
plot!(p1, vec(xt), vec(npu(xt)), label="GatedNPU $(Flux.mse(npu(xt),yt))")
plot!(p1, vec(xt), vec(nalu(xt)), label="NALU $(Flux.mse(nalu(xt),yt))")
#plot!(p1, vec(xt), vec(nmu(xt)), label="NMU $(Flux.mse(nmu(xt),yt))")
plot!(p1, vec(xt), vec(dense(xt)), label="Dense $(Flux.mse(dense(xt),yt))")
display(p1)
error()


model = res[:model]
history = res[:history]
c = res[:config]
@unpack lowlim,uplim = c
(x,y,xt,yt) = generate(f,lowlim,uplim)
display(validationplot(xt,yt,model(xt),lowlim,uplim))
error()
println("")
heat(model)

# function concat(m::Chain{<:Tuple{<:GatedNPUX,<:NAU}})
#     Re = m[1].Re[end:-1:1,:]
#     Im = m[1].Im[end:-1:1,:]
#     W  = m[2].W'
#     h  = cat(Re,Im,W,dims=2)
# end
# h = concat(model)
# display(h)
# p1 = heatmap(h)


# ps = reduce(hcat, get(history,:μz)[2])'
# p2 = plot(ps)










# function run(c::Dict, f::Function)
#     @unpack dim, lowlim, uplim, niter, lr, βl1 = c
#     x = Float32.(reshape(lowlim:0.1:uplim, 1, :))
#     y = f.(x)
#     xt = Float32.(reshape((lowlim*1.5):0.2:(uplim*1.5),1,:))
#     yt = f.(xt)
# 
#     h = MVHistory()
#     data = Iterators.repeated((x,y), niter)
#     opt = ADAM(lr)
#     model = Chain(GatedNPUX(1,dim), NAU(dim,1))
#     ps = params(model)
#     mse(x,y) = sum(abs2, model(x) .- y)
#     #loss(x,y) = mse(x,y) + βl1*norm(model[1].Im)
#     loss(x,y) = mse(x,y) + βl1*norm(ps)
# 
#     cb = [Flux.throttle(()->(
#                p1 = plot(vec(xt), vec(yt));
#                plot!(p1, vec(xt), vec(model(xt)));
#                display(p1);
#                @info loss(x,y) loss(xt,yt)
#               ), 1),
#           Flux.throttle(() -> (push!(h, :μz, Flux.destructure(model)[1]);
#                                push!(h, :mse, mse(xt,yt))), 0.1)
#          ]
#     Flux.train!(loss, ps, data, opt, cb=cb)
#     return Dict(:model=>model, :config=>c, :history=>h)
# end
# 
# res, _ = produce_or_load(datadir("polynomial"),
#                          Dict(:lowlim=>-5, :uplim=>5, :dim=>3, :niter=>10000, :lr=>1e-2, :βl1=>0.1),
#                          c -> run(c, f), prefix="x2",
#                          force=true, digits=8)
# 
# model = res[:model]
# 
# pyplot()
# function concat(m::Chain{<:Tuple{<:GatedNPUX,<:NAU}})
#     Re = m[1].Re[end:-1:1,:]
#     Im = m[1].Im[end:-1:1,:]
#     W  = m[2].W'
#     h  = cat(Re,Im,W,dims=2)
# end
# h = concat(model)
# heatmap(h)
# # heat(m::Chain) =
# #

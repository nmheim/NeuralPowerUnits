using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Flux
using Parameters
using LinearAlgebra
using NeuralArithmetic
using ValueHistories
using Plots
#unicodeplots()
pyplot()

f(x) = x^2
f(x) = x^4/10 + x^3/10 - (13x^2)/10 - x/10 + 6/5
f(x) = (x^3-2x)/2(x^2-5)
#f(x) = (x-2)/(x-3)

function validationplot(xt,yt,tt,lowlim,uplim)
    p1 = plot(vec(xt), vec(yt))
    plot!(p1, vec(xt), vec(tt))
    vline!([lowlim,uplim],c=:gray)
end



function run(c::Dict, f::Function)
    @unpack dim, lowlim, uplim, niter, lr, βpsl1, βiml1 = c
    x = Float32.(reshape(lowlim:0.1:uplim, 1, :))
    y = f.(x)
    xt = Float32.(reshape((lowlim*1.1):0.1:(uplim*1.1),1,:))
    yt = f.(xt)

    h = MVHistory()
    data = Iterators.repeated((x,y), niter)
    opt = ADAM(lr)
    #model = Chain(NAU(1,dim), GatedNPUX(dim,dim), NAU(dim,1))
    model = Chain(GatedNPUX(1,dim), NAU(dim,dim), GatedNPUX(dim,1))
    iml1(model::GatedNPUX) = norm(model.Im,1)
    iml1(model::NAU) = 0
    iml1(model::Chain) = sum(iml1, model)
    #model = Chain(NALU(1,dim), NALU(dim,1))
    ps = params(model)
    mse(x,y) = sum(abs2, model(x) .- y)
    iml1() = βiml1*iml1(model)
    psl1() = βpsl1*norm(ps,1)
    loss(x,y) = mse(x,y) #+ iml1() + psl1()

    cb = [Flux.throttle(()->(
               p1 = validationplot(xt,yt,model(xt),lowlim,uplim);
               display(p1);
               @info loss(x,y) loss(xt,yt)
              ), 1),
          Flux.throttle(() -> (push!(h, :μz, Flux.destructure(model)[1]);
                               push!(h, :mse, mse(xt,yt))), 0.1)
         ]
    Flux.train!(loss, ps, data, opt, cb=cb)
    return Dict(:model=>model, :config=>c, :history=>h)
end


res, _ = produce_or_load(datadir("polynomial"),
                         Dict(:lowlim=>-5, :uplim=>5, :dim=>50, :niter=>100000, :lr=>1e-4, :βiml1=>0.1, :βpsl1=>0.01),
                         c -> run(c, f), prefix="rational",
                         force=true, digits=8)

model = res[:model]
history = res[:history]

pyplot()
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

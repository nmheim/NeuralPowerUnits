using DrWatson
@quickactivate "NIPS_2020_NMUX"

using DiffEqFlux
using DiffEqFlux: paramlength
using OrdinaryDiffEq
using Flux
using Optim
using LinearAlgebra
using NeuralArithmetic
using Plots
unicodeplots()

include(joinpath(@__DIR__, "odedata.jl"))

init(a,b) = rand(Float64,a,b)/10

function plot_cb(t,ode_data,pred)
    pl = scatter(t,ode_data[1,:],label="S")
    scatter!(pl,t,ode_data[2,:],label="I")
    scatter!(pl,t,ode_data[3,:],label="R")
    plot!(pl,t,pred[1,:],label="Ŝ")
    plot!(pl,t,pred[2,:],label="Î")
    plot!(pl,t,pred[3,:],label="R̂")
end

nrparams(p, thresh) = sum(abs.(p) .> thresh)

function run(d::Dict)
    idim = 3
    @unpack hdim, αinit, βps, βim, niters, lr = d

    # TODO: does it still work with standard glorot?
    init(a,b) = Float64.(Flux.glorot_uniform(a,b))*αinit

    ode_data,u0,t,tspan = fracsir_data()
    dudt = FastChain(
        NeuralArithmetic.FastGatedNPUX(idim,hdim, initRe=init,initIm=zeros),
        NeuralArithmetic.FastNAU(hdim,idim,init=init))
    node = NeuralODE(dudt,tspan,Euler(),saveat=t,dt=1)
    predict(p) = node(u0,p)

    reg_loss(p) = norm(p,1)
    mse_loss(x) = Flux.mse(x,ode_data)
    function img_loss(p)
        npu = dudt.layers[1]
        (_,Im,_) = NeuralArithmetic._restructure(npu, p[1:paramlength(npu)])
        norm(Im,1)
    end

    function node_loss(p)
        pred = predict(p)
        loss = mse_loss(pred) + βps * reg_loss(p) + βim * img_loss(p)
        return loss, pred
    end

    function plot_chain(p)
        npu = dudt.layers[1]
        Re,Im,_ = NeuralArithmetic._restructure(npu, p[1:paramlength(npu)])
        W = reshape(p[(paramlength(npu)+1):end], idim, hdim)
        UnicodePlots.heatmap(cat(Re,Im,W',dims=1))
    end

    function cb(p,l,pred;doplot=true)
        @info l mse_loss(pred) reg_loss(p) img_loss(p)
        display(plot_cb(t,ode_data,pred))
        println("\n")
        display(plot_chain(p))
        println("\n")
        return false
    end

    cb(node.p, node_loss(node.p)...)

    res1 = DiffEqFlux.sciml_train(node_loss, node.p, RMSProp(lr),
                                  cb=Flux.throttle(cb,1), maxiters=niters)
    cb(res1.minimizer,node_loss(res1.minimizer)...)

    res2 = DiffEqFlux.sciml_train(node_loss, res1.minimizer, LBFGS(),
                                  cb=(p,l,pred)->cb(p,l,pred,doplot=false))
    cb(res2.minimizer,node_loss(res2.minimizer)...)

    ϵ = 0.001
    ps = node.p
    pred = predict(ps)
    mse = mse_loss(pred)
    nrps = nrparams(res2.minimizer, ϵ)
    @info "# params" nrps
    @dict(dudt, ps, pred, mse, nrps)
end

produce_or_load(datadir("fracsir"),
                Dict(:hdim=>3,
                     :βim=>1,
                     :βps=>1,
                     :lr=>0.005,
                     :niters=>2000,
                     :αinit=>0.2),
                run,
                prefix="gatednpux",
                digits=10,
                force=false)
error()


res2 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, LBFGS(), cb = cb)
cb(res2.minimizer,loss_n_ode(res2.minimizer)...;doplot=true)
@info "# params" nrparams(res1.minimizer, 0.001)

npu = dudt.layers[1]
l = paramlength(npu)
Re,Im,_ = NeuralArithmetic._restructure(npu, res2.minimizer[1:l])
W = reshape(res2.minimizer[(l+1):end], indim, hdim)
@info "matrices" Re Im W

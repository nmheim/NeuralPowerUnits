using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Logging
using TerminalLoggers
using ProgressLogging
global_logger(TerminalLogger(right_justify=80))

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
    @unpack hdim, αinit, βps, niters, lr = d

    init(a,b) = Float64.(Flux.glorot_uniform(a,b))*αinit

    ode_data,u0,t,tspan = fracsir_data()
    act = Flux.tanh
    dudt = FastChain(
        FastDense(idim,hdim,act,initW=init),
        FastDense(hdim,hdim,act,initW=init),
        FastDense(hdim,idim,initW=init))
    node = NeuralODE(dudt,tspan,Euler(),saveat=t,dt=1)
    predict(p) = node(u0,p)

    reg_loss(p) = norm(p,1)
    mse_loss(x) = Flux.mse(x,ode_data)

    function node_loss(p)
        pred = predict(p)
        loss = mse_loss(pred) + βps * reg_loss(p)
        return loss, pred
    end

    function cb(p,l,pred;doplot=true)
        @info l mse_loss(pred) reg_loss(p)
        if doplot
            display(plot_cb(t,ode_data,pred))
            println("\n")
        end
        return false
    end

    cb(node.p, node_loss(node.p)...)

    res = DiffEqFlux.sciml_train(node_loss, node.p, RMSProp(lr),
                                  cb=Flux.throttle(cb,1), maxiters=niters)
    cb(res.minimizer,node_loss(res.minimizer)...)

    try
        res = DiffEqFlux.sciml_train(node_loss, res.minimizer, LBFGS(),
                                      cb=(p,l,pred)->cb(p,l,pred,doplot=false))
        cb(res.minimizer,node_loss(res.minimizer)...)
    catch e
        println("Error during LBFGS training!")
        for (exc,bt) in Base.catch_stack()
            showerror(stdout,exc,bt)
            println()
        end
    end

    ϵ = 0.001
    d[:ps]   = res.minimizer
    d[:dudt] = dudt
    d[:pred] = predict(ps)
    d[:mse]  = mse_loss(pred)
    d[:nrps] = nrparams(res.minimizer, ϵ)
    return d
end

@progress for nr in 1:10
    produce_or_load(datadir("fracsir"),
                    Dict(:hdim=>20,
                         :βps=>1.1,
                         :lr=>0.005,
                         :niters=>2000,
                         :αinit=>1,
                         :run=>nr),
                    run,
                    prefix="dense",
                    digits=10,
                    force=false)
end

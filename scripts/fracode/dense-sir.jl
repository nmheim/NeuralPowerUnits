using DiffEqFlux
using DiffEqFlux: paramlength
using OrdinaryDiffEq
using Flux
using LinearAlgebra
using NeuralArithmetic
using Plots
unicodeplots()

datasize = 40


function fracsir!(du,u,p,t)
    S,I,R = u
    α,β,η,γ,κ = p

    r = β * (I^γ) * (S^κ)

    dS = -r + η*R
    dI = r - α*I
    dR = α*I -η*R

    du .= (dS, dI, dR)
end
tspan = (0.0,200.0)
#tspan = (0.0,4.0)
u0 = [100.; 0.01; 0.]
α  = 0.05
β  = 0.06
η  = 0.01
γ  = 0.5
κ  = 1-γ
p  = [α,β,η,γ,κ]
t  = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(fracsir!,u0,tspan,p)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

init(a,b) = rand(Float64,a,b)/10
init(a,b) = Float64.(Flux.glorot_uniform(a,b))/3

hdim = 20
indim = 3

dudt = FastChain(FastDense(indim,hdim,σ),
                 FastDense(hdim,hdim,σ),
                 FastDense(hdim,indim))
n_ode = NeuralODE(dudt,tspan,Euler(),saveat=t, dt=1)

function predict_n_ode(p)
  n_ode(u0,p)
end

reg_loss(p) = norm(p,1)
mse_loss(pred) = sum(abs2, ode_data .- pred)
function img_loss(p)
    npu = dudt.layers[1]
    if npu isa NeuralArithmetic.FastGatedNPUX
        (_,Re,_) = NeuralArithmetic._restructure(npu, p[1:paramlength(npu)])
        norm(Re,1)
    else
        0
    end
end


function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = mse_loss(pred) #+ 1e0 * reg_loss(p)
    loss,pred
end

function plot_chain(p)
    if dudt.layers[1] isa NeuralArithmetic.FastGatedNPUX
        npu = dudt.layers[1]
        Re,Im,_ = NeuralArithmetic._restructure(npu, p[1:paramlength(npu)])
        W = reshape(p[(paramlength(npu)+1):end], indim, hdim)
        UnicodePlots.heatmap(cat(Re,Im,W',dims=1))
    end
end

nrparams(p, thresh) = sum(abs.(p) .> thresh)

cb = function (p,l,pred;doplot=true) #callback function to observe training
  # plot current prediction against data
  @info l mse_loss(pred) reg_loss(p) img_loss(p) minimum(p) maximum(p)
  if doplot
    display(plot_chain(p))
    pl = scatter(t,ode_data[1,:],label="S")
    scatter!(pl,t,ode_data[2,:],label="I")
    scatter!(pl,t,ode_data[3,:],label="R")
    plot!(pl,t,pred[1,:],label="Ŝ")
    plot!(pl,t,pred[2,:],label="Î")
    plot!(pl,t,pred[3,:],label="R̂")
    display(plot(pl))
    println("\n")
  end
  return false
end

# Display the ODE with the initial parameter values.
cb(n_ode.p,loss_n_ode(n_ode.p)...)

#res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, RMSProp(0.005), cb = Flux.throttle(cb,1), maxiters = 2000)
res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(0.01), cb = Flux.throttle(cb,1), maxiters = 2000)
cb(res1.minimizer,loss_n_ode(res1.minimizer)...;doplot=true)

# res2 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, RMSProp(0.0005), cb = cb, maxiters = 2000)
# cb(res2.minimizer,loss_n_ode(res2.minimizer)...;doplot=true)

using Optim
res2 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, LBFGS(), cb = cb)
cb(res2.minimizer,loss_n_ode(res2.minimizer)...;doplot=true)
@info "# params"nrparams(res2.minimizer, 0.001)

npu = dudt.layers[1]
l = paramlength(npu)
Re,Im,_ = NeuralArithmetic._restructure(npu, res2.minimizer[1:l])
W = reshape(res2.minimizer[(l+1):end], indim, hdim)
@info "matrices" Re Im W

# result is res2 as an Optim.jl object
# res2.minimizer are the best parameters
# res2.minimum is the best loss

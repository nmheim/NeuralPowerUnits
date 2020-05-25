using DiffEqFlux
using OrdinaryDiffEq
using Flux
using LinearAlgebra
using NeuralArithmetic
using Plots
unicodeplots()

datasize = 40

u0 = [1.; 0.]

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

function trueODEfunc(du,u,p,t)
  du[1] = u[2]
  #du[2] = -2sqrt(u[1])
  #du[2] = -2u[1]^2
  du[2] = -u[2] - 0u[1] - u[1]^(-1)
end
tspan = (0.0,2.5)
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

init(a,b) = rand(Float64,a,b)/10
init(a,b) = Float64.(Flux.glorot_uniform(a,b))/3
# dudt = Chain(NAU(2,10),
#              GatedNPU(10,10,init=init),
#              NAU(10,2,init=init))


hdim = 5
dudt = Chain(GatedNPUX(2,hdim,initRe=init, initIm=zeros),
             NAU(hdim,2,init=init))

dudt = FastChain(NeuralArithmetic.FastGatedNPUX(2,hdim,initRe=init, initIm=zeros),
                 NeuralArithmetic.FastNAU(hdim,2,init=init))
#dudt = Chain(NALU(2,hdim),NALU(hdim,2))

# dudt = Chain(GatedNPUX([3.1 0.0; 0.0 3.1],
#                        zeros(2,2), ones(2)),
#              NAU([-0.1 2.0; -2.0 -0.1]))
# dudt = Chain(NPUX([3.0 0.0; 0.0 3.0],
#                   zeros(2,2)),
#              NAU([-0.1 2.0; -2.0 -0.1]'))



# dudt = FastChain(FastDense(2,hdim,tanh),
#                  FastDense(hdim,hdim,tanh),
#                  FastDense(hdim,2))
n_ode = NeuralODE(dudt,tspan,Euler(),saveat=t, dt=0.01)

function predict_n_ode(p)
  n_ode(u0,p)
end

reg_loss(p) = 1e-4*norm(p,1)
mse_loss(pred) = sum(abs2, ode_data .- pred)
img_loss(p) = 1e-1*norm(p[hdim*2+1:hdim*4],1)

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = mse_loss(pred) + reg_loss(p) + img_loss(p)
    loss,pred
end

function plot_chain(p)
    npuw = p[1:(hdim*4)]
    nauw = p[(hdim*4+1+2):end]
    UnicodePlots.heatmap(reshape(vcat(npuw,nauw), hdim, 6), height=hdim)
end


cb = function (p,l,pred;doplot=true) #callback function to observe training
  # plot current prediction against data
  @info l mse_loss(pred) reg_loss(p) minimum(p) maximum(p)
  if doplot
    display(plot_chain(p))
    pl = scatter(t,ode_data[1,:],label="data")
    scatter!(pl,t,ode_data[2,:],label="data")
    plot!(pl,t,pred[1,:],label="prediction")
    plot!(pl,t,pred[2,:],label="prediction")
    display(plot(pl))
    println("\n")
  end
  return false
end

# Display the ODE with the initial parameter values.
cb(n_ode.p,loss_n_ode(n_ode.p)...)
#error()

res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, RMSProp(0.0005), cb = Flux.throttle(cb,1), maxiters = 10000)
cb(res1.minimizer,loss_n_ode(res1.minimizer)...;doplot=true)

# res2 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, RMSProp(0.0005), cb = cb, maxiters = 2000)
# cb(res2.minimizer,loss_n_ode(res2.minimizer)...;doplot=true)

using Optim
res2 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, LBFGS(), cb = cb)
cb(res2.minimizer,loss_n_ode(res2.minimizer)...;doplot=true)

# result is res2 as an Optim.jl object
# res2.minimizer are the best parameters
# res2.minimum is the best loss

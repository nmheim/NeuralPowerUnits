using DiffEqFlux
using OrdinaryDiffEq
using Flux
using NeuralArithmetic
using Plots
unicodeplots()

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

# function trueODEfunc(du,u,p,t)
#     true_A = [-0.1 2.0; -2.0 -0.1]
#     du .= ((u.^3)'true_A)'
# end
# t = range(tspan[1],tspan[2],length=datasize)
# prob = ODEProblem(trueODEfunc,u0,tspan)
# ode_data = Array(solve(prob,Tsit5(),saveat=t))

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,7.0)
truep = [1.1,1.0,1.3,1.0]
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(lotka_volterra,u0,tspan,truep)
ode_data = Array(solve(prob, Tsit5(), saveat=t))


init(a,b) = rand(Float64,a,b)/10
# dudt = Chain(NAU(2,10),
#              GatedNPU(10,10,init=init),
#              NAU(10,2,init=init))
dudt = Chain(GatedNPU(2,2,init=init),
             NAU(2,2,init=init))
# dudt = FastChain(FastDense(2,50,tanh),
#                  FastDense(50,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t)

function predict_n_ode(p)
  n_ode(u0,p)
end

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end


cb = function (p,l,pred;doplot=true) #callback function to observe training
  # plot current prediction against data
  if doplot
    pl = scatter(t,ode_data[1,:],label="data", title="$l")
    scatter!(pl,t,pred[1,:],label="prediction")
    display(plot(pl))
  end
  return false
end

# Display the ODE with the initial parameter values.
cb(n_ode.p,loss_n_ode(n_ode.p)...)

res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, RMSProp(0.005), cb = cb, maxiters = 100)
cb(res1.minimizer,loss_n_ode(res1.minimizer)...;doplot=true)

res2 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, RMSProp(0.005), cb = cb, maxiters = 300)
cb(res2.minimizer,loss_n_ode(res2.minimizer)...;doplot=true)

res3 = DiffEqFlux.sciml_train(loss_n_ode, res2.minimizer, RMSProp(0.005), cb = cb, maxiters = 3000)
cb(res3.minimizer,loss_n_ode(res3.minimizer)...;doplot=true)
error()

using Optim
res3 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, LBFGS(), cb = cb)
cb(res2.minimizer,loss_n_ode(res2.minimizer)...;doplot=true)

# result is res2 as an Optim.jl object
# res2.minimizer are the best parameters
# res2.minimum is the best loss

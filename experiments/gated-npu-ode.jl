using DiffEqFlux
using OrdinaryDiffEq
using Flux
using LinearAlgebra
using NeuralArithmetic
using Plots
unicodeplots()

u0 = [2.; 0.]
datasize = 30
tspan = (0.0,1.5)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^1)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [0.5,1.0]
tspan = (0.0,7.0)
truep = [1.1,1.0,1.3,1.0]
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(lotka_volterra,u0,tspan,truep)
ode_data = Array(solve(prob, Tsit5(), saveat=t))


init(a,b) = rand(Float64,a,b)/2
init(a,b) = Float64.(Flux.glorot_uniform(a,b))
# dudt = Chain(NAU(2,10),
#              GatedNPU(10,10,init=init),
#              NAU(10,2,init=init))

dudt = Chain(GatedNPU(2,10,init=init),
             NAU(10,2,init=init))

hdim = 20
dudt = Chain(GatedNPUX(2,hdim,initRe=init, initIm=zeros),
             NAU(hdim,2,init=init))

#dudt = Chain(NALU(2,hdim),NALU(hdim,2))

# dudt = Chain(GatedNPUX([3.1 0.0; 0.0 3.1],
#                        zeros(2,2), ones(2)),
#              NAU([-0.1 2.0; -2.0 -0.1]))
# dudt = Chain(NPUX([3.0 0.0; 0.0 3.0],
#                   zeros(2,2)),
#              NAU([-0.1 2.0; -2.0 -0.1]'))



# dudt = FastChain(FastDense(2,50,tanh),
#                  FastDense(50,2))
n_ode = NeuralODE(dudt,tspan,Euler(),saveat=t, dt=0.01)

function predict_n_ode(p)
  n_ode(u0,p)
end

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred) + 1e-4*norm(p,1) + 0.1norm(p[hdim*2+1:hdim*4],1)
    loss,pred
end


cb = function (p,l,pred;doplot=true) #callback function to observe training
  # plot current prediction against data
  @info l p'
  if doplot
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

res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, RMSProp(0.005), cb = cb, maxiters = 1000)
cb(res1.minimizer,loss_n_ode(res1.minimizer)...;doplot=true)
@info "opt 1 done ---------------------------------------------------------"
@info "opt 1 done ---------------------------------------------------------"
@info "opt 1 done ---------------------------------------------------------"
@info "opt 1 done ---------------------------------------------------------"
@info "opt 1 done ---------------------------------------------------------"

res2 = DiffEqFlux.sciml_train(loss_n_ode, res1.minimizer, RMSProp(0.0005), cb = cb, maxiters = 2000)
cb(res2.minimizer,loss_n_ode(res2.minimizer)...;doplot=true)
error()

using Optim
res3 = DiffEqFlux.sciml_train(loss_n_ode, res2.minimizer, LBFGS(), cb = cb)
cb(res3.minimizer,loss_n_ode(res3.minimizer)...;doplot=true)

# result is res2 as an Optim.jl object
# res2.minimizer are the best parameters
# res2.minimum is the best loss

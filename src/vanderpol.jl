# using DrWatson
# @quickactivate "NIPS_2020_NMUX"

using DiffEqBase
using OrdinaryDiffEq
using Parameters

function vanderpol(du, u, p, t)
    x, y = u
    μ = p[1]
    du[1] = y
    du[2] = -x -μ*(x^2 - 1)*y
end

function vanderpol_dataset(setup::Dict)
    @unpack x0, y0, steps, dt, μ = setup
    init_steps = 100

    u0 = [x0, y0]
    tspan = (0.0, (steps+init_steps)*dt)
    p = [μ]

    prob = ODEProblem(vanderpol, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=dt)

    u = hcat(sol.u...)[:,init_steps:end]
    t = sol.t[init_steps:end]

    @dict u t
end

function vanderpol_dataset(; x0=-1.0, y0=1.0, dt=0.1, steps=10000, μ=1.0)
    setup = @dict x0 y0 dt steps μ
    produce_or_load(datadir("vanderpol"), setup, vanderpol_dataset)[1]
end

# using Plots
# pyplot()
# @unpack u = vanderpol_dataset()
# plot(u')

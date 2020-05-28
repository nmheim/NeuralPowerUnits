function fracsir!(du,u,p,t)
    S,I,R = u
    α,β,η,γ,κ = p

    r = β * (I^γ) * (S^κ)

    dS = -r + η*R
    dI = r - α*I
    dR = α*I -η*R

    du .= (dS, dI, dR)
end

function fracsir_data()
    datasize = 40

    α  = 0.05
    β  = 0.06
    η  = 0.01
    γ  = 0.5
    κ  = 1-γ

    tspan = (0.0,200.0)
    u0 = [100.; 0.01; 0.]

    t  = range(tspan[1],tspan[2],length=datasize)
    prob = ODEProblem(fracsir!,u0,tspan,[α,β,η,γ,κ])
    ode_data = Array(solve(prob,Tsit5(),saveat=t))
    ode_data, u0, t, tspan
end

function fracosc!(du,u,p,t)
  du[1] = u[2]
  du[2] = -u[2] - 0u[1] - u[1]^(-1)
end

function fracosc_data()
    datasize = 40
    
    u0 = [1.; 0.]
    tspan = (0.0,3.0)

    t = range(tspan[1],tspan[2],length=datasize)
    prob = ODEProblem(fracosc!,u0,tspan)
    ode_data = Array(solve(prob,Tsit5(),saveat=t))
    ode_data, u0, t, tspan
end


using LaTeXStrings
using OrdinaryDiffEq
using Plots
pgfplotsx()

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

(x,u0,t,_) = fracsir_data()
display(x)

p1 = scatter(t, x[1,:], markersize=3, c=1, label=L"True $S$", title="Fractional SIR", size=(450,300))
scatter!(p1, t, x[2,:], markersize=3, c=2, label=L"True $I$")
scatter!(p1, t, x[3,:], markersize=3, c=3, label=L"True $R$")
savefig(p1, "fsir-data.tikz")

plot!(p1, t, x[1,:], lw=3, c=1, label=L"Predicted $S$")
plot!(p1, t, x[2,:], lw=3, c=2, label=L"Predicted $I$")
plot!(p1, t, x[3,:], lw=3, c=3, label=L"Predicted $R$")
savefig(p1, "fsir-fit.tikz")


display(p1)


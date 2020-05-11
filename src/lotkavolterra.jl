function generate_lotka_volterra_func(;dt=0.1, tlen=15, p=[1.5,1.0,1.0,1.0], noise=0.1)
    function lotka_volterra(du,u,p,t)
        x, y = u
        α, β, δ, γ = p
        du[1] = dx = α*x - β*x*y
        du[2] = dy = -δ*y + γ*x*y
    end
    u0 = [1.0,1.0]
    tspan = (0.0,100.0)
    prob = ODEProblem(lotka_volterra,u0,tspan,p)
    sol = solve(prob,Tsit5(),saveat=dt)
    ODE_DATA = hcat(sol.u...)
    ODE_DATA .+= randn(size(ODE_DATA)) * noise

    xlen = tlen*2

    function generate(batch)
        U = Array{Float32}(undef, xlen, batch)
        for ii in 1:batch
            s = rand(1:size(ODE_DATA,2)-tlen)
            u = reshape(ODE_DATA[:, s:s+tlen-1], :)
            U[:,ii] = u
        end
        return U
    end

    return ODE_DATA, generate
end

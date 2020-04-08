"""
Generate pairs of sine waves and their generating frequency
"""
function generate_harmonic(nr_examples; steps=30, dt=pi/10,
                        freq_range=[1.0, 1.2], T=Float32)
    U = Array{T,2}(undef, steps, nr_examples)
    Ω = Array{T,1}(undef, nr_examples)
    t = T.(range(0, length=steps, step=dt))
    for ii in 1:nr_examples
        ω = T(freq_range[1] + rand() * abs(freq_range[2] - freq_range[1]))
        ϕ = rand(T) * 2pi
        u = sin.(ω * t .+ ϕ)

        U[:,ii] .= u
        Ω[ii]    = ω
    end
    reshape(U, 1, steps, nr_examples), Ω
end

function generate_harmonic(ω, batch; ω0=0.5, noise=0.01, dt=0.1, steps=20, T=Float32)
    U, Ω = generate_harmonic(
        batch, steps=steps, dt=dt, freq_range=[ω0, ω], T=T)
    U .+= randn(T, size(U)) * T(noise)
    return U, Ω
end



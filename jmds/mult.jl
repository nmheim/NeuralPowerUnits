
using DrWatson
@quickactivate "NIPS_2020_NMUX"

using LinearAlgebra
using Flux
using NeuralArithmetic
using Distributions: Uniform
using Plots
# pyplot()
plotly()

include(srcdir("arithmetic_st_models.jl"))


w1 = Float32.(-1:0.04:1)
w2 = Float32.(-1:0.04:1)
r  = Uniform(-3,3)

X  = Float32.(rand(r, 4, 300))

function f(x::Vector)
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    x4 = x[4]
    (x1 + x2) * (x1 + x2 + x3 + x4)
end;


nacplus(x::Matrix{T}, w::Matrix{T}) where T = w*x
function nacplus(x::Matrix{T}, w::T) where T
    W = [w w T(0) T(0);
         w w w w]
    nacplus(x, W)
end

nacmult(x::Matrix{T}, w::Matrix{T}; ϵ=T(1e-8)) where T = exp.(w * log.(abs.(x) .+ ϵ))
nacmult(x::Matrix{T}, w::T) where T = nacmult(x, [w w]);

nalu(x::Matrix{T}, w1::T, w2::T) where T = nacmult(nacplus(x, w1), w2)


function mse(model::Function, x::Matrix{T}, w1::T, w2::T; zmin=0, zmax=1000) where T
    y = mapslices(f, x, dims=1)
    ŷ = model(x, w1, w2)
    max(min(Flux.mse(y,ŷ), zmax), zmin)
end

# p1 = surface(w1, w2, (w1,w2)-> mse(nalu,X,w1,w2),
#     title="NALU", xlabel="w1", ylabel="w2")
# display(p1)


function nmu(x::Vector{T}, W::Matrix{T}) where T
    z = W .* reshape(x,1,:) .+ 1 .- W
    dropdims(prod(z, dims=2), dims=2)
end
function nmu(x::AbstractMatrix, W::Matrix{T}) where T
    buf = zeros(size(W,1), size(x,2))
    for i in 1:size(x,2)
        buf[:,i] = nmu(x[:,i], W)
    end
    return buf
end
nmu(x::Matrix{T}, w::T) where T = nmu(x, [w w])

naunmu(x::Matrix{T}, w1, w2) where T = nmu(nacplus(x,w1),w2)

# p1 = surface(w1, w2, (w1,w2)-> mse(naunmu,X,w1,w2),
#     title="NAU/NMU", xlabel="w1", ylabel="w2")
# display(p1)


function npu(x::Matrix{T}, W::Matrix{T}) where T
    r = abs.(x)
    k = map(i -> T(i < 0 ? pi : 0.0), x)
    z = exp.(W * log.(r)) .* cos.(W*k)
    z[vec(r .< 1f-1)] .= 0
end
npu(x::Matrix{T}, w::T) where T = npu(x, [w w])

naunpu(x::Matrix{T}, w1, w2) where T = npu(nacplus(x,w1),w2)

p1 = surface(w1, w2, (w1,w2)-> mse(naunpu,X,w1,w2),
    title="NAU/NPU", xlabel="w1", ylabel="w2")
display(p1)


function mse_l1(model::Function, x::Matrix{T}, w1::T, w2::T; zmin=0, zmax=30) where T
    nau = [w1 w1 T(0) T(0);
           w1 w1  w1   w1]
    npu = [w2 w2]

    y = mapslices(f, x, dims=1)
    ŷ = model(x, w1, w2)
    mse = Flux.mse(y,ŷ)
    l1  = 2norm(cat(vec(nau), vec(npu), dims=1), 1)
    l = mse + l1
    max(min(l, zmax), zmin)
end

p1 = surface(w1, w2, (w1,w2)-> mse_l1(naunpu,X,w1,w2),
    title="NAU/NPU", xlabel="w1", ylabel="w2")
display(p1)


function mse_logst(model::Function, x::Matrix{T}, w1::T, w2::T; zmax=27) where T
    nau = [w1 w1 T(0) T(0);
           w1 w1  w1   w1]
    npu = [w2 w2]

    y = mapslices(f, x, dims=1)
    ŷ = model(x, w1, w2)
    mse = Flux.mse(y,ŷ)
    lst = sum(logSt(cat(vec(nau), vec(npu), dims=1), 0.5, 2f0))
    l = mse - lst
    min(l, zmax)
end

p1 = surface(w1, w2, (w1,w2)-> mse_logst(naunpu,X,w1,w2),
    title="NAU/NPU", xlabel="w1", ylabel="w2")
display(p1)


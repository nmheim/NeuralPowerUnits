using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Plots
using Zygote
using Distributions
using LaTeXStrings
#pgfplotsx()
pyplot()

include(srcdir("turbocmap.jl"))

function nmu(x::Vector, W::Matrix)
     z = W .* reshape(x,1,:) .+ 1 .- W
    dropdims(prod(z, dims=2), dims=2)
end
nmu(X::Matrix, W::Matrix) = vec(mapslices(x->nmu(x,W), X, dims=1))

function npu(x::AbstractArray{T}, W::Matrix{T}) where T
    r = abs.(x) .+ eps(T)
    k = max.(-sign.(x), 0) .* T(pi)
    z = exp.(W * log.(r)) .* cos.(W*k)
end
#npu(X::Matrix, W::Matrix) = vec(mapslices(x->npu(x,W), X, dims=1))

function gatednpu(x::AbstractArray{T}, W::AbstractMatrix{T}, g::AbstractVector{T}) where T
    g = min.(max.(g, 0), 1)
    r = abs.(x) .+ eps(T)
    r = g .* r .+ (1 .- g) .* T(1)
    k = max.(-sign.(x), 0) .* T(pi)
    z = exp.(W * log.(r)) .* cos.(W*k)
end

batch = 500
X = Array{Float64,2}(undef,2,batch)
X[1,:] .= rand(Uniform(-3,3),batch)
X[2,:] .= rand(Uniform(-0.05,0.05),batch)
# X[1,:] .= rand(Uniform(-3,3),batch)
# X[2,:] .= rand(Uniform(-0.001,0.001), batch)


npuloss(w1,w2) = mean(abs2, vec(npu(X, [w1 w2])) .- X[1,:])
nmuloss(w1,w2) = mean(abs2, vec(nmu(X, [w1 w2])) .- X[1,:])
gatednpuloss(w1,w2,g) = mean(abs2, vec(gatednpu(X, [w1 w2], [g,g])) .- X[1,:])
# gatednpuloss_zero(w1,w2) = mean(abs2, gatednpu(X, [w1 w2], zeros(2)) .- X[1,:])

w1 = -1:0.05:1
w2 = -1:0.05:1

zlim = clim = (0,4.2)
p1 = heatmap(w1, w2, (w1,w2)->min(gatednpuloss(w1,w2,0.5), clim[2]),
             xlabel=L"w_1", ylabel=L"w_2", c=turbo_cgrad, title="Gated NPU", colorbar=false)
p2 = heatmap(w1, w2, (w1,w2)->min(npuloss(w1,w2), clim[2]),
             xlabel=L"w_1", c=turbo_cgrad, title="NPU", yticks=false, colorbar=false)
p3 = heatmap(w1, w2, (w1,w2)->min(nmuloss(w1,w2), clim[2]),
             xlabel=L"w_1", c=turbo_cgrad, title="NMU", yticks=false)
plt = plot(p1,p2,p3,layout=(1,3),size=(1000,333))

#wsave(plotsdir("test.tex"), plt)
#savefig(plt, plotsdir("test.tikz"))

using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Plots
using Distributions
using LaTeXStrings
using Random
Random.seed!(2)
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
    r = g .* r .+ (1 .- g)

    k = max.(-sign.(x), 0) .* T(pi)
    k = g .* k

    z = exp.(W * log.(r)) .* cos.(W*k)
end

function gatednpux(x::AbstractArray{T}, Re::AbstractMatrix{T}, Im::AbstractMatrix{T}, g::AbstractVector{T}) where T
    g = min.(max.(g, 0), 1)

    r = abs.(x) .+ eps(T)
    r = g .* r .+ (1 .- g)

    k = max.(-sign.(x), 0) .* T(pi)
    k = g .* k

    exp.(Re*log.(r) - Im*k) .* cos.(Re*k + Im*log.(r))
end


batch = 32
X = Array{Float64,2}(undef,2,batch)
X[1,:] .= rand(Uniform(0,2),batch)
X[2,:] .= rand(Uniform(-0.05,0.05),batch)
# X[1,:] .= rand(Uniform(-3,3),batch)
# X[2,:] .= rand(Uniform(-0.001,0.001), batch)


npuloss(w1,w2) = mean(abs2, vec(npu(X, [w1 w2])) .- X[1,:])
nmuloss(w1,w2) = mean(abs2, vec(nmu(X, [w1 w2])) .- X[1,:])
gatednpuloss(w1,w2,g1,g2) = mean(abs2, vec(gatednpu(X, [w1 w2], [g1,g2])) .- X[1,:])
gatednpuloss(w1,w2,g) = gatednpuloss(w1,w2,g,g)

gatednpuxloss(w1,w2,i1,i2,g1,g2) = mean(abs2, vec(gatednpux(X, [w1 w2], [i1 i2], [g1,g2])) .- X[1,:])
gatednpuxloss(w1,w2,i,g) = gatednpuxloss(w1,w2,i,i,g,g)

w1 = -1:0.05:1
w2 = -1:0.05:1

zlim = clim = (1,4.2)
# cmap = cgrad(:inferno, rev=true)
# p1 = contour(w1, w2, (w1,w2)->min(gatednpuloss(w1,w2,0.5), clim[2]),
#              xlabel=L"w_1", ylabel=L"w_2", title="GatedNPU", fill=true, colorbar=false,aspect_ratio=:equal, c=cmap)
# p2 = contour(w1, w2, (w1,w2)->min(npuloss(w1,w2), clim[2]),
#              xlabel=L"w_1", title="NPU", yticks=false, fill=true,aspect_ratio=:equal, c=cmap)
# plt = plot(p1,p2,layout=grid(1,2,widths=[0.465,0.5]),size=(600,300))
# display(plt)
# savefig(plt, plotsdir("npu_gatednpu_id_loss.pdf"))

using Flux
gated_dLdw1(w1,w2) = Flux.gradient(w->gatednpuloss(w,w2,0.5), w1)[1]
gated_dLdw2(w1,w2) = Flux.gradient(w->gatednpuloss(w1,w,0.5), w2)[1]
dLdw1(w1,w2) = Flux.gradient(w->npuloss(w,w2), w1)[1]
dLdw2(w1,w2) = Flux.gradient(w->npuloss(w1,w), w2)[1]

w1 = -1:0.05:2.5
w2 = -0.2:0.02:1
cmap = cgrad(:inferno, rev=true)
clim = (0.,2.5)
levels = range(clim..., length=15)
clip(v,l,u) = max(min(v,u),l)
clip(v) = clip(v,clim[1],clim[2])

function dnpu(w1,w2)
    d1,d2 = Flux.gradient(npuloss,w1,w2)
    d = sqrt(d1^2+d2^2)
    clip(d)
end
function dgated(w1,w2,g1,g2)
    d1,d2 = Flux.gradient((a,b)->gatednpuloss(a,b,g1,g2),w1,w2)
    d = sqrt(d1^2+d2^2)
    clip(d)
end
dgated(w1,w2) = dgated(w1,w2,0.5,0.5)

function dgatedx(w1,w2,i1,i2,g1,g2)
    d1,d2,d3,d4 = Flux.gradient((a,b,c,d)->gatednpuxloss(a,b,c,d,g1,g2),
                                w1,w2,i1,i2)
    d = sqrt(d1^2+d2^2+d3^2+d4^2)
end
dgatedx(i1,i2) = dgatedx(1.,0.,i1,i2,0.5,0.5)

p1 = contour(w1,w2,dnpu,title="NaiveNPU", colorbar=false,
             xlabel=L"w_1", ylabel=L"w_2",
             clim=clim,c=cmap, fill=true,levels=levels
             )
p2 = contour(w1,w2,(a,b)->dgatedx(a,b,0.,0.,0.5,0.5),title=L"NPU $g_1=g_2=0.5$", yticks=false,
             xlabel=L"w_1", colorbar=false,
             clim=clim,c=cmap, fill=true,levels=levels)
p3 = contour(w1,w2,(a,b)->dgated(a,b,1.0,0.0),title=L"NPU $g_1=1$; $g_2=0$", yticks=false,
             xlabel=L"w_1",
             clim=clim,c=cmap, fill=true,levels=levels)

scatter!(p1, [1.], [0.], m=:circle, c=:white, ms=7, label="Solution",
         xlim=(w1[1],w1[end]), ylim=(w2[1],w2[end]))
scatter!(p2, [1.], [0.], m=:circle, c=:white, ms=7, label="Solution",
         xlim=(w1[1],w1[end]), ylim=(w2[1],w2[end]))
scatter!(p3, [1.], [0.], m=:circle, c=:white, ms=7, label="Solution",
         xlim=(w1[1],w1[end]), ylim=(w2[1],w2[end]))

#plt = plot(p1,p2,layout=grid(1,2,widths=[0.465,0.5]),size=(600,300))
plt = plot(p1,p2,p3,layout=grid(1,3,widths=[0.3,0.3,0.33]),size=(900,300))
#plt = plot(p1,p2,p3,p4,p5)
savefig(plt, plotsdir("npu_gatednpu_id_grad.pdf"))
display(plt)

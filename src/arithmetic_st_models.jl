normalize_logSt(v::Real) = loggamma((v+1)/2) -loggamma(v/2) -log(π*v)/2
normalize_logSt(v::Real, σ::Real) = normalize_logSt(v) - log(σ)

function logSt(t::Real, v::Real)
    normalize_logSt(v) -(v+1)/2*log(1+t^2/v)
end

_logSt(t::Real, v::Real, σ::Real) = -(v+1)/2 * log(1 + t^2 /(v*σ^2))
_logSt(t::AbstractArray, v::Real, σ::Real) = -(v+1)/2 .* log.(1 .+ t.^2 ./(v*σ^2))

function logSt(t::Real, v::Real, σ::Real)
    normalize_logSt(v,σ) -(v+1)/2*log(1+t^2/(v*σ^2))
end

using Zygote: @adjoint, pull_block_vert
@adjoint function reduce(::typeof(vcat), As::AbstractVector{<:AbstractVecOrMat})
  cumsizes = cumsum(size.(As, 1))
  return reduce(vcat, As), Δ -> (nothing, map((sz, A) -> pull_block_vert(sz, Δ, A), cumsizes, As))
end

# @adjoint loggamma(x::Real) = loggamma(x), Δ -> Δ*digamma(x)

function logSt(t::Flux.Params, v::Real)
    n0 = normalize_logSt(v)
    ls = map(p -> -(v+1)/2 .* log.(1 .+ p.^2 ./v), t)
    sum(reduce(vcat, map(vec, ls)))
end

function logSt(t::Flux.Params, v::Real, σ::Real)
    #n0 = normalize_logSt(v,σ)
    #ls = map(p -> n0 .+ _logSt(p,v,σ), t)
    ls = map(p -> _logSt(p,v,σ), t)
    s = sum(reduce(vcat, map(vec, ls)))
end

struct StModel
    m
    v::Array{Float32,1}
    σ::Array{Float32,1}
end

using GenerativeModels: AbstractGM
using ConditionalDists: ACGaussian

include(srcdir("flux_decoder.jl"))

FDCMeanGaussian = CMeanGaussian{V,<:FluxDecoder} where V

struct BayesNet{P<:Gaussian, E<:Gaussian, D<:CMeanGaussian} <: AbstractGM
    prior::P
    encoder::E
    decoder::D
end

Flux.@functor BayesNet

function ConditionalDists.logpdf(p::ACGaussian, x::AbstractArray{T}, z::AbstractArray{T},
                                 ps::AbstractVector{T}) where T
    μ = mean(p, z, ps)
    σ2 = var(p, z)
    d = x - μ
    y = d .* d
    y = (1 ./ σ2) .* y .+ log.(σ2) .+ T(log(2π))
    -sum(y, dims=1) / 2
end

function GenerativeModels.mean(p::FDCMeanGaussian, z::AbstractArray, ps::AbstractVector)
    p.mapping(z, ps)
end

function elbo(m::BayesNet, x, y; β=1)
    ps = reshape(rand(m.encoder),:)
    llh = sum(logpdf(m.decoder, y, x, ps))
    kld = sum(kl_divergence(m.encoder, m.prior))
    llh - β*kld
end

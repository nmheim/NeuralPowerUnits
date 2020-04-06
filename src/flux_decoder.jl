struct FluxDecoder{M}
    model::M
    restructure::Function
end

FluxDecoder(m) = FluxDecoder(m, Flux.destructure(m)[2])

(d::FluxDecoder)(x::AbstractMatrix, ps::AbstractVector) = d.restructure(ps)(x)
(d::FluxDecoder)(x::AbstractMatrix) = d.model(x)

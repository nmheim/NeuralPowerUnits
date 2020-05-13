using SpecialFunctions

normalize_logSt(v::Real) = loggamma((v+1)/2) -loggamma(v/2) -log(π*v)/2
normalize_logSt(v::Real, σ::Real) = normalize_logSt(v) - log(σ)

function logSt(t::Real, v::Real)
    normalize_logSt(v) -(v+1)/2*log(1+t^2/v)
end

_logSt(t::Real, v::Real, σ::Real) = -(v+1)/2 * log(1 + t^2 /(v*σ^2))
_logSt(t::AbstractArray, v::Real, σ::Real) = -(v+1)/2 .* log.(1 .+ t.^2 ./(v*σ^2))

logSt(t, v::Real, σ::Real) = normalize_logSt(v,σ) .+ _logSt(t,v,σ)

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

(m::StModel)(x) = m.m(x)

Flux.@functor StModel (m,σ)

get_mapping(m::StModel) = m.m

function train!(loss, model::StModel, data, val_data, opt, history=MVHistory(); log=true)
    ps = params(model)
    trn_loss, val_loss, mse_loss, reg_loss = 0f0, 0f0, 0f0, 0f0

    logging = Flux.throttle((i)->(
            @info("Step $i | σ=$(model.σ[1]) $(model.v[1])", trn_loss, mse_loss, reg_loss, val_loss);
            m = get_mapping(model);
            (h,w) = size(m[1].W);
            p1 = UnicodePlots.heatmap(m[1].W[end:-1:1,:], height=h, width=w);
            display(p1);
        ),
    5)
    pushhist = Flux.throttle((i)->(
            push!(history, :μz, i, copy(Flux.destructure(model.m)[1]));
            val_loss = Flux.mse(model(val_data[1]), val_data[2]);
            push!(history, :loss, i, [trn_loss,mse_loss,reg_loss,val_loss]);
       ),
    5)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1
    try 
        @progress for d in data
            gs = gradient(ps) do
                trn_loss, mse_loss, reg_loss = loss(d...)
                return trn_loss
            end
            if log logging(i) end
            pushhist(i)
            Flux.Optimise.update!(opt, ps, gs)
            i += 1
        end
    catch e
        println("Error during training!")
        for (exc,bt) in Base.catch_stack()
            showerror(stdout,exc,bt)
            println()
        end
    end

    history
end



include(srcdir("unicodeheat.jl"))

struct SingleGatedNPU
    W::AbstractMatrix
    g::AbstractVector
end

SingleGatedNPU(in::Int, out::Int; init=Flux.glorot_uniform) =
    SingleGatedNPU(init(out,in), Flux.ones(in)/2)

Flux.@functor SingleGatedNPU

function mult(W::AbstractMatrix{T}, g::AbstractVector{T}, x::AbstractArray{T}) where T
    g = min.(max.(g, 0), 1)

    r = abs.(x) .+ eps(T)
    r = g .* r .+ (1 .- g) .* T(1)

    k = max.(-sign.(x), 0) .* T(pi)
    #k = g .* k .+ (1 .- g) .* T(0)

    z = exp.(W * log.(r)) .* cos.(W*k)
end

(l::SingleGatedNPU)(x) = mult(l.W, l.g, x)
function init_diag(T, a::Int, b::Int)
    m = zeros(T,a,b)
    m[diagind(m,0)] .= 1f0
    return m
end

function initf(s::String)
    if s == "diag"
        return (a,b) -> init_diag(Float32,a,b)
    elseif s == "glorotuniform"
        return (a,b) -> Flux.glorot_uniform(a,b)
    elseif s == "rand"
        return (a,b) -> rand(Float32,a,b)/10
    elseif s == "randn"
        return (a,b) -> randn(Float32,a,b)/10
    elseif s == "zero"
        return (a,b) -> zeros(Float32,a,b)
    elseif s== "one"
        return (a,b) -> ones(Float32,a,b)
    else
        throw(ArgumentError("Unkown init: $s"))
    end
end

function get_model(model::String, inlen::Int, fstinit::String, sndinit::String)
    if model == "npu"
        return Chain(NAU(inlen, inlen, init=initf(fstinit)),
                     NPU(inlen, 1, init=initf(sndinit)))
    elseif model == "nmu"
        return Chain(NAU(inlen, inlen, init=initf(fstinit)),
                     NMU(inlen, 1, init=initf(sndinit)))
    elseif model == "nalu"
        return Chain(NALU(inlen, inlen,
                          initNAC=initf(fstinit),
                          initG=initf(fstinit),
                          initb=Flux.zeros),
                     NALU(inlen, 1,
                          initNAC=initf(sndinit),
                          initG=initf(sndinit),
                          initb=Flux.zeros))
    elseif model == "inalu"
        return Chain(iNALU(inlen, inlen,
                          initNAC=initf(fstinit),
                          initG=initf(fstinit)),
                     iNALU(inlen, 1,
                          initNAC=initf(sndinit),
                          initG=initf(sndinit)))
    elseif model == "npux"
        return Chain(NAU(inlen, inlen, init=initf(fstinit)),
                     NPUX(inlen, 1, initRe=initf(sndinit)))
    elseif model == "gatednpu"
        nau = NAU(inlen, inlen, init=initf(fstinit))
        W   = initf(sndinit)(1,inlen)
        g   = ones(Float32, inlen) #.* Float32(0.99)
        #g   = rand(Float32, inlen)
        npu = GatedNPU(W, g)
        return Chain(nau,npu)
        # return Chain(NAU(inlen, inlen, init=initf(fstinit)),
        #              GatedNPU(inlen, 1, init=initf(sndinit)))
    elseif model == "gatednpux"
        nau = NAU(inlen, inlen, init=initf(fstinit))
        Re  = initf(sndinit)(1,inlen)
        Im  = zeros(Float32,1,inlen)
        g   = ones(Float32, inlen) #.* Float32(0.99)
        #g   = rand(Float32, inlen)
        npu = GatedNPUX(Re, Im, g)
        return Chain(nau,npu)
    else
        error("Unknown model string: $model")
    end
end

function train!(loss, model, data, val_data, opt, sch::Schedule, history=MVHistory(); log=true)
    ps = params(model)
    trn_loss, val_loss, mse_loss, reg_loss = 0f0, 0f0, 0f0, 0f0

    logging = Flux.throttle((i)->(
            @info("Step $i | β=$(sch.eta)", trn_loss, mse_loss, reg_loss, val_loss);
            m = get_mapping(model) |> cpu;
            p1 = heat(m);
            display(p1);
        ),
    5)
    pushhist = Flux.throttle((i)->(
            push!(history, :μz, i, copy(Flux.destructure(model)[1] |> cpu));
            val_loss = Flux.mse(model(val_data[1]), val_data[2]);
            push!(history, :loss, i, [trn_loss,mse_loss,reg_loss,val_loss]);
       ),
    5)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1
    try 
        @progress for d in data
            factor = step!(sch)
            gs = gradient(ps) do
                trn_loss, mse_loss, reg_loss = loss(d..., factor)
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

get_mapping(m) = m

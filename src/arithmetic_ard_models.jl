function ardnet(mapping, α0, β0, out)
    μz = Flux.destructure(mapping)[1]
    zlen = length(μz)
    λz = ones(Float32,zlen)/10
    σz = ones(Float32,zlen)/10
    σx = ones(Float32,1)

    e = Gaussian(μz, σz)
    p = Gaussian(NoGradArray(zeros(Float32,zlen)), λz)
    h = InverseGamma(α0,β0,zlen,true)
    d = CMeanGaussian{ScalarVar}(FluxDecoder(mapping), σx, out)
    ARDNet(h, p, e, d)
end

function ardnet_αβ(mapping, α0, β0, out)
    μz = Flux.destructure(mapping)[1]
    zlen = length(μz)
    λz = ones(Float32,zlen)/10
    σz = ones(Float32,zlen)/10
    σx = ones(Float32,1)

    e = Gaussian(μz, σz)
    p = Gaussian(NoGradArray(zeros(Float32,zlen)), λz)
    h = InverseGamma(α0,β0,zlen)
    d = CMeanGaussian{ScalarVar}(FluxDecoder(mapping), σx, out)
    a = ARDNet(h, p, e, d)
end


function train!(loss, model::ARDNet, data, val_data, opt, history=MVHistory())
    ps = params(model)
    tot, llh, kld, lpλ, tot_val = 0f0, 0f0, 0f0, 0f0, 0f0

    logging = Flux.throttle((i)->(
        @info "Step $i" [tot llh kld lpλ] tot_val;
        p1 = UnicodePlots.heatmap(get_mapping(model)[1].W[end:-1:1,:]);
        display(p1)
    ), 1)
    pushhist = Flux.throttle((i)->(
            push!(history, :μz, i, copy(mean(model.encoder)));
            push!(history, :σz, i, copy(var(model.encoder)));
            push!(history, :λz, i, copy(var(model.prior)));
            push!(history, :σx, i, copy(var(model.decoder,ones(1))));
            push!(history, :αβ, i, copy([model.hyperprior.α[1], model.hyperprior.β[1]]));
            tot_val = loss(val_data...);
            push!(history, :loss, i, [tot, llh, kld, lpλ, tot_val]);
       ),
    5)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1
    try 
        @progress for d in data
            gs = gradient(ps) do
                (tot, llh, kld, lpλ) = loss(d...)
                return tot
            end
            logging(i)
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

function notelbo(m::ARDNet, x, y)
    ps = reshape(rand(m.encoder),:)
    llh = sum(logpdf(m.decoder, y, x, ps))
    kld = sum(kl_divergence(m.encoder, m.prior))
    lpλ = sum(logpdf(m.hyperprior, var(m.prior)))

    _elbo = -llh + kld - lpλ
    _elbo, -llh, kld, -lpλ
end

get_mapping(m::ARDNet) = m.decoder.mapping.restructure(mean(m.encoder))

function train!(loss, model, data, opt, history=MVHistory())
    ps = params(model)
    tot, llh, kld, lpλ = 0f0, 0f0, 0f0, 0f0

    logging = Flux.throttle((i)->(@info "Step $i" [tot llh kld lpλ]), 1)
    pushhist = Flux.throttle((i)->(
            push!(history, :μz, i, copy(mean(model.encoder)));
            push!(history, :σz, i, copy(var(model.encoder)));
            push!(history, :λz, i, copy(var(model.prior)));
            push!(history, :σx, i, copy(var(model.decoder,ones(1))));
            push!(history, :loss, i, [tot, llh, kld, lpλ]);
       ),
    0.5)

    i = haskey(history,:loss) ? get(history,:loss)[1][end]+1 : 1
    for d in data
        gs = gradient(ps) do
            (tot, llh, kld, lpλ) = loss(d...)
            return tot
        end
        logging(i)
        pushhist(i)
        Flux.Optimise.update!(opt, ps, gs)
        i += 1
    end

    history
end

function notelbo(m::ARDNet, x, y; α0=1, β0=0, esamples=1)
    llh = T(0)
    for _ in 1:esamples
        ps = reshape(rand(m.encoder),:)
        llh += sum(logpdf(m.decoder, y, x, ps))
    end
    llh /= esamples
    kld = sum(kl_divergence(m.encoder, m.prior))
    lpλ = sum(logpdf(m.hyperprior, var(m.prior)))

    _elbo = -llh + kld - lpλ
    _elbo, -llh, kld, -lpλ
end

function Plots.plot(h::MVHistory; logscale=true)
    idx, ls = get(h, :loss)
    ls = reduce(hcat, ls)
    tot = ls[1,:]
    mse = ls[2,:]
    L1  = ls[3,:]
    tot_val = ls[4,:]

    yscale = logscale ? :log10 : :identity

    p1 = plot(idx, tot, label="Loss", yscale=yscale, lw=2)
    plot!(p1, idx, mse, label="MSE", lw=2)
    plot!(p1, idx, L1,  label="L1", lw=2)
    plot!(p1, idx, tot_val,  label="Val.", lw=2)

    ps = reduce(hcat, get(h, :μz)[2])'
    p2 = plot(idx, ps, legend=false, lw=2)

    plot(p1,p2,layout=(2,1))
end

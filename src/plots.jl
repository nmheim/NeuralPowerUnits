function Plots.plot(h::MVHistory; logscale=true)
    idx, ls = get(h, :loss)
    _,   ps = get(h, :μz)
    ls = reduce(hcat, ls)
    tot = ls[1,:]
    mse = ls[2,:]
    reg = ls[3,:]
    L1  = map(p->norm(p,1), ps)
    tot_val = ls[4,:]

    yscale = logscale ? :log10 : :identity

    p1 = plot(idx, tot, label="Loss", yscale=yscale, lw=2)
    plot!(p1, idx, mse, label="MSE", lw=2)
    plot!(p1, idx, reg,  label="Reg.", lw=2)
    plot!(p1, idx, L1,  label="L1.", lw=2)
    plot!(p1, idx, tot_val,  label="Val.", lw=2)

    p2 = plot(idx, reduce(hcat, ps)', legend=false, lw=2)

    plot(p1,p2,layout=(2,1))
end

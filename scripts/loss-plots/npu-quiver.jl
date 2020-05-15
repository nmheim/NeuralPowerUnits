meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))


nacmult(x,w;ϵ=1f-7) = exp(w * log(abs(x + ϵ)))

function npu(x::T, w::T; e::T=T(1e-7)) where T
    r = abs(x) + e
    k = T(x < 0 ? pi : 0.0)
    # if r < e && abs(w) < e
    #     T(1)
    # else
    #     return exp(w * log(r)) * cos(w*k)
    # end
    return exp(w * log(r)) * cos(w*k)
end;


id_mse(x,w) = abs(x - npu(x,w))

dx(x,w) = -Zygote.gradient(x->id_mse(x,w), x)[1]
dw(x,w) = -Zygote.gradient(w->id_mse(x,w), w)[1]

x = -1:0.1:2
w = -1:0.1:2
x,w = meshgrid(x,w)
u = dx.(x,w)
v = dw.(x,w)
r = sqrt.(u.^2 + v.^2) * 10
u = u ./ r
v = v ./ r

# p1 = heatmap(x, w, (x,w)->dx(x,w), clim=(-10,10))
# p2 = heatmap(x, w, (x,w)->dw(x,w), clim=(-10,10))
# plot(p1,p2,size=(1000,500))
plt = heatmap(x, w, id_mse, clim=(0,2), c=:viridis)
quiver!(plt, x, w, quiver=(u,v), c=:white)



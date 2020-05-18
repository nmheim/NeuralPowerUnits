using Flux
using Plots
pyplot()


f(x, w) = exp(w*log(x))

loss(x,w) = abs2(f(x,w) - x^2)

Flux.gradient(loss, 3, 1)

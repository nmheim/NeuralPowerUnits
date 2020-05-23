using DrWatson
@quickactivate "NIPS_2020_NMUX"

using Plots
using LatinHypercubeSampling

pyplot()


n = 1000
x = rand(n)
y = rand(n)


@info "creating plan"
plan, _ = LHCoptim(300,2,1000)
@info "scaling plan"
scaled_plan = scaleLHC(plan,[(-5.0,5.0),(-5.0,5.0)])

rosenbrock_2D(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
#samples = mapslices(rosenbrock_2D,scaled_plan; dims=2)

(x,y) = eachcol(scaled_plan)
# x = -5:0.2:5
# y = -5:0.2:5

@info "plotting"
heatmap(x,y,(x,y)->log10(rosenbrock_2D([x,y])))

# (hx,hy) = eachcol(cube)
# 
# p1 = scatter(x,y)
# p2 = scatter(hx,hy)
# 
# plot(p1,p2)

T = Float32
batch  = 50
inlen  = 4
outlen = 1
r = Uniform(-2,2)

function f(x)
    x1 = x[1,:]
    x2 = x[2,:]
    x3 = x[3,:]
    x4 = x[4,:]
    y = (x1 .+ x2)
    reshape(y, 1, :)
end

function generate()
    x = Float32.(rand(r, inlen, batch))
    y = f(x)
    (x,y)
end

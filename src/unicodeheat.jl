concat(m::Chain{<:Tuple{<:NAU,<:GatedNPUX}}) =
    cat(model[1].W[end:-1:1,:], model[2].Re', model[2].Im',dims=2)
concat(m::Chain) = cat(model[1].W[end:-1:1,:], model[2].W', dims=2)

function heat(m::Chain)
    arr = concat(m)
    (h,w) = size(arr)
    UnicodePlots.heatmap(arr,height=h,width=w)
end

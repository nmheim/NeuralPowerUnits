concat(m::Chain{<:Tuple{<:NAU,<:GatedNPUX}}) =
    cat(m[1].W, m[2].Re', m[2].Im',dims=2)

concat(m::Chain) = cat(m[1].W, m[2].W', dims=2)

function concat(m::Chain{<:Tuple{<:NALU,<:NALU,<:NALU}})
    cat(m[1].nac.M,
        m[1].nac.W,
        m[2].nac.M,
        m[2].nac.W,
        m[3].nac.M',
        m[3].nac.W', dims=2)
end

function concat(m::Chain{<:Tuple{<:GatedNPUX,<:NAU,<:GatedNPUX}})
    R1 = m[1].Re
    I1 = m[1].Im
    W  = m[2].W
    R3 = m[3].Re'
    I3 = m[3].Im'
    h  = cat(R1,I1,W,R3,I3,dims=2)
end

function concat(m::Chain{<:Tuple{<:NPUX,<:NAU,<:NPUX}})
    R1 = m[1].Re
    I1 = m[1].Im
    W  = m[2].W
    R3 = m[3].Re'
    I3 = m[3].Im'
    h  = cat(R1,I1,W,R3,I3,dims=2)
end

concat(m::NAU) = m.W

function heat(m)
    arr = concat(m)
    (h,w) = size(arr)
    UnicodePlots.heatmap(arr,height=h,width=w)
end

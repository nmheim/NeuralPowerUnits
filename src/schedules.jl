abstract type Schedule end

"""
    ExpDecay(ηstart, ηend, decay, decay_step)

Discount the parameter `η` by the factor `decay` every `decay_step` steps till
a minimum of `clip`.

# Parameters
- Parameter (`η`): parameter that is scheduled
- `decay`: Factor by which the parameter is discounted.
- `decay_step`: Schedule decay operations by setting the number of steps between
                two decay operations.
- `clip`: Minimum value of parameter.
"""
mutable struct ExpDecay{T<:Real} <: Schedule
  eta::T
  clip::T
  decay::T
  step::Int
  current::Int
end

ExpDecay(ηstart::T, ηstop::T, decay::T, decay_step::Int) where T =
    ExpDecay(ηstart, ηstop, decay, decay_step, 0)

function step!(sch::ExpDecay)
  η, s, decay = sch.eta, sch.step, sch.decay
  n = sch.current = sch.current + 1
  if n%s == 0
    η = max(η * decay, sch.clip)
    sch.eta = η
  end
  return sch.eta
end


mutable struct ExpGrowth{T<:Real} <: Schedule
  eta::T
  clip::T
  growth::T
  step::Int
  current::Int
end

ExpGrowth(ηstart::T, ηstop::T, growth::T, growth_step::Int) where T =
    ExpGrowth(ηstart, ηstop, growth, growth_step, 0)

function step!(sch::ExpGrowth)
  η, s, growth = sch.eta, sch.step, sch.growth
  n = sch.current = sch.current + 1
  if n%s == 0
    η = min(η * growth, sch.clip)
    sch.eta = η
  end
  return sch.eta
end


function ExpSchedule(ηstart::T, ηend::T, factor::T, step::Int) where T
    if ηstart < ηend && factor > 1
        return ExpGrowth(ηstart, ηend, factor, step)
    elseif (ηstart >= ηend && factor <= 1)
        return ExpDecay(ηstart, ηend, factor, step)
    else
        error("ηstart, ηend, and exponential factor are inconsistent.")
    end
end

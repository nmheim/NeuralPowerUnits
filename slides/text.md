# Slide 0
HI! I am Niklas, an author of the paper **Neural Power Units**, which was
written together with Vasek & Tomas at AIC in Prague.

# Slide 1
We looked at a new reasearch area sometimes called **arithmetic
extrapolation**/**neural arithmetic**, which tries to increase the
extrapolation performance of neural nets on certain tasks. NA assumes that the
underlying function is *partially* composed of arithmetic operations and uses this
bias to extrapolate beyond the training range.
Possible applications lie for example in the areas of financial and physical
modelling as well as equation discovery.

# Slide 2
NA was initiated in 2018 with the NALU by Trask et al. It can perform
addition/mult and division, but has problems with convergence and cannot handle
negative numbers.

# Slide 3
This year Madsen & Johansen published the NMU and the NAU, which converge
consistently for addition and multiplication. The NMU correctly handles
negative numbers but cannot perform divsion.

# Slide 4
In our work, we took the multiplication path of NALU and lifted it to complex
space. This enables correct processing of negative numbers and can peform
multiplication, division, and fractional powers.

# Slide 5
After some algebra one arrives at the NaiveNPU, which you can see here.  Note
that the NaiveNPU can be used as a pulg-and-play replacement for another layer
without requiring the rest of the network to be complex.
Unfortunately, the NaiveNPU suffers from similar convergence issues as the NALU
for small inputs, which is why we introduced the **relevance gate**.

# Slide 6
The relevance gate (shaded in grey) can choose to ignore small inputs, which
helps with convergence.

# Slide 7
In our experiments we compared the currently available arithmetic layers.
The plot shows the predicition error for the different models trained to learn
the function `f`. Bright colors indicate a low error. We trained the models
with examples in a range from 0 to 2, which can be seen nicely in the `Dense`
network plots.  Inside the training range the dense net performs great and
outside it deteriorates.  NALU also has problems to extrapolate, due to the
convergence issues mentioned earlier.  The NMU is best at addition and
multiplication, but fails at div/sqrt.  Only the NPU manages to learn all
operations including the fractinoal power `sqrt` which is a fractional power.

# Slide 8
Finally, we want to show how the NPU could be used for equation
discovery.  Note that our approach is not a finished EQD framework, much more
verification is needed for real world application.  However, we can try to
learn an ODE like the fSIR model shown here.
It features a prodcut of franctional powers of the variables S and I.

# Slide 9
We trained an NPU without complex weights (called RealNPU)
with L1 regularization on a trajectory from the fSIR model.
ONe can see here that the RealNPU actually discovers a product of S and I.
The following NAU looks very similar to the matrix above.

# Slide 10
Thank you slide with links to repo/papers.

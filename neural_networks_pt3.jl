# Neural Networks from Scratch Pt.3 -> Dot Product
# Also transitioning from lists into vectors and matricies.

inputs  = [1, 2, 3, 2.5]

weights = [0.2 0.8 -0.5 1.0;
           0.5 -0.91 0.26 -0.5;
           -0.26 -0.27 0.17 0.87]

biases = [2, 3, 0.5]


layer_outputs = []  # Output of current layer

for (neuron_weights, neuron_bias) in zip(weights, biases)
    neuron_output = 0   # Output of given neuron

    for (n_input, weight) in zip(inputs, neuron_weights)
        neuron_output += n_input*weight
    end

    neuron_output += neuron_bias
    append!(layer_outputs, neuron_output)
end

println(layer_outputs)


# The Dot Product multiplies vectors element-wise.
# e.g. [1,2,3] [4,5,6] would be 1*4 + 2*5 + 3*6
# In Julia the euqivilant of Numpy is LinearAlgebra
using LinearAlgebra

# Inputs to neuron:
inputs  = [1, 2, 3, 2.5]

weights = [0.2 0.8 -0.5 1.0;
           0.5 -0.91 0.26 -0.5;
           -0.26 -0.27 0.17 0.87]

biases = [2, 3, 0.5]

# Note: dot() in Julia is for Vectors. Matrix * Matrix inner product is done like below.
# Ref: https://web.stanford.edu/class/engr108/julia_slides/julia_matrices_slides.pdf
output = weights * inputs + biases
println(output)

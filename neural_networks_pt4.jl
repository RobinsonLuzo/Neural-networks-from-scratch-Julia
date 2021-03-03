# Neural Networks from Scratch Pt.4 -> Batches, Layers and Objects

# In the below example we need to transpose weights so we can get the dot product of it and inputs.
# As defined: inputs and weights are both 3x4. We have to convert weights into 4x3.
using LinearAlgebra
using Random

Random.seed!(0)

inputs  = [1 2 3 2.5;
           2.0 5.0 -1.0 2.0;
          -1.5 2.7 3.3 -0.8]

weights = [0.2 0.8 -0.5 1.0;
           0.5 -0.91 0.26 -0.5;
           -0.26 -0.27 0.17 0.87]       

biases = [2 3 0.5]

# 2nd layer:
weights2 = [0.1 -0.14 0.5;
           -0.5 0.12 -0.33;
           -0.44 0.73 -0.13]       

biases2 = [-1 2 -0.5]

# Neuron layers:
layer1_outputs = inputs * weights' .+ biases
layer2_outputs = layer1_outputs * weights2' .+ biases2
#println(layer2_outputs)


# Now we create a layer node
X  = [1 2 3 2.5;
    2.0 5.0 -1.0 2.0;
    -1.5 2.7 3.3 -0.8]

# Julia doesn't have classes so create a struct and methods to operate on the data in there.
mutable struct Layer_Dense
    weights::Array{Float64, 2}
    biases::Vector{Float64}
end

"""
Initialize the struct Layer_Dense.

Arguments:
- n_inputs: size of inputs coming in.
- n_neurons: number of neurons wanted.

Returns:
- weights: a matrix of random Gaussian distribution of numbers, size: n_inputs, n_neurons.
- biases: matrix initialized as 0s of size n_neurons.

"""
function Layer_Dense(n_inputs::Int, n_neurons::Int)
    weights = 0.10 * randn(n_inputs, n_neurons)
    biases = zeros(n_neurons)

    return Layer_Dense(weights, biases)
end

"""
Derives the output of a layer of neurons.

Arguments:
- layer: a LayerDense consisting of a matrix of weights and matrix of biases.
- inputs: an input matrix. Can be inital matrix or output of a previous forward().

"""
function forward(layer::Layer_Dense, inputs)
    return inputs * layer.weights .+ layer.biases'
end


# Layer creation and outputs:
# Input of a layer must match the output of the previous layer.
layer1 = Layer_Dense(4,5)   # Input: 4, Output:5
layer2 = Layer_Dense(5,2)   # Input: 5, Output:2

output1 = forward(layer1, X)
output2 = forward(layer2, output1)

#println(output2)
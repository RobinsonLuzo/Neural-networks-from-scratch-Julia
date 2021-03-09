# Neural Networks from Scratch Pt.2 -> Hidden Layer Activation Functions

using LinearAlgebra
using Random

Random.seed!(0)

X  = [1 2 3 2.5;
    2.0 5.0 -1.0 2.0;
    -1.5 2.7 3.3 -0.8]

inputs = [0 2 -1 3.3 -2.7 1.1 2.2 -100]

outputs = []

for i in inputs
    if i > 0
        append!(outputs, i)
    elseif i <= 0
        append!(outputs, 0)
    end
end

println(outputs)


# Spiral_data method counterpart to Python.
# Optimised by borrowing for spiral macro from: https://github.com/Sentdex/NNfSiX/blob/master/Julia/p005-ReLU-Activation.jl

function spiral_data(points::Int, classes::Int)
    X = zeros(Float64, points * classes, 2)
    y = zeros(Int64, points * classes)

    for class_number = 1:classes
        ix = points*(class_number-1)+1:points*class_number
        r = range(0, 1, length=points)
        t = range((class_number-1)*4, class_number*4, length=points) .+ randn(points)*0.2

        # The @. macro makes all calculations element-wise.
        @. X[ix, :] = [r*sin(t*2.5) r*cos(t*2.5)]
        @. y[ix] = class_number
    end
    return X, y
end

(X, y) = spiral_data(100, 3)



# Julia doesn't have classes, so instead we make a struct.
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
function Layer_Dense(n_inputs, n_neurons)
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


"""
Activation function.

Takes output from layer of neurons and produces the activation for the entire layer.
"""
function activation_ReLU(inputs)

    # Using the dot takes the element-wise maximum.
    return max.(0., inputs)
end


# First layer from our input data.
# Number of input, number of neurons
layer1 = Layer_Dense(2, 5)

output1 = forward(layer1, X)
activation_output1 = activation_ReLU(output1)

println(activation_output1)
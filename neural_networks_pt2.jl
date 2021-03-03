# Neural Networks from Scratch Pt.2 -> Neural Network layer


# Input to neuron:
inputs  = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias    = 2

# resulting output from neuron
output = inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + inputs[4]*weights[4] + bias
println(output)
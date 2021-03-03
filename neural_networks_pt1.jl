# Neural Networks from Scratch Pt.1 -> Neural Network code

# Creating an individual neuron somewhere in a network.
# Consists of 3 neurons in previous layer and 1 output.

# Input to neuron:
# Every unique neron has it's own bias
inputs  = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias    = 3

# resulting output from neuron
output = inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
println(output)
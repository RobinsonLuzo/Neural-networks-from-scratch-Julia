# Neural Networks from Scratch Pt.2 -> Neural Network layer

# We will create 4 inputs into 3 neurons -> as if it were the final layer.

# Inputs to neurons:
inputs  = [1, 2, 3, 2.5]

weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1    = 2
bias2    = 3
bias3    = 0.5

# resulting output from neurons
neuron1_output = inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + inputs[4]*weights1[4] + bias1
neuron2_output = inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + inputs[4]*weights2[4] + bias2
neuron3_output = inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + inputs[4]*weights3[4] + bias3

# Final output
output = [neuron1_output, neuron2_output, neuron3_output]
println(output)
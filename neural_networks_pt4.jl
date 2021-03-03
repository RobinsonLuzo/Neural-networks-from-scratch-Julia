# Neural Networks from Scratch Pt.4 -> Batches, Layers and Objects

# In the below example we need to transpose weights so we can get the dot product of it and inputs.
# As defined: inputs and weights are both 3x4. We have to convert weights into 4x3.
using LinearAlgebra

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
import numpy as np
import random
class Network:
    def __init__(self, layers):
        """
        Initialize a Network with a list of layers.
        
        :param layers: List of Layer objects that make up the network.
        """
        self.layers = layers

    def forward(self, inputs):
        """
        Perform a forward pass through the network.
        
        :param inputs: Input data to the network.
        :return: Output of the network after passing through all layers.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, inputs, output_gradient, learning_rate):
        """
        Perform a backward pass through the network.
        
        :param inputs: Input data to the network.
        :param output_gradient: Gradient of the loss with respect to the output of the network.
        :return: Gradient of the loss with respect to the inputs of the network.
        """
        for layer in reversed(self.layers):
            output_gradient = layer.backward(inputs, output_gradient, learning_rate)
            inputs = layer.forward(inputs)
        return output_gradient
    
    def calculate_loss(self, output, target):
        """
        Calculate the loss between the network output and the target.
        
        :param output: Output of the network.
        :param target: Target values for comparison.
        :return: Loss value.
        """
        return np.mean((output - target) ** 2)
 
    def train(self, training_data, epochs, learning_rate):
        """
        Train the network using the provided training data.
        
        :param training_data: List of tuples (input, target) for training.
        :param epochs: Number of epochs to train the network.
        :param learning_rate: Learning rate for weight updates.
        """
        for epoch in range(epochs):
            random.shuffle(training_data)
            for input_data, target in training_data:
                output = self.forward(input_data)
                loss_gradient = output - target
                self.backward(input_data, loss_gradient, learning_rate)
                
    def __str__(self):
        """
        String representation of the Network object.
        
        :return: String representation of the network's layers.
        """
        return "\n".join(str(layer) for layer in self.layers)
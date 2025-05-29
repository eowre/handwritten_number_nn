import numpy as np
class Layer:
    def __init__(self, number_inputs, number_neurons):
        """
        Initialize a Layer with a specified number of inputs and neurons.
        
        :param number_inputs: Number of inputs to the layer.
        :param number_neurons: Number of neurons in the layer.
        """
        self.weights = np.random.rand(number_neurons, number_inputs)
        self.biases = np.random.rand(number_neurons)
        self.number_neurons = number_neurons
        self.number_inputs = number_inputs

    def forward(self, inputs):
        """
        Perform a forward pass through the layer.
        
        :param inputs: Input data to the layer.
        :return: Output of the layer after applying weights and biases.
        """
        z = np.dot(self.weights, inputs) + self.biases
        return self.sigmoid(z)
    
    def backward(self, inputs, output_gradient, learning_rate):
        """
        Perform a backward pass through the layer.
        
        :param inputs: Input data to the layer.
        :param output_gradient: Gradient of the loss with respect to the output of this layer.
        :return: Gradient of the loss with respect to the inputs of this layer.
        """
        z = np.dot(self.weights, inputs) + self.biases
        sigmoid_derivative = self.sigmoid_derivative(z)
        delta = output_gradient * sigmoid_derivative

        weights_gradient = np.outer(delta, inputs)
        biases_gradient = delta

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return np.dot(self.weights.T, delta)
    
    def __str__(self):
        """
        String representation of the Layer object.
        :return: String representation of the layer's weights and biases.
        """
        return f"Layer(weights={self.weights}, biases={self.biases})"
    
    def sigmoid(self, x):
        """
        Apply the sigmoid activation function to the input.
        
        :param x: Input data to the activation function.
        :return: Output after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Compute the derivative of the sigmoid function.
        
        :param x: Input data to the derivative function.
        :return: Derivative of the sigmoid function applied to x.
        """
        sig = self.sigmoid(x)
        return sig * (1 - sig)
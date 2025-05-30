import numpy as np
class Layer:
    def __init__(self, number_inputs, number_neurons, activation="sigmoid"):
        """
        Initialize a Layer with a specified number of inputs and neurons.
        
        :param number_inputs: Number of inputs to the layer.
        :param number_neurons: Number of neurons in the layer.
        """
        self.weights = np.random.rand(number_neurons, number_inputs) * np.sqrt(2. / number_inputs)
        self.biases = np.zeros(number_neurons)
        self.activation = activation
        self.number_neurons = number_neurons
        self.number_inputs = number_inputs

    def forward(self, inputs):
        """
        Perform a forward pass through the layer.
        
        :param inputs: Input data to the layer.
        :return: Output of the layer after applying weights and biases.
        """
        self.last_input = inputs
        self.last_z = np.dot(self.weights, inputs) + self.biases

        if self.activation == "sigmoid":
            return self.sigmoid(self.last_z)
        elif self.activation == "softmax":
            return self.softmax(self.last_z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def backward(self, inputs, output_gradient, learning_rate):
        """
        Perform a backward pass through the layer.
        
        :param inputs: Input data to the layer.
        :param output_gradient: Gradient of the loss with respect to the output of this layer.
        :return: Gradient of the loss with respect to the inputs of this layer.
        """
        if self.activation == "sigmoid":
            sigmoid_derivative = self.sigmoid_derivative(self.last_z)
            delta = output_gradient * sigmoid_derivative
        elif self.activation == "softmax":
            delta = output_gradient

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
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    """
    FOF FUTURE BATCH TRAINING 
        shift_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    """

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
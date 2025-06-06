import numpy as np

class Layer:
    def __init__(self, number_inputs, number_neurons, activation="sigmoid"):
        """
        Initialize a Layer with a specified number of inputs and neurons.
        
        :param number_inputs: Number of inputs to the layer.
        :param number_neurons: Number of neurons in the layer.
        """
        self.biases = np.zeros(number_neurons)
        self.activation = activation
        self.number_neurons = number_neurons
        self.number_inputs = number_inputs
        
        if activation == "relu":
            # He initialization
            self.weights = np.random.randn(number_neurons, number_inputs) * np.sqrt(2. / number_inputs)
        elif activation == "softmax":
            # Xavier initialization
            self.weights = np.random.randn(number_neurons, number_inputs) * np.sqrt(1. / number_inputs)
        else:
            # Small random weights for sigmoid/tanh
            self.weights = np.random.randn(number_neurons, number_inputs) * 0.01

    def forward(self, inputs):
        """
        Perform a forward pass through the layer.
        
        :param inputs: Input data to the layer.
        :return: Output of the layer after applying weights and biases.
        """
        self.last_input = inputs
        # Clip to prevent overflow in exp for activations
        self.last_z = np.clip(np.dot(self.weights, inputs) + self.biases, -100, 100)

        if self.activation == "sigmoid":
            return self.sigmoid(self.last_z)
        elif self.activation == "softmax":
            return self.softmax(self.last_z)
        elif self.activation == "relu":
            return self.relu(self.last_z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
    
    def backward(self, output_gradient, learning_rate, clip_norm=1.0):
        """
        Perform a backward pass through the layer.
        
        :param output_gradient: Gradient of the loss w.r.t. the output of this layer.
                                For softmax + cross-entropy, this should be (y_pred - y_true).
        :param learning_rate: Learning rate for weight updates.
        :param clip_norm: Maximum norm for gradient clipping.
        :return: Gradient of the loss w.r.t. the inputs of this layer.
        """
        if self.activation == "sigmoid":
            sigmoid_derivative = self.sigmoid_derivative(self.last_z)
            delta = output_gradient * sigmoid_derivative
        elif self.activation == "relu":
            relu_derivative = self.relu_derivative(self.last_z)
            delta = output_gradient * relu_derivative
        elif self.activation == "softmax":
            # For softmax + cross-entropy, output_gradient is already (y_pred - y_true)
            delta = output_gradient
        else:
            raise ValueError(f"Unsupported activation in backward: {self.activation}")

        weights_gradient = np.outer(delta, self.last_input)
        biases_gradient = delta

        # Gradient clipping
        grad_norm = np.linalg.norm(weights_gradient)
        if grad_norm > clip_norm:
            weights_gradient = weights_gradient * (clip_norm / grad_norm)
            biases_gradient = biases_gradient * (clip_norm / grad_norm)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        # Return gradient for previous layer inputs
        return np.dot(self.weights.T, delta)
    
    def __str__(self):
        return f"Layer(weights={self.weights}, biases={self.biases})"
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def softmax(self, x):
        shift_x = x - np.max(x)  # numerical stability
        exp_x = np.exp(shift_x)
        return exp_x / (np.sum(exp_x) + 1e-9)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
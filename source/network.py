import numpy as np
import layer
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

    def __str__(self):
        """
        String representation of the Network object.
        
        :return: String representation of the network's layers.
        """
        return "\n".join(str(layer) for layer in self.layers)
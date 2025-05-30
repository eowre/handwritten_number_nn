import numpy as np
import random
import time
from async_logger import BufferedLogger, AsyncLogger

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
        self.inputs = [inputs]
        for layer in self.layers:
            inputs = layer.forward(inputs)
            self.inputs.append(inputs)
        return inputs
    
    def backward(self, output_gradient, learning_rate):
        """
        Perform a backward pass through the network.
        
        :param inputs: Input data to the network.
        :param output_gradient: Gradient of the loss with respect to the output of the network.
        :return: Gradient of the loss with respect to the inputs of the network.
        """
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            inputs = self.inputs[i]
            output_gradient = layer.backward(inputs, output_gradient, learning_rate)
        return output_gradient
    
    def calculate_loss(self, output, target):
        """
        Calculate the loss between the network output and the target.
        
        :param output: Output of the network.
        :param target: Target values for comparison.
        :return: Loss value.
        """
        return -np.sum(target * np.log(output + 1e-12))
 
    def train(self, training_data, epochs, learning_rate):
        """
        Train the network using the provided training data.
        
        :param training_data: List of tuples (input, target) for training.
        :param epochs: Number of epochs to train the network.
        :param learning_rate: Learning rate for weight updates.
        """
        _logger = AsyncLogger("training_async.log")

        batch_size = 1000
        for epoch in range(epochs):
            random.shuffle(training_data)
            total_loss = 0
            correct = 0
            batch_start_time = time.time()

            for i, (input_data, target) in enumerate(training_data):
                output = self.forward(input_data)
                loss = self.calculate_loss(output, target)
                loss_gradient = output - target
                self.backward(loss_gradient, learning_rate)
                total_loss += loss

                prediction = np.argmax(output)
                true = np.argmax(target)
                if prediction == true:
                    correct += 1

                if (i + 1) % 1000 == 0 or (i + 1) == len(training_data):
                    batch_end_time = time.time()    
                    batch_duration = batch_end_time - batch_start_time
                    images_left = len(training_data) - (i + 1)
                    accuracy = correct / (i + 1)
                    _logger.log(
                        f"Epoch {epoch + 1}/{epochs} | "
                        f"Processed: {i + 1}/{len(training_data)} | "
                        f"Average Loss: {total_loss / (i + 1):.4f} | "
                        f"Accuracy: {accuracy*100:.4f}% | "
                        f"Images Left: {images_left} | "
                        f"Batch Time: {batch_duration:.2f}s " 
                    )
                    batch_start_time = time.time()  # reset for next batch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_data):.4f}")
            _logger.log(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_data):.4f}")
            
                
                
    def __str__(self):
        """
        String representation of the Network object.
        
        :return: String representation of the network's layers.
        """
        return "\n".join(str(layer) for layer in self.layers)
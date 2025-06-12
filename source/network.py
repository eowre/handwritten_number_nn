import numpy as np
import random
import time
import pickle
from visHandler import visHandler
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
            output_gradient = layer.backward(output_gradient, learning_rate)
        return output_gradient
    
    def calculate_loss(self, output, target):
        """
        Calculate the loss between the network output and the target.
        
        :param output: Output of the network.
        :param target: Target values for comparison.
        :return: Loss value.
        """
        return -np.sum(target * np.log(output + 1e-12))
 
    def train(self, training_data, epochs, learning_rate, log_file_name="training_async"):
        """
        Train the network using the provided training data.
        
        :param training_data: List of tuples (input, target) for training.
        :param epochs: Number of epochs to train the network.
        :param learning_rate: Learning rate for weight updates.
        """
        _logger = AsyncLogger(f"logs/{log_file_name}.log")

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
        _logger.close()  # Close the logger at the end of training
        print("Training complete.") 
    def test(self, test_data, visualize=False, num_visuals=5):
        """
        Test the network using the provided test data.
        
        :param test_data: List of tuples (input, target) for testing.
        :param visualize: Whether to visualize the predictions.
        :return: Accuracy of the network on the test data.
        """
        correct = 0
        total = len(test_data)
        
        for i, (input_data, label) in enumerate(test_data):
            output = self.forward(input_data)
            prediction = np.argmax(output)
            true_label = np.argmax(label)

            if prediction == true_label:
                correct += 1

            if visualize and i < num_visuals:
                handler = visHandler(input_image=input_data, label=true_label)
                handler.show_prediction(output)
        
        accuracy = correct / len(test_data)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy
                
                
    def __str__(self):
        """
        String representation of the Network object.
        
        :return: String representation of the network's layers.
        """
        return "\n".join(str(layer) for layer in self.layers)

    def save(self, file_path):
        """
        Save the network to a file.
        
        :param file_path: Path to the file where the network will be saved.
        """
        parameters = []
        for layer in self.layers:
            params = {
                'weights': layer.weights,
                'biases': layer.biases,
                'activation_function': layer.activation,
                'number_inputs': layer.number_inputs,
                'number_neurons': layer.number_neurons
            }
            parameters.append(params)
        with open(file_path, 'wb') as f:
            pickle.dump(parameters, f)
    
    @classmethod
    def load(cls, file_path):
        """
        Load the network from a file.
        
        :param file_path: Path to the file from which the network will be loaded.
        """
        from layer import Layer  # Ensure Layer is imported from the correct module
        with open(file_path, 'rb') as f:
            parameters = pickle.load(f)
        layers = []
        for params in parameters:
            layer = Layer(
                number_neurons=params['number_neurons'],
                number_inputs=params['number_inputs'],
                activation=params['activation_function']
            )
            layer.weights = params['weights']
            layer.biases = params['biases']
            layers.append(layer)
        # print(f"Network loaded from {file_path}")
        return cls(layers)
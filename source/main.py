import struct
import numpy as np
import layer
import network

def parse_idx1_ubyte(file_path):
    """
    Parse IDX1 byte file format (labels).
    """
    try:
        with open(file_path, 'rb') as f:
            magic, num_items = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f'Invalid magic number {magic} for IDX1 file.')

            data = np.frombuffer(f.read(), dtype=np.uint8)
            if data.shape[0] != num_items:
                raise ValueError(f'Expected {num_items} labels, got {data.shape[0]}.')

            return data

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    
def parse_idx3_ubyte(file_path):
    """
    Parse IDX3 byte file format (images).
    """
    try:
        with open(file_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f'Invalid magic number {magic} for IDX3 file.')

            data = np.frombuffer(f.read(), dtype=np.uint8)
            expected_size = num_images * rows * cols
            if data.shape[0] != expected_size:
                raise ValueError(f'Expected {expected_size} bytes, got {data.shape[0]}.')

            return data.reshape((num_images, rows, cols))

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    
def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot

def main():

    # Load MNIST dataset
    train_images_file_path = "/home/eowre/Documents/Projects/Python/handwritten_number_nn/Data/archive/train-images.idx3-ubyte"
    train_labels_file_path = "/home/eowre/Documents/Projects/Python/handwritten_number_nn/Data/archive/train-labels.idx1-ubyte"
    test_images_file_path = "/home/eowre/Documents/Projects/Python/handwritten_number_nn/Data/archive/t10k-images.idx3-ubyte"
    test_labels_file_path = "/home/eowre/Documents/Projects/Python/handwritten_number_nn/Data/archive/t10k-labels.idx1-ubyte"
    train_images = parse_idx3_ubyte(train_images_file_path)  # Load training images
    train_labels = parse_idx1_ubyte(train_labels_file_path)  # Load training labels
    test_images = parse_idx3_ubyte(test_images_file_path)  # Load test images
    test_labels = parse_idx1_ubyte(test_labels_file_path)  # Load test labels
    if train_images is None or train_labels is None or test_images is None or test_labels is None:
        print("Error loading dataset. Please check the file paths and formats.")
        return

    # Normalize the images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Flatten the images: (num_samples, 28, 28) → (num_samples, 784)
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    train_labels_one_hot = one_hot_encode(train_labels, num_classes=10)
    test_labels_one_hot = one_hot_encode(test_labels, num_classes=10)


    # Create a simple neural network
    layers = [
        layer.Layer(784, 728),  # Input layer with 784 inputs (28x28 pixels) and 728 neurons
        layer.Layer(728, 16),   # Hidden layer with 16 neurons
        layer.Layer(16, 16),   # Hidden layer with 16 neurons
        layer.Layer(16, 10, activation="softmax")     # Output layer with 10 neurons (one for each digit)
    ]
    
    neural_network = network.Network(layers)
    


    neural_network.train(
        training_data=list(zip(train_images, train_labels_one_hot)),
        epochs=10,  # Number of epochs to train
        learning_rate=0.01  # Learning rate for weight updates
    )

    # Evaluate the network on the test set
    correct_predictions = 0
    for i in range(len(test_images)):
        output = neural_network.forward(test_images[i])
        predicted_label = np.argmax(output)
        true_label = test_labels[i]  # already an integer
        if predicted_label == true_label:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_images)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
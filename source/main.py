import numpy as np
import layer
import network


def parse_idx1_ubyte(file_path):
    """
    Parse IDX1 byte file format.
    
    :param file_path: Path to the IDX1 byte file.
    :return: Parsed data as a numpy array.
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        magic = int.from_bytes(data[0:4], byteorder='big')
        dims = int.from_bytes(data[4:8], byteorder='big')
        offset = 8

        size = int.from_bytes(data[offset:offset + 4], byteorder='big')
        offset += 4

        data = np.frombuffer(data[offset:], dtype=np.uint8).reshape(size)
        return data

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    
def parse_idx3_ubyte(file_path):
    """
    Parse IDX3 byte file format.
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            
        print(f"File size: {len(data)} bytes")
        print(f"Magic number: {int.from_bytes(data[0:4], byteorder='big'):08x}")
        print(f"Number of dimensions: {int.from_bytes(data[4:8], byteorder='big')}")
        
        dims = int.from_bytes(data[4:8], byteorder='big')
        offset = 8
        
        shape = []
        for i in range(dims):
            dim_size = int.from_bytes(data[offset:offset + 4], byteorder='big')
            print(f"Dimension {i + 1} size: {dim_size}")
            shape.append(dim_size)
            offset += 4
            
        print(f"Final shape: {shape}")
        data_array = np.frombuffer(data[offset:], dtype=np.uint8).reshape(shape)
        print(f"Loaded array shape: {data_array.shape}")
        return data_array

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error parsing {file_path}")
        print(f"Error details: {str(e)}")
        print(f"Error type: {type(e)}")
        return None
def main():

    # Load MNIST dataset
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

    # Create a simple neural network
    layers = [
        layer.Layer(784, 728),  # Input layer with 784 inputs (28x28 pixels) and 728 neurons
        layer.Layer(728, 16),   # Hidden layer with 16 neurons
        layer.Layer(16, 16),   # Hidden layer with 16 neurons
        layer.Layer(16, 10)     # Output layer with 10 neurons (one for each digit)
    ]
    
    neural_network = network.Network(layers)

    neural_network.train(
        training_data=list(zip(train_images, train_labels)),
        epochs=10,  # Number of epochs to train
        learning_rate=0.01  # Learning rate for weight updates
    )

    # Evaluate the network on the test set
    correct_predictions = 0
    for i in range(len(test_images)):
        output = neural_network.forward(test_images[i])
        predicted_label = np.argmax(output)
        if predicted_label == test_labels[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_images)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
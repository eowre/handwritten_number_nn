import numpy as np 
import matplotlib.pyplot as plt

class visHandler:
    def __init__(self, input_image=None, label=None):
        """
        Initializes the VisHandler with an optional input image and label.
        :param input_image: The image to be visualized.
        :param label: The label associated with the image.
        """
        self.input_image = input_image
        self.label = label

    def show_prediction(self, output_probabilities):
        """
        Displays the input image and the predicted label with probabilities.
        :param output_probabilities: The probabilities of each class.
        """
        predicted_Digit = np.argmax(output_probabilities)
        confidence = output_probabilities[predicted_Digit]

        print(f"Predicted Digit: {predicted_Digit} with confidence: {confidence:.2f}")

        if self.label is not None:
            print(f"Actual Label: {self.label}")
         
        print("\nProbablities")
        for i, prob in enumerate(output_probabilities):
            print(f"Digit {i}: {prob:.2f}")

        if self.input_image is not None:
            plt.imshow(self.input_image.reshape(28,28), cmap='gray')
            plt.title(f"Predicted: {predicted_Digit}, Actual: {self.label if self.label is not None else 'N/A'}")
            plt.axis('off')
            plt.show()
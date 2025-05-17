import cv2
import joblib
import numpy as np
from PIL import Image

from .core import extract_signature


class SignatureExtractor:
    def __init__(self, model_path="models/decision-tree.pkl"):
        # Load the pre-trained model
        self.model = joblib.load(model_path)

    def detect_signature(self, img_path):
        im = cv2.imread(img_path, 0)
        mask = extract_signature(im, self.model, preprocess=True)
        im = cv2.imread(img_path)
        im[np.where(mask == 255)] = (0, 0, 255)
        # Draw bounding box on image
        points = np.argwhere(mask == 255)  # find where the black pixels are
        points = np.fliplr(
            points
        )  # store them in x,y coordinates instead of row,col indices
        x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def predict(self, input_image_path, output_image_path):
        try:
            image_array = self.detect_signature(input_image_path)
            cv2.imwrite(output_image_path, image_array)
        except Exception as e:
            print(f"Error processing {input_image_path}: {e}")


if __name__ == "__main__":
    app = SignatureExtractor()
    app.predict(
        r"/content/signature-extraction/images/img1.jpg",
        "output_signature.png",
    )

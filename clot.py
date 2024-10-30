import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class ClotDetector:
    def __init__(self, resnet_weights_path='D:\\testindian\\resnet18-f37072fd.pth', custom_weights_path='D:\\testindian\\clot.pth'):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the ResNet18 model architecture
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)  # Binary classification head
        
        # Load the weights from the specified paths
        self.load_weights(resnet_weights_path, custom_weights_path)

        # Move model to the device and set it to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define the same transform used during training
        self.transform = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_weights(self, resnet_weights_path, custom_weights_path):
        """Load model weights from the specified paths."""
        try:
            # Load the ResNet18 pretrained weights
            state_dict = torch.load(resnet_weights_path, map_location=self.device)
            
            # Filter out the keys that do not match the current model's architecture
            filtered_state_dict = {k: v for k, v in state_dict.items() if k not in ['fc.weight', 'fc.bias']}
            
            self.model.load_state_dict(filtered_state_dict, strict=False)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error loading the ResNet model weights: {e}")
            exit(1)

        try:
            # Load custom classifier weights
            custom_state_dict = torch.load(custom_weights_path, map_location=self.device)
            self.model.load_state_dict(custom_state_dict, strict=False)
        except RuntimeError as e:
            print(f"Error loading the custom model state dictionary: {e}")
            exit(1)

    def predict(self, image_path):
        """
        Perform inference on a single image.
        
        Args:
            image_path (str): Path to the input image.
        
        Returns:
            int: Predicted label (0 or 1).
            float: Confidence score (between 0 and 1).
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image).view(-1)
            confidence = torch.sigmoid(output).item()

        prediction = 1 if confidence >= 0.5 else 0
        return prediction, confidence

# Example usage
if __name__ == '__main__':
    detector = ClotDetector()
    image_path = '827.jpg'  # Replace with your image path
    prediction, confidence = detector.predict(image_path)

    print(f"Predicted Label: {prediction}, Confidence: {confidence:.4f}")

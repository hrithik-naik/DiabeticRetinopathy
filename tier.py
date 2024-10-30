import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from timm import create_model
from torchvision import models

# --- DEVICE SETUP ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
NUM_CLASSES = 5

# --- ENSEMBLE CLASSIFIER ---
class EnsembleClassifier:
    def __init__(self, weights_path='best_ensemble_model.pth'):
        self.device = DEVICE
        self.num_classes = NUM_CLASSES
        self.img_size = IMG_SIZE
        self.model = self.build_model()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device), strict=False)
        self.model.eval()
    
    def build_model(self):
        # Build the ensemble model with EfficientNet and ViT
        class EnsembleModel(nn.Module):
            def __init__(self):
                super(EnsembleModel, self).__init__()
                # Load EfficientNet-b0
                self.efficientnet = models.efficientnet_b0(pretrained=True)
                self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, NUM_CLASSES)
                
                # Load ViT
                self.vit = create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)

            def forward(self, x):
                # Pass through both models and average the outputs
                out1 = self.efficientnet(x)
                out2 = self.vit(x)
                return (out1 + out2) / 2  # Averaging the outputs

        return EnsembleModel().to(self.device)

    def get_inference_transforms(self):
        # Return the transformations needed for inference
        return Compose([
            Resize(self.img_size, self.img_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def infer(self, image_path, class_names):
        # Perform inference on a given image
        transform = self.get_inference_transforms()
        image = datasets.folder.default_loader(image_path)  # Load image using torchvision's loader
        image = transform(image=np.array(image))['image'].unsqueeze(0).to(self.device)  # Apply transforms and add batch dim

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        class_name = class_names[predicted_class]
        highest_prob = probabilities[0, predicted_class].item()

        return predicted_class, class_name, highest_prob, probabilities.cpu().numpy()

    def get_class_names(self, root_folder):
        # Load class names from the dataset
        dataset = datasets.ImageFolder(root_folder)  # Load the dataset to get the class-to-index mapping
        return dataset.classes  # List of class names in alphabetical order


# --- MAIN FUNCTION FOR TESTING ---
if __name__ == '__main__':
    # Replace with the path to the training folders
    class_names = ['mild', 'moderate', 'no_dr', 'poliferate', 'severe']  # Get class names from training data

    # Create an instance of the classifier and load the model
    classifier = EnsembleClassifier(weights_path='best_ensemble_model.pth')

    # Test image for inference
    test_image_path = 'IDRiD_01.jpg'  # Replace with your test image path

    # Perform inference
    predicted_class, class_name, highest_prob, probabilities = classifier.infer(test_image_path, class_names)

    # Output the results
    print(f'Predicted Class: {predicted_class} ({class_name})')
    print(f'Highest Probability: {highest_prob:.4f}')
    print(f'Class Probabilities: {probabilities}')

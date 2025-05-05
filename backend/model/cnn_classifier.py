import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import copy
import time
import json
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

class FruitDataset(Dataset):
    """Fruit dataset for PyTorch DataLoader"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in class folders
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.imgs = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.imgs.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
        
        print(f"Found {len(self.imgs)} images in {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.imgs[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, class_idx
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a small black image and the same class if there's an error
            dummy_img = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
            return dummy_img, class_idx

class CNNClassifier:
    def __init__(self, num_classes=None, device=None):
        self.num_classes = num_classes
        self.model = None
        self.class_names = None
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.history = None
        self.img_size = (224, 224)
        
        print(f"Using device: {self.device}")
        
        # Define data transforms
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
    def build_model(self, feature_extract=True):
        """
        Build a transfer learning model using ResNet18
        
        Args:
            feature_extract (bool): Flag for feature extracting. 
                                    When False, we finetune the whole model, 
                                    when True we only update the classifier weights
        """
        # If num_classes is not set, default to 3 (apple, banana, orange)
        if self.num_classes is None:
            self.num_classes = 3
            
        # Load the pretrained model
        model = models.resnet18(weights='DEFAULT')
        
        # Set gradients for feature extraction or fine-tuning
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, self.num_classes)
        )
        
        # Move model to the device
        self.model = model.to(self.device)
        
        return self.model
    
    def train(self, train_dir, validation_dir, epochs=15, batch_size=32, learning_rate=0.001):
        """
        Train the model using transfer learning
        
        Args:
            train_dir (str): Path to training data
            validation_dir (str): Path to validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for the optimizer
        """
        # Create datasets
        train_dataset = FruitDataset(train_dir, transform=self.data_transforms['train'])
        val_dataset = FruitDataset(validation_dir, transform=self.data_transforms['val'])
        
        if not self.class_names:
            self.class_names = train_dataset.classes
            self.num_classes = len(self.class_names)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        dataset_sizes = {
            'train': len(train_dataset),
            'val': len(val_dataset)
        }
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Save history
                if phase == 'train':
                    self.history['train_loss'].append(epoch_loss)
                    self.history['train_acc'].append(epoch_acc.item())
                else:
                    self.history['val_loss'].append(epoch_loss)
                    self.history['val_acc'].append(epoch_acc.item())
                
                # Deep copy the model if it's the best validation accuracy so far
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            
            print()
        
        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        return self.history
    
    def evaluate(self, test_dir, batch_size=32):
        """
        Evaluate the model on a test set
        
        Args:
            test_dir (str): Path to test data
            batch_size (int): Batch size for evaluation
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        # Create test dataset
        test_dataset = FruitDataset(test_dir, transform=self.data_transforms['val'])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Set model to evaluate mode
        self.model.eval()
        
        running_corrects = 0
        all_preds = []
        all_labels = []
        
        # Iterate over test data
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Statistics
                running_corrects += torch.sum(preds == labels.data)
                
                # Save predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = running_corrects.double() / len(test_dataset)
        print(f'Test Acc: {test_acc:.4f}')
        
        return test_acc.item(), all_preds, all_labels
    
    def predict(self, img_array):
        """
        Predict the class of an image
        
        Args:
            img_array: numpy array of the image
            
        Returns:
            predictions: numpy array of class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Convert numpy array to PIL Image
        if isinstance(img_array, np.ndarray):
            img = Image.fromarray((img_array * 255).astype(np.uint8))
        else:
            img = img_array
        
        # Apply transformations
        transform = self.data_transforms['val']
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Set model to evaluate mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        return probabilities.cpu().numpy()
    
    def save_model(self, save_path):
        """
        Save the model to disk
        
        Args:
            save_path: Path where to save the model
        """
        if self.model is None:
            print("No model to save. Train a model first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model data
        model_data = {
            'state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'history': self.history,
            'img_size': self.img_size
        }
        
        torch.save(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """
        Load a model from disk
        
        Args:
            model_path: Path to the saved model
        """
        # Load model data
        model_data = torch.load(model_path, map_location=self.device)
        
        # Get class names and set num_classes
        self.class_names = model_data['class_names']
        self.num_classes = len(self.class_names)
        self.history = model_data.get('history')
        self.img_size = model_data.get('img_size', (224, 224))
        
        # Build model architecture
        self.build_model()
        
        # Load model weights
        self.model.load_state_dict(model_data['state_dict'])
        
        print(f"Model loaded from {model_path}")
        print(f"Classes: {self.class_names}")
        
        return self
    
    def plot_training_history(self):
        """
        Plot training and validation accuracy and loss
        """
        if not self.history:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'], label='Train')
        plt.plot(self.history['val_acc'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def grade_fruit(self, img, predicted_class, grading_criteria=None):
        """
        Grade the fruit based on features extracted from the image
        
        Parameters:
        - img: The PIL Image object
        - predicted_class: The predicted fruit class
        - grading_criteria: Dictionary with grading criteria for each fruit type
        
        Returns:
        - grade: A, B, or C grade
        - score: Numerical score
        - features: Dictionary of extracted features
        """
        if grading_criteria is None:
            # Default grading criteria if none provided
            grading_criteria = {
                'all': {
                    'color_threshold': {'A': 0.8, 'B': 0.6},
                    'size_threshold': {'A': 0.8, 'B': 0.6},
                    'defect_threshold': {'A': 0.1, 'B': 0.3},
                    'texture_threshold': {'A': 0.75, 'B': 0.5}
                },
                'apple': {
                    'color_threshold': {'A': 0.85, 'B': 0.65},  # Apples need better color uniformity
                    'size_threshold': {'A': 0.75, 'B': 0.55},   # Size thresholds for apples
                    'defect_threshold': {'A': 0.08, 'B': 0.25}, # Lower tolerance for defects in apples
                    'texture_threshold': {'A': 0.8, 'B': 0.6}   # Texture is important for apples
                },
                'banana': {
                    'color_threshold': {'A': 0.9, 'B': 0.7},    # Bananas need very uniform yellow for A grade
                    'size_threshold': {'A': 0.7, 'B': 0.5},     # Size is less critical for bananas
                    'defect_threshold': {'A': 0.05, 'B': 0.3},  # Very low tolerance for spots in A grade
                    'texture_threshold': {'A': 0.7, 'B': 0.5}   # Texture is moderately important
                },
                'orange': {
                    'color_threshold': {'A': 0.9, 'B': 0.7},    # Oranges need uniform orange coloration
                    'size_threshold': {'A': 0.8, 'B': 0.6},     # Size is important for oranges
                    'defect_threshold': {'A': 0.05, 'B': 0.2},  # Very low tolerance for black spots or defects
                    'texture_threshold': {'A': 0.85, 'B': 0.65} # Texture is very important (smooth skin)
                }
            }
        
        # Convert image to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Extract features for grading
        features = {}
        
        # 1. Color analysis for apples - with special handling
        color_mean = np.mean(img_array, axis=(0, 1))
        color_std = np.std(img_array, axis=(0, 1))
        color_uniformity = 1 - np.mean(color_std)
        
        # For apples, check if the color is appropriate (red, green, or yellow)
        if predicted_class.lower() == 'apple':
            # Calculate red-to-green ratio for red apples
            r_g_ratio = color_mean[0] / (color_mean[1] + 0.001)  # Avoid division by zero
            
            # Examine dominant color for apple classification
            is_red_apple = color_mean[0] > 0.5 and r_g_ratio > 1.2
            is_green_apple = color_mean[1] > 0.4 and color_mean[1] > color_mean[0]
            is_yellow_apple = color_mean[0] > 0.5 and color_mean[1] > 0.5 and color_mean[2] < 0.4
            
            # Store apple type information
            features['apple_type'] = 'red' if is_red_apple else ('green' if is_green_apple else ('yellow' if is_yellow_apple else 'mixed'))
            
            # Give bonus for good coloration based on apple type
            color_bonus = 0
            if (is_red_apple and r_g_ratio > 1.5) or \
               (is_green_apple and color_mean[1] > 0.5) or \
               (is_yellow_apple and color_mean[0] > 0.6 and color_mean[1] > 0.6):
                color_bonus = 0.1
            
            # Apply bonus to color uniformity
            color_uniformity = min(1.0, color_uniformity + color_bonus)
        
        features['color_score'] = color_uniformity
        
        # 2. Size estimation - More specific for apples
        mask = np.mean(img_array, axis=2) > 0.05
        size_ratio = np.sum(mask) / (img_array.shape[0] * img_array.shape[1])
        
        # For apples, evaluate shape circularity
        if predicted_class.lower() == 'apple':
            # Calculate roundness/circularity
            from scipy import ndimage
            labeled_mask, num_features = ndimage.label(mask)
            if num_features > 0:
                # Find largest connected component (the apple)
                largest_label = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
                apple_mask = labeled_mask == largest_label
                
                # Get properties
                props = ndimage.find_objects(apple_mask)
                if props:
                    # Calculate aspect ratio
                    y_slice, x_slice = props[0]
                    height = y_slice.stop - y_slice.start
                    width = x_slice.stop - x_slice.start
                    aspect_ratio = min(height, width) / max(height, width)
                    
                    # A perfect circle has aspect ratio of 1
                    # Apply shape bonus for circular apples
                    shape_bonus = aspect_ratio * 0.2  # Up to 0.2 bonus for perfect circle
                    size_ratio = min(1.0, size_ratio + shape_bonus)
                    
                    # Store aspect ratio for display
                    features['aspect_ratio'] = aspect_ratio
                    
        # For bananas, evaluate yellow color and black spots
        if predicted_class.lower() == 'banana':
            # Convert to HSV for better color detection
            from skimage.color import rgb2hsv
            hsv_img = rgb2hsv(img_array)
            
            # Yellow detection in HSV (hue around 0.12-0.18)
            yellow_hue_low = 0.10
            yellow_hue_high = 0.2
            yellow_sat_min = 0.4
            
            # Create yellow mask
            yellow_mask = (hsv_img[:,:,0] >= yellow_hue_low) & (hsv_img[:,:,0] <= yellow_hue_high) & (hsv_img[:,:,1] >= yellow_sat_min) & mask
            
            # Calculate percentage of yellow within the banana
            yellow_percentage = np.sum(yellow_mask) / np.sum(mask) if np.sum(mask) > 0 else 0
            
            # Black spots detection (very low value in HSV)
            black_threshold = 0.3  # Black spots have low V value
            black_spots = (hsv_img[:,:,2] < black_threshold) & mask
            
            # Calculate percentage of black spots
            black_spot_percentage = np.sum(black_spots) / np.sum(mask) if np.sum(mask) > 0 else 0
            
            # Store metrics for display
            features['yellow_percentage'] = yellow_percentage * 100
            features['black_spot_percentage'] = black_spot_percentage * 100
            
            # Adjust color score based on yellow percentage and lack of black spots
            # A grade: Pure yellow (high yellow %, very low black spot %)
            # B grade: Good yellow with some spots
            # C grade: Lots of black spots (overripe)
            if black_spot_percentage < 0.05 and yellow_percentage > 0.9:
                # A grade - perfect yellow banana
                features['color_score'] = 0.95
            elif black_spot_percentage < 0.3 and yellow_percentage > 0.7:
                # B grade - good yellow with some spots
                features['color_score'] = 0.75
            else:
                # C grade - either too many black spots or not yellow enough
                features['color_score'] = 0.4
                
            # Adjust defect score based on black spots (higher weight than standard)
            features['defect_score'] = max(0, 1.0 - (black_spot_percentage * 3))
            
        # For oranges, evaluate orange color and detect defects/black spots
        if predicted_class.lower() == 'orange':
            # Convert to HSV for better color detection
            from skimage.color import rgb2hsv
            hsv_img = rgb2hsv(img_array)
            
            # Orange detection in HSV (hue around 0.05-0.11)
            orange_hue_low = 0.04
            orange_hue_high = 0.12
            orange_sat_min = 0.4
            
            # Create orange mask
            orange_mask = (hsv_img[:,:,0] >= orange_hue_low) & (hsv_img[:,:,0] <= orange_hue_high) & (hsv_img[:,:,1] >= orange_sat_min) & mask
            
            # Calculate percentage of orange within the orange fruit
            orange_percentage = np.sum(orange_mask) / np.sum(mask) if np.sum(mask) > 0 else 0
            
            # Detect defects - black spots or green patches
            # Black spots detection (very low value in HSV)
            black_threshold = 0.3  # Dark spots have low V value
            black_spots = (hsv_img[:,:,2] < black_threshold) & mask
            
            # Green patches detection (unripe areas)
            green_hue_low = 0.25
            green_hue_high = 0.4
            green_patches = (hsv_img[:,:,0] >= green_hue_low) & (hsv_img[:,:,0] <= green_hue_high) & mask
            
            # Combined defects
            defect_mask = black_spots | green_patches
            
            # Calculate percentage of defects
            defect_percentage = np.sum(defect_mask) / np.sum(mask) if np.sum(mask) > 0 else 0
            
            # Store metrics for display
            features['orange_percentage'] = orange_percentage * 100
            features['defect_percentage'] = defect_percentage * 100
            
            # Adjust color score based on orange percentage
            if orange_percentage > 0.9:
                # A grade - uniform orange color
                features['color_score'] = 0.95
            elif orange_percentage > 0.7:
                # B grade - mostly orange
                features['color_score'] = 0.75
            else:
                # C grade - inconsistent orange color
                features['color_score'] = 0.4
                
            # Adjust defect score based on defect percentage (severe penalty for any defects)
            # Oranges with black spots are severely downgraded
            if defect_percentage < 0.02:
                # A grade - virtually no defects
                features['defect_score'] = 0.95
            elif defect_percentage < 0.1:
                # B grade - minor defects
                features['defect_score'] = 0.6
            else:
                # C grade - significant defects
                features['defect_score'] = 0.2
        
        features['size_score'] = size_ratio
        
        # 3. Defect detection
        from scipy import ndimage
        edges = ndimage.sobel(np.mean(img_array, axis=2))
        edge_ratio = np.sum(edges > 0.2) / np.sum(mask) if np.sum(mask) > 0 else 1.0
        features['defect_score'] = 1 - min(edge_ratio * 5, 1.0)
        
        # For apples, check for blemishes and bruises
        if predicted_class.lower() == 'apple':
            # Convert to HSV for better defect detection
            from skimage.color import rgb2hsv
            hsv_img = rgb2hsv(img_array)
            
            # Look for dark spots within the apple region
            # These could be bruises or blemishes
            saturation = hsv_img[:,:,1]
            value = hsv_img[:,:,2]
            
            # Dark areas have low value
            dark_spots = (value < 0.5) & (mask)
            
            # Calculate percentage of dark spots
            dark_spot_ratio = np.sum(dark_spots) / np.sum(mask) if np.sum(mask) > 0 else 0
            
            # Adjust defect score based on dark spots (higher dark_spot_ratio means more defects)
            defect_penalty = dark_spot_ratio * 2  # Scale the penalty
            features['defect_score'] = max(0, features['defect_score'] - defect_penalty)
            
            # Store percentage of blemishes for display
            features['blemish_percentage'] = dark_spot_ratio * 100
        
        # 4. Texture analysis
        # Convert to grayscale
        gray = rgb2gray(img_array)
        
        # Only analyze pixels that are part of the fruit
        gray_fruit = gray.copy()
        gray_fruit[~mask] = 0
        
        # Quantize to fewer gray levels to make computation more efficient
        bins = 32
        gray_quantized = np.floor(gray_fruit * (bins - 1)).astype(np.uint8)
        
        # Calculate GLCM features where there are enough fruit pixels
        if np.sum(mask) > 100:  # Only if we have enough fruit pixels
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray_quantized, distances, angles, levels=bins, symmetric=True, normed=True)
            
            # Calculate texture properties
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            # For apples, smooth texture is preferred
            if predicted_class.lower() == 'apple':
                # Higher weight to homogeneity and energy for apples
                texture_score = (homogeneity * 1.5 + energy + max(0, correlation)) / 3.5
                
                # Store raw texture metrics for display
                features['texture_metrics'] = {
                    'homogeneity': float(homogeneity),
                    'energy': float(energy),
                    'contrast': float(contrast),
                    'correlation': float(correlation)
                }
            else:
                # Standard texture scoring for other fruits
                texture_score = (homogeneity + energy + max(0, correlation)) / 3
            
            # Scale to 0-1 range with a bias toward higher scores for good textures
            texture_score = min(1.0, texture_score * 1.5)
        else:
            # If not enough pixels, assign a mediocre texture score
            texture_score = 0.5
            
        features['texture_score'] = texture_score
        
        # Get criteria for this fruit type or use default
        criteria = grading_criteria.get(predicted_class.lower(), grading_criteria['all'])
        
        # Calculate overall score with type-specific weighting
        if predicted_class.lower() == 'apple':
            # For apples, color and texture are more important
            overall_score = (
                features['color_score'] * 0.35 + 
                features['size_score'] * 0.20 + 
                features['texture_score'] * 0.30 +
                features['defect_score'] * 0.15
            )
        elif predicted_class.lower() == 'orange':
            # For oranges, defect score is weighted heavily
            overall_score = (
                features['color_score'] * 0.30 + 
                features['size_score'] * 0.20 + 
                features['texture_score'] * 0.20 +
                features['defect_score'] * 0.30  # High weight for defects/black spots
            )
        else:
            # Default weighting for other fruits
            overall_score = (
                features['color_score'] * 0.35 + 
                features['size_score'] * 0.25 + 
                features['texture_score'] * 0.25 +
                features['defect_score'] * 0.15
            )
        
        # Determine grade with stricter criteria for apples
        if predicted_class.lower() == 'apple':
            if (overall_score >= criteria['color_threshold']['A'] and
                features['texture_score'] >= criteria['texture_threshold']['A'] and
                features['defect_score'] >= 0.92):  # Very low defects for grade A apples
                grade = 'A'
            elif (overall_score >= criteria['color_threshold']['B'] and
                  features['texture_score'] >= criteria['texture_threshold']['B'] and
                  features['defect_score'] >= 0.75):  # Low defects for grade B apples
                grade = 'B'
            else:
                grade = 'C'
        elif predicted_class.lower() == 'orange':
            # For oranges, any significant defects/black spots result in lower grade
            if (overall_score >= criteria['color_threshold']['A'] and
                features['defect_score'] >= 0.9):  # Virtually no defects for grade A oranges
                grade = 'A'
            elif (overall_score >= criteria['color_threshold']['B'] and
                  features['defect_score'] >= 0.5):  # Few defects for grade B oranges
                grade = 'B'
            else:
                grade = 'C'
        else:
            # Standard grading for other fruits
            if (overall_score >= criteria['color_threshold']['A'] and
                features['texture_score'] >= criteria['texture_threshold']['A']):
                grade = 'A'
            elif (overall_score >= criteria['color_threshold']['B'] and
                  features['texture_score'] >= criteria['texture_threshold']['B']):
                grade = 'B'
            else:
                grade = 'C'
            
        return grade, overall_score, features 
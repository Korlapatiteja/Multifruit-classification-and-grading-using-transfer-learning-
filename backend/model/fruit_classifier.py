import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from skimage import feature, color
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

class FruitClassifier:
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = None
        self.history = None
        self.img_size = (224, 224)  # Fixed size for image processing
        
    def build_model(self, n_estimators=100, max_depth=None):
        """
        Build a Random Forest classifier model
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        return self.model
    
    def _extract_features(self, img_array):
        """
        Extract features from an image
        
        Args:
            img_array: A numpy array of shape (height, width, 3) representing the image
            
        Returns:
            features: A numpy array of extracted features
        """
        # Resize to fixed size if not already
        if img_array.shape[:2] != self.img_size:
            from skimage.transform import resize
            img_array = resize(img_array, self.img_size, anti_aliasing=True)
        
        # Convert to grayscale for some features
        gray_img = color.rgb2gray(img_array)
        
        # Color features - mean and std of each channel
        r_mean, g_mean, b_mean = np.mean(img_array, axis=(0, 1))
        r_std, g_std, b_std = np.std(img_array, axis=(0, 1))
        
        # Texture features - Histogram of Oriented Gradients (HOG)
        hog_features = feature.hog(
            gray_img, 
            orientations=9, 
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=False
        )
        
        # Shape features - simple metrics
        edges = feature.canny(gray_img)
        edge_ratio = np.sum(edges) / edges.size
        
        # Combine features
        color_features = [r_mean, g_mean, b_mean, r_std, g_std, b_std]
        shape_features = [edge_ratio]
        
        # Use a subset of HOG features to keep the dimensionality reasonable
        # Take every 10th feature
        hog_subset = hog_features[::10]
        
        # Combine all features
        combined_features = np.concatenate([
            color_features,
            shape_features,
            hog_subset
        ])
        
        return combined_features
    
    def _process_directory(self, directory):
        """
        Process all images in a directory and extract features
        
        Args:
            directory: Path to directory containing class subdirectories
            
        Returns:
            features: List of feature vectors
            labels: List of corresponding labels
        """
        features = []
        labels = []
        class_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        if not self.class_names:
            self.class_names = sorted(class_dirs)
            self.num_classes = len(self.class_names)
            print(f"Found {self.num_classes} classes: {self.class_names}")
        
        for i, class_dir in enumerate(class_dirs):
            class_path = os.path.join(directory, class_dir)
            class_idx = self.class_names.index(class_dir)
            
            print(f"Processing class {class_dir} ({i+1}/{len(class_dirs)})")
            
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                        feature_vector = self._extract_features(img_array)
                        features.append(feature_vector)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        return np.array(features), np.array(labels)
    
    def train(self, train_dir, validation_dir, param_grid=None):
        """
        Train the classifier on images in the training directory
        
        Args:
            train_dir: Directory containing training images
            validation_dir: Directory containing validation images
            param_grid: Optional parameter grid for GridSearchCV
        """
        # Extract features from training and validation sets
        print("Extracting features from training set...")
        X_train, y_train = self._process_directory(train_dir)
        
        print("Extracting features from validation set...")
        X_val, y_val = self._process_directory(validation_dir)
        
        # Fit the scaler to training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build the model if not already built
        if self.model is None:
            self.build_model()
        
        # Train with grid search if param_grid is provided
        if param_grid:
            print("Performing grid search for hyperparameter tuning...")
            grid_search = GridSearchCV(
                self.model, 
                param_grid, 
                cv=3, 
                n_jobs=-1, 
                verbose=1,
                scoring='accuracy'
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # Set the best model
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train the model directly
            print("Training the model...")
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        val_score = self.model.score(X_val_scaled, y_val)
        print(f"Validation accuracy: {val_score:.4f}")
        
        # Store training history
        self.history = {
            'val_accuracy': val_score
        }
        
        return self.history
    
    def evaluate(self, test_dir):
        """
        Evaluate the model on a test set
        
        Args:
            test_dir: Directory containing test images
            
        Returns:
            accuracy: Test set accuracy
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features from test set
        print("Extracting features from test set...")
        X_test, y_test = self._process_directory(test_dir)
        
        # Scale the features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Evaluate
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"Test accuracy: {test_score:.4f}")
        
        return test_score
    
    def predict(self, img_array):
        """
        Predict the class of an image
        
        Args:
            img_array: A numpy array representing the image
            
        Returns:
            predictions: A numpy array of class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")
        
        # Extract features
        feature_vector = self._extract_features(img_array)
        
        # Scale features
        scaled_features = self.scaler.transform([feature_vector])
        
        # Get class probabilities
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        return probabilities
    
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
        
        # Save classifier, scaler, and class names
        model_data = {
            'classifier': self.model,
            'scaler': self.scaler,
            'class_names': self.class_names,
            'img_size': self.img_size
        }
        
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path):
        """
        Load a model from disk
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            self: The classifier instance
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['classifier']
        self.scaler = model_data['scaler']
        self.class_names = model_data['class_names']
        self.num_classes = len(self.class_names)
        self.img_size = model_data.get('img_size', (224, 224))
        
        print(f"Model loaded from {model_path}")
        print(f"Classes: {self.class_names}")
        
        return self
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importances for the Random Forest classifier
        
        Args:
            top_n: Number of top features to show
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            print("No trained Random Forest model available.")
            return
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create feature names (this is a simplification)
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(top_n), importances[indices], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
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
        from skimage.feature import graycomatrix, graycoprops
        from skimage.color import rgb2gray
        
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
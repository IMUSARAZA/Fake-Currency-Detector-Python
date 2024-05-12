import os
import cv2
from preprocess import crop_top_right, preprocess_image
from features import divide_into_blocks, calculate_derivative, calculate_features
from train import train_svm
from sklearn.metrics import accuracy_score

# Directory paths
train_dir = 'train/'
test_dir = 'test/'
output_dir = 'preprocessed_images/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Preprocess training images
train_image_paths = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
for image_path in train_image_paths:
    preprocess_image(image_path, output_dir)

# Load and preprocess test images
test_image_paths = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]
test_images = []
test_labels = []
for image_path in test_image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed_image = crop_top_right(cv2.medianBlur(image, 5))
    test_images.append(preprocessed_image)
    if "FAKE" in image_path:
        test_labels.append(1)
    else:
        test_labels.append(0)

# Process preprocessed training images
preprocessed_train_image_paths = [os.path.join(output_dir, img) for img in os.listdir(output_dir)]
train_features = []
train_labels = []
for image_path in preprocessed_train_image_paths:
    # Read the preprocessed image
    preprocessed_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Divide the preprocessed image into four blocks
    blocks = divide_into_blocks(preprocessed_image)
    fd = calculate_derivative(preprocessed_image)  # First derivative
    sd = calculate_derivative(fd)  # Second derivative

    # Calculate roughness for each block and aggregate features
    roughness_features = []
    for block in blocks:
        roughness_features.extend(calculate_features(block, fd, sd))
    
    # Append features and labels
    train_features.append(roughness_features)
    if "FAKE" in image_path:
        train_labels.append(1)  # Fake note
    else:
        train_labels.append(0)  # Genuine note

# Train SVM classifier
svm = train_svm(train_features, train_labels)

# Test the classifier
test_features = []
for test_image in test_images:
    # Divide the preprocessed image into four blocks
    blocks = divide_into_blocks(test_image)
    fd = calculate_derivative(test_image)  # First derivative
    sd = calculate_derivative(fd)  # Second derivative

    # Calculate roughness for each block and aggregate features
    roughness_features = []
    for block in blocks:
        roughness_features.extend(calculate_features(block, fd, sd))
    test_features.append(roughness_features)

# Predict labels for test images
predicted_labels = svm.predict(test_features)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predicted_labels) * 100
print("Accuracy:", accuracy, "%")

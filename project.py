
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def crop_top_right(image):
    # Get image dimensions
    height, width = image.shape[:2]

    # Crop top right portion
    crop_height = height // 2
    crop_width = width // 2
    top_right = image[:crop_height, -crop_width:]

    return top_right

def calculate_derivative(image):
    # Calculate derivatives in x and y directions
    fx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Derivative in x direction
    fy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Derivative in y direction
    
    # Combine the derivatives to get the magnitude
    gradient_magnitude = np.sqrt(fx**2 + fy**2)
    
    return gradient_magnitude

def divide_into_blocks(image):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate block size
    block_width = width // 2
    block_height = height // 2
    
    # Divide the image into four equal blocks
    blocks = []
    for i in range(2):
        for j in range(2):
            block = image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            blocks.append(block)
    
    return blocks

def calculate_features(block, fd, sd):
    features = []
    m = np.zeros(8) # mean vector
    c = np.zeros(8) # pixel counter
    
    block_cols = block.shape[1]
    f_block = block.flatten()
    f_fd = fd.flatten()
    f_sd = sd.flatten()
    
    for i in range(block_cols + 2, len(f_block) - 2):
        p_pixel = i - block_cols
        
        if f_fd[i] >= 0 and f_fd[p_pixel] < 0:
            c[0] += 1
            m[0] += f_block[i]
        
        if f_fd[i] < 0 and f_fd[p_pixel] < 0:
            c[1] += 1
            m[1] += f_block[i]
        
        if f_fd[i] < 0 and f_fd[p_pixel] >= 0:
            c[2] += 1
            m[2] += f_block[i]
        
        if f_fd[i] > 0 and f_fd[p_pixel] > 0:
            c[3] += 1
            m[3] += f_block[i]
        
        if f_sd[i] >= 0 and f_sd[p_pixel] < 0:
            c[4] += 1
            m[4] += f_block[i]
        
        if f_sd[i] < 0 and f_sd[p_pixel] < 0:
            c[5] += 1
            m[5] += f_block[i]
        
        if f_sd[i] < 0 and f_sd[p_pixel] >= 0:
            c[6] += 1
            m[6] += f_block[i]
        
        if f_sd[i] > 0 and f_sd[p_pixel] > 0:
            c[7] += 1
            m[7] += f_block[i]
    
    for i in range(8):
        if c[i] != 0:
            m[i] /= c[i]
    
    return m

def preprocess_image(image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Crop top right portion
    cropped_image = crop_top_right(image)

    # Apply median filter to reduce noise
    filtered_image = cv2.medianBlur(cropped_image, 5)  # Adjust the kernel size as needed
    
    # Save preprocessed image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, filtered_image)

    return output_path

def calculate_roughness(block, fd, sd):
    roughness_features = calculate_features(block, fd, sd)
    return roughness_features

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
for image_path in test_image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed_image = crop_top_right(cv2.medianBlur(image, 5))
    test_images.append(preprocessed_image)

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
        roughness_features.extend(calculate_roughness(block, fd, sd))
    
    # Append features and labels
    train_features.append(roughness_features)
    if "FAKE" in image_path:
        train_labels.append(1)  # Fake note
    else:
        train_labels.append(0)  # Genuine note

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(train_features, train_labels)

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
        roughness_features.extend(calculate_roughness(block, fd, sd))
    test_features.append(roughness_features)

# Predict labels for test images
predicted_labels = svm.predict(test_features)
# Print unique values in train_labels
print("Unique values in train_labels:", np.unique(train_labels))

# Print train_features
print("Number of train features:", len(train_features))
print("Example train feature vector:", train_features[0])

# Calculate accuracy
test_labels = [1, 1, 0, 0]  # Assuming first two are genuine and last two are counterfeit

# Print which test image is genuine or fake
for i, image_path in enumerate(test_image_paths):
    if predicted_labels[i] == 1:
        prediction = "Fake"
    else:
        prediction = "Genuine"
    print(f"{image_path} is predicted as {prediction}")



accuracy = accuracy_score(test_labels, predicted_labels) * 100
print("Accuracy:", accuracy, "%")
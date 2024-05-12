import os
import cv2

def crop_top_right(image):
    # Get image dimensions
    height, width = image.shape[:2]

    # Crop top right portion
    crop_height = height // 2
    crop_width = width // 2
    top_right = image[:crop_height, -crop_width:]

    return top_right

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

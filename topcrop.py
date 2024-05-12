import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop_top_right(image):
    # Get image dimensions
    height, width = image.shape[:2]

    # Define the region to crop (top right portion, adjust as needed)
    crop_height = int(height * 0.5)  # Adjust the percentage as needed
    crop_width = int(width * 0.5)    # Adjust the percentage as needed
    top_right = image[:crop_height, -crop_width:]

    return top_right




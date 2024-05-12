import numpy as np
import cv2

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

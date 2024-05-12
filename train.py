import os
import cv2
import numpy as np
from features import divide_into_blocks, calculate_derivative, calculate_features

def calculate_roughness(block, fd, sd):
    roughness_features = calculate_features(block, fd, sd)
    return roughness_features

def train_svm(train_features, train_labels):
    from sklearn.svm import SVC
    
    # Train SVM classifier
    svm = SVC(kernel='linear')
    svm.fit(train_features, train_labels)
    return svm

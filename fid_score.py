
# example of calculating the frechet inception distance
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import tensorflow as tf
from scipy.linalg import sqrtm

def preprocess_array(array):
    # Normalize array to range [-1, 1]
    normalized_array = (array - 0.5) * 2.0
    # Add third channel by replicating the second channel
    extra_channel = np.zeros_like(array[..., :1])
    expanded_array = np.concatenate([normalized_array, extra_channel], axis=-1)

    return expanded_array

# calculate frechet inception distance
def calculate_fid(real_images, generated_images):
    # Load pre-trained InceptionV3 model
    inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')
    print('Predicting..')
    # Generate feature vectors for both arrays
    features1 = inception_model.predict(real_images)
    features2 = inception_model.predict(generated_images)

    # Compute mean and covariance matrices
    mean1 = np.mean(features1, axis=0)
    mean2 = np.mean(features2, axis=0)
    cov1 = np.cov(features1, rowvar=False)
    cov2 = np.cov(features2, rowvar=False)
    
    # Add a small positive constant to the covariance matrices
    eps = 1e-6
    cov1 += eps * np.eye(cov1.shape[0])
    cov2 += eps * np.eye(cov2.shape[0])
    # Calculate square root of the product of covariances (covariance square root)
    diff = mean1 - mean2
    cov_sqrt = sqrtm(cov1.dot(cov2)).real

    # Calculate FID
    fid = np.sum(diff**2) + np.trace(cov1 + cov2 - 2*cov_sqrt)

    return fid

if __name__ == "__main__":
    # define two collections of activations
    act1 = random(10*2048)
    act1 = act1.reshape((10,2048))
    act2 = random(10*2048)
    act2 = act2.reshape((10,2048))
    # fid between act1 and act1
    fid = calculate_fid(act1, act1)
    print('FID (same): %.3f' % fid)
    # fid between act1 and act2
    fid = calculate_fid(act1, act2)
    print('FID (different): %.3f' % fid)
import os
import pickle
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors



# Load ResNet50 Feature Extractor

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])



# Feature Extraction Function

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0).flatten()
    features = features / norm(features)
    return features



# Build Feature Database

def build_feature_database(image_folder):
    filenames = []
    feature_list = []

    for file in tqdm(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, file)
        features = extract_features(img_path, model)

        filenames.append(img_path)
        feature_list.append(features)

    return np.array(feature_list), filenames



# Recommendation Function

def recommend(image_path, feature_list, filenames, top_k=5):
    query_features = extract_features(image_path, model)

    knn = NearestNeighbors(
        n_neighbors=top_k + 1,
        algorithm="brute",
        metric="euclidean"
    )
    knn.fit(feature_list)

    distances, indices = knn.kneighbors([query_features])

    # Exclude the query image itself
    recommended_indices = indices[0][1:]
    return [filenames[i] for i in recommended_indices]


    IMAGE_FOLDER = "images"

    # Step 1: Create feature embeddings
    feature_list, filenames = build_feature_database(IMAGE_FOLDER)

    # Optional: Save embeddings
    pickle.dump(feature_list, open("embeddings.pkl", "wb"))
    pickle.dump(filenames, open("filenames.pkl", "wb"))

    # Step 2: Test recommendation
    test_image = "test.jpg"  # replace with your image path
    recommendations = recommend(test_image, feature_list, filenames)

    print("Recommended images:")
    for img in recommendations:
        print(img)

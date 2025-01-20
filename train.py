# train_model.py
import numpy as np
import pandas as pd
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import joblib

def preprocess_images(image_paths):
    # Preprocess the images for InceptionV3
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    model = Model(inputs=inception_model.inputs, outputs=inception_model.output)
    
    image_features = []
    for image_path in image_paths:
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image)
        image_features.append(feature.flatten())  # Flatten the feature vector
    return np.array(image_features)

def build_captioning_model(vocab_size, max_length):
    # Define the model architecture
    inputs1 = Input(shape=(2048,))  # InceptionV3 features
    fe1 = Dense(256, activation='relu')(inputs1)
    inputs2 = Input(shape=(max_length,))  # Tokenized captions
    se1 = Embedding(vocab_size, 256)(inputs2)
    se2 = LSTM(256)(se1)
    decoder_input = Add()([fe1, se2])
    outputs = Dense(vocab_size, activation='softmax')(decoder_input)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# After training, save the model and tokenizer
model.save("captioning_model.h5")
joblib.dump(tokenizer, "tokenizer.joblib")

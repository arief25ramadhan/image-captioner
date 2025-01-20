# app.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = load_model('captioning_model.h5')
tokenizer = joblib.load('tokenizer.joblib')

# Define constants
max_length = 34  # Example, adjust based on your dataset

app = FastAPI()

def preprocess_image(image: UploadFile):
    # Preprocess the uploaded image
    img = load_img(image.file, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

def generate_caption(image):
    # Extract features using the InceptionV3 model
    image_features = model.predict(image)
    image_features = image_features.flatten()  # Flatten the feature vector
    
    # Generate a caption
    caption = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])
        seq = pad_sequences(seq, maxlen=max_length)
        pred = model.predict([image_features, seq])
        pred_idx = np.argmax(pred)
        word = tokenizer.index_word.get(pred_idx)
        if word is None:
            break
        caption += " " + word
        if word == 'endseq':
            break
    return caption.replace('startseq', '').replace('endseq', '')

class Item(BaseModel):
    image: UploadFile

@app.post("/caption/")
def create_caption(item: Item):
    img = preprocess_image(item.image)
    caption = generate_caption(img)
    return {"caption": caption}

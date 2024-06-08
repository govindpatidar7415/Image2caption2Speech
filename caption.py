from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.optimizers import Adam
import pickle
import numpy as np
from keras.models import load_model, Model
from keras.initializers import Orthogonal

# Import the gTTS library for text-to-speech conversion
from gtts import gTTS

# Define the optimizer with the desired learning rate
from keras.optimizers import Adam

# Define the optimizer with the desired learning rate
optimizer = Adam(learning_rate=0.001)

# Load the model and specify custom objects including Orthogonal initializer
model = load_model("./model_weights/model_9.h5", custom_objects={'Orthogonal': Orthogonal})

# Define ResNet50 without top layer
model_temp = ResNet50(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

# Create a new model by removing the last layer (output layer of 1000 classes) from ResNet50
model_resnet = Model(model_temp.input, model_temp.layers[-1].output)

# Load the word_to_idx and idx_to_word from disk
with open("./storage/word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)

with open("./storage/idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)

max_len = 35

def preprocess_image(img):
    img = image.load_img(img, target_size=(32,32))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, -1)
    return feature_vector

def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def caption_this_image(input_img): 
    photo = encode_image(input_img)
    caption = predict_caption(photo)
    return caption

def caption_and_speech(input_img):
    # Generate caption text
    caption_text = caption_this_image(input_img)
  
    return caption_text

# Example usage
input_img_path = "static\images.jpg"  # Provide the correct path to your image file
caption_text= caption_and_speech(input_img_path)
print("Generated Caption:", caption_text)
#print("Audio Path:", audio_path)

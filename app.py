import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import load_model, Model
import pickle

# Display the title of the application centered on the page
st.markdown(
    "<h1 style='text-align: center;'>🖼️✍️ CAPT-IMAGE 🖼️✍️</h1>",
    unsafe_allow_html=True
)

# File uploader to allow the user to upload images with specified extensions
image = st.file_uploader('Upload Image', ['jpg', 'jpeg', 'png', 'webp', 'svg'])

# Load pre-trained VGG16 model for feature extraction
vgg_model = load_model("C:\\Users\\Mitesh Gupta\\Downloads\\vgg_model.keras")

# Load the trained image captioning model
model = load_model("C:\\Users\\Mitesh Gupta\\Downloads\\temp_model.keras")

# Load the tokenizer used during training
with open('C:\\Users\\Mitesh Gupta\\Downloads\\Tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def preprocess_image(image):
    """
    Preprocesses the uploaded image for feature extraction.
    """
    # Load and resize the image to the required dimensions
    image = load_img(image, target_size=(224, 224))

    # Convert image to numpy array
    image = img_to_array(image)

    # Reshape the image to add a batch dimension
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    # Apply preprocessing specific to VGG16
    image = preprocess_input(image)

    # Extract features from the image using the VGG16 model
    feature = vgg_model.predict(image, verbose=0)
    
    return feature

def indx_to_word(integer, tokenizer):
    """
    Converts an integer index to its corresponding word using the tokenizer.
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    """
    Predicts the caption for the given image using the trained model.
    """
    in_text = 'startseq'  # Start sequence for caption generation
    for i in range(max_length):
        # Convert the input sequence to integers and pad it
        seq = tokenizer.texts_to_sequences([in_text])[0]
        pad_seq = pad_sequences([seq], max_length)
        
        # Predict the next word in the sequence
        y_pred = model.predict([image, pad_seq], verbose=0)
        y_pred = np.argmax(y_pred)  # Get the word index with highest probability
        
        # Convert the predicted word index to the corresponding word
        word = indx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
        
        # Append the predicted word to the sequence
        in_text += " " + word
        
        # Stop if the end of sequence token is predicted
        if word == 'endseq':
            break
    
    return in_text

# If an image is uploaded, process it and generate a caption
if image is not None:
    # Display the uploaded image
    st.image(image)
    
    # Preprocess the uploaded image
    image = preprocess_image(image)
    
    # Predict the caption for the image
    caption = predict_caption(model, image, tokenizer, 35)
    
    # Clean up the caption by removing start and end tokens
    caption = caption.replace('startseq ', '')
    caption = caption.replace(' endseq', '')
    
    # Display the generated caption
    st.write('## *Generated Caption for this Image*')
    st.text_area(label="Caption", value=caption)
import streamlit as st
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input,VGG16
from keras.models import load_model, Model
import pickle

st.title("Image Caption Generator")

image=st.file_uploader('Upload Image',['jpg','jpeg','png'])

vgg_model=VGG16() # Loading pre-trained VGG-16 Model
vgg_model=Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2] .output)

model=load_model("D:/model.h5")

with open('Tokenizer.pkl','rb') as f:
    tokenizer=pickle.load(f)

def preprocess_image(image):

    image=load_img(image,target_size=(224,224))

    # Converting image into numpy array
    image=img_to_array(image)

    # Reshaping the Image
    image=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])

    # Preprocessing the image
    image=preprocess_input(image)
    # print(image.shape)
    # Feature Extraction
    feature=vgg_model.predict(image,verbose=0)
    
    return feature

def indx_to_word(integer,tokenizer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def predict_caption(model,image,tokenizer,max_length):
    in_text=['Hello','bye','mitesh']
    for i in range(max_length):
        seq=tokenizer.texts_to_sequences(in_text)[0]
        pad_seq=pad_sequences([seq],max_length)
        y_pred=model.predict([image,pad_seq],verbose=0)
        y_pred=np.argmax(y_pred)
        word=indx_to_word(y_pred,tokenizer)
        if word is None:
            break
        in_text+=" "+word
        if word=='endseq':
            break
    return in_text

if image is not None:
    st.image(image)
    image=preprocess_image(image)
    caption=predict_caption(model,image,tokenizer,35)
    st.write(caption)
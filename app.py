import streamlit as st
from pydub import AudioSegment 
import cv2
from PIL import Image, ImageOps
import numpy as np
import pickle
import librosa 
import matplotlib.pyplot as plt
import librosa.display

st.write("""
         # Speech to Text Model
         """
         )
st.write("This is a simple speech to text system that converts audio to text")
file = st.file_uploader("Please upload a wav file",type=['wav'])
# sample,sr=librosa.load(file)
# st.write(sample)
# st.write(file)

if file is not None:
    audio_bytes=file.read()
    st.audio(audio_bytes,format='wav')
    h=st.audio(audio_bytes,format='wav')
    


# def create_spect():
#     sample,sr=librosa.load(file)
#     mels=librosa.features.melspectrogram(sample,sr)
#     fig=plt.figure()
#     # canvas=FigureCavas(fig)
#     p=plt.imshow(librosa.power_to_db(mels,ref=np.max))
#     plt.show()

# create_spect()


    
        

model = pickle.load(open(r"model_1.pickle",'rb')) 







def import_and_predict(file, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload a wav file")
else:
    audio = Image.open(file)
    st.audio(audio)
    # prediction = import_and_predict(image, model)
    
    
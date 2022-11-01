import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})
sequential_model = load_model('./sequential.h5')
st.title("Check for Endometriosis")
uploaded_files = st.file_uploader("Please choose a photo file",type=["png","jpg","jpeg"])
if uploaded_files is not None:
    st.image(uploaded_files, channels="RGB", output_format="auto")
    with st.spinner('Wait for it...'):
      image1=load_img(uploaded_files)
      img_np_array=np.asarray(image1)
      resize=tf.image.resize(img_np_array,(256,256))
      yhat=sequential_model.predict(np.expand_dims(resize/255,0))
      
    st.success('Done!')
    if yhat>0.5:
      pathology='Pathology detected'
    else:
      pathology='No pathology detected'
    st.markdown(f"<h1 style='text-align: center; color: red;'>{pathology}</h1>", unsafe_allow_html=True)



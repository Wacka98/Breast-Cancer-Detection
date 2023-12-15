# streamlit_app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the pre-trained model
model_path = './Best_model'
model = load_model(model_path)

# Function to preprocess the image for the model
def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make a prediction
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

# Streamlit app
def main():
    st.title('Breast Cancer Detection Web App')
    st.write('Upload a histology image to check for malignancy.')

    # File uploader for image
    uploaded_file = st.file_uploader("Choose a histology image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the selected image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make a prediction
        prediction = predict(image)

        # Display the prediction result
        st.write("Prediction Probability:", prediction[0][0])
        if prediction[0][0] > 0.5:
            st.success("Prediction: Malignant")
        else:
            st.success("Prediction: Benign")
            # Signature or Remark

    st.markdown("""
    ---\n
    *Created by Ethan Liew Jia Wei*\n
    """)

# Run the app
if __name__ == '__main__':
    main()

import streamlit as st

import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imageio import imread
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from streamlit_image_select import image_select


st.sidebar.title("Table of Content")
pages = [
    "Flower Image Classification",
]
page = st.sidebar.radio("Index", pages)

# if page == pages[0]:
#     st.write("Introduction")
#     if st.checkbox("Display"):
#         st.write("Streamlit continuation")


# if page == pages[1]:
#     pass


def load_image(url):
    np_image = plt.imread(url)
    np_image = resize(np_image, (254, 254)).numpy()
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def load_image_url(url):
    np_image = imread(url)
    np_image = resize(np_image, (254, 254)).numpy() / 256
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


img = None
if page == pages[0]:
    st.title("Flower Prediction")

    assets = "static/flower_photos"
    model_flower_photos = load_model("static/flower_photos/model.keras")
    df = pd.read_csv("static/flower_photos/images.csv")

    features = []
    for file in df.feature:
        fc = file.split("/")
        features.append(f"{assets}/{fc[-2]}/{fc[-1]}")

    img_feature = image_select(
        "Flowers", features, captions=df.label_name.values, use_container_width=False
    )
    if img_feature != None:
        img = load_image(img_feature)
        predictions = model_flower_photos.predict(img)
        predictions = np.argmax(predictions, axis=1)[0]
        print(predictions)
        print(img_feature)

        st.image(load_image(img_feature) / 255)
        names = ["Tullips", "Sunflower", "Rose", "Daisy", "Dandelion"]
        st.write(names[predictions])


# if page == pages[3]:
#     model = load_model("models/b/efficientNetB1/20230930232958_10/model.keras")
#     url = st.text_input("Image URL", "")

#     if st.button("Predict"):
#         img = load_image_url(url)
#         st.image(img, caption="Online Mushroom", width=256)

#         predictions = model.predict(img)
#         predictions = np.argmax(predictions, axis=1)
#         print(predictions)

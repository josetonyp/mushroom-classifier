import streamlit as st

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imageio import imread
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from streamlit_image_select import image_select


st.sidebar.title("Table of Content")
pages = [
    "Flower Set 1",
    "Flower Set 2",
    "Mushrooms Set 1",
]
page = st.sidebar.radio("Index", pages)


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
    st.title("Flower Set 1")

    assets = "static/flower_photos"
    model_flower_photos = load_model(f"{assets}/model.keras")
    df = pd.read_csv(f"{assets}/images.csv")

    features = []
    for file in df.feature:
        fc = file.split("/")
        features.append(f"{assets}/{fc[-2]}/{fc[-1]}")

    img_feature = image_select(
        "Flowers",
        features,
        captions=list(df.label_name.values),
        use_container_width=False,
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


if page == pages[1]:
    st.title("Flower Set 2")

    assets = "static/flower_images"
    model_flower_images = load_model(f"{assets}/model.keras")
    df = pd.read_csv(f"{assets}/images.csv")

    features = []
    for file in df.feature:
        fc = file.split("/")
        features.append(f"{assets}/{fc[-2]}/{fc[-1]}")

    img_feature = image_select(
        "Flowers",
        features,
        captions=list(df.label_name.values),
        use_container_width=False,
    )
    if img_feature != None:
        img = load_image(img_feature)
        predictions = model_flower_images.predict(img)
        predictions = np.argmax(predictions, axis=1)[0]
        print(predictions)
        print(img_feature)

        st.image(load_image(img_feature) / 255)
        names = [
            "Sunflower",
            "Tulip",
            "Orchid",
            "Lotus",
            "Lilly",
        ]
        st.write(names[predictions])

if page == pages[2]:
    st.title("Mushrooms Set 1")

    assets = "static/Kaggle Mushrooms"
    model_mushset1 = load_model(f"{assets}/model.keras")
    df = pd.read_csv(f"{assets}/images.csv")

    features = []
    for file in df.feature:
        fc = file.split("/")
        features.append(f"{assets}/{fc[-2]}/{fc[-1]}")

    img_feature = image_select(
        "Flowers",
        features,
        captions=list(df.label_name.values),
        use_container_width=False,
    )
    if img_feature != None:
        img = load_image(img_feature)
        predictions = model_mushset1.predict(img)
        predictions = np.argmax(predictions, axis=1)[0]
        print(predictions)
        print(img_feature)

        st.image(load_image(img_feature) / 255)
        names = [
            "Cortinarius",
            "Entoloma",
            "Lactarius",
            "Hygrocybe",
            "Boletus",
            "Agaricus",
            "Suillus",
            "Russula",
            "Amanita",
        ]
        st.write(names[predictions])

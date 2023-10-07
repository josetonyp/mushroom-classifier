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
    "Mushrooms Psilocybe",
    "Mushrooms Observer",
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


def render_images(folder):
    df = pd.read_csv(f"{folder}/images.csv")
    features = []
    for file in df.feature:
        fc = file.split("/")
        features.append(f"{folder}/{fc[-2]}/{fc[-1]}")

    return image_select(
        "Images",
        features,
        captions=list(df.label_name.values),
        use_container_width=False,
    )


def make_sorter(l):
    """
    Create a dict from the list to map to 0..len(l)
    Returns a mapper to map a series to this custom sort order
    """
    sort_order = {k: v for k, v in zip(l, range(len(l)))}
    return lambda s: s.map(lambda x: sort_order[x])


def render_images_df(folder, classes):
    df = pd.read_csv(f"{folder}/images.csv")
    df = df[df.label.isin(classes)]
    df = df.sort_values("label", key=make_sorter(classes))

    features = []
    for file in df.image_lien:
        features.append(f"{folder}/images/{file}")

    return image_select(
        "Images",
        features,
        captions=list(df.label.values),
        use_container_width=False,
    )


if page == pages[0]:
    st.title("Flower Set 1")

    assets = "static/flower_photos"
    model_flower_photos = load_model(f"{assets}/model.keras")

    img_feature = render_images(assets)
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

    img_feature = render_images(assets)
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

    assets = "static/Kaggle_Mushrooms"
    model_mushset1 = load_model(f"{assets}/model.keras")

    img_feature = render_images(assets)
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

if page == pages[3]:
    st.title("Psilocybe Images Set")
    names = [
        "Psilocybe zapotecorum",
        "Psilocybe cyanescens",
        "Psilocybe allenii",
        "Psilocybe ovoideocystidiata",
        "Psilocybe pelliculosa",
        "Psilocybe caerulescens",
        "Psilocybe neoxalapensis",
        "Psilocybe cubensis",
        "Psilocybe stuntzii",
        "Psilocybe azurescens",
        "Psilocybe baeocystis",
        "Deconica coprophila",
        "Psilocybe semilanceata",
        "Psilocybe muliercula",
        "Psilocybe caerulipes",
        "Psilocybe mexicana",
        "Psilocybe yungensis",
        "Psilocybe aztecorum",
        "Psilocybe subaeruginosa",
        "Psilocybe fagicola",
    ]

    n_class_select = st.radio(
        "Select the number of Classes to load", ["3 Classes", "5 Classes"]
    )

    if n_class_select == "3 Classes":
        n_class = 3
    else:
        n_class = 5

    assets = "static/psilocybe"

    img_feature = render_images_df(assets, names[:n_class])
    model_psilocybe = load_model(f"{assets}/model_{n_class}.keras")

    if img_feature != None:
        img = load_image(img_feature)
        predictions = model_psilocybe.predict(img)
        predictions = np.argmax(predictions, axis=1)[0]
        print(predictions)
        print(img_feature)

        st.image(load_image(img_feature) / 255)
        st.write(names[predictions])

if page == pages[4]:
    st.title("Mushrooms Observer Set")
    names = [
        "Agaricales",
        "Russula",
        "Cortinarius",
        "Amanita",
        "Polyporales",
        "Psathyrella",
        "Agaricus",
        "Inocybe",
        "Mycena",
        "Entoloma",
    ]

    n_class_select = st.radio(
        "Select the number of Classes to load", ["3 Classes", "5 Classes"]
    )

    if n_class_select == "3 Classes":
        n_class = 3
    else:
        n_class = 5

    assets = "static/mushrooms"
    model_musshy = load_model(f"{assets}/model_{n_class}.keras")
    img_feature = render_images_df(assets, names[:n_class])

    if img_feature != None:
        img = load_image(img_feature)
        predictions = model_musshy.predict(img)
        predictions = np.argmax(predictions, axis=1)[0]
        print(predictions)
        print(img_feature)

        st.image(load_image(img_feature) / 255)
        st.write(names[predictions])

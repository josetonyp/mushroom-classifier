# Mushroom Classifier CLI App

The objective of this project is to present a Command Line Application as a Framework to automate the training of several Convolutional Networks and document the output of the training and evaluation process. The training standarize the execution of CNN for computer vision and object classifications. 


## Datasets Used

The trainig process was run in the following datasets:

### **Flower Images:** 

A balanced 5 classes flower images data set disposed into 5 folders with the name of the class/flower as folder name. No CSV file present. 

URL: https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset

![Flower Images Data Set](/static/flower_images/samples2.jpg "Flower Images Data Set")


### **Flower Photos:** 

An unbalanced 5 classes flower images data set disposed into 5 folders with the name of the class/flower as folder name. No CSV file present. 

URL: https://www.tensorflow.org/datasets/catalog/tf_flowers

![Flower Photos Data Set](/static/flower_photos/samples2.jpg "Flower Photos Data Set")

### **Kaggle Mushroom Images Dataset:** 

An unbalanced 9 class mushrooms images dataset disposed into 9 folders with the name of each mushroom as a folder name. No CSV file present.

URL: https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images

![Kaggle Mushroom Images Dataset](/static/psilocybe/samples.jpg)


### **Mushroom Observer's Images Dataset:** 

A considerable big database of 648_000 images divided in aproximately 1_000 especies. The Database is well documented biologically and location based.  


URL: https://mushroomobserver.org/

![Kaggle Mushroom Images Dataset](/static/mushrooms/samples.jpg)


## Install

To install the CLI App clone the repositor in your folder

```bash
git clone https://github.com/josetonyp/mushroom-classifier.git
mkdir models # Output folder where models will be documented
mkdir input # Folder where to locate images to train the model
```


## Execute

### Trainers

Trainers are endpoints that takes a Dataset and runs model training automatically saving it's result in a folder structured based on training parameters. Given a project name the output folder will look like:

```bash
- models
  - project_name
    - architecture
      - base_model
        - datetime
          -- report.txt
          -- confusion_matrix.json
          -- classification_report.txt
          -- history.csv
          -- model.keras
```
For Example:

```bash
- models
  - mushrooms
    - b
      - vgg16
        - 20230925200055
          -- report.txt
          -- confusion_matrix.json
          -- classification_report.txt
          -- history.csv
          -- model.keras
      - efficientNetB1
        - 20230927200055
          -- report.txt
          -- confusion_matrix.json
          -- classification_report.txt
          -- history.csv
          -- model.keras
```

###### Folders DataSets
Trains a Model based on an image folder with ouput in a target folder with a given model base name (ex: VGG16) and using a given architecture (a|b|c currently).

```bash
BATCH_SIZE=256 EPOCHS=20 ./learning train-folder  \
      --folder_name=input/flower_images \
      --model_name=efficientNetB1 \
      --architecture=b
```

###### CSV DataSets

Trains a Model based on a CSV Dataset with images in a target folder with a given model base name (ex: VGG16) and using a given architecture (a|b|c currently).

```bash
BATCH_SIZE=300 EPOCHS=3 PD_LABEL_COUNT=5 ./learning train-dataset \
      --name=<project-name> \
      --dataframe_file=input/old_pd_files/mushrooms_top_15_psilocybe.csv\t \
      --images_folder=input/images \
      --model_name=efficientNetB1 \
      --architecture=b
```

### Renderers

Convertst the classification report, confussion matrix and training history from the training evaluation into an image. 

###### Render a Classification Report

Reads the classification report files from the training ouput folder and converts it into an image

```bash
./learning render-report \
      --file_name=<path_to_file> \
      --title="Desired Title" \
      --label_names="<Comma,separated,label,names>" \
      --figsize=60,60
```
###### Render a Confusion Matrix

Reads the confusion matrix json file from the training ouput folder and converts it into an image

```bash
./learning render-cfn-matrix \
      --file_name=<path_to_file> \
      --title="Desired Title" \
      --label_names="<Comma,separated,label,names>" \
      --figsize=60,60

```

###### Render a Training History

Reads the training history from the training ouput folder and plots it into an image 

```bash

./learning render-history \
      --file_name=<path_to_file> \
      --title="Desired Title" \
      --label_names="<Comma,separated,label,names>" \
      --figsize=60,60
```

###### Renders a CSV Sampler

Takes a DataSet from a CSV file, samples 1 image per class and renders a sampler image.

Renders a 1 x 6 image sampler
```bash
PD_LABEL_COUNT=6 ./learning csv-sampler \
      --dataframe_file=<path_to_file> \
      --images_folder=input/images \
      --format=1,6 \
      --figsize=65,12
```

Render a 3 x 5 image sampler
```bash
PD_LABEL_COUNT=15 ./learning csv-sampler \
      --dataframe_file=<path_to_file> \
      --images_folder=input/images \
      --format=3,5 \
      --figsize=60,36
```

###### Renders a Folder Sampler

Takes a DataSet from a foler with images classified into subfolders as class, samples 1 image per class and renders a sampler image.

Render a 3 x n_classes image sampler
```bash
./learning folder-sampler \
      --images_folder=<path_to_folder> \
      --reows=3 \
      --figsize=60,36
```

## Predicting Application

Our best trainined models were documented and deployed as Streamlit applications. The streamlit app is configured in `streamlit.py` and the statict files to execute that project are hosted in `./static/`

Run the application in local:

```bash
streamlit run streamlit.py
```

## Test
The current architecture is build as POC and there is no intention to Unit Test the code.


## Collaboration and Updates

The code is deliver as it is. We do not have plans to upgrade this code commercially beyond our Data Science studies but we are open to help you understand and improve given our availability. The code has been tested in Mac and Linux only and only. The Tensorflow design was run only in CPU with lmited capacities. 


## Licence

#### MIT License

Copyright (c) 2023 [Jose Pio](https://github.com/josetonyp) and [Sushovan Dam](https://github.com/sdam89)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
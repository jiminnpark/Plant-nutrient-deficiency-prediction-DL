import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing import image
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("generated_dataset.csv")
dataset.dropna(inplace=True)
X = dataset.drop(columns=["Recommended Fertilizer"])
y = dataset["Recommended Fertilizer"]
y = pd.factorize(y)[0]
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier.fit(X_train, y_train)


custom_model = load_model("shivanshu_dl_model.h5")


def show_prob(pred):
    d=[]
    for i in range(len(pred)):
        d.append(round(pred[i]*100,2))
    nutrients = ["BORON", "CALCIUM", "HEALTHY", "IRON", "MAGNESIUM", "MANGANESE", "POTASSIUM", "SULPHUR", "ZINC"]
    max_deficient=nutrients[d.index(max(d))]
    if max_deficient!="HEALTHY":
        max_deficient+=" deficient"
    return d,max_deficient
  
def find_deficiency(imgname):
    img_width, img_height = 66, 66
    img = image.load_img(imgname, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    prediction=custom_model.predict(img)
    return show_prob(prediction[0])
  
def fertilizer(pred):
    pred=pred[:2]+pred[3:]
    pred=[pred]
    new_data_df = pd.DataFrame(pred, columns=["BORON", "CALCIUM", "IRON", "MAGNESIUM", "MANGANESE", "POTASSIUM", "SULPHUR", "ZINC"])
    predicted_fertilizers = rf_classifier.predict(new_data_df)
    predicted_fertilizers = pd.Series(predicted_fertilizers).map({label: fertilizer for label, fertilizer in enumerate(dataset["Recommended Fertilizer"].unique())})
    return(predicted_fertilizers.values)
  
 
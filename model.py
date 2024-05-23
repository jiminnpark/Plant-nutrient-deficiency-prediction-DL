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


def get_nutrient_actions(nutrient):
    nutrient_actions = {
        "boron": [
            "Apply boron-containing fertilizers such as borax or boric acid.",
            "Ensure uniform soil moisture to prevent boron leaching in sandy soils.",
            "Avoid excessive boron application, as it can be toxic to plants at high concentrations."
        ],
        "calcium": [
            "Apply calcium-rich fertilizers such as gypsum or calcium nitrate.",
            "Lime acidic soils to increase calcium availability and improve soil structure.",
            "Incorporate calcium-containing organic matter like eggshells or bone meal into the soil."
        ],
        "iron": [
            "Apply iron chelate or iron sulfate as a soil drench or foliar spray.",
            "Adjust soil pH to slightly acidic conditions to enhance iron availability.",
            "Use iron-containing organic amendments like blood meal or fish emulsion."
        ],
        "magnesium": [
            "Apply magnesium-rich fertilizers such as magnesium sulfate (Epsom salt) or dolomitic lime.",
            "Ensure proper soil pH and drainage to prevent magnesium leaching.",
            "Use magnesium-containing foliar sprays for rapid correction of deficiencies."
        ],
        "manganese": [
            "Apply manganese sulfate or manganese chelate to the soil or as a foliar spray.",
            "Maintain soil pH between 5.5 and 6.5 for optimal manganese uptake.",
            "Incorporate organic materials like compost or manure to increase manganese availability."
        ],
        "potassium": [
            "Apply potassium-rich fertilizers such as potassium chloride or potassium sulfate.",
            "Maintain balanced soil pH to prevent potassium fixation in alkaline soils.",
            "Use potassium-containing organic materials such as wood ash or kelp meal as soil amendments."
        ],
        "sulphur": [
            "Apply sulfur-rich fertilizers such as elemental sulfur or ammonium sulfate.",
            "Use sulfur-containing organic materials like gypsum or composted manure.",
            "Avoid excessive liming, as it can reduce sulfur availability in the soil."
        ],
        "zinc": [
            "Apply zinc sulfate or zinc chelate to the soil or as a foliar spray.",
            "Use zinc-containing fertilizers or organic materials like zinc oxide or zinc lignosulfonate.",
            "Ensure proper soil pH and organic matter content for optimal zinc uptake."
        ]
    }

    nutrient_lower = nutrient.lower()
    if nutrient_lower in nutrient_actions:
        return nutrient_actions[nutrient_lower]
    else:
        return [
        "Regularly inspect banana plants for signs of stress, disease, or nutrient deficiencies.",
        "Ensure balanced nutrition by providing appropriate fertilization based on soil testing and plant nutrient requirements.",
        "Implement integrated pest and disease management practices to control pests and diseases effectively."
    ]


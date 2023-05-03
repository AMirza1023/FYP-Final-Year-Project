from tensorflow.keras.models import load_model
import tensorflow.keras.optimizers
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from skimage.feature import graycomatrix, graycoprops
from sklearn import preprocessing
import pickle as pk

#arr = np.genfromtxt("labels.csv", delimiter=",")

mymodel = load_model("/Users/alimi/PycharmProjects/GAN/1.h5",compile=False)
images = []
# for predimages in
predimg = cv2.imread("/Users/alimi/Downloads/Glioma.jpg")
predimg = cv2.cvtColor(predimg, cv2.COLOR_BGR2GRAY)
predimg = cv2.resize(predimg, (225, 225))
images.append(predimg)
images_array = np.array(images)


def glcmfeatureExtraction(dataset):
    imageDataset = pd.DataFrame()
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()
        img = dataset[image, :, :]
        for i, j in [(1, 0), (3, 0), (5, 0), (0, np.pi / 4), (0, np.pi / 2)]:
            GLCM = graycomatrix(img, [i], [j])
            Energy = graycoprops(GLCM, 'energy')[0]
            df['Energy'] = Energy
            Correlation = graycoprops(GLCM, 'correlation')[0]
            df['Correlation'] = Correlation
            Dissimilarity = graycoprops(GLCM, 'dissimilarity')[0]
            df['Dissimilarity'] = Dissimilarity
            Homogeneity = graycoprops(GLCM, 'homogeneity')[0]
            df['Homogeneity'] = Homogeneity
            Contrast = graycoprops(GLCM, 'contrast')[0]
            df['Contrast'] = Contrast
            imageDataset = imageDataset.append(df)

    return imageDataset

modelVGG16 = load_model("/Users/alimi/PycharmProjects/GAN/vgg16.h5")


features = modelVGG16.predict(images_array, verbose=1, batch_size=100)
featureExtractionGLCM = glcmfeatureExtraction(images_array)
featureExtractionGLCM.to_numpy()
featureExtractionGLCM = np.expand_dims(featureExtractionGLCM, axis=0)
featureExtractionGLCM = np.reshape(featureExtractionGLCM, (images_array.shape[0], -1))
features = np.concatenate((features, featureExtractionGLCM), axis=1)


#pca = pk.load(open("/Users/alimi/PycharmProjects/GAN/pcaFINAL3.pkl", 'rb'))
normalizer = pk.load(open("/Users/alimi/PycharmProjects/GAN/scaler.pkl",'rb'))
Data = normalizer.transform(features)

pred = mymodel.predict(Data, verbose=1, batch_size=30)

out = np.argmax(pred, axis=1)
score = tensorflow.nn.softmax(pred[0])

for classes in out:
    if classes == 0:
        print("This image most likely belongs to GLIOMA with a {:.2f} percent confidence.".format(100 * np.max(pred)))
    if classes == 1:
        print("This image most likely belongs to HEALTHY with a {:.2f} percent confidence.".format(100 * np.max(pred)))
    if classes == 2:
        print(
            "This image most likely belongs to MENINGIOMA with a {:.2f} percent confidence.".format(100 * np.max(pred)))
    if classes == 3:
        print(
            "This image most likely belongs to PITUITARY with a {:.2f} percent confidence.".format(100 * np.max(pred)))

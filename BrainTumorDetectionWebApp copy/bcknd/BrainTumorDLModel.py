import matplotlib.pyplot as plt
import tensorflow.keras.optimizers
from sklearn.decomposition import PCA
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras import Model, layers, models
from sklearn.model_selection import train_test_split
from scipy import ndimage as nd
import numpy as np
import seaborn as sns
import pandas as pd
import cv2
import os
from imutils import paths
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score, roc_curve
import pickle as pk

images = []
labels = []
path = "/Users/alimi/Desktop/BT/"
list(os.listdir(path))
imagePaths = list(paths.list_images(path))

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    #image = nd.gaussian_filter(image, sigma=0.1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #kernel = np.ones((3, 3), np.uint8)
    #image = cv2.erode(image, kernel, iterations=1)
    #image = cv2.dilate(image, kernel, iterations=1)
    #T,image = cv2.threshold(image, 5, 255,cv2.THRESH_BINARY)
    image = cv2.resize(image, (225, 225))
    images.append(image)
    labels.append(label)

images_array = np.array(images)
labels_array = np.array(labels)
print(images_array.shape)
print(labels_array[1])

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(labels_array)
encoded_labels = le.transform(labels_array)
print(encoded_labels.size)
np.savetxt("labels.csv", labels_array,
              delimiter = ",", fmt="%s")

check = le.inverse_transform(encoded_labels)









modelVGG16 = VGG16(weights='imagenet', include_top=False)
for layer in modelVGG16.layers:
    layer.trainable = False
modelVGG16.summary()

modelVGG16Config = modelVGG16.get_config()

modelVGG16Config["layers"][0]["config"]["batch_input_shape"] = (None, 225, 225, 1)
modelVGG16Updated = Model.from_config(modelVGG16Config)


def avg_wts(weights):
    average_wts = np.mean(weights, axis=-2).reshape(weights[:, :, -1:, :].shape)
    return (average_wts)


vgg_updated_config = modelVGG16Updated.get_config()
vgg_updated_layer_names = [vgg_updated_config['layers'][x]['name'] for x in range(len(vgg_updated_config['layers']))]

first_conv_name = vgg_updated_layer_names[1]

for layer in modelVGG16.layers:
    if layer.name in vgg_updated_layer_names:
        if layer.get_weights() != []:
            targetLayer = modelVGG16Updated.get_layer(layer.name)
            if layer.name in first_conv_name:
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                weightsForSingleChannel = avg_wts(weights)
                targetLayer.set_weights([weightsForSingleChannel, biases])
                targetLayer.trainable = False
            else:
                targetLayer.set_weights(layer.get_weights())
                targetLayer.trainable = False

VGG16WtsUpdatedModel = modelVGG16Updated.layers[1].get_weights()[0]
modelVGG16Updated.summary()
print(VGG16WtsUpdatedModel[:, :, 0, 0])


new_model = Model(inputs = modelVGG16Updated.input, outputs=modelVGG16Updated.get_layer('block4_pool').output)
output = new_model.output
output = GlobalAveragePooling2D()(output)
new_model = Model(new_model.input, output)
new_model.summary()
modelVGG16Updated = new_model



x_train, x_test, y_train, y_test = train_test_split(images_array, encoded_labels, train_size=0.75, random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


y_trainEnc = to_categorical(y_train)
y_testEnc = to_categorical(y_test)






featureExtractionTrain = modelVGG16Updated.predict(x_train, verbose=1, batch_size=100)
featureExtractionTrain = featureExtractionTrain.reshape(featureExtractionTrain.shape[0], -1)
print("Done with train feature extraction ")
print(featureExtractionTrain.shape)
print(featureExtractionTrain)


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


featureExtractionGLCMTrain = glcmfeatureExtraction(x_train)
featureExtractionGLCMTrain.to_numpy()
featureExtractionGLCMTrain = np.expand_dims(featureExtractionGLCMTrain, axis=0)
featureExtractionGLCMTrain = np.reshape(featureExtractionGLCMTrain, (x_train.shape[0], -1))
featureExtractionTrain = np.concatenate((featureExtractionTrain,featureExtractionGLCMTrain), axis=1)
print(featureExtractionTrain.shape)




standardization = preprocessing.scale(featureExtractionTrain, axis=1)
x = standardization
print(x)

pcaData = PCA(n_components = 537)
pcaData.fit(x)
plt.plot(np.cumsum(pcaData.explained_variance_ratio_))
plt.xlabel('No of Components')
plt.ylabel('Variance')
plt.show()

pca = PCA(n_components = 400)
ComponentsTrain = pca.fit_transform(x)
print(ComponentsTrain)
print(ComponentsTrain.shape)
print((1 - np.sum(pca.explained_variance_ratio_))*100, '%')






model = models.Sequential()
opt = Adam()
model.add(layers.Flatten(input_shape=(400,1)))
model.add(layers.Dense(70, activation='relu', kernel_initializer= 'uniform'))
model.add(layers.Dropout(0.1))#0.08
model.add(layers.Dense(4, activation='softmax', kernel_initializer='uniform'))
model.compile(optimizer= opt,
              loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()



hist = model.fit(ComponentsTrain, y_trainEnc, verbose=1, batch_size= 40, epochs= 30)

import matplotlib
matplotlib.pyplot.style.use("ggplot")
matplotlib.pyplot.figure()
matplotlib.pyplot.plot(np.arange(0, 30), hist.history['loss'], label="train_loss")
matplotlib.pyplot.plot(np.arange(0, 30), hist.history['acc'], label="train_acc")
matplotlib.pyplot.title("Training Loss and Accuracy")
matplotlib.pyplot.xlabel("Epochs")
matplotlib.pyplot.ylabel("Loss and Accuracy")
matplotlib.pyplot.legend(loc="lower left")
plt.show()





featureExtractionTest = modelVGG16Updated.predict(x_test, verbose=1, batch_size=100)
featureExtractionTest = featureExtractionTest.reshape(featureExtractionTest.shape[0], -1)
print("Done with test feature extraction ")
print(featureExtractionTest.shape)
print(featureExtractionTest)
featureExtractionGLCMTest = glcmfeatureExtraction(x_test)
#featureExtractionGLCM.to_csv('GLCM.csv', encoding='utf-8', index=False)
featureExtractionGLCMTest.to_numpy()
print(featureExtractionGLCMTest.shape)
featureExtractionGLCMTest = np.expand_dims(featureExtractionGLCMTest, axis=0)
featureExtractionGLCMTest = np.reshape(featureExtractionGLCMTest, (x_test.shape[0], -1))
#featureExtractionGLCM.to_csv('GLCM.csv', encoding='utf-8', index=False)
featureExtractionTest = np.concatenate((featureExtractionTest,featureExtractionGLCMTest), axis=1)
print(featureExtractionTest.shape)
standardization = preprocessing.scale(featureExtractionTest, axis=1)
testData = standardization
print(testData)

ComponentsTest = pca.transform(testData)
print(ComponentsTest)
print(ComponentsTest.shape)






predictionRF = model.predict(ComponentsTest, verbose=1, batch_size= 30)
out = np.argmax(predictionRF, axis=1)
predictions_test = le.inverse_transform(out)
print(predictions_test)
y_pred=np.argmax(predictionRF, axis=1)
y_test=np.argmax(y_testEnc, axis=1)
print("Acc:", metrics.accuracy_score(y_test, y_pred))
print("F1 scores:", metrics.f1_score(y_test, y_pred, average= None))
print("Recall Scores:", metrics.recall_score(y_test,y_pred, average= None))
print("Matthews correlation coefficient:",matthews_corrcoef(y_test, y_pred))
y_probabilities = model.predict_proba(ComponentsTest)
roc_auc_score(y_test,y_probabilities, multi_class= "ovr", average=None)
#roc_curve(y_test,y_probabilities)

cmat = confusion_matrix(y_test, y_pred, labels=[0, 1,2,3])
s = sns.heatmap(cmat, annot=True, fmt='g', xticklabels=[0, 1,2,3], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.xlabel("Actual")
plt.title("Confusion Matrix")
plt.show()
acc = cmat.trace() / cmat.sum()
print('Accuracy: {0:5.2f}%'.format(acc * 100))
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
                     index = ['Pituitary','Glioma','Meningioma','Healthy'],
                     columns = ['Pituitary','Glioma','Meningioma','Healthy'])
#Plotting the confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()
acc = cm.trace() / cm.sum()
print('Accuracy: {0:5.2f}%'.format(acc * 100))

#model.save('BrainTumorClassifier.h5')
modelVGG16Updated.save('FeatureExtractor.h5')

from flask import Flask
app = Flask(__name__)
@app.route('/sample')
def running():
    return 'Flask Works'


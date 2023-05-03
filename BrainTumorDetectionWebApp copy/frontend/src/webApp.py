import os


from imutils import paths



from flask import Flask, render_template, request

app = Flask(__name__)
@app.route('/', methods = ['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods = ['POST'])
def saveInputImg():

    if request.method == 'POST':
        if request.files['imgFile'].filename != '':

            imgFile = request.files['imgFile']
            imgPath = "./images/" + imgFile.filename
            imgFile.save(imgPath)
            alertofUpload = "Image(s) has/have been uploaded so can now proceed with prediction."
            return render_template('index.html', alert = alertofUpload)
        return render_template('index.html')


    render_template('index.html')





@app.route('/pred', methods = ['POST', 'GET'])
def makePrediction():


    if request.method == 'POST':
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
        import os
        from imutils import paths

        # arr = np.genfromtxt("labels.csv", delimiter=",")

        mymodel = load_model(
            "/Users/alimi/PycharmProjects/GAN/BrainTumorClassifier.h5")
        images = []

        path = "/Users/alimi/Desktop/FinalYearProject-ML/BrainTumorDetectionWebApp/bcknd/images/"

        list(os.listdir(path))
        direc = os.listdir(path)
        if len(direc) == 1:
            return render_template('index.html')
        imagePaths = list(paths.list_images(path))
        imgFilename = []
        for imagepath in imagePaths:
            image = cv2.imread(imagepath)
            imgFilename.append(os.path.basename(imagepath))

            # image = nd.gaussian_filter(image, sigma=0.1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # kernel = np.ones((3, 3), np.uint8)
            # image = cv2.erode(image, kernel, iterations=1)
            # image = cv2.dilate(image, kernel, iterations=1)
            # T,image = cv2.threshold(image, 5, 255,cv2.THRESH_BINARY)
            image = cv2.resize(image, (225, 225))
            images.append(image)

        images_array = np.array(images)
        print(images_array.shape)
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
        vgg_updated_layer_names = [vgg_updated_config['layers'][x]['name'] for x in
                                   range(len(vgg_updated_config['layers']))]

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

        new_model = Model(inputs=modelVGG16Updated.input, outputs=modelVGG16Updated.get_layer('block4_pool').output)
        output = new_model.output
        output = GlobalAveragePooling2D()(output)
        new_model = Model(new_model.input, output)
        new_model.summary()
        modelVGG16Updated = new_model

        features = modelVGG16Updated.predict(images_array, verbose=1, batch_size=100)
        features = features.reshape(features.shape[0], -1)
        featureExtractionGLCM = glcmfeatureExtraction(images_array)
        featureExtractionGLCM.to_numpy()
        featureExtractionGLCM = np.expand_dims(featureExtractionGLCM, axis=0)
        featureExtractionGLCM = np.reshape(featureExtractionGLCM, (images_array.shape[0], -1))
        features = np.concatenate((features, featureExtractionGLCM), axis=1)
        #pca = pk.load(open("/Users/alimi/PycharmProjects/GAN/pca3.pkl", 'rb'))
        pca = pk.load(open("pca.pkl", 'rb'))
        standardization = preprocessing.scale(features, axis=1)
        Data = standardization
        ComponentsTest = pca.transform(Data)

        pred = mymodel.predict(ComponentsTest, verbose=1, batch_size=30)
        classifications = []
        for name, predictions in zip(imgFilename,pred):
            out = np.argmax(predictions)

            # @app.route('/pred', methods = ['GET'])

            if out == 0:
                classifications.append(name+" " + "This image most likely belongs to GLIOMA with a {:.2f} % confidence.".format(
                    100 * np.max(predictions)))

            if out == 1:
                classifications.append(name+" " + "This image most likely belongs to HEALTHY with a {:.2f} % confidence.".format(
                    100 * np.max(predictions)))
            if out == 2:
                classifications.append(name + " " + "This image most likely belongs to MENINGIOMA with a {:.2f} % confidence.".format(
                    100 * np.max(predictions)))
            if out == 3:
                classifications.append(name + " " + "This image most likely belongs to PITUITARY with a {:.2f} % confidence.".format(
                    100 * np.max(predictions)))
        #return classifications

        #result = classifications
        #print(classifications)
        return render_template('index.html',result=classifications)

    if request.method == 'GET':
        return render_template('index.html')








if __name__ == '__main__':
 app.run(port=3000, debug= True)
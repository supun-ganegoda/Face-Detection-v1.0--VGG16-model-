import os, time, uuid, json, shutil
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import albumentations as alb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16
from face_recognizer import FaceRecognizer

# Collect image samples 
def collectImageSamples():
    imageSource = os.path.join('data', 'faces')
    sampleImageCount = 30
    cap = cv2.VideoCapture(0)
    for img in range(sampleImageCount):
        ret, frame = cap.read()
        if(ret):
            print('Collecting samples {}'.format(img+1))
            imgName = os.path.join(imageSource, f'{str(uuid.uuid1())}.jpg')
            cv2.imwrite(imgName, frame)
            cv2.imshow('frame',frame)
            time.sleep(1)
        else:
            print('Error accessing the camera')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def limitGPU():
    # Limit the GPU usage tensorflow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('GPU :',tf.config.experimental.list_physical_devices('GPU'))

# Tensor flow image loading function
def load_image(x):
        byteImg = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byteImg)
        return img

# Define the image loading function
def loadSampleImages():
    images = tf.data.Dataset.list_files('data\\faces\\*.jpg', shuffle=False)
    print('First loaded image', images.as_numpy_iterator().next())

    images = images.map(load_image)
    print('First tensorFlow transformed image', images.as_numpy_iterator().next())

    imageGenerator = images.batch(4).as_numpy_iterator()
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, image in enumerate(imageGenerator.next()):
        ax[idx].imshow(image)
    plt.show()

# Splitting the dataset
def dataSplit():
    # Automatically split the dataset into train, test and val
    fileNames = os.listdir('data\\faces')
    trainPaths, testPaths = train_test_split(fileNames, test_size=0.2, random_state=42)
    trainPaths, valPaths = train_test_split(trainPaths, test_size=0.1, random_state=42)

    # Create directories if they don't exist
    os.makedirs('data/train/faces', exist_ok=True)
    os.makedirs('data/val/faces', exist_ok=True)
    os.makedirs('data/test/faces', exist_ok=True)

    for path in trainPaths:
        shutil.copy(os.path.join('data','faces',path), os.path.join('data/train/faces', path))

    for path in testPaths:
        shutil.copy(os.path.join('data','faces',path), os.path.join('data/test/faces', path))

    for path in valPaths:
        shutil.copy(os.path.join('data','faces',path), os.path.join('data/val/faces', path))

    # Matching the labels
    for folder in ['train', 'test', 'val']:
        for file in os.listdir(os.path.join('data', folder,'faces')):
            fileName = file.split('.')[0]+'.json'
            extFilePath = os.path.join('data','labels',fileName)
            if os.path.exists(extFilePath):
                newFilePath = os.path.join('data',folder,'labels',fileName)
                os.replace(extFilePath, newFilePath)

def generateAugmentedData():
    augmenter = alb.Compose([alb.RandomCrop(width=450, height=450),
                 alb.HorizontalFlip(p=0.5),
                 alb.RandomBrightnessContrast(p=0.2),
                 alb.RandomGamma(p=0.2),
                 alb.RGBShift(p=0.2),
                 alb.VerticalFlip(p=0.5)],
                 bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

    os.makedirs('data/augData/train/faces', exist_ok=True)
    os.makedirs('data/augData/val/faces', exist_ok=True)
    os.makedirs('data/augData/test/faces', exist_ok=True)
    os.makedirs('data/augData/train/labels', exist_ok=True)
    os.makedirs('data/augData/val/labels', exist_ok=True)
    os.makedirs('data/augData/test/labels', exist_ok=True)

    for partition in ['train', 'test', 'val']:
        for image in os.listdir(os.path.join('data',partition,'faces')):
            img = cv2.imread(os.path.join('data',partition,'faces',image))

            coords = [0,0,0.00001,0.00001]
            labelPath = os.path.join('data', partition,'labels',f'{image.split(".")[0]}.json')
            if os.path.exists(labelPath):
                with open(labelPath, 'r') as f:
                    label = json.load(f)

                    coords[0] = label['shapes'][0]['points'][0][0]
                    coords[1] = label['shapes'][0]['points'][0][1]
                    coords[2] = label['shapes'][0]['points'][1][0]
                    coords[3] = label['shapes'][0]['points'][1][1]
                    coords = list(np.divide(coords, [640,480,640,480]))
            try:
                for x in range(60):
                    augmented = augmenter(image=img, bboxes=[coords], class_labels=['face'])
                    cv2.imwrite(os.path.join('data','augData',partition,'faces',f'{image.split(".")[0]}.{x}.jpg'),augmented['image'])

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(labelPath):
                        if len(augmented['bboxes']) == 0:
                            annotation['bbox'] = [0,0,0,0]
                            annotation['class'] = 0
                        else:
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else:
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0
                    
                    with open(os.path.join('data','augData', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                        json.dump(annotation, f)
            except Exception as e:
                print(e)

# Pre-process images 
def preProcessImage(image):
    image = tf.image.resize(image, (120, 120))
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Label loading function for the tensor-flow
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']

# Develop the model
def buildModel():
    # vgg = VGG16(include_top = False)
    # print(vgg.summary())

    inputLayer = Input(shape=(120,120,3))
    vgg = VGG16(include_top=False)(inputLayer)

    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation = 'relu')(f1)
    class2 = Dense(1, activation = 'sigmoid')(class1)

    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation = 'relu')(f2)
    regress2 = Dense(4, activation = 'sigmoid')(regress1)

    faceRecognizer = Model(inputs=inputLayer, outputs=[class2, regress2])
    return faceRecognizer

# Calculate localization lost
def localizationLoss(yTrue, yHat):
    delta_coord = tf.reduce_sum(tf.square(yTrue[:,:2] - yHat[:,:2]))
                  
    h_true = yTrue[:,3] - yTrue[:,1] 
    w_true = yTrue[:,2] - yTrue[:,0] 

    h_pred = yHat[:,3] - yHat[:,1] 
    w_pred = yHat[:,2] - yHat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size


# Defining the main function
def main(): 
    limitGPU()

    # following needs to run once ------------------

    # collectImageSamples()
    # loadSampleImages()
    # dataSplit()
    # generateAugmentedData()
    # ----------------------------------------------

    # Load data into tensor-flow dataset
    def loadAndPreProcessImages(img):
        return preProcessImage(load_image(img))
    
    dataDirs = ['data\\augData\\train\\faces\\', 'data\\augData\\test\\faces\\', 'data\\augData\\val\\faces\\']
    labelDirs = ['data\\augData\\train\\labels\\', 'data\\augData\\test\\labels\\', 'data\\augData\\val\\labels\\']
    datasets = []
    labelSets = []

    # Loop over directories
    for dataDir in dataDirs:
        # Load files from directory
        dataset = tf.data.Dataset.list_files(dataDir + '*.jpg', shuffle=False)
        # Map loading and preprocessing function
        dataset = dataset.map(loadAndPreProcessImages)
        datasets.append(dataset)

    # Split into train, test, and val datasets
    trainFaces, testFaces, valFaces = datasets

    
    # Load labels to tensorflow pipeline
    for labelDir in labelDirs:
        labelSet = tf.data.Dataset.list_files(labelDir+ '*.json', shuffle=False)
        labelSet = labelSet.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
        labelSets.append(labelSet)
    
    # Split into train, test, and val datasets
    trainLabels, testLabels, valLabels = labelSets
    
    # Printing the pipeline results
    print('Train set: ', len(trainFaces), ' Labels: ',len(trainLabels))
    print('Test set: ', len(testFaces), ' Labels: ',len(testLabels))
    print('Val set: ', len(valFaces), ' Labels: ',len(valLabels))
    
    
    # Generate final dataset
    trainFaces = tf.data.Dataset.zip((trainFaces, trainLabels))
    trainFaces = trainFaces.shuffle(5000).batch(8).prefetch(4)
    testFaces = tf.data.Dataset.zip((testFaces, testLabels))
    testFaces = testFaces.shuffle(5000).batch(8).prefetch(4)
    valFaces = tf.data.Dataset.zip((valFaces, valLabels))
    valFaces = valFaces.shuffle(5000).batch(8).prefetch(4)

    # Visualize the final data samples
    res = trainFaces.as_numpy_iterator().next()

    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx in range(4): 
        sample_image = res[0][idx]
        sample_coords = res[1][1][idx]
        sample_image_umat = cv2.UMat(sample_image)

        cv2.rectangle(sample_image_umat, 
                    tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                    tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                            (255,0,0), 2)

        ax[idx].imshow(cv2.UMat.get(sample_image_umat))
    plt.show()

    # Building the neural network ----------------------
    faceRecognizer = buildModel()
    print(faceRecognizer.summary())

    # Define the optimizers and losses
    batchesPerEpoch = len(trainFaces)
    lrDecay = (1./0.75 - 1)/batchesPerEpoch
    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001, decay = lrDecay)

    # Initialize the loss functions
    classLoss = tf.keras.losses.BinaryCrossentropy()
    regressLoss = localizationLoss

    model = FaceRecognizer(faceRecognizer)
    model.compile(opt,classLoss, regressLoss)

    
    
  
if __name__=="__main__": 
    main() 
import os, time, uuid, json, shutil
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

imageSource = os.path.join('data', 'faces')
sampleImageCount = 30

# Collect image samples 
def collectImageSamples():
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

# Define the image loading function
def loadSampleImages():
    images = tf.data.Dataset.list_files('data\\faces\\*.jpg', shuffle=False)
    print('First loaded image', images.as_numpy_iterator().next())

    def load_image(x):
        byteImg = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byteImg)
        return img

    images = images.map(load_image)
    print('First tensorFlow transformed image', images.as_numpy_iterator().next())

    imageGenerator = images.batch(4).as_numpy_iterator()
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, image in enumerate(imageGenerator.next()):
        ax[idx].imshow(image)
    plt.show()
    return images

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

# Defining the main function
def main(): 
    # collectImageSamples()
    limitGPU()
    images = loadSampleImages()
    dataSplit()
  
  
if __name__=="__main__": 
    main() 
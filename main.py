import os, time, uuid
import cv2

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

# Defining the main function
def main(): 
    collectImageSamples()
  
  
if __name__=="__main__": 
    main() 
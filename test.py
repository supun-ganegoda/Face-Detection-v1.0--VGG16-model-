from tensorflow.keras.models import load_model
import cv2
import numpy as np

faceRecognizerLoaded = load_model('faceRecognizer_epoch_5.keras')

cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = frame[50:500, 50:500, :]
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (120, 120))

    yhat = faceRecognizerLoaded.predict(np.expand_dims(resized/255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.8: 
        # Controls the main rectangle
        x = int(np.multiply(sample_coords[0], 450))
        y = int(np.multiply(sample_coords[1], 450))
        width = int(np.multiply(sample_coords[2], 450)) - x
        height = int(np.multiply(sample_coords[3], 450)) - y

        # Draw the rectangle
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)

        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('Face Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
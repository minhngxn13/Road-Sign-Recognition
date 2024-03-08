import cv2
import numpy as np
# from tensorflow.keras.models import load_weights
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import Sequential

IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,channels)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),

    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),

    keras.layers.Dense(43, activation='softmax')
])


model_path = 'C:/Users/Admin/Desktop/Road-Sign-Recognition/src/model_weights.h5'
model.load_weights(model_path)

# Define a function to preprocess the captured frame
def preprocess_frame(frame):
    image_fromarray = Image.fromarray(frame, 'RGB')
    resize_image = image_fromarray.resize((30, 30))
    expand_input = np.expand_dims(resize_image,axis=0)
    input_data = np.array(expand_input)
    input_data = input_data / 255
    return input_data

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Preprocess the captured frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Make prediction
    predictions = model.predict(preprocessed_frame)
    predicted_index = np.argmax(predictions, axis=1)[0]  # Assuming the batch size is 1
    predicted_label = classes[predicted_index]
    
    # Add prediction to the frame
    cv2.putText(frame, 
                f'Predicted class: {predicted_label}', 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

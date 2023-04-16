import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data_dir = r"C:\Users\G Vishnu Kailash\OneDrive\Desktop\trainig data"


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


class_labels = ['CELOSIA ARGENTEA L', 'CROWFOOT GRASS', 'PURPLE CHLORIS']


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

test_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(class_labels), activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=36, validation_data=test_data)


_, accuracy = model.evaluate(test_data)
print('Test accuracy:', accuracy)


def predict_weed():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if ret:

            img = cv2.resize(frame, IMG_SIZE)
            img = np.expand_dims(img, axis=0)
            img = img / 255.


            pred = model.predict(img)
            class_idx = np.argmax(pred[0])
            class_label = class_labels[class_idx]


            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2
            cv2.putText(frame, class_label, org, font, fontScale, color, thickness, cv2.LINE_AA)


            cv2.imshow('Webcam', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()

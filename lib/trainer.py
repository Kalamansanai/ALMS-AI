from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    
    def __init__(self, cnf):
        self.dataset_path = os.getcwd() + cnf.dataset_path

        self.model = None
        self.history = None

        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None

        self.image_width = cnf.image_width
        self.image_height = cnf.image_height
        self.item_number = cnf.item_number

        self.resize_width = cnf.resize_width
        self.resize_height = cnf.resize_height

        self.epochs = cnf.epochs
        self.batch_size = cnf.batch_size
        self.learning_rate = cnf.learning_rate
        self.loss = cnf.loss
        self.metrics = cnf.metrics
        self.validation_split = cnf.validation_split
        self.validation_data = None
        self.shuffle = cnf.shuffle

    def start(self):
        print("Loading dataset")
        self.load_dataset()
        print("Creating model")
        self.create_model()
        print("Training model")
        self.train_model()
        print("Evaluating model")
        #TODO
        print("Saving model")
        self.save_model()
        print("Summary:")
        self.model.summary()

    def preprocess_image(self, img):

        img = img[110:712 ,180:1000]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img= cv2.resize(img, (self.resize_width, self.resize_height))
        _, img = cv2.threshold(img, 90 , 255, cv2.THRESH_BINARY)
        img = img / 255.0
        return img    

    def load_dataset(self):

        self.train_data = []
        files = os.listdir(self.dataset_path + "/taken_images")
        files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))

        for image in files:
            image_path = os.path.join(self.dataset_path+"/taken_images/", image)
            img = cv2.imread(image_path)
            img = self.preprocess_image(img)
            self.train_data.append(img)
        self.train_data = np.array(self.train_data)
        self.train_data = np.expand_dims(self.train_data, axis=-1)
        

        self.train_labels = []
        f = open(self.dataset_path + "/taken_labels/labels.txt", "r")
        for line in f:
            if (len(line)==0):
                continue
            label = line.strip()[1:-1].split(',')
            label = [int(x) for x in label] 
            self.train_labels.append(label)
        f.close()
        self.train_labels = np.array(self.train_labels)


        print(self.train_data.shape)
        print(self.train_labels.shape)
    
    def create_model(self):
        self.model = Sequential([
            
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.resize_width, self.resize_height, 1)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
                      
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.item_number, activation='sigmoid')  # Output layer
            
        ])

        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss=self.loss,
                           metrics=self.metrics)
        
    def train_model(self):

        early_stopping = EarlyStopping(
            monitor='binary_accuracy',        
            patience=5,                
            restore_best_weights=True, 
            verbose=1                  
        )

        self.model.fit(
            self.train_data, 
            self.train_labels,
            epochs=self.epochs, 
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            shuffle=self.shuffle,
            callbacks = [early_stopping]
        )

        self.history = self.model.history.history

    def evaluate_model(self):
        self.model.evaluate(self.test_data, self.test_labels
                            , batch_size=self.batch_size, verbose=1)
        
    def clear_history(self):
        self.history = None

    def save_model(self):
        self.model.save('model.h5')
        print("Model saved")



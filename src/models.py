import tensorflow
import os
import cv2
import numpy as np
from video import Video
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, BatchNormalization
from keras.models import load_model

# input_shape = (12, 32, 24, 1)
#
# model = Sequential()
# model.add(Conv3D(48, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
# model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
# model.add(Flatten())
# model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
# model.add(Dense(5, activation='softmax'))
#
# model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
#               optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001),
#               metrics=['accuracy'])
# model.summary()

model = load_model('current_model')


def get_model():
    return model


def train(main_dir, epochs=1):
    words = ['ABOUT', 'DIFFERENT', 'EVERY', 'GOING', 'HUMAN']
    label_map = {'ABOUT': np.array([1, 0, 0, 0, 0]), 'DIFFERENT': np.array([0, 1, 0, 0, 0]),
                 'EVERY': np.array([0, 0, 1, 0, 0]), 'GOING': np.array([0, 0, 0, 1, 0]),
                 'HUMAN': np.array([0, 0, 0, 0, 1])}
    about_files = sorted(os.listdir(os.path.join(main_dir, 'ABOUT')))
    different_files = sorted(os.listdir(os.path.join(main_dir, 'DIFFERENT')))
    every_files = sorted(os.listdir(os.path.join(main_dir, 'EVERY')))
    going_files = sorted(os.listdir(os.path.join(main_dir, 'GOING')))
    human_files = sorted(os.listdir(os.path.join(main_dir, 'HUMAN')))
    file_list_map = {'ABOUT': about_files, 'DIFFERENT': different_files, 'EVERY': every_files,
                     'GOING': going_files, 'HUMAN': human_files}
    for i in range(epochs):
        train_data = None
        train_labels = None
        counter = 0
        while counter < 2048:
            word = np.random.choice(words)
            word_file = np.random.choice(file_list_map[word])
            label = label_map[word]
            # print("{} {} {}".format(word, word_file, label))
            vid = Video()
            vid.load(os.path.join(main_dir, word, word_file))
            data = None
            for frame in vid.get_frames_from_source():
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if data is None:
                    data = frame[np.newaxis, :, :]
                else:
                    data = np.vstack((data, frame[np.newaxis, :, :]))
            if train_data is None:
                train_data = data[np.newaxis, :, :, :]
            else:
                train_data = np.vstack((train_data, data[np.newaxis, :, :, :]))
            if train_labels is None:
                train_labels = label[np.newaxis, :]
            else:
                train_labels = np.vstack((train_labels, label[np.newaxis, :]))
            counter += 1
        # counter = 0
        # while counter < 50:
        #     word = np.random.choice(words)
        #     word_file = np.random.choice(file_list_map[word])
        #     label = label_map[word]
        #     # print("{} {} {}".format(word, word_file, label))
        #     vid = Video()
        #     vid.load(os.path.join(main_dir, word, word_file))
        #     data = None
        #     for frame in vid.get_frames_from_source():
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #         if data is None:
        #             data = frame[np.newaxis, :, :]
        #         else:
        #             data = np.vstack((data, frame[np.newaxis, :, :]))
        #     if test_data is None:
        #         test_data = data[np.newaxis, :, :, :]
        #     else:
        #         test_data = np.vstack((test_data, data[np.newaxis, :, :, :]))
        #     if test_labels is None:
        #         test_labels = label[np.newaxis, :]
        #     else:
        #         test_labels = np.vstack((test_labels, label[np.newaxis, :]))
        #     counter += 1
        print("Epoch = {}".format(i))
        model.fit(train_data, train_labels, batch_size=16)
        # model.save('current_model')






# train('word_data', 5)
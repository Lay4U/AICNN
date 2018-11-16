from numpy.random import seed
import random
from tensorflow import set_random_seed
import os
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Activation, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras import optimizers
from keras import regularizers
from sklearn.metrics import confusion_matrix, log_loss
import numpy as np
from util import load_data, draw_result, n2c
import time
import tensorflow as tf
import winsound
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

duration = 1000  # millisecond
freq = 440  # Hz

config = tf.ConfigProto()

config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.8

session = tf.Session(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed(0)
random.seed(1)
set_random_seed(2)
os.environ['PYTHONHASHSEED'] = '0'
f = open("result.txt", 'w')


class ModelMgr():
    def __init__(self, target_class=[3, 5], use_validation=True):
        self.target_class = target_class
        self.use_validation = use_validation
        print('\nload dataset')
        if use_validation:
            (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = \
                load_data(target_class, use_validation=use_validation)
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = \
                load_data(target_class, use_validation=use_validation)

    def train(self):
        print('\ntrain model')

        model = self.get_model()  # 모델 가져오기
        # model = self.get_model_sample_1()  # 예시 모델1
        # model = self.get_model_sample_2()  # 예시 모델2

        hp = self.get_hyperparameter()  # 파이퍼파라미터 로드

        temp = model.summary()  # 모델 구조 출력
        print('hyperparameters :')
        print('\tbatch size :', hp['batch_size'])
        print('\tepochs :', hp['epochs'])
        print('\toptimizer :', hp['optimizer'].__class__.__name__)
        print('\tlearning rate :', hp['learning_rate'])
        # 모델의 손실 함수 및 최적화 알고리즘 설정
        model.compile(optimizer=hp['optimizer'],
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # if hp['epochs'] > 20:  # epochs은 최대 20로 설정 !!
        # hp['epochs'] = 20
        if self.use_validation:
            validation_data = (self.x_val, self.y_val)
        else:
            validation_data = (self.x_test, self.y_test)

        # 모델 학습
        history = model.fit(self.x_train, self.y_train,
                            batch_size=hp['batch_size'],
                            epochs=hp['epochs'],
                            validation_data=validation_data,
                            shuffle=False,
                            verbose=2)

        history.history['hypers'] = hp
        self.model = model
        self.history = history

    def get_hyperparameter(self):
        hyper = dict()
        ############################
        '''
        (1) 파라미터 값들을 수정해주세요
       '''
        hyper['batch_size'] = 32  # 배치 사이즈
        hyper['epochs'] = 20  # epochs은 최대 20 설정 !!
        # hyper['learning_rate'] = 0.01  # 학습률
        hyper['learning_rate'] = 0.01  # 학습률
        # 최적화 알고리즘 선택 [sgd, rmsprop, adagrad, adam 등]
        # hyper['optimizer'] = optimizers.sgd(lr=hyper['learning_rate'])  # default: SGD
        # hyper['optimizer'] = optimizers.rmsprop(lr=0.0001, decay=1e-6)
        hyper['optimizer'] = optimizers.adam()
        result = 'batch_size: {}\nepochs: {}\nlearning_rage: {}\noptimizer: {}\n'.format(hyper['batch_size'],
                                                                                         hyper['epochs'], \
                                                                                         hyper['learning_rate'],
                                                                                         hyper['optimizer'])
        f.write(result)
        ############################
        '''
        optimizer = Adam(lr=1e-3)
        '''
        return hyper

    def get_model(self):
        nDropout = 0.25
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        model.add(Dropout(nDropout))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Dropout(nDropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(nDropout))
        '''
        non
        Last Accuracy: 0.955875

Valid error: 0.7702255444414914
        mapool
        Last Accuracy: 0.975875
        

Valid error: 1.0640612498112023

down conv
        Last Accuracy: 0.971125

Valid error: 0.9136082481145859
        '''
        # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        nDropout = 0.5
        '''
0.5
Last Accuracy: 0.883

Valid error: 0.42263877797126775

 0.6
Last Accuracy: 0.663125

Valid error: 0.0037601342201233345

0.4
Last Accuracy: 0.956875

Valid error: 0.9670988367386162

        '''

        nUnit = 1024
        model.add(Flatten())
        model.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        '''
                model.add(Dense(2048, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
                model.add(Dropout(nDropout))
                model.add(Dense(1024, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
                model.add(Dropout(nDropout))
                model.add(Dense(512, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
                model.add(Dropout(nDropout))
                model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
                model.add(Dropout(nDropout))
                model.add(Dense(128, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
                model.add(Dropout(nDropout))
        
Last Accuracy: 0.87975

Valid error: 0.5065860754251481

        model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(2048, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
Last Accuracy: 0.890125

Valid error: 0.5184675998985767
        '''

        '''
        
        128
        Last Accuracy: 0.747125
        
        Valid error: 0.05675763714313509
        
        256
        Last Accuracy: 0.842625
        
        Valid error: 0.4056897324025631
        
        512
        Last Accuracy: 0.9385
        
        Valid error: 0.7423957723230123
        
        1024
        Last Accuracy: 0.966875
        
        Valid error: 1.5335488475980237
        '''


        # model.add(Dropout(nDropout))
        # model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        # model.add(Dropout(nDropout))
        # model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        # model.add(Dropout(nDropout))

        # model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal', bias_initializer='glorot_uniform'))
        # model.add(Dropout(nDropout))
        model.add(Dense(len(self.target_class)))
        model.add(Activation('softmax'))

        '''
        model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(516, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        사용시 
        Last Accuracy: 0.930375
        Valid error: 0.63189967565611
        
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        사용시
        Last Accuracy: 0.84875

        Valid error: 0.2601667023301124
        
        
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        사용시
        Last Accuracy: 0.949625

        Valid error: 0.691403053805232
        
                model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dense(256, activation='relu', kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform'))
        사용시
        Last Accuracy: 0.94575

Valid error: 0.7000599619597196
        '''

        # model = Sequential()
        ################
        '''
        (2) 모델 코드를 완성해주세요.
        model.add(...)
        '''
        ################
        '''
        주의사항
        1. 모델의 입력 데이터 크기는 (batch_size, 32, 32, 1) # 고양이 or 강아지 흑백 사진
           출력 데이터 크기는 (batch_size, 2) # 고양이일 확률, 강아지일 확률
        2. 최초 Dense() 사용 시, Flatten()을 먼저 사용해야함
        3. out of memory 오류 시,
            메모리 부족에 의한 오류임.
            batch_size를 줄이거나, 모델 구조의 파라미터(ex. 유닛수)를 줄여야함
        4. BatchNormalization() 사용 금지

        기타 문의 : sdh9446@gmail.com (수업조교)
        '''
        return model

    def get_modezq(self):
        '''
        Last Accuracy: 0.927125

        Valid error: 0.5721124948859215
        '''
        nDropout = 0.3
        model = Sequential()
        model.add(Conv2D(16, (2, 2), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        model.add(Conv2D(16, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(nDropout))

        model.add(Conv2D(32, (2, 2), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(nDropout))

        model.add(Conv2D(64, (2, 2), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(nDropout))
        # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.3))

        nDropout = 0.4  # 0.3 88,0.4     0.4 93,0.78     #0.2 97,1.04
        model.add(Flatten())
        # model.add(Dense(4096, activation='relu'))
        # model.add(Dropout(nDropout))
        # model.add(Dense(2048, activation='relu'))
        # model.add(Dropout(nDropout))
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dropout(nDropout))

        # model.add(Dense(64, activation='relu')) #128: 0.6045 256: 0.74
        # model.add(Dropout(nDropout))
        # model.add(Dense(64, activation='relu')) #128: 0.6045 256: 0.74
        # model.add(Dropout(nDropout))
        model.add(Dense(2048, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform'))
        # model.add(Dense(1024, activation='relu')) #128: 0.6045 256: 0.74
        model.add(Dropout(nDropout))
        # model.add(Dense(128, activation='relu')) #128: 0.6045 256: 0.74
        # model.add(Dropout(nDropout))
        model.add(Dense(len(self.target_class)))
        model.add(Activation('softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # model = Sequential()
        # model.add(Conv2D(16, (5, 5),strides=(1,1), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        #
        # model.add(Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu'))
        # model.add(Conv2D(32, (3, 3), strides=(1,1), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.2))
        #
        # model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        #
        #
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(len(self.target_class)))
        # model.add(Activation('softmax'))

        # model = Sequential()
        ################
        '''
        (2) 모델 코드를 완성해주세요.
        model.add(...)
        '''
        ################
        '''
        주의사항
        1. 모델의 입력 데이터 크기는 (batch_size, 32, 32, 1) # 고양이 or 강아지 흑백 사진
           출력 데이터 크기는 (batch_size, 2) # 고양이일 확률, 강아지일 확률
        2. 최초 Dense() 사용 시, Flatten()을 먼저 사용해야함
        3. out of memory 오류 시,
            메모리 부족에 의한 오류임.
            batch_size를 줄이거나, 모델 구조의 파라미터(ex. 유닛수)를 줄여야함
        4. BatchNormalization() 사용 금지
        기타 문의 : sdh9446@gmail.com (수업조교)
        '''
        return model

    def test(self, model=None):
        print('\ntest model')
        if model is None:
            model = self.model
        start = time.time()
        y_pred = model.predict(self.x_test, batch_size=1, verbose=0)
        end = time.time()

        y_true = np.argmax(self.y_test, -1)
        loss = log_loss(y_true, y_pred)
        y_pred = np.argmax(y_pred, -1)
        y_true = np.argmax(self.y_test, -1)
        cmat = confusion_matrix(y_pred, y_true)
        acc_per_class = cmat.diagonal() / cmat.sum(axis=1)

        print('\n===== TEST RESULTS ====')
        print('Test loss:', str(loss)[:6])
        print('Test Accuracy:')
        print('\tTotal: {}%'.format(str(acc_per_class.mean() * 100)[:5]))
        for idx, label in n2c.items():
            print('\t{}: {}%'.format(label, str(acc_per_class[idx] * 100)[:5]))
        print('Test FPS:', str(1 / ((end - start) / len(self.x_test)))[:6])
        print('=======================')
        if hasattr(self, 'history'):
            self.history.history['test_acc'] = acc_per_class.mean()

        # model_path = './trained_model.h5'
        result = 'Test loss: {}\nTotal: {}\n'.format(str(loss), str(acc_per_class.mean() * 100)[:5])
        f.write(result)

        return acc_per_class.mean()

    def save_model(self, model_path='./trained_model.h5'):
        print('\nsave model : \"{}\"'.format(model_path))
        self.model.save(model_path)

    def load_model(self, model_path='./trained_model.h5'):
        print('\nload model : \"{}\"'.format(model_path))
        self.model = load_model(model_path)

    def draw_history(self, file_path='./result.png'):
        print('\nvisualize results : \"{}\"'.format(file_path))
        draw_result(self.history.history, self.use_validation, file_path=file_path)
        result = '\n\nAccuracy: {}\nLoss: {}\n'.format(
            self.history.history['acc'][len(self.history.history['acc']) - 1],
            self.history.history['loss'][len(self.history.history['loss']) - 1])
        f.write(result)
        print('\nLast Accuracy: {}'.format(self.history.history['acc'][len(self.history.history['acc']) - 1]))
        print('\nValid error: {}'.format(
            self.history.history['val_loss'][len(self.history.history['val_loss']) - 1] - self.history.history['loss'][
                len(self.history.history['loss']) - 1]))


if __name__ == '__main__':
    trained_model = None
    # trained_model = './trained_model.h5'  # 학습된 모델 테스트 시 사용

    modelMgr = ModelMgr()
    if trained_model is None:
        modelMgr.train()
        # modelMgr.save_model('./trained_model.h5')  # 모델 저장 (이름이 같으면 덮어씀)
        modelMgr.test()
        modelMgr.draw_history('./result.png')  # 학습 결과 그래프 저장 (./result.png)
    else:
        modelMgr.load_model(trained_model)
        modelMgr.test()

    f.close()
    winsound.Beep(freq, duration)  # 코드 실행 끝나면 비프음 나도록

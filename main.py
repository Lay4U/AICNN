from tensorflow import keras
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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.optimizers import SGD

duration = 1000  # millisecond
freq = 440  # Hz

config = tf.ConfigProto()

config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.7

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
        a, b, c, d = self.x_train.shape
        train_x = self.x_train.reshape((a, b * c))
        e, f = self.y_train.shape
        train_y = self.y_train.reshape((e, f))

        # rf = RandomForestRegressor()
        # rf = rf.fit(train_x, train_y)

        # https://tykimos.github.io/2017/07/09/Early_Stopping/
        from keras.callbacks import EarlyStopping

        early_stopping = EarlyStopping(monitor='val_loss', verbose=2, patience=5)
        # rf = RandomForestRegressor()
        # rf.fit(train_x, train_y)
        history = model.fit(np.array(self.x_train), np.array(self.y_train),
                            batch_size=hp['batch_size'],
                            epochs=hp['epochs'],
                            validation_data=validation_data,
                            shuffle=False,
                            verbose=2,
                            # callbacks=[early_stopping]
                            )
        history.history['hypers'] = hp
        self.model = model
        self.history = history

        # rf = KerasRegressor(build_fn=self.get_model)
        # rf = rf.fit(self.x_train, self.y_train,
        #                     batch_size=hp['batch_size'],
        #                     epochs=hp['epochs'],
        #                     validation_data=validation_data,
        #                     shuffle=False,
        #                     verbose=2)

    def get_hyperparameter(self):
        hyper = dict()
        ############################
        '''
        (1) 파라미터 값들을 수정해주세요
       '''
        hyper['batch_size'] = 32  # 배치 사이즈
        hyper['epochs'] = 20  # epochs은 최대 20 설정 !!
        # hyper['learning_rate'] = 0.01  # 학습률
        hyper['learning_rate'] = 1  # 학습률
        # 최적화 알고리즘 선택 [sgd, rmsprop, adagrad, adam 등]
        # hyper['optimizer'] = optimizers.sgd(lr=hyper['learning_rate'])  # default: SGD
        # hyper['optimizer'] = optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # deep learning libraries generally use the default parameters recommended by the paper.
        # Keras: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
        # adam parameter
        # def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
        #              epsilon=None, decay=0., amsgrad=False, **kwargs):
        # lr: float >= 0. Learning rate.
        # beta_1: float, 0 < beta < 1. Generally close to 1.
        # beta_2: float, 0 < beta < 1. Generally close to 1.
        # epsilon: float >= 0. Fuzz factor. If None, defaults to K.epsilon().
        # decay: float >= 0. Learning rate decay over each update.
        # amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond".

        hyper['optimizer'] = optimizers.adam(epsilon=1e-04, decay=0.001)
        result = 'batch_size: {}\nepochs: {}\nlearning_rage: {}\noptimizer: {}\n'.format(hyper['batch_size'],
                                                                                         hyper['epochs'], \
                                                                                         hyper['learning_rate'],
                                                                                         hyper['optimizer'])
        f.write(result)
        ############################

        return hyper

    def get_model(self):

        model = Sequential()
        nDropout = 0.1

        model.add(Conv2D(32, (2, 2), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))

        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))

        model.add(Flatten())
        model.add(Dense(8192, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))
        model.add(Dropout(nDropout))

        model.add(Dense(2))
        model.add(Activation('softmax'))
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

    def wtf(self):
        kr = KerasRegressor(build_fn=self.get_model, nb_epoch=100, batch_size=5, verbose=2)
        kr = kr.fit(self.x_train, self.y_train)
        return kr


if __name__ == '__main__':
    trained_model = None
    # trained_model = './trained_model.h5'  # 학습된 모델 테스트 시 사용

    modelMgr = ModelMgr()
    if trained_model is None:
        modelMgr.train()
        modelMgr.save_model('./trained_model.h5')  # 모델 저장 (이름이 같으면 덮어씀)
        modelMgr.test()
        modelMgr.draw_history('./result.png')  # 학습 결과 그래프 저장 (./result.png)
    else:
        modelMgr.load_model(trained_model)
        modelMgr.test()

    f.close()
    # winsound.Beep(freq, duration)  # 코드 실행 끝나면 비프음 나도록
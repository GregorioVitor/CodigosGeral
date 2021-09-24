"""
Created on Fri Sep 18 22:52:21 2020

@author: gregorio

Aluno: Vitor Gregorio
R.A.:1827588

"""


import keras
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from skimage.io import imread
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv
import time
import os

batch= 32 # Exemplos utilizados para um determinado processo
epochs = 200
lr = 0.001 #taxa de aprendizado ou otimização de erro
train_model = 'gnn'
path_train = '/home/gregorio/Área de Trabalho/Linux/Trabalho final/archive/data/natural_images'
path_test = '/home/gregorio/Área de Trabalho/Linux/Trabalho final/archive/natural_images'

#função para abrir as imagens para os deep features
def load_image_files(container_path):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    count = 0
    train_img = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            count += 1
            img = imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_pred = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
            img_pred = image.img_to_array(img_pred)
            img_pred = img_pred / 255
            train_img.append(img_pred)

    X = np.array(train_img)

    return X


base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape = (224,224,3) )

#x recebe o final da VGG19

x=base_model.output
x=tf.keras.layers.Flatten()(x)

#adiciona apos x duas camada densa com 4096 neuronios com funcao de ativacao relu. Atribui este no a x novamente
x=tf.keras.layers.Dense(4096,activation='relu')(x)
x=tf.keras.layers.Dense(4096,activation='relu')(x)

#adiciona apos x uma camada densa com 7 neuronios (sete classes) com funcao de ativacao softmax (distribuicao de probabilidade). Atribui este no a preds
preds=tf.keras.layers.Dense(8,activation='softmax', name='predictions')(x)

#definindo modelo final
model=tf.keras.models.Model(inputs=base_model.input,outputs=preds)

#mostrando modelo final e sua estrutura
model.summary()

#congelando os neuronios já treinados na ImageNet, queremos retreinar somente a ultima camada
for l in model.layers:
  if l.name.split('_')[0] != 'dense':
    l.trainable=False
  else:
    l.trainable=True

lr = keras.optimizers.Adam(learning_rate=lr)

model.compile(optimizer=lr, loss='categorical_crossentropy', metrics=['accuracy'])


#chamada para abrir as imagens
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input) #included in our dependencies
test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg19.preprocess_input)
train_generator = train_data_gen.flow_from_directory(path_train, target_size=(224,224),color_mode='rgb',batch_size=batch,class_mode='categorical',shuffle=True)
test_generator = train_data_gen.flow_from_directory(path_test, target_size=(224,224),color_mode='rgb', batch_size=batch,class_mode='categorical',shuffle=True)

step_size_train = train_generator.n // train_generator.batch_size
step_size_test = test_generator.n // test_generator.batch_size

# treinando e testando o modelo
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=step_size_train,
                              epochs=epochs,
                              validation_data=test_generator,
                              validation_steps=step_size_test)




#Avaliando o modelo
loss_train, train_acc = model.evaluate_generator(train_generator, steps=step_size_train)
loss_test, test_acc = model.evaluate_generator(test_generator, steps=step_size_test)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
#Apresentando resultados em graficos
plt.title('Loss')
plt.plot(history.history['loss'], label=path_train)
plt.plot(history.history['val_loss'], label=path_test)
plt.legend()
plt.show()


# Criando graficos para visualização dos resultados
print("\n\n")
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label=path_train)
plt.plot(history.history['val_accuracy'], label=path_test)
plt.legend()
plt.show()

print('Criando classificações..')
labels = os.listdir(path_train)
print('Rótulos', labels)


#criando estruturas para métricas de avaliação, processo um pouco mais demorado
Y_pred = model.predict_generator(test_generator)
print('Preds Created')
y_pred = np.argmax(Y_pred, axis=1)
print('Preds 1D created')

classification = classification_report(test_generator.classes, y_pred, target_names=labels)
print('----------------CLASSIFICATION--------------')
print(classification)
matrix = confusion_matrix(test_generator.classes, y_pred)
df_cm = pd.DataFrame(matrix, index = [i for i in range(7)],
                  columns = [i for i in range(7)])
fig2 = plt.figure(figsize = (10,7))
fig2.suptitle('Matriz de confusão')
sn.heatmap(df_cm, annot=True, linewidths=2.5)


#abrir as imagens para o deep 
index=0
X = []
X = load_image_files(path_train)
Y = train_generator.classes
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=327)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, random_state=42, test_size=0.5)


num_training = X_train.shape[0]
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
acc_train = []
acc_test = []
precision = []
recall = []

print("\n\n")
#Avaliando o modelo

if (train_model == 'gnn'):
    # Gaussian Naive Bayes
    t = time.time()
    gaussian = GaussianNB()
    model1 = gaussian.fit(X_train, y_train)
    print('Tempo do treino do Gaussian Naive Bayes: {}'.format(time.time() - t))
        
    t = time.time()
    aux = gaussian.predict(X_test)
    cm = confusion_matrix(y_test, aux)
    precision.append(precision_score(y_test, aux, average='macro'))
    recall.append(recall_score(y_test, aux, average='macro'))
    acc_train.append(gaussian.score(X_train, y_train))
    acc_test.append(gaussian.score(X_test, y_test))
    
    print('\nAcuracia do Gaussian Naive Bayes no Treinamento: {:.2f}'.format(acc_train[0]))
    print('Acuracia do Gaussian Naive Bayes no Teste: {:.2f}'.format(acc_test[0]))
    print('Precisão: {:.5f}'.format(precision[0]))
    print('Revocação: {:.5f}'.format(recall[0]))
    print('(Tempo de teste: {:.5f})'.format(time.time() - t))
    df_cm = pd.DataFrame(cm, index = [i for i in range(7)],columns = [i for i in range(7)])
    fig = plt.figure(figsize = (10,7))
    fig.suptitle('Gaussian Naive Bayes Matriz de confusão')
    sn.heatmap(df_cm, annot=True, linewidths=2.5)
    print('\n')
    
elif(train_model == 'knn'):   
    # K-Nearest Neighbors
    t = time.time()
    knn = KNeighborsClassifier(26)
    model2 = knn.fit(X_train, y_train)
    print('Tempo do treino do K-Nearest Neighbors terminado: {}'.format(time.time() - t))
        
    t = time.time()
    aux = knn.predict(X_test)
    cm = confusion_matrix(y_test, aux)
    precision.append(precision_score(y_test, aux, average='macro'))
    recall.append(recall_score(y_test, aux, average='macro'))
    acc_train.append(knn.score(X_train, y_train))
    acc_test.append(knn.score(X_test, y_test))
    
    print('\nAcuracia do K-Nearest Neighbors no Treinamento: {:.2f}'.format(acc_train[0]))
    print('Acuracia do K-Nearest Neighbors no Teste: {:.2f}'.format(acc_test[0]))
    print('Precisão: {:.5f}'.format(precision[0]))
    print('Revocação: {:.5f}'.format(recall[0]))
    print('(Tempo de teste: {:.5f})'.format(time.time() - t))
    df_cm = pd.DataFrame(cm, index = [i for i in range(7)],columns = [i for i in range(7)])
    fig = plt.figure(figsize = (10,7))
    fig.suptitle('K-Nearest Neighbors Matriz de confusão')
    sn.heatmap(df_cm, annot=True, linewidths=2.5)
    print('\n')
    
elif(train_model =='dtc'):
    # Decision Tree
    t = time.time()
    dtc = DecisionTreeClassifier()
    model3 = dtc.fit(X_train, y_train)
    print('Treino do Decision Tree Terminado. (Tempo de execucao: {})'.format(time.time() - t))
        
    t = time.time()
    aux = dtc.predict(X_test)
    cm = confusion_matrix(y_test, aux)
    precision.append(precision_score(y_test, aux, average='macro'))
    recall.append(recall_score(y_test, aux, average='macro'))
    acc_train.append(dtc.score(X_train, y_train))
    acc_test.append(dtc.score(X_test, y_test))
    
    print('\nAcuracia do Decision Tree no Treinamento: {:.2f}'.format(acc_train[0]))
    print('Acuracia do Decision Tree no Teste: {:.2f}'.format(acc_test[0]))
    print('Precisão: {:.5f}'.format(precision[0]))
    print('Revocação: {:.5f}'.format(recall[0]))
    print('(Tempo de teste: {:.5f})'.format(time.time() - t))
    df_cm = pd.DataFrame(cm, index = [i for i in range(7)],columns = [i for i in range(7)])
    fig = plt.figure(figsize = (10,7))
    fig.suptitle('Decision Tree Matriz de confusão')
    sn.heatmap(df_cm, annot=True, linewidths=2.5)
    print('\n')
    

import os
import csv
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dense, Flatten, BatchNormalization, LayerNormalization
from keras.optimizers import SGD, Adam, Adamax
from tensorflow.keras.initializers import he_normal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

DATASET_PATH = '/gpfs/projects/nct00/nct00001/mame/data_256'

np.random.seed(42)
tf.random.set_seed(314)

batch_size = 128
n_epochs = 50


train_datagen = ImageDataGenerator(  
    rescale=1. / 255,
    rotation_range = 15,
    width_shift_range=30.0,
    height_shift_range=30.0,
    brightness_range=[0.5, 1.3],
    shear_range=10, # potser baixar a 5
    zoom_range=(0.9, 1.3),
    channel_shift_range=20,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATASET_PATH,'train'),
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42)

val_datagen = ImageDataGenerator(
    rescale=1 / 255.0)

val_generator = val_datagen.flow_from_directory(
    directory=os.path.join(DATASET_PATH,'val'),
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)

test_datagen = ImageDataGenerator(
    rescale=1 / 255.0)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(DATASET_PATH,'test'),
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)

def load_data_labels(filename):
    labels = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(row[1])
    return labels

def plot_training_curve(history):
    # Plot the training and validation loss and accuracy
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(history.history['loss'], label='train_loss')
    ax[0].plot(history.history['val_loss'], label='val_loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(history.history['accuracy'], label='train_acc')
    ax[1].plot(history.history['val_accuracy'], label='val_acc')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].grid(True)
    ax[1].legend()
    
    # Save the training curves with a timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('Timestamp curves of this experiment:',timestamp)
    plt.savefig(f'training_curves_{timestamp}.pdf')

def train_perceptron():
    # Create model architecture
    img_rows, img_cols, channels = 256, 256, 3
    input_shape = (img_rows, img_cols, channels)

    model = Sequential()  # [B, 256, 256, 3]
    # 256 x 256
    model.add(
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=he_normal(),
               strides=2))  # [B, 128, 128, 32]
    # 128 x 128
#    model.add(Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer=he_normal())) # [B, 32, 256, 256]  // # [B, 128, 128, 64]
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_initializer=he_normal()))  # [B, 32, 256, 256]  // # [B, 128, 128, 64]
    model.add(MaxPooling2D(pool_size=2))
    # 64 x 64
    #model.add(Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer=he_normal())) # [B, 32, 256, 256]  // # [B, 64, 64, 64]
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_initializer=he_normal()))  # [B, 32, 256, 256]  // # [B, 64, 64, 128]
    model.add(MaxPooling2D(pool_size=2))
    # 32 x 32
    #model.add(Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer=he_normal())) # [B, 32, 256, 256]  // # [B, 32, 32, 128]
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_initializer=he_normal()))  # [B, 32, 256, 256]  // # [B, 32, 32, 256]
    model.add(MaxPooling2D(pool_size=2))
    # 16 x 16
    #model.add(Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer=he_normal())) # [B, 32, 256, 256]  // # [B, 16, 16, 256]
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                     kernel_initializer=he_normal()))  # [B, 32, 256, 256]  // # [B, 16, 16, 256]
    model.add(MaxPooling2D(pool_size=2))
    # 8 x 8
    #model.add(Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer=he_normal())) # [B, 32, 256, 256]  // # [B, 8, 8, 256]
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_initializer=he_normal()))  # [B, 32, 256, 256]  // # [B, 8, 8, 512]
    model.add(MaxPooling2D(pool_size=2))
    # 4 x 4
    #model.add(Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer=he_normal())) # [B, 32, 256, 256]  // # [B, 4, 4, 512]
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_initializer=he_normal()))  # [B, 32, 256, 256]  // # [B, 4, 4, 512]
    # Flat:
    model.add(GlobalAveragePooling2D())  # [B, 1, 1, 32]
    model.add(Flatten())  # [B, 32]  # 256
    model.add(Dense(128, activation='relu',
                    kernel_initializer=he_normal()))  # [B, 16] # ALARMA ! Reduim a 16 quan despres volem 29  // 128
    model.add(Dense(29, activation='softmax', kernel_initializer=he_normal()))  # [B, 29]

    # Choose optimizer and compile the model
    learning_rate = 0.001
    #adam = Adam(lr=learning_rate)
    adam = keras.optimizers.Adam(learning_rate=learning_rate) #, weight_decay= learning_rate / epochs) , decay= learning_rate / epochs)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    #Check model summary!
    print(model.summary())
    
    #Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

    # Train the model
    history = model.fit(train_generator, validation_data = val_generator, epochs=n_epochs, verbose=1, callbacks=[early_stop],  steps_per_epoch=len(train_generator), validation_steps=len(val_generator))
    
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate_generator(val_generator, verbose=1)
    print("Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(loss, accuracy))
    #Classification outputs
    y_pred = model.predict_generator(val_generator)
    y_true = val_generator.classes
    print('y_pred:',y_pred)
    #Assign most likely label
    y_pred = np.argmax(y_pred, axis=1)
    #Read data labels
    labels = load_data_labels(os.path.join(DATASET_PATH,'..','MAMe_labels.csv'))
    print(labels) 
    print(classification_report(y_true, y_pred,target_names=labels))
    print(confusion_matrix(y_true, y_pred))
    #Curves
    print('Plotting curves')
    plot_training_curve(history)
    print('Plotting curves done')

if __name__ == "__main__":
    train_perceptron()

# 0. Les primeres intuicions:
#   - A partir del train/val loss inicial, està clar que no fa overfitting. S'ha d'incrementar la complexitat
#   - Al començar el projecte ja ens hem de plantejar si podem fer augmentation. Al principi només aplica simetria horitzontal, però s'ha de mirar si es pot fer més (depen de la tasca)
#   - L'optimització es amb SGD però ho canviariem a Adam
#   - Mirant la xarxa, no processem bé la informació. No reduim adecuadament la dimensionalitat (Amb el avg-pooling)
#   - Learning rate massa gran! Encara que no exploti durant l'entrenament amb la xarxa inicial, en general 0.1 es massa gran. Provem amb 0.001 de moment. Podriem afegir un schedule per baixar-lo progressivament.
# 1. Visualitzar les imatges i entendre la tasca (que estem predint? El preu del quadre? El pintor?)
# 2. Calcular el tamany de la imatge al final de cada filtre. Abans del pooling no ha


# DEBUG:
#img_rows, img_cols, channels = 256, 256, 3
#input_shape = (img_rows, img_cols, channels)
#layer1 = Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer=he_normal()) # [B, 256, 256, 32]
#layer2 = Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer=he_normal()) # [B, 32, 256, 256]  // # [B, 128, 128, 64]
#layer3 = GlobalAveragePooling2D() # [B, 1, 1, 32]   # ALARMA! Reduint dimensionalitat 32x256x256 -> 32
#layer4 = Flatten() # [B, 32]  # 256
#layer5 = Dense(16, activation='relu', kernel_initializer=he_normal()) # [B, 16] # ALARMA ! Reduim a 16 quan despres volem 29  // 128
#layer6 = Dense(29, activation='softmax', kernel_initializer=he_normal()) # [B, 29]
#image_input = tf.convert_to_tensor(np.zeros((1, 256, 256, 3)))
#image2 = layer1(image)

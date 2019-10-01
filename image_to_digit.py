import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
from PIL import Image, ImageFilter
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import image_slicer

'''
(1) Model Build
'''

# Import the datasets to train the model
file = r'C:\Data\YOUELLT\My Documents\Digit Recognizer\data\\'
train = pd.read_csv(file + 'train.csv')
test = pd.read_csv(file + 'test.csv')

# check for nulls
train.isnull().any().describe()
test.isnull().any().describe()

# check the counts of the labels
print(Counter(train['label']))
sns.countplot(train['label'])

# split dataset into features (X) and target variable (Y)
X_train = (train.iloc[:, 1:].values).astype('float32')
Y_train = train.iloc[:, 0].values.astype('int32')

X_test = test.values.astype('float32')

# Observe some digit examples
plt.figure(figsize=(12, 6))
x, y = 10, 4
for i in range(40):
    plt.subplot(y, x, i + 1)
    plt.imshow(X_train[i].reshape((28, 28)), interpolation='nearest')
plt.show()


# See an example of how pixel values are defined on the gray scale
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')

# convert the training set into integer values
X_train = (train.iloc[:, 1:].values).astype('int32')
# define figure size
fig = plt.figure(figsize=(12, 12))
# define subplots
ax = fig.add_subplot(111)
# execute visualize_input function with first element in training set
visualize_input(X_train[1].reshape(28, 28), ax)
# converts training set back into float values
X_train = (train.iloc[:, 1:].values).astype('float32')

# Normalize the data (CNNs converg faster on [0->1] rather than [0->255])
X_train = X_train/255.0
X_test = X_test/255.0

# Reshape the data into 28pixelx28pixel 'images'
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
# Dummy transforming the labels. A 2 in the label column would give:
# label = 0: 0, label = 1: 0, label = 2: 1, label = 3: 0, etc...
Y_train = to_categorical(Y_train, num_classes=10)

# Splitting training into train and validation (test).
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                  test_size=0.1,
                                                  random_state=42)

# Set the CNN model
# this CNN architechture is:
# In->[[Conv2D->relu]*2->MaxPool2D->Dropout]*2->Flatten->Dense->Dropout->Out

# batch size = dividing the dataset into a number of batches.
batch_size = 64
# epoch = number of iterations over entire dataset.
epochs = 20
# input shape of 28x28pixels
input_shape = (28, 28, 1)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_normal', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

# Define the optimizer and compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)

# Data augmentation to alter the images to imporve the model + stop overfitting
datagen = ImageDataGenerator(  # set input mean to 0 over the dataset
                             featurewise_center=False,
                               # set each sample mean to 0
                             samplewise_center=False,
                               # divide inputs by std of the dataset
                             featurewise_std_normalization=False,
                               # divide each input by its std
                             samplewise_std_normalization=False,
                               # apply ZCA whitening
                             zca_whitening=False,
                               # randomly rotate images in the range (0 to 180)
                             rotation_range=15,
                               # Randomly zoom image
                             zoom_range=0.1,
                               # randomly shift images horizontally
                               # (fraction of total width)
                             width_shift_range=0.1,
                               # randomly shift images vertically
                               # (fraction of total height)
                             height_shift_range=0.1,
                               # randomly flip images
                             horizontal_flip=False,
                               # randomly flip images - we do not want this as
                               # it  messes up the digits 6 and 9.
                             vertical_flip=False)

# View a summary of the model
# model.summary()

datagen.fit(X_train)
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(X_val, Y_val),
                        verbose=1,
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        callbacks=[learning_rate_reduction],)


final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)
print("validation loss: {0:.6f}, validation accuracy: {1:.6f}".format(final_loss, final_acc))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=range(11))


print(h.history.keys())
accuracy = h.history['acc']
val_accuracy = h.history['val_acc']
loss = h.history['loss']
val_loss = h.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')

plt.title('training and validation accuracy')

plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')

predicted_classes = model.predict_classes(X_test)
submissions = pd.DataFrame({"ImageId": list(range(1, len(predicted_classes)+1)),
                           "Label": predicted_classes})

submissions.to_csv(file + 'submission\TY_submission.csv', index=False)

    
model.save('my_model_1.h5')
json_string = model.to_json()


errors = (Y_pred_classes - Y_true != 0)
Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
            n += 1


# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


def imageprepare(argv):
    '''
    This function returns the pixel values.
    The input is a png file location.
    '''
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    # create white canvas of 28x28pixels.
    newImage = Image.new('L', (28, 28), (255))

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        # resize height according to ratio width
        nheight = int(round((20.0 / width * height), 0))
        if (nheight == 0):
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS) \
            .filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS) \
            .filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())

    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva


x = [imageprepare(r"C:\Data\YOUELLT\My Documents\Digit Recognizer\data\5x5split_04_04.png")]
cols = test.columns.tolist()

x_df = pd.DataFrame(data=x, columns=cols)
x_df = (x_df.iloc[:, 0:].values).astype('float32')
x_df_reshaped = x_df.reshape(x_df.shape[0], 28, 28, 1)

prediction = model.predict_classes(x_df_reshaped)
answer = pd.DataFrame({"ImageId": list(range(1, len(prediction)+1)),
                       "Label": prediction})
print(answer['Label'])

newArr = [[0 for d in range(28)] for y in range(28)]

k = 0
for i in range(28):
    for j in range(28):
        newArr[i][j] = x[0][k]
        k = k + 1
'''
for i in range(28):
    for j in range(28):
        print(newArr[i][j])
    print('\n')
'''
plt.imshow(newArr, interpolation='nearest')
plt.savefig('MNIST_IMAGE_TEST.png')
plt.show()

'''
4x4 images attempt
'''


def prediction_printer(image_name, grid_size):
    splits = grid_size**2
    data = np.array(image_slicer.slice(file + image_name + '.png', splits))
    i_df = pd.DataFrame(data=data)
    i_df[0] = i_df[0].astype(str).str.strip('.png>').str.split(' - ').str[1]
    image_list = i_df[0].tolist()
    df = pd.DataFrame([])
    for img in image_list:
        df = df.append([imageprepare(file + img + ".png")])
    df = (df.iloc[:, 0:].values).astype('float32')
    df = df.reshape(df.shape[0], 28, 28, 1)
    predictions = model.predict_classes(df)
    new_df = pd.DataFrame(np.array_split(predictions.tolist(), grid_size))
    print(new_df.to_string(index=False, header=None))


prediction_printer(image_name='split', grid_size=4)  # 13/16 correct

prediction_printer(image_name='5x5split', grid_size=5)  # 25/25 correct


# serialize model to JSON
model_json = model.to_json()
with open(file + "20_Epoch_CNN.json", "w") as json_file:
    json_file.write(model_json)

load_json_file = open(file + "20_Epoch_CNN.json", 'r')
loaded_model_json = load_json_file.read()

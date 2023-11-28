import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# Load the dataset
# Replace 'your_dataset.csv' with the actual path to your dataset file
dataset2_path = r"C:\Users\prajw\Downloads\fer2013new.csv"
df = pd.read_csv(dataset2_path)



# Function to convert pixel values to images
def pixels_to_images(pixel_str):
    pixel_values = np.array(pixel_str.split(' '), dtype=int)
    return pixel_values.reshape(48, 48)

# Apply the pixels_to_images function to create the 'Images' column
df['Images'] = df["pixels"].apply(pixels_to_images)


# Convert emotion labels to categorical

one_hot_encoded = pd.get_dummies(df['emotion'], prefix='emotion')
df = pd.concat([df, one_hot_encoded], axis=1)
df = df.drop('emotion', axis=1)

print(one_hot_encoded)
for col in df.columns:
    print(col)

tester = ["emotion_0","emotion_1","emotion_2","emotion_3","emotion_4","emotion_5","emotion_6"]



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Images'], df[tester], test_size=0.2, random_state=42)


# Convert images to numpy arrays
X_train = np.vstack(X_train.values).reshape(-1, 48, 48, 1)
X_test = np.vstack(X_test.values).reshape(-1, 48, 48, 1)

# Display the first 5 images from X_train with corresponding emotions from y_train
"""for i in range(10):
    plt.figure(figsize=(3, 3))
    plt.imshow(X_train[i].reshape(48, 48), cmap='gray')  # Assuming grayscale images
    emotion_label = np.argmax(y_train.values[i])
    plt.title(f'Emotion: {emotion_label} ({tester[emotion_label]})')
    plt.show()"""

# number of possible label values
nb_classes = 7

# Initialising the CNN
model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert predictions and true labels back to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test.values, axis=1)

# Display classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(y_test_classes, y_pred_classes))

# Save the model
model.save(r'C:\Users\prajw\PycharmProjects\python_athul\ann project/facial_expression_model.h5')
print("model saved successfully")


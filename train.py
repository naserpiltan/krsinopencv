import matplotlib
import numpy as np
from  untitled1 import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

matplotlib.use("Agg")

INIT_LR = 1e-4
BS = 8
EPOCHS = 120
 

print("[INFO] loading images...")
imagePaths = list(paths.list_images("/media/piltan/9EE80C20E80BF4F5/liveness-detection-opencv/dataset"))

data = []
labels = []
for imagePath in imagePaths:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))
 
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)
data = np.array(data, dtype="float") / 255.0
le = LabelEncoder()
print(labels)
labels = le.fit_transform(labels)

labels = np_utils.to_categorical(labels, 2)
print(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
 
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))
 
# save the network to disk
#print("[INFO] serializing network to '{}'...".format("/media/piltan/9EE80C20E80BF4F5/liveness-detection-opencv/model"))
#model.save("/media/piltan/9EE80C20E80BF4F5/liveness-detection-opencv/liveness.model")
#model.save("/media/piltan/9EE80C20E80BF4F5/liveness-detection-opencv/liveness.h5")
 
# save the label encoder to disk
f = open("/media/piltan/9EE80C20E80BF4F5/liveness-detection-opencv/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
 
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
#plt.savefig("/media/piltan/9EE80C20E80BF4F5/liveness-detection-opencv/model")
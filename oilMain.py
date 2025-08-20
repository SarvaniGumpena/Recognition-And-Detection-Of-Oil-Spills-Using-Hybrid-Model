from PIL import Image
import shutil
import matplotlib.pyplot as plt
import glob
import os
import tensorflow as tf
# from keras.applications import ImageDataGenerators
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

train_nospill_img_list = glob.glob("./kaggle/input/spill-data/Spill_Data/Train/NoSpill*.jpg")
train_spill_img_list = glob.glob("./kaggle/input/spill-data/Spill_Data/Train/Oilspill*.jpg")
test_nospill_img_list = glob.glob("./kaggle/input/spill-data/Spill_Data/Test/NoSpill*.jpg")
test_spill_img_list = glob.glob("./kaggle/input/spill-data/Spill_Data/Test/Oilspill*.jpg")
print(f"Number of No Spill images in train dataset: {len(train_nospill_img_list)}")
print(f"Number of Spill images in train dataset: {len(train_spill_img_list)}")
print(f"Number of No Spill images in test dataset: {len(test_nospill_img_list)}")
print(f"Number of Spill images in test dataset: {len(test_nospill_img_list)}")


# mkdir /kaggle/working/train
# mkdir /kaggle/working/train/spill
# mkdir /kaggle/working/train/nospill
# mkdir /kaggle/working/test
# mkdir /kaggle/working/test/spill
# mkdir /kaggle/working/test/nospill

for train_file in train_nospill_img_list:
    file_name = train_file.split("/")[-1]
    file_name = file_name.split("\\")[-1]
    # print(file_name)
    new_path = f"./kaggle/working/train/nospill/{file_name}"
    shutil.copy(train_file, new_path)

for train_file in train_spill_img_list:
    file_name = train_file.split("/")[-1]
    file_name = file_name.split("\\")[-1]
    # print(file_name)
    new_path = f"./kaggle/working/train/spill/{file_name}"
    shutil.copy(train_file, new_path)

for test_file in test_nospill_img_list:
    file_name = test_file.split("/")[-1]
    file_name = file_name.split("\\")[-1]
    # print(file_name)
    new_path = f"./kaggle/working/test/nospill/{file_name}"
    shutil.copy(test_file, new_path)

for test_file in test_spill_img_list:
    file_name = test_file.split("/")[-1]
    file_name = file_name.split("\\")[-1]
    # print(file_name)
    new_path = f"./kaggle/working/test/spill/{file_name}"
    shutil.copy(test_file, new_path)


i=0
for oilspill_train_file in os.listdir("./kaggle/working/train/spill"):
    img = Image.open(f"./kaggle/working/train/spill/{oilspill_train_file}")
    plt.imshow(img)
    plt.title("train-spill")
    plt.show()
    i+=1
    if i==5:
        break


i=0
for nospill_train_file in os.listdir("./kaggle/working/train/nospill"):
    img = Image.open(f"./kaggle/working/train/nospill/{nospill_train_file}")
    plt.imshow(img)
    plt.title("train-nospill")
    plt.show()
    i+=1
    if i==5:
        break



i=0
for spill_test_file in os.listdir("./kaggle/working/test/spill"):
    img = Image.open(f"./kaggle/working/test/spill/{spill_test_file}")
    plt.imshow(img)
    plt.title("test-spill")
    plt.show()
    i+=1
    if i==5:
        break


i=0
for nospill_test_file in os.listdir("./kaggle/working/test/nospill"):
    img = Image.open(f"./kaggle/working/test/nospill/{nospill_test_file}")
    plt.imshow(img)
    plt.title("test-nospill")
    plt.show()
    i+=1
    if i==5:
        break

train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    "./kaggle/working/train/",
    target_size=(150,150),
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    "./kaggle/working/test/",
    target_size=(150,150),
    class_mode="binary"
)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch,logs={}):
    if(logs.get('accuracy')>=0.95):
      print("\nReached 95% accuracy so cancelling training1")
      self.model.stop_training=True
callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation = "relu", input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation = "relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(1,activation = 'sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch = 8,
    epochs = 100,
    verbose=1,
    validation_data=validation_generator,
    callbacks=[callbacks]
)

model.save_weights("modeloil.h5")

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

# accuracy
fig = plt.figure(figsize=(25,8))
ax1 = fig.add_subplot(121)
ax1.plot(epochs, acc, "b-", label="Training acc")
ax1.plot(epochs, val_acc, "g-", label="Validation acc")
ax1.set_title("Training and validation accuracy")
ax1.legend()
plt.show()

# loss
ax2 = fig.add_subplot(122)
ax2.plot(epochs, loss, "b-", label="Training loss")
ax2.plot(epochs, val_loss, "g-", label="Validation loss")
ax2.set_title("Training and validation loss")
ax2.legend()
plt.show()



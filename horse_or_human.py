import os
import signal
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf
from keras.layers import Conv2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

my_path = 'horse_or_human'

train_horses = os.path.join('horse_or_human/horses')
train_humans = os.path.join('horse_or_human/humans')

train_horses_names = os.listdir(train_horses)
train_humans_names = os.listdir(train_humans)

print(train_horses_names[:10])
print(train_humans_names[:10])

print('total horses images in training data::', len(train_horses_names))
print('total humans image in training data: ', len(train_humans_names))

nrows = 4
ncols = 4
pic_index = 8
fig = plt.gcf()
fig.set_size_inches(8, 8)
next_horse_pic = [os.path.join(train_horses, fname)
                  for fname in train_horses_names[:pic_index]]

next_human_pic = [os.path.join(train_humans, fname)
                  for fname in train_humans_names[:pic_index]]

for i, img_path in enumerate(next_horse_pic+next_human_pic):
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off')
    
    imgs = img.imread(img_path)
    plt.imshow(imgs)
    
plt.show()

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.summary()
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale = 1/255)
train_generator = train_datagen.flow_from_directory(my_path, target_size=(300, 300), batch_size=16, class_mode='binary')

history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1)




for i in range(1000):
    path = input("please enter correct location and name of image\n")
    if path == "end":
        print("exiting program")
        break
    else:
        imgs =image.load_img(path, target_size=(300, 300))
        x = image.img_to_array(imgs)
        x = np.expand_dims(x, axis = 0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size = 10)
        print(classes[0])
        if classes[0]>0.5:
            print("this is a image of human")
        else:
            print("this is a image of horse")

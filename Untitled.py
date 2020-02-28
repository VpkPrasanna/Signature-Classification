#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 


# In[25]:


ls


# In[3]:



train_data_dir = '/home/pcuser/Downloads/Task/db task/sample_Signature/train'
validation_data_dir = '/home/pcuser/Downloads/Task/db task/sample_Signature/test'
nb_train_samples =400 
nb_validation_samples = 100
epochs = 50
batch_size = 16


# In[4]:


train_data_dir


# In[5]:


img_width, img_height = 224, 224
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 


# In[6]:


model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 


# In[7]:


model.compile(loss ='binary_crossentropy', 
                     optimizer ='rmsprop', 
                   metrics =['accuracy']) 


# In[8]:


train_datagen = ImageDataGenerator( 
                rescale = 1. / 255, 
                 shear_range = 0.2, 
                  zoom_range = 0.2, 
            horizontal_flip = True) 


# In[9]:


test_datagen = ImageDataGenerator(rescale = 1. / 255) 


# In[10]:


train_generator = train_datagen.flow_from_directory(train_data_dir, 
                              target_size =(img_width, img_height), 
                     batch_size = batch_size, class_mode ='binary') 


# In[11]:


validation_generator = test_datagen.flow_from_directory( 
                                    validation_data_dir, 
                   target_size =(img_width, img_height), 
          batch_size = batch_size, class_mode ='binary') 


# In[12]:


model.fit_generator(train_generator, 
    steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = nb_validation_samples // batch_size) 
  
model.save_weights('weight_model_weights.h5')
model.save('saved_model.h5')


# In[14]:


# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# #Train and validation accuracy
# plt.plot(epochs, acc, 'b', label='Training accurarcy')
# plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
# plt.title('Training and Validation accurarcy')
# plt.legend()

# plt.figure()
# #Train and validation loss
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()

# plt.show()


# In[26]:


model.save_weights('weight_model_weights.hdf5')
model.save('saved_model.hdf5')


# In[31]:


ls


# In[53]:


from PIL import Image
import numpy as np
get_ipython().system('pip install scikit-image')
from skimage import transform
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('test.png')
model.predict(image)


# In[54]:


# from keras.preprocessing import image
# img = image.load_img(path="/home/pcuser/Downloads/Task/db task/sample_Signature/test/forged/NFI-01301017.png",grayscale=True,target_size=(28,28,1))
# img = image.img_to_array(img)
# test_img = img.reshape([-1,720, 1280,1])
# img_class = model.predict_classes(test_img)


# In[55]:


# from keras.models import load_model

# import cv2

# import numpy as np

# model = load_model('/home/pcuser/Downloads/Task/db task/sample_Signature/saved_model.hdf5')
# print(model)

# model.compile(loss='binary_crossentropy',

#               optimizer='rmsprop',

#               metrics=['accuracy'])

# img = cv2.imread('test.jpg')

# img = cv2.resize(img,(224,224))

# img = np.reshape(img,[1,320,240,3])

# classes = model.predict_classes(img)

# print(classes)


# In[56]:


# img_width, img_height = 224,224

# # load the model we saved
# loaded_model = load_model('saved_model.hdf5')

# # Get test image ready
# # test_image = image.load_img('/home/pcuser/Downloads/Task/db task/sample_Signature/train/genuine/NFI-00101001.png', target_size=(img_width, img_height))
# test_img = cv2.imread('test.jpg')
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)

# test_image = test_image.reshape(img_width, img_height*3)    # Ambiguity!
# # Should this instead be: test_image.reshape(img_width, img_height, 3) ??
# result = model.predict(test_image, batch_size=1)
# print(result)


# In[ ]:





# to activate venv run
# .\env\Scripts\activate
# to deactivate run
# deactivate

import functools
import os

from PIL import Image
from matplotlib import gridspec
import matplotlib.image as matImage
import matplotlib.pylab as plt
import matplotlib
import png
# from matplotlib._png import read_png
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import base64

from unrealcv import client
client.connect() # Connect to the game
if not client.isconnected(): # Check if the connection is successfully established
  print ('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
else:
  res = client.request('vget /camera/0/lit lita.png')
  filename = client.request('vget /camera/0/lit lol.jpg')
# res = client.request('vget /camera/0/lit jpg')

im1 = Image.open(r'RealisticRendering-Win-0.3.10\RealisticRendering\WindowsNoEditor\RealisticRendering\Binaries\Win64\lita.png')
im1 = im1.convert('RGB')
im1.save(r'img\lita.jpg')


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5096)])
  except RuntimeError as e:
    print(e)


# ALL THIS WORKS TO DISPLAY IMAGE FROM UE4-----------------
# res = client.request('vget /camera/0/lit png')
# res = np.frombuffer(res, np.uint8)
# res = cv2.imdecode(res, -1)
# cv2.imshow("Render", res)
# cv2.waitKey(0)
# exit()
# ---------------------------------------------------------


# plt.imshow(filename)
# plt.savefig('test2.png')
# print(filename.shape)
# res = client.request('vget /camera/0/lit lit.png')
# print('The image is saved to %s' % res)
# im = read_png(res)
# print(im.shape)

# from tensorflow.compat.v1 import InteractiveSession
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# Load content and style images (see example in the attached colab).
# content_image = res
# content_image = plt.imread('img/pexels-caio-cardenas-2101839.jpg')
style_image = plt.imread('img\pexels-steve-johnson-4208443.jpg')

# Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

# Optionally resize the images. It is recommended that the style image is about
# 256 pixels (this size was used when training the style transfer network).
# The content image can be any size.
style_image = tf.image.resize(style_image, (256, 256))

# Load image stylization module.
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Stylize image.


# print(content_image.shape[0])
# plt.savefig(stylized_image)
# print(stylized_image)


while 1:
  #get img from UE4
  # res = client.request('vget /camera/0/lit lita.png')
  res = client.request('vget /camera/0/lit png')
  # res = matImage.imread(res).convert('RGB')
  # res = np.vstack(map(np.uint16, res))
  res = base64.b64decode(res)
  print(res)
  res = res.convert('RGB')
  # res = plt.imread(res).convert('RGB')
  #convert to jpg
  # im1 = Image.open(r'RealisticRendering-Win-0.3.10\RealisticRendering\WindowsNoEditor\RealisticRendering\Binaries\Win64\lita.png')
  # im1 = im1.convert('RGB')
  # im1.save(r'img\lita.jpg')
  #stylize
  # content_image = plt.imread('img\lita.jpg')
  content_image = res
  content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
  outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
  stylized_image = outputs[0]
  plt.imshow(stylized_image[0])
  plt.savefig('test.png')

  # WORKING PLT SHOW -----------------------------------  
  # plt.imshow(stylized_image[0])
  # plt.show(block = False)
  # plt.pause(0.05)
  # WORKING PLT SHOW -----------------------------------

# plt.imshow(filename)

# img = Image.fromarray(stylized_image[0], 'RGB')
# img.save('test.png')

# plt.savefig()
# matplotlib.image.imsave('out00.png', stylized_image)
# plt.imsave('out01.jpg', stylized_image[0])
# plt.imload(stylized_image)

client.disconnect()



# start the model
# connect to unreal
# a while loop
    # get frame
    # stylize
    # save to file
#



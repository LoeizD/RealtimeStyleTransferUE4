from PIL import Image
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime, timezone, timedelta
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5096)])
  except RuntimeError as e:
    print(e)

hub_module = hub.load('magenta_arbitrary-image-stylization-v1-256_2')
res = plt.imread('../StyleRnd/ue4.jpg')

# Change style image here
import os
for filename in os.listdir('../StyleRnd/dl'):
  style_image = plt.imread('../StyleRnd/dl/' + str(filename))

  style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
  style_image = tf.image.resize(style_image, (256, 256)) # Optionally resize the images. It is recommended that the style image is about 256px (this size was used when training the style transfer network).


  # lastTime = startTime = time_ms()
  # timeDiff( 'ue4 req' )
  # res = np.frombuffer(res, np.uint8)
  # content_image = cv2.imdecode(res, -1)
  content_image = res
  # timeDiff( 'RGB conv' )
  content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
  # timeDiff( 'Resize' )
  outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
  # timeDiff( 'Algo' )
  stylized_image = outputs[0]
  pil_img = tf.keras.preprocessing.image.array_to_img(stylized_image[0])
  # pil_img = np.array(pil_img)
  # pil_img = cv2.resize(pil_img, (1280, 720))
  # cv2.imshow("Render", cv2.cvtColor(pil_img, cv2.COLOR_RGB2BGR))
  # timeDiff( 'Render' )
  # cv2.waitKey(1)
  pil_img.save('../StyleRnd/style/'+ str(filename) + 'Style.jpg')
  # cv2.imwrite('../StyleRnd/dl/'+ str(filename) + 'Style.jpg', cv2.IMREAD_COLOR)
  # cv2.imwrite('../StyleRnd/dl/'+ str(filename) + 'Style.jpg', cv2.COLOR_RGB2BGR)
  print( filename )
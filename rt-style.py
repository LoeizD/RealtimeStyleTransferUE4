from PIL import Image
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime, timezone, timedelta
import cv2

from unrealcv import client
client.connect() # Connect to the game
if not client.isconnected():
  print ('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
  exit()

# cap the memory use for the model (5096 works on a 2070 super), as we are running Unreal too
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5096)])
  except RuntimeError as e:
    print(e)

# Change style image here
style_image = plt.imread('img\p13.jpg')

style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
style_image = tf.image.resize(style_image, (256, 256)) # Optionally resize the images. It is recommended that the style image is about 256px (this size was used when training the style transfer network).
hub_module = hub.load('magenta_arbitrary-image-stylization-v1-256_2')
# hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

startTime = 0
lastTime = 0

def time_ms():
  now = datetime.now(timezone.utc)
  epoch = datetime(1970, 1, 1, tzinfo=timezone.utc) # use POSIX epoch
  return (now - epoch) // timedelta(microseconds=1)

def timeDiff( msg = '' ):
  global lastTime
  timeNow = time_ms()
  print( str( timeNow - lastTime ) + ' ' + msg )
  lastTime = timeNow

while 1:
  lastTime = startTime = time_ms()
  res = client.request('vget /camera/0/lit png') # lit can be changed to normal
  timeDiff( 'ue4 req' )
  res = np.frombuffer(res, np.uint8)
  content_image = cv2.imdecode(res, -1)
  content_image = content_image[:,:,:3]
  timeDiff( 'RGB conv' )
  content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
  timeDiff( 'Resize' )
  outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
  timeDiff( 'Algo' )
  stylized_image = outputs[0]
  pil_img = tf.keras.preprocessing.image.array_to_img(stylized_image[0])
  pil_img = np.array(pil_img)
  pil_img = cv2.resize(pil_img, (1280, 720))
  cv2.imshow("Render", cv2.cvtColor(pil_img, cv2.COLOR_RGB2BGR))
  timeDiff( 'Render' )
  cv2.waitKey(1)
  print( '-------------------------------' )
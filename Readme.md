# Real time style transfer UE4

![ue4-style-transfer-opencv](demo.gif)

the correct file to run is `rt-style.py`

make sure to run an Unreal OpenCV exec before executing the python file 

made to run on a 2070 Super, lower end GPUs will be slower/crash

## Setup
to activate venv run

`.\env\Scripts\activate`

to deactivate run

`deactivate`

make sure to have cuda 10.1 installed and cudnn 7 (I had to copy a few dlls into system32 on Windows 10)

change screencap resolution in unrealcv.ini

change game resolution in GameUserSettings.ini (you don't need a big game resolution as it will take VRAM, which is also used by the style transfer)

## Program Architecture

connect to unreal

start the model

a while loop to:
    get ue4 frame, 
    stylize, 
    display



## Credits

Using https://unrealcv.org/ to get images out of Unreal Engine 4

The style transfer model used is https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2 

@LoeizD https://bergamot.digital

with help from @shramee https://leastimperfect.com/ for the timing functions

![ue4-style-transfer](test.png)

# ZED Camera + YOLO model for human only tracking and detection
**Data sent via websocket to Unreal Engine**

There are two main scripts to this repo:
1. objectDetectionGPU1cam_ZMQ_IMUUE.py
2. ZMQ_UEwebSocket_sleep.py

`objectDetectionGPU1cam_ZMQ_IMUUE.py` is to start up the recognition system. It requires a ZED camera, ZED API and YOLO v8. You can switch the YOLOv8 model for your own custom YOLO model.

`ZMQ_UEwebSocket_sleep.py` is a webocket server that sends out the recognition data to Unreal Engine via websocket.

There's a third file. `ZMQ_UEwebSocket_clientTest.py`, this is just a simple websocket client testing script to check the stability of the websocket server.


## Order of Startup

1st `objectDetectionGPU1cam_ZMQ_IMUUE.py`
2nd `ZMQ_UEwebSocket_sleep.py`

## To-dos
- [ ] add the requirement.txt to the repo (remove from gitignore)
- [ ] rename files for easier understanding
- [ ] create a .bat to startup both main files (remove from gitignore)

## ZED Installation on Windows 11

I strongly suggest to use ZED cameras and ZED API with windows 11 or Linux. Otherwise, you'll have to do the below steps for Windows 10.

Works with CUDA 11.7 and 11.8 (tested)

## ZED Installation on Win10

You need to use Anaconda.

0. Install the ZED SDK 4.0+
1. Create an Python 3.7 environment
2. Open Conda Terminal
3. CD to your project folder
4. Run the get_python_api.py from this folder with OpenVPN connection
5. Close Conda Terminal
6. Copy the build 'sl.cp37-win_amd64.pyd' from the src build folder in the ZED ZED_API_installation folder on the D: Drive to your version of:
C:\Users\remko\anaconda3\envs\ZED5gpu_env\Lib\site-packages\pyzed
7. Yes replace the existing one.
8. Copy all the .dll files from C:\Program Files (x86)\ZED SDK\bin to your version of:
C:\Users\remko\anaconda3\envs\ZED5gpu_env\Lib\site-packages\pyzed
9. Go back to Anaconda Navigator and update Index on the environment you created
10. Open the Conda Terminal
11. Launch VSCode via Code . or from within Anaconda Navigator. Both work, but you may have to resign in to your plug-ins if you use the one from Anaconda Navigator.
12. Make sure you're using the correct python interpreter inside of VSCode. Check the bottom VSCode.

**Note:** If you don’t have the already built 'sl.cp37-win_amd64.pyd' from the src build folder in the ZED ZED_API_installation folder on the D: Drive. Then you can rebuild it, following the instructions below.

### Building a Cython extension

1. Copy the original sl.cp37-win_amd64.pyd file from your version of:
`C:\Users\remko\anaconda3\envs\ZED5gpu_env\Lib\site-packages\pyzed` to your just downloaded and unzipped version of:
`D:\Development\ZED_API_installation\zed-python-api-4.0\src\pyzed` 
2. **Install Cython inside your python environment and a C++ Compiler**: If you don’t have Cython installed, you can install it using pip:

```bash
pip install Cython
```

Additionally, you will need a C++ compiler. Since you are on Windows, you can install Visual Studio with C++ support.

1. **Build the Extension**: Now, open a command prompt in the root of your project directory where **`setup.py`** is located. Then run the following command to build the extension:

```
python setup.py build_ext --inplace
```

This command tells **`setup.py`** to build the extension in place, which means the compiled **`.pyd`** file will be created in the current directory.

## Links

You can find the GitHub repo to the Cython Extension API build here:

[ZED python API](https://github.com/stereolabs/zed-python-api)

## ZED on GPU (YOLO)
After installing the regular version of ZED API and doing all the  necessary changes.

You need to install pytorch version with 11.7 CUDA. It doesn’t come with installer script. It’s an additional installation.

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

That should do it. Update index within Anaconda Navigator, and always run Anaconda prompt as administrator


## YOLO Models

Download the necessary models for YOLOv8
- yolov8m.pt

[Ultralytics Github Repo](https://github.com/ultralytics/ultralytics)

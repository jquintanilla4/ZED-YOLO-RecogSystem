# 2 ZED cameras, each with its own ZED SDK instance
# 1 YOLOv8 instance on GPU for ZED camera 1
# YOLOv8 bounding box from ZED camera 1 used for ZED camera 2
# Results are fused into a world coordinate system
# All actions happen on one python script
# The detection results are published to a ZMQ socket

import os
import numpy as np
import keyboard
import argparse
import json

import torch
print("PyTorch version: ", torch.__version__)
print("CUDA available: ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)

import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import cv_viewer.tracking_viewer as cv_viewer

import zmq
context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind('tcp://*:5555')

# from class_dicts import dict_365, dict_coco, dict_imagenet, dict_voc # additonial class dictionaries for YOLOv8

# Setting up environment variable to allow for multiple instances of OpenMP to run simultaneously
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# function to get the serail numbers and their order from the ZED SDK
def get_camera_index_list():
    camera_index_list = []
    device_list = sl.Camera.get_device_list()
    for device in device_list:
        camera_index_list.append(device.serial_number)
    return camera_index_list

# VARIABLES
# camera serial numbers
camera_index_list = get_camera_index_list()
print(f"Camera index list: {camera_index_list}")
# number of cameras
num_cameras = len(camera_index_list)

# Create a list of lists for images an detections
image_list = [[] for _ in range(num_cameras)]
detection_list = [[] for _ in range(num_cameras)]
# Create a list of locks for each camera
lock_list = [Lock() for _ in range(num_cameras)]

# Global run signal for all threads
global_run_signal = False
# Global exit signal for all threads
global_exit_signal = False


# Function to calculate the transformation matrix from camera 1 to camera 2
def calculate_transformation_matrix(rot1, trans1, rot2, trans2):
    # convert rotations to rotation matrices
    rot1_matrix = cv2.Rodrigues(np.array(rot1))[0]
    rot2_matrix = cv2.Rodrigues(np.array(rot2))[0]

    # convert translations to column vectors
    trans1_matrix = np.array(trans1).reshape(3, 1)
    trans2_matrix = np.array(trans2).reshape(3, 1)

    # build the transformation matrices
    trans1_full = np.hstack((rot1_matrix, trans1_matrix))
    trans1_full = np.vstack((trans1_full, [0, 0, 0, 1]))
    trans2_full = np.hstack((rot2_matrix, trans2_matrix))
    trans2_full = np.vstack((trans2_full, [0, 0, 0, 1]))

    # calcualte the transformation from camera 1 to camera 2
    trans1to2 = np.linalg.inv(trans1_full).dot(trans2_full)

    return trans1to2


# Function to calculate the transformation matrix from camera1 to camera2
def calculate_transformation_matrix_for_camera(calibration, camera1, camera2):
    return calculate_transformation_matrix(
        calibration[str(camera1)]['world']['rotation'],
        calibration[str(camera1)]['world']['translation'],
        calibration[str(camera2)]['world']['rotation'],
        calibration[str(camera2)]['world']['translation'])


# Function to convert bounding box from (center x, center y, width, height) format
# to (top left, top right, bottom left, bottom right) format
def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output


# function to convert detection from YOLO to a ZED SDK ingestable format
def detections_to_custom_box(detections, im0): # input is a list of detections and an image
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData() # detection converted into a ZED class object
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape) # converted the bounding box coordinates
        obj.label = det.cls # class label of object detection
        obj.probability = det.conf # confidence level of object detection
        obj.is_grounded = False
        output.append(obj) # output is a list of CustomBoxObjectData objects (ZED SDK class)
    return output


# Function that runs YOLO on a separate thread
def torch_thread(weights, img_size, conf_thres=0.7, iou_thres=0.7, i=0): # conf originally 0.4, iou originally 0.45
    global class_names, yolo_output_label, global_run_signal, global_exit_signal, yolo_output_id
    try:
        print(f"Starting Torch Thread for camera {i}")
        print("Initializing Network...")

        # create a yolov8 model object
        model = YOLO(weights)        
        model.to('cuda') # moves the model to GPU

        # get the list of class names
        class_names = model.names
        print("Class names: ", class_names)

        # Loop continues indefinitely until the exit signal is true
        while not global_exit_signal:
            # if the camera index is camera 1, run YOLO detection
            if camera_index_list[i] == camera_index_list[0]:
                # When the run signal is true, acquire the lock to prevent simultaneous access to shared resources
                if global_run_signal:
                    # Acquire the lock
                    lock_list[i].acquire()

                    # Convert the image to RGB
                    img = cv2.cvtColor(image_list[i], cv2.COLOR_BGRA2RGB)
                        
                    # Run YOLO on the image on the GPU
                    det = model.track(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, tracker='bytetrack.yaml', verbose=False)[0].cpu().numpy().boxes

                    # Filter detections to only keep humans, class 0
                    det = [det for det in det if det.cls == 0]

                    # Convert the YOLO detections to a ZED SDK ingestable format
                    detection_list[i] = detections_to_custom_box(det, image_list[i])

                    # Pass the yolo output label to a variable for the main thread to access
                    try:
                        for yolo_output in detection_list[i]:
                            yolo_output_label = yolo_output.label
                    except Exception as e:
                        print("Error in yolo output label: ", e)

                    for yolo_output in det:
                        # print(f"Yolo output: {yolo_output}")
                        yolo_output_id = int(yolo_output.id.item())

                    # Release the lock
                    lock_list[i].release()

                    # Reset the run signal to false
                    global_run_signal = False

                # Pause for a short time
                sleep(0.01)
        
        # Print statement when the thread is completed, part of try-except block
        print(f"Completed Torch Thread for camera {i}")

    except Exception as e:
        print(f"Error in torch thread for camera {i}: ", e)
        # Add traceback printout
        import traceback
        traceback.print_exc()


def open_camera(camera_index, svo_filepath=None):
        # Create a ZED camera object
        zed = sl.Camera()

        # Create InputType object
        input_type = sl.InputType()

        # if an SVO file path is provided, set the input type to use it
        if svo_filepath is not None:
            input_type.set_from_svo_file(svo_filepath)
            print(f"Using SVO file {svo_filepath}")
        
        # Set up camera parameters
        init_params = sl.InitParameters(svo_real_time_mode=True)
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_maximum_distance = 50

        # Open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Error {err} opening camera {camera_index}")
            exit(1)
        
        print(f"Initialized camera {camera_index}")
        return zed


# Function for the main loop of the program
def main_loop(i, svo_filepath=None, trans1to2=None):
    global global_run_signal, global_exit_signal

    # Get the serial number of the camera to be used
    camera_index = camera_index_list[i]
    # Create ZED camera object via the open_camera function
    zed = open_camera(camera_index, svo_filepath)

    # Initialize CUDA for each thread
    print(f"Initializing CUDA for camera {camera_index}...")
    if torch.cuda.is_available():
        _ = torch.cuda.FloatTensor(1)

    if camera_index == camera_index_list[0]: #Only runs YOLO on camera 1
        # Create a new thread for each camera to run YOLO model
        print(f"Starting YOLO torch thread for camera {camera_index}...")
        capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, 'conf_thres': opt.conf_thres, 'i': i})
        capture_thread.start()
        print(f"YOLO torch thread started for camera {camera_index}...")

    # Set runtime parameters
    runtime_params = sl.RuntimeParameters()

    # Create a temporary image matrix for the left camera
    image_left_tmp = sl.Mat()

    # Enable positional tracking parameters, required for object tracking
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    # Enable object detection parameters
    detection_parameters = sl.ObjectDetectionParameters()
    detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    detection_parameters.enable_tracking = True
    zed.enable_object_detection(detection_parameters)

    # Set runtime parameters
    detection_runtime_parameters = sl.ObjectDetectionRuntimeParameters()

    # To store the detected objects
    objects = sl.Objects()

    # Get camera info
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    # Display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8) # 4 channels, light grey RGBA

    # Detect objects loop
    while not global_exit_signal:
        # If able to grab image from camera
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            # Acquire lock
            lock_list[i].acquire()
            # Retrieve image from left camera
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_list[i] = image_left_tmp.get_data()
            # Release lock
            lock_list[i].release()

            # RUN YOLO DETECTION IN THE OTHER THREAD
            global_run_signal = True

            # Wait for detections on the other thread
            while global_run_signal:
                sleep(0.001)

            # Acquire lock
            lock_list[i].acquire()
            # Ingest detections into ZED SDK
            zed.ingest_custom_box_objects(detection_list[i])
            # Release lock
            lock_list[i].release()

            # Retrieve the detected objects
            zed.retrieve_objects(objects, detection_runtime_parameters)

            # ACTIONS
            # Retrieve object list, where all the fun information is
            if objects.is_new:
                obj_array = objects.object_list
                # Print out the detected objects x, y, z coordinates in 3D space
                if len(obj_array) > 0:
                    data_list = [] # List to hold dictionaries of each object's data TESTING
                    for obj in obj_array:
                        # Print out the detected objects x, y, z coordinates, its ID, and its label in camera 1's space
                        # print(f"Camera {camera_index} detected object | {class_names[yolo_output_label]} {obj.id}, 3D location: x: {obj.position[0]} y: {obj.position[1]} z: {obj.position[2]}")
                        
                        # Transform the 3D coordinates of the detected objects from camera 1 to camera 2
                        point_cam1 = np.array([obj.position[0], obj.position[1], obj.position[2], 1])
                        point_cam2 = np.dot(trans1to2, point_cam1)
                        
                        # Now point_cam2 contains the coordinates of the detected object from camera 1 in camera 2's space
                        # print(f"Camera 2 {camera_index_list[1]} detected object | {class_names[yolo_output_label]} {obj.id}, 3D location: x: {point_cam2[0]} y: {point_cam2[1]} z: {point_cam2[2]}")

                        # Calculate the shared 3D location of the detected object
                        points_shared = (point_cam1 + point_cam2) / 2

                        # Print the shared(world) coordinates
                        # print(f"Object ID: {obj.id}, X: {points_shared[0]}, Y: {points_shared[1]}, Z: {points_shared[2]}")
                        # print(f"Yolo ID: {yolo_output_id}, Object ID: {obj.id}, X: {points_shared[0]}, Y: {points_shared[1]}, Z: {points_shared[2]}")

                        # Create a dictionary TESTING
                        data = {
                            'obj_id': yolo_output_id,
                            'type': 'coord',
                            'x': points_shared[0],
                            'y': points_shared[1],
                            'z': points_shared[2]
                        }
                        data_list.append(data) # TESTING

                        # Convert the list of dictionaries to a JSON string
                        json_string = json.dumps(data_list)
                        # Send the JSON string to the ZeroMQ socket
                        publisher.send_string(json_string)

            # 2D Display
            # Copy the left image data to image_left_ocv (numpy array)
            np.copyto(image_left_ocv, image_left_tmp.get_data())
            # renders the 2d view of the image
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, detection_parameters.enable_tracking)
            
            if opt.dev:
                # Display the image using OpenCV
                cv2.imshow(f"2D View Camera {camera_index}", image_left_ocv)
                # This allows OpenCV to process its event queue
                cv2.waitKey(10)

            # If 'q' is pressed, exit the loop
            if keyboard.is_pressed('q'):
                global_exit_signal = True
                break

        else:
            global_exit_signal = True
            break

    # Close the camera
    global_exit_signal = True
    zed.close()

    # if the last camera closes, send "end" message
    if i == num_cameras -1:
        publisher.send_string('{"end": true}')


# Parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='models/yolov8m.pt', help='model.pt path(s)')
parser.add_argument('--svo', type=str, default=None, help='optional svo file')
parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--conf_thres', type=float, default=0.7, help='object confidence threshold')
parser.add_argument('--dev', action='store_true', help='dev mode gives you OpenCV windows') # remember to revert back, no default
opt = parser.parse_args()


def main():
    # Load calibration data
    with open('calibration/calibration03.json', 'r') as f:
        calibration = json.load(f)

    # Determine the order of cameras
    camera1 = camera_index_list[0]
    print(f"Camera 1: {camera1}")
    camera2 = camera_index_list[1]
    print(f"Camera 2: {camera2}")

    # Calculate the transformation matrix between the cameras, based on the order of the cameras
    trans1to2 = calculate_transformation_matrix_for_camera(calibration, camera1, camera2)
 
    # THE THREAD BLOCK
    # Create a list for threads
    thread_list = []

    # The thread will run the main loop function over the number of cameras
    for i in range(num_cameras):
        main_thread = Thread(target=main_loop, args=(i, opt.svo, trans1to2))
        main_thread.start()

        thread_list.append(main_thread)

    # Wait for all threads to finish and the join them
    for thread in thread_list:
        thread.join()


if __name__ == '__main__':
    # Run main function without calculating gradients
    # This is more memory-efficient and is necessary because we are not training the model
    with torch.no_grad():
        main()

# YOLOV8 OBJECT DETECTION ON GPU -> ZED SDK GPU SINGLE CAMERA

import os
import numpy as np
import keyboard
import argparse

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

# from class_dicts import dict_365, dict_coco, dict_imagenet, dict_voc

# Setting up environment variable to allow for multiple instances of OpenMP to run simultaneously
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Creating a lock to prevent simultaneous access to shared resources from different threads
lock = Lock()
# Control execution of the threads
run_signal = False
exit_signal = False

# Function to convert bounding box from (center x, center y, width, height) format
# to (top left, top right, bottom left, bottom right) format
def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    x_min = (xywh[0] - 0.5*xywh[2])
    x_max = (xywh[0] + 0.5*xywh[2])
    y_min = (xywh[1] - 0.5*xywh[3])
    y_max = (xywh[1] + 0.5*xywh[3])

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
def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections, class_names, class_dict,yolo_output_label
    try:
        print("Initializing Network...")

        # create a yolov8 model object
        model = YOLO(weights)        
        model.to('cuda') # moves the model to GPU

        # get the list of class names
        class_names = model.names
        print("Class names: ", class_names)

        # Loop continues indefinitely until the exit signal is true
        while not exit_signal:
            # When the run signal is true, acquire the lock to prevent simultaneous access to shared resources
            if run_signal:
                lock.acquire()

                # Convert the image to RGB
                img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)

                # Run YOLO on the image on the GPU
                det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)[0].cpu().numpy().boxes

                # Convert the YOLO detections to a ZED SDK ingestable format
                detections = detections_to_custom_box(det, image_net)

                # Pass the yolo output label to a variable for the main thread to access
                for yolo_output in detections:
                    yolo_output_label = yolo_output.label

                # Release the lock
                lock.release()

                # Reset the run signal to false
                run_signal = False

            # Pause for a short time
            sleep(0.01)

    except Exception as e:
        print("Error in torch thread: ", e)


# Main function
def main():
    global image_net, exit_signal, run_signal, detections

    # Initialize CUDA in the main thread
    print("Initializing CUDA...")
    if torch.cuda.is_available():
        _ = torch.cuda.FloatTensor(1)

    # Create a new thread for the YOLO model
    print("Starting YOLO torch thread...")
    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()
    print("YOLO torch thread started...")

    # Initialize Camera
    print("Initializing Camera...")

    # Create camera object
    zed = sl.Camera()

    # Detects if there's a SVO file for YOLO detection. But if there's none, it keeps the default, which is live video
    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Set up camera parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    # Set runtime parameters
    runtime_params = sl.RuntimeParameters()

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    # Create a temporary image matrix for the left camera
    image_left_tmp = sl.Mat()

    print("Initialized Camera")

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
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Detect objects loop
    while not exit_signal:
        # If able to grab image from camera
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            # Acquire lock
            lock.acquire()
            # Retrieve image from left camera
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            # Release lock
            lock.release()

            # RUN YOLO DETECTION IN THE OTHER THREAD
            run_signal = True

            # Wait for detections on the other thread
            while run_signal:
                sleep(0.001)

            # Acquire lock
            lock.acquire()
            # Ingest detections into ZED SDK
            zed.ingest_custom_box_objects(detections)
            # Release lock
            lock.release()

            # Retrieve the detected objects
            zed.retrieve_objects(objects, detection_runtime_parameters)

            # Retrieve object list
            if objects.is_new:
                obj_array = objects.object_list
                # Print out the detected objects x, y, z coordinates in 3D space
                if len(obj_array) > 0:
                    for obj in obj_array:
                         # Print out the detected objects x, y, z coordinates, its ID, and its label
                         print(f"Object: {class_names[yolo_output_label]} {obj.id}, 3D location: x: {obj.position[0]} y: {obj.position[1]} z: {obj.position[2]}")

                        # print(f"Position in 3D space, x: {obj.position[0]} y: {obj.position[1]} z: {obj.position[2]}")

            # 2D Display
            # Copy the left image data to image_left_ocv (numpy array)
            np.copyto(image_left_ocv, image_left_tmp.get_data())
            # renders the 2d view of the image
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, detection_parameters.enable_tracking)
            
            # Display the image using OpenCV
            cv2.imshow("2D View", image_left_ocv)
            
            # This allows OpenCV to process its event queue
            cv2.waitKey(10)

            # If 'q' is pressed, exit the loop
            if keyboard.is_pressed('q'):
                exit_signal = True
        else:
            exit_signal = True

    # Close the camera
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models\yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()

    # Run main function without calculating gradients
    # This is more memory-efficient and is necessary because we are not training the model
    with torch.no_grad():
        main()

# HEAVY WITH ANNOTATIONS, FIRST GPU VERSION

import os
# Allow for multiple instances of OpenMP to run simultaneously
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# import sys
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
# from ultralytics.yolo.engine.model import YOLO # In case the above import doesn't work

# For multi-threading and delays
from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

# to prevent simulteanous access to shared resources from different threads
lock = Lock()
# to control execution of the threads
run_signal = False
exit_signal = False

# Function to convert from (center x, center y, width, height) to (top left,top right, bottom left, bottom right)
# input array is of shape (4,2) and output array is of shape (4,2)
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

# Function runs YOLO on a separate thread
# Input: YOLO weights, image size, confidence threshold, Interest over union threshold for detections
def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections
    try:
        print("Intializing Network...")

        # create a yolov8 model object
        model = YOLO(weights)        
        model.to('cuda') # moves the model to GPU

        while not exit_signal:
            if run_signal:
                lock.acquire() # when the run signal is true, acquire the lock to prevent simultaneous access to shared resources

                img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB) # converts the image to RGB
                # Run YOLO on the image on the GPU
                det = model.predict(img, save=True, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

                # ZED CustomBox format (with inverse letterboxing tf applied)
                detections = detections_to_custom_box(det, image_net) # converts the YOLO detections to a ZED SDK ingestable format
                lock.release() # releases lock
                run_signal = False # resets the run signal to false
            sleep(0.01)
            # the loop continues indefinitely until the exit signal is true
    except Exception as e:
        print("Error in torch thread: ", e)

print("Starting main function...")
def main():
    global image_net, exit_signal, run_signal, detections

    # Initialize CUDA in the main thread
    print("Initializing CUDA...")
    if torch.cuda.is_available():
        _ = torch.cuda.FloatTensor(1)

    # Create a new thread  for the YOLO model
    # passing all the arguments to the thread
    print("Starting YOLO torch thread...")
    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()
    print("YOLO torch thread started...")

    print("Initializing Camera...")

    # Create camera object
    zed = sl.Camera()

    # Detects if there's a SVO file for YOLO detection. But if there's none, it keeps the default, which is live video
    input_type = sl.InputType() # creates an input type object
    if opt.svo is not None: # if the input is a SVO file (steroscopic video file recorded by the ZED camera)
        input_type.set_from_svo_file(opt.svo) # sets the input type to SVO file

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use UlTRA depth mode
    init_params.coordinate_units = sl.UNIT.METER # Use meter units (for depth measurements)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50 # sets the maximum depth distance to 50 meters

    # Set runtime parameters
    runtime_params = sl.RuntimeParameters()

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    image_left_tmp = sl.Mat() # creates a temporary image matrix for the left camera

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is moving, comment the following line to have better performances and boxes sticked to the ground.
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters) # enable tracking parameters, required for object tracking

    # Enable object detection parameters
    detection_parameters = sl.ObjectDetectionParameters()
    detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    detection_parameters.enable_tracking = True
    zed.enable_object_detection(detection_parameters)

    # Set runtime parameters
    detection_runtime_parameters = sl.ObjectDetectionRuntimeParameters()

    objects = sl.Objects() # to store the detected objects

    # DISPLAY
    # Get camera info
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404)) # point cloud resolution
    point_cloud_render = sl.Mat() # to store the point cloud render
    viewer.init(camera_infos.camera_model, point_cloud_res, detection_parameters.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU) # to store the point cloud
    image_left = sl.Mat() # to store the left image

    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    # Creates a numpy arrary for the left image, same resolution as display resolution, with 4 channels (RGBA)
    # Array is filled with 245, 239, 239, 255 (RGBA) values which is the color light gray in RGBA format
    # Unit8 is used for the color values because it has a range of 0-255
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height) # tracks resolution 400 by the display resolution height (720)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    # Creates a numpy arrary for the tracks view, same resolution as display resolution, with 4 channels (RGBA)
    # Array is filled with zeros, which represents black in RGBA format
    # Unit8 is used for the color values because it has a range of 0-255
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
   
    # Camera pose
    cam_w_pose = sl.Pose() # camera world pose


    # Detect objects loop
    while viewer.is_available() and not exit_signal: # while the viewer is available and exit signal is false
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True # if successful in retrieving the left image, its data, it sets the run signal to true
            # run signal is used to execute the detection running on another the other thread

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections on the other thread
            lock.acquire() # when the run_signal resets, acquires the lock
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections) # ingest detection into the ZED SDK
            lock.release()
            zed.retrieve_objects(objects, detection_runtime_parameters) # retrieves the detected objects

            # DISPLAY
            # Retrieve display data
            # retrieves xyz data and color data for eacpoint in the pcd
            # retrieves the data from the CPU memory
            # Specify the resolution of the point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            # copies the point cloud data to the point cloud render
            # PCD for data retrieval, PCD_render for display
            point_cloud.copy_to(point_cloud_render)
            # retrieves the left image
            # retreives the data from the CPU memory
            # Specify the resolution of the left image
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # retrieves position of the camera, camera_w_pose holds the camera position data
            # positionl is relative to the world frame, a fixed reference frame that doesn't move with the camera
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud_render, objects) # updates the data in the 3d viewer (openGL)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data()) # copies the left image data to image_left_ocv (numpy array)
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, detection_parameters.enable_tracking) # renders the 2d view of the image
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked) # renders the 3d track view
            # concatenates the 2d view and the birds eye view (track view) horizontally into a new image
            # image_left_ocv on the left, image_track_ocv on the right
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            # Process openCV events
            cv2.waitKey(10)
            # Press q to exit the loop, no matter which windows is active
            if keyboard.is_pressed('q'):
                exit_signal = True
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models\yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
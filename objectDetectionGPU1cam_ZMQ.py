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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

zed = sl.Camera()  # create a camera object
lock = Lock()  # create a lock object

global_run_signal = False
global_exit_signal = False

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
def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output

# Function that runs YOLO on a separate thread
def torch_thread(weights, img_size, conf_thres=0.7, iou_thres=0.7):
    global image, class_names, yolo_output_label, global_run_signal, global_exit_signal, yolo_output_id, detections
    try:
        print("Starting Torch Thread")
        print("Initializing Network...")

        model = YOLO(weights)        
        model.to('cuda')

        class_names = model.names
        print("Class names: ", class_names)

        while not global_exit_signal:
            if global_run_signal:
                lock.acquire()

                img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

                det = model.track(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres, tracker='bytetrack.yaml', verbose=False)[0].cpu().numpy().boxes

                # Filter for only humans
                det = [det for det in det if det.cls == 0]

                # Convert the YOLO detections to a ZED SDK ingestable format
                detections = detections_to_custom_box(det, image)

                # Get YOLO output label
                try:
                    for yolo_output in detections:
                        yolo_output_label = yolo_output.label
                except Exception as e:
                    print("Error in yolo output label: ", e)

                # Get YOLO output id
                for yolo_output in det:
                    yolo_output_id = int(yolo_output.id.item())

                lock.release()

                # reset the global run signal to false
                global_run_signal = False

            sleep(0.01)
    
        print("Completed Torch Thread")

    except Exception as e:
        print("Error in torch thread: ", e)
        import traceback
        traceback.print_exc()

def open_camera(svo_filepath=None):
    input_type = sl.InputType()

    if svo_filepath is not None:
        input_type.set_from_svo_file(svo_filepath)
        print(f"Using SVO file {svo_filepath}")
    
    init_params = sl.InitParameters(svo_real_time_mode=True)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error {err} opening camera")
        exit(1)
    
    print("Initialized camera")
    return zed

def main(svo_filepath=None):
    global image, global_run_signal, global_exit_signal, detections


    print("Initializing CUDA...")
    if torch.cuda.is_available():
        _ = torch.cuda.FloatTensor(1)

    print("Starting YOLO torch thread...")
    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, 'conf_thres': opt.conf_thres})
    capture_thread.start()
    print("YOLO torch thread started...")

    zed = open_camera(svo_filepath)
    
    # set runtime parameters
    runtime_params = sl.RuntimeParameters()

    # create a temp image matrix for the left camera
    image_left_tmp = sl.Mat()

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    detection_parameters = sl.ObjectDetectionParameters()
    detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    detection_parameters.enable_tracking = True
    zed.enable_object_detection(detection_parameters)

    # set object deteciton runtime parameters
    detection_runtime_parameters = sl.ObjectDetectionRuntimeParameters()
    
    # to store detected objects
    objects = sl.Objects()

    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    display_resolution = sl.Resolution(min(camera_res.width, 1920), min(camera_res.height, 1080))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    while not global_exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image = image_left_tmp.get_data()
            lock.release()

            global_run_signal = True

            while global_run_signal:
                sleep(0.001)

            lock.acquire()
            zed.ingest_custom_box_objects(detections)
            lock.release()

            zed.retrieve_objects(objects, detection_runtime_parameters)

            if objects.is_new:
                obj_array = objects.object_list
                if len(obj_array) > 0:
                    data_list = []
                    for obj in obj_array:
                        # print the coordinates and yolo id
                        print(f"Yolo ID: {yolo_output_id} | X: {obj.position[0]} | Y: {obj.position[1]} | Z: {obj.position[2]}")
                        
                        # Dictionary to send over ZMQ
                        data = {
                            'obj_id': yolo_output_id,
                            'type': 'coord',
                            'x': obj.position[0],
                            'y': obj.position[1],
                            'z': obj.position[2]
                        }
                        data_list.append(data)

                        json_string = json.dumps(data_list)
                        publisher.send_string(json_string)

            np.copyto(image_left_ocv, image_left_tmp.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, detection_parameters.enable_tracking)
            
            if opt.dev:
                cv2.imshow("2D View Camera", image_left_ocv)
                cv2.waitKey(10)

            if keyboard.is_pressed('q'):
                global_exit_signal = True
                break

        else:
            global_exit_signal = True
            break

    global_exit_signal = True
    zed.close()

    publisher.send_string('{"end": true}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--dev', action='store_true', help='dev mode gives you OpenCV windows')
    opt = parser.parse_args()
    
    with torch.no_grad():
        main()

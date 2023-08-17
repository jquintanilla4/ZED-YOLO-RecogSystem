import time

static_objects = {} # Dictionary to keep track of static objects
STATIC_OBJ_TIME_THRESHOLD_IN_SECS = 10 # number of seconds before an object is considered static

def log_static_object(obj_id: str, x: float, y: float, z: float):
      global static_objects
      if obj_id not in static_objects:
            static_objects[obj_id] = {"x": x, "y": y, "z": z, "prev_time": 0, "total_time": 0}
      elif abs(static_objects[obj_id]["x"] - x) > 0.01 or abs(static_objects[obj_id]["y"] - y) > 0.01 or abs(static_objects[obj_id]["z"] - z) > 0.01:
            # If the object has moved, reset the timer
            static_objects[obj_id]["x"] = x
            static_objects[obj_id]["y"] = y
            static_objects[obj_id]["z"] = z
            static_objects[obj_id]["prev_time"] = time.time()
            static_objects[obj_id]["total_time"] = 0
      else:
            static_objects[obj_id]["total_time"] += time.time() - static_objects[obj_id]["prev_time"]
            static_objects[obj_id]["prev_time"] = time.time()
            
def is_static_obj(obj_id: str):
      global static_objects
      # Check if the object has barely moved given a threshold
      if obj_id in static_objects and static_objects[obj_id]["total_time"] > STATIC_OBJ_TIME_THRESHOLD_IN_SECS:
            return False
      else:
            return True
      
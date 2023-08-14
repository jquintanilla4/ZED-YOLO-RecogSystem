import aioconsole
import zmq
import asyncio
import websockets
import json
from math import isnan

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect('tcp://localhost:5555')
subscriber.setsockopt_string(zmq.SUBSCRIBE, '')

server = None # Global variable for the server
seen_objects = {} # Dictionary to keep track of seen objects and loops since last seen
unreal_disconnected = False # Flag for when Unreal Engine disconnects

async def echo(websocket, _):
      global seen_objects, unreal_disconnected
      try:
            while True:
                  try: # Exception for sending no data
                        message = subscriber.recv_string(zmq.NOBLOCK)
                  except zmq.Again:
                        # print("No data to send")
                        continue

                  if message == '{"end": true}':
                        print("Stopping server...")
                        server.close()
                        break

                  # The message is a JSON string of a list of dictionaries
                  data_list = json.loads(message) # converts a string into a list of python dictionaries

                  current_ids = [data['obj_id'] for data in data_list] # List of ids in the current message

                  # Destory object mesaage logic
                  for obj_id in list(seen_objects.keys()): # create a copy of the keys to avoid dictionary size changing during iteration
                        if obj_id not in current_ids: # If an object is not in the current message
                              seen_objects[obj_id] += 1 # increment the count of loops since last seen
                              if seen_objects[obj_id] > 15: # send a message to Unreal Engine to delete the object if hasn't been seen in 15 loops
                                    await websocket.send(json.dumps({"obj_id": obj_id, "type": "create"}))
                                    seen_objects.pop(obj_id) # Remove the object id from the dictionary
                        else:
                              seen_objects[obj_id] = 0 # reset count if object has been seen

                  # Create object message logic
                  for obj_id in current_ids:
                        if obj_id not in seen_objects: # send message when new object is seen
                              await websocket.send(json.dumps({"obj_id": obj_id, "type": "destroy"}))
                              seen_objects[obj_id] = 0 # add object id to dictionary and set count to 0


                  for data in data_list: # Send each object's data
                        if any(isnan(value) for value in data.values()): # Check if any values are NaN
                              continue # Skip the object if it contains a value NaN
                        await websocket.send(json.dumps(data)) # converts it back to a json string

                  await asyncio.sleep(0.001)

      except websockets.exceptions.ConnectionClosed:
            print("Unreal Engine disconnected\n ")
            unreal_disconnected = True
      except asyncio.CancelledError: # Exception for stopping the server early with the exit command
            return 
            

async def main():
    global server, unreal_disconnected
    server = await websockets.serve(echo, "0.0.0.0", 8765)

    while True:
        try:
            cmd = await aioconsole.ainput(">>> ")
            if cmd == 'exit':
                  print("Stopping server...")
                  server.close()
                  break

            if unreal_disconnected: # Check the unreal flag here
                  print(">>> ", end='')
                  unreal_disconnected = False

            _, pending = await asyncio.wait(
                  [server.wait_closed()],
                  return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                  task.cancel()

        except asyncio.CancelledError:
              continue


# starting the server
asyncio.run(main())

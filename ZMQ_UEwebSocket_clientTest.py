import asyncio
import websockets
import json
import logging

# async def test():
#     uri = "ws://localhost:8765"
#     async with websockets.connect(uri) as websocket:
#         while True:
#             try:
#                 message = await websocket.recv()
#                 data = json.loads(message)
#                 print(data)
#             except websockets.exceptions.ConnectionClosed:
#                 print("Unreal Engine disconnected")
#                 break
#             except Exception as e:
#                 print(e)
#                 continue

# asyncio.run(test())

logging.basicConfig(level=logging.INFO)

async def test():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                logging.info(data)
            except websockets.exceptions.ConnectionClosed:
                logging.error("Unreal Engine disconnected")
                break
            except Exception as e:
                logging.error(e)
                continue

asyncio.run(test())

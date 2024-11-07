# timer_client.py

import grpc
import timer_pb2
import timer_pb2_grpc
import time
from datetime import datetime

def run():
    # Connect to the server
    with grpc.insecure_channel('192.168.64.212:50051') as channel:
        stub = timer_pb2_grpc.TimerServiceStub(channel)
        response = stub.GetCurrentTime(timer_pb2.TimeRequest())
        print("Current Time from Server:", response.current_time)
        # First request to get initial time
        response1 = stub.GetCurrentTime(timer_pb2.TimeRequest())
        print("First Execution Time from Server:", response1.current_time)
        
        # Sleep for a short duration to simulate a delay
        time.sleep(2)  # you can adjust the sleep duration as needed

        # Second request to get time again
        response2 = stub.GetCurrentTime(timer_pb2.TimeRequest())
        print("Second Execution Time from Server:", response2.current_time)

        # Parse the received time strings to datetime objects
        time1 = datetime.fromisoformat(response1.current_time)
        time2 = datetime.fromisoformat(response2.current_time)

        # Calculate the difference between the two times
        time_difference = time2 - time1
        print("Time Difference Between Executions:", time_difference)

if __name__ == '__main__':
    run()

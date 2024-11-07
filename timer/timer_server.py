# timer_server.py

from concurrent import futures
import grpc
import timer_pb2
import timer_pb2_grpc
from datetime import datetime

class TimerService(timer_pb2_grpc.TimerServiceServicer):
    def GetCurrentTime(self, request, context):
        # Get the current server time in ISO format
        current_time = datetime.now().isoformat()
        return timer_pb2.TimeResponse(current_time=current_time)

def serve():
    # Create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    timer_pb2_grpc.add_TimerServiceServicer_to_server(TimerService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started at port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

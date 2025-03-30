import grpc
from concurrent import futures
import time
import sys
import os
import base64
import logging

# Import the function from your existing module
from cuff_detection import detect_cuff_position_base64

# Add paths for the generated proto files
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto_gen'))

# Import the generated proto classes
import service_pb2
import service_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define model paths
POSE_MODEL_PATH = "yolo11n-pose.pt"
SEGMENT_MODEL_PATH = "yoloe-v8l-seg.pt"
CUFF_CLASS_NAME = "sphygmomanometer cuff"
DEFAULT_THRESHOLD = 10

class CuffDetectionServicer(service_pb2_grpc.CuffDetectionServiceServicer):
    def __init__(self):
        logging.info("Initializing CuffDetectionServicer")
        # You can load models here if you want to avoid loading them for each request
        # self.pose_model = YOLO(POSE_MODEL_PATH)
        # self.segment_model = YOLOE(SEGMENT_MODEL_PATH)

    def DetectCuffPosition(self, request, context):
        logging.info("Received detection request")
        try:
            # Extract parameters from request
            base64_image = request.base64_image
            threshold = request.threshold if request.HasField("threshold") else DEFAULT_THRESHOLD
            
            # Call the detection function
            position = detect_cuff_position_base64(
                base64_image, 
                POSE_MODEL_PATH, 
                SEGMENT_MODEL_PATH, 
                cuff_class_name=CUFF_CLASS_NAME, 
                threshold=threshold
            )
            
            # Create and return response
            return service_pb2.CuffDetectionResponse(
                position=position,
                success=True,
                error_message=""
            )
            
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            return service_pb2.CuffDetectionResponse(
                position="Error",
                success=False,
                error_message=str(e)
            )

def serve():
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add our servicer to the server
    service_pb2_grpc.add_CuffDetectionServiceServicer_to_server(
        CuffDetectionServicer(), server
    )
    
    # Add secure/insecure port
    server_address = '[::]:50051'
    server.add_insecure_port(server_address)
    
    # Start server
    server.start()
    logging.info(f"Server started on {server_address}")
    
    try:
        # Keep server running
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        logging.info("Server shutting down")
        server.stop(0)

if __name__ == '__main__':
    # Generate proto files if needed first - can be a separate script too
    # os.system('python -m grpc_tools.protoc -I./proto --python_out=./proto_gen --grpc_python_out=./proto_gen ./proto/service.proto')
    
    # Start server
    serve()
import grpc
from concurrent import futures
import time
import sys
import os
import logging

# Import the enhanced detection function
from cuff_detection import detect_health_metrics

# Add paths for the generated proto files
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto_gen'))

# Import the generated proto classes
import service_pb2
import service_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define model paths
POSE_MODEL_PATH = "yolo11n-pose.pt"
SEGMENT_MODEL_PATH = "yoloe-11l-seg.pt"
CUFF_CLASS_NAME = "arm cuff"
DEFAULT_THRESHOLD = 10

class CuffDetectionServicer(service_pb2_grpc.CuffDetectionServiceServicer):
    def __init__(self):
        logging.info("Initializing CuffDetectionServicer")

    def DetectCuffPosition(self, request, context):
        logging.info("Received detection request")
        try:
            # Extract parameters from request
            base64_image = request.base64_image
            threshold = request.threshold if request.HasField("threshold") else DEFAULT_THRESHOLD
            
            # Call the enhanced detection function
            result = detect_health_metrics(
                base64_image, 
                POSE_MODEL_PATH, 
                SEGMENT_MODEL_PATH, 
                cuff_class_name=CUFF_CLASS_NAME, 
                threshold=threshold
            )
            
            # Return single response
            return service_pb2.CuffDetectionResponse(
                position=result.position,
                success=result.success,
                error_message=result.error_message,
                posture_correct=result.posture_correct,
                position_correct=result.position_correct,
                is_talking=result.is_talking
            )
            
        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            return service_pb2.CuffDetectionResponse(
                position="Error",
                success=False,
                error_message=str(e)
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_CuffDetectionServiceServicer_to_server(
        CuffDetectionServicer(), server
    )
    
    server_address = '[::]:50051'
    server.add_insecure_port(server_address)
    server.start()
    logging.info(f"Server started on {server_address}")
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logging.info("Server shutting down")
        server.stop(0)

if __name__ == '__main__':
    serve()

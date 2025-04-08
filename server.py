import grpc
from concurrent import futures
import time
import sys
import os
import logging
import openai
from dotenv import load_dotenv

# Import the enhanced detection function
from cuff_detection import detect_health_metrics

# Add paths for the generated proto files
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto_gen'))

# Import the generated proto classes
import service_pb2
import service_pb2_grpc

from chat import ChatService  # Add this import at the top

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

class ChatServicer(service_pb2_grpc.ChatServiceServicer):
    def __init__(self):
        self.chat_service = ChatService()

    def Chat(self, request_iterator, context):
        """
        Handle streaming chat via gRPC
        """
        try:
            for request in request_iterator:
                # Use ChatService to handle messages
                async for response in self.chat_service.chat(request.message):
                    if response["is_error"]:
                        yield service_pb2.ChatResponse(
                            message="",
                            is_error=True,
                            error_message=response["error_message"]
                        )
                    else:
                        yield service_pb2.ChatResponse(
                            message=response["content"],
                            is_error=False
                        )
                    
        except Exception as e:
            yield service_pb2.ChatResponse(
                message="",
                is_error=True, 
                error_message=f"Stream error: {str(e)}"
            )

def serve():
    """Start the combined server with both services"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add both services to the same server
    service_pb2_grpc.add_CuffDetectionServiceServicer_to_server(
        CuffDetectionServicer(), server
    )
    service_pb2_grpc.add_ChatServiceServicer_to_server(
        ChatServicer(), server
    )
    
    server_address = '[::]:50051'
    server.add_insecure_port(server_address)
    server.start()
    
    logging.info(f"Server started on {server_address} with both Cuff Detection and Chat services")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
        server.stop(0)

def serve_chat():
    """Start the chat server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_ChatServiceServicer_to_server(ChatServicer(), server)
    
    server_address = '[::]:50052'  # Use different port from cuff detection
    server.add_insecure_port(server_address)
    server.start()
    
    logging.info(f"Chat server started on {server_address}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Chat server shutting down...")
        server.stop(0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve_chat()

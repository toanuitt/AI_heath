import grpc
import sys
import os
import base64
import logging

# Add paths for the generated proto files
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto_gen'))

# Import the generated proto classes
import service_pb2
import service_pb2_grpc

def detect_cuff_position(image_path, threshold=10):
    """
    Connect to gRPC server and detect cuff position
    
    Args:
        image_path: Path to the image file
        threshold: Detection threshold
        
    Returns:
        The detected position
    """
    # Read image and encode to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Connect to server
    with grpc.insecure_channel('localhost:50051') as channel:
        # Create stub
        stub = service_pb2_grpc.CuffDetectionServiceStub(channel)
        
        # Create request
        request = service_pb2.CuffDetectionRequest(
            base64_image=base64_image,
            threshold=threshold
        )
        
        # Call service
        response = stub.DetectCuffPosition(request)
        
        return response

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Get image path from command line or use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "./d.jpg"
    
    # Call detection
    response = detect_cuff_position(image_path)
    
    # Print results
    if response.success:
        print(f"Cuff position: {response.position}")
    else:
        print(f"Error: {response.error_message}")
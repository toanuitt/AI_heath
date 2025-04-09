import grpc
import base64
import os
import logging
from proto_gen import service_pb2
from proto_gen import service_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cuff_detection(stub, image_path):
    """Test the cuff detection service with a sample image"""
    try:
        # Read and encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Create request
        request = service_pb2.CuffDetectionRequest(
            base64_image=base64_image,
            threshold=10
        )
        
        # Make the call
        response = stub.DetectCuffPosition(request)
        
        # Log results
        logger.info("Cuff Detection Results:")
        logger.info(f"Position: {response.position}")
        logger.info(f"Success: {response.success}")
        logger.info(f"Error Message: {response.error_message}")
        logger.info(f"Posture Correct: {response.posture_correct}")
        logger.info(f"Position Correct: {response.position_correct}")
        logger.info(f"Is Talking: {response.is_talking}")
        
        return response.success
        
    except Exception as e:
        logger.error(f"Error testing cuff detection: {e}")
        return False

def test_chat(stub):
    """Test the chat service"""
    try:
        # Create a test message
        def generate_messages():
            yield service_pb2.ChatRequest(message="Hello, how are you?")
        
        # Make the streaming call
        responses = stub.Chat(generate_messages())
        
        # Process responses
        for response in responses:
            if response.is_error:
                logger.error(f"Chat Error: {response.error_message}")
                return False
            else:
                logger.info(f"Chat Response: {response.message}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing chat service: {e}")
        return False

def main():
    try:
        # Connect to the server
        channel = grpc.insecure_channel('localhost:50051')
        
        # Create stubs
        cuff_stub = service_pb2_grpc.CuffDetectionServiceStub(channel)
        chat_stub = service_pb2_grpc.ChatServiceStub(channel)
        
        # Test image path (modify this to your test image path)
        test_image_path = "/home/ductoan/AI_health/tuthe.jpg"
        
        # Ensure test image exists
        if not os.path.exists(test_image_path):
            logger.error(f"Test image not found at {test_image_path}")
            return False
        
        # Run tests
        logger.info("Starting Docker container tests...")
        
        cuff_success = test_cuff_detection(cuff_stub, test_image_path)
        logger.info(f"Cuff detection test {'passed' if cuff_success else 'failed'}")
        
        chat_success = test_chat(chat_stub)
        logger.info(f"Chat service test {'passed' if chat_success else 'failed'}")
        
        # Overall test result
        all_passed = cuff_success and chat_success
        logger.info(f"All tests {'passed' if all_passed else 'failed'}")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False
    finally:
        if 'channel' in locals():
            channel.close()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
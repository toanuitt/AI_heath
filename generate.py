import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_proto():
    """Generate Python code from proto file"""
    # Define paths
    proto_dir = os.path.join(os.path.dirname(__file__), 'proto')
    output_dir = os.path.join(os.path.dirname(__file__), 'proto_gen')
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if proto directory exists
    if not os.path.exists(proto_dir):
        logger.error(f"Proto directory not found at {proto_dir}")
        logger.info("Make sure you have cloned the proto submodule")
        return False
    
    # Check if service.proto exists
    proto_file = os.path.join(proto_dir, 'service.proto')
    if not os.path.exists(proto_file):
        logger.error(f"Proto file not found at {proto_file}")
        return False
    
    # Generate Python code from proto file
    command = f'python -m grpc_tools.protoc -I{proto_dir} --python_out={output_dir} --grpc_python_out={output_dir} {proto_file}'
    logger.info(f"Running command: {command}")
    
    result = os.system(command)
    if result != 0:
        logger.error("Failed to generate proto files")
        return False
    
    # Create __init__.py file to make it a proper package
    init_file = os.path.join(output_dir, '__init__.py')
    with open(init_file, 'w') as f:
        pass
    
    logger.info("Proto files generated successfully!")
    return True

if __name__ == '__main__':
    if not generate_proto():
        sys.exit(1)
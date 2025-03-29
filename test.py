import numpy as np
import cv2
from ultralytics import YOLO, YOLOE
import supervision as sv
from PIL import Image
import base64
import io

def detect_cuff_position_base64(base64_image, pose_model_path, segment_model_path, cuff_class_name="sphygmomanometer cuff", threshold=10):
    """
    Detects whether a blood pressure cuff is positioned up, down, or centered (OK)
    relative to the line between left shoulder and left elbow.
    
    Args:
        base64_image (str): Base64 encoded image string
        pose_model_path (str): Path to the YOLO pose estimation model
        segment_model_path (str): Path to the YOLO segmentation model for cuff detection
        cuff_class_name (str): Class name for the blood pressure cuff
        threshold (int): Pixel threshold for considering the cuff centered (OK)
        
    Returns:
        str: "up", "down", or "OK" indicating the cuff position relative to arm line
        bytes: The visualization image as bytes (can be converted to base64 if needed)
    """
    # Load models
    pose_model = YOLO(pose_model_path)
    segment_model = YOLOE(segment_model_path)
    segment_model.set_classes([cuff_class_name], segment_model.get_text_pe([cuff_class_name]))
    
    # Decode base64 image
    img_bytes = base64.b64decode(base64_image)
    
    # Convert to OpenCV and PIL formats
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    pil_image = Image.open(io.BytesIO(img_bytes))
    
    # Get pose keypoints
    pose_results = pose_model(image, conf=0.6)[0]
    keypoints = pose_results.keypoints.data.cpu().numpy() if pose_results.keypoints is not None else []
    
    if len(keypoints) == 0:
        return "No person detected", None
    
    # Get left shoulder and left elbow coordinates
    left_shoulder = keypoints[0][5][:2]  # x, y coordinates of left shoulder (keypoint index 5)
    left_elbow = keypoints[0][7][:2]     # x, y coordinates of left elbow (keypoint index 7)
    
    # Detect cuff
    seg_results = segment_model.predict(pil_image, conf=0.3, verbose=False)
    detections = sv.Detections.from_ultralytics(seg_results[0])
    
    if len(detections.xyxy) == 0:
        return "No cuff detected", None
    
    # Get cuff bounding box and calculate center
    cuff_box = detections.xyxy[0]  # Get the first cuff detected
    cuff_center = np.array([
        (cuff_box[0] + cuff_box[2]) / 2,  # x center
        (cuff_box[1] + cuff_box[3]) / 2   # y center
    ])
    
    # Calculate arm line vector (from shoulder to elbow)
    arm_vector = left_elbow - left_shoulder
    arm_length = np.linalg.norm(arm_vector)
    arm_unit_vector = arm_vector / arm_length
    
    # Calculate perpendicular vector (90 degrees counterclockwise rotation)
    perp_vector = np.array([-arm_unit_vector[1], arm_unit_vector[0]])
    
    # Vector from shoulder to cuff center
    cuff_vector = cuff_center - left_shoulder
    
    # Project cuff vector onto perpendicular direction
    perp_projection = np.dot(cuff_vector, perp_vector)
    
    # Determine if cuff is up, down, or centered (OK) relative to arm line
    if abs(perp_projection) <= threshold:
        position = "OK"
    else:
        position = "up" if perp_projection > 0 else "down"
    
    # Visualize the results
    vis_image_bytes = visualize_result(image, left_shoulder, left_elbow, cuff_center, position, perp_projection)
    
    return position, vis_image_bytes

# def visualize_result(image, left_shoulder, left_elbow, cuff_center, position, distance):
#     """
#     Visualizes the position analysis on the image
    
#     Args:
#         image: Input image
#         left_shoulder: Coordinates of left shoulder
#         left_elbow: Coordinates of left elbow
#         cuff_center: Coordinates of cuff center
#         position: "up", "down", or "OK" string
#         distance: Perpendicular distance from arm line
        
#     Returns:
#         bytes: Image as bytes that can be converted to base64 if needed
#     """
#     # Create a copy of the image for visualization
#     vis_image = image.copy()
    
#     # Draw shoulder and elbow points
#     cv2.circle(vis_image, (int(left_shoulder[0]), int(left_shoulder[1])), 5, (0, 0, 255), -1)
#     cv2.circle(vis_image, (int(left_elbow[0]), int(left_elbow[1])), 5, (0, 0, 255), -1)
    
#     # Draw line between shoulder and elbow
#     cv2.line(vis_image, 
#              (int(left_shoulder[0]), int(left_shoulder[1])), 
#              (int(left_elbow[0]), int(left_elbow[1])), 
#              (0, 255, 0), 2)
    
#     # Draw cuff center
#     cv2.circle(vis_image, (int(cuff_center[0]), int(cuff_center[1])), 5, (255, 0, 0), -1)
    
#     # Choose color based on position
#     if position == "OK":
#         color = (0, 255, 0)  # Green for OK
#     else:
#         color = (0, 0, 255)  # Red for up/down
    
#     # Add position information as text
#     cv2.putText(vis_image, f"Cuff position: {position}", (10, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#     cv2.putText(vis_image, f"Distance: {abs(distance):.1f} pixels", (10, 60), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
#     # Convert image to bytes (can be encoded to base64 later if needed)
#     is_success, buffer = cv2.imencode(".jpg", vis_image)
#     if is_success:
#         return buffer.tobytes()
#     return None

# Example usage
if __name__ == "__main__":
    # Load image and convert to base64
    with open("/path/to/your/image.jpg", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    pose_model_path = "/path/to/yolo11n-pose.pt"
    segment_model_path = "/path/to/yoloe-v8l-seg.pt"
    
    position, vis_image_bytes = detect_cuff_position_base64(base64_image, pose_model_path, segment_model_path, threshold=10)
    print(f"Cuff position: {position}")
    
    # If you need the visualization as base64 for web display
    if vis_image_bytes:
        vis_base64 = base64.b64encode(vis_image_bytes).decode("utf-8")
        print(f"Visualization base64: {vis_base64[:30]}...") # Print just the beginning
        
        # To display in HTML you would use:
        # <img src="data:image/jpeg;base64,{vis_base64}" />
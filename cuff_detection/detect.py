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
        return "No person detected"
    
    # Get left shoulder and left elbow coordinates
    left_shoulder = keypoints[0][5][:2]  # x, y coordinates of left shoulder (keypoint index 5)
    left_elbow = keypoints[0][7][:2]     # x, y coordinates of left elbow (keypoint index 7)
    
    # Detect cuff
    seg_results = segment_model.predict(pil_image, conf=0.1, verbose=False)
    detections = sv.Detections.from_ultralytics(seg_results[0])
    
    if len(detections.xyxy) == 0:
        return "No cuff detected"
    
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
    
    return position

# # Test function when running this file directly
# if __name__ == "__main__":
#     # Load image and convert to base64
#     with open("./d.jpg", "rb") as image_file:
#         base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
#     pose_model_path = "yolo11n-pose.pt"
#     segment_model_path = "yoloe-v8l-seg.pt"
    
#     position = detect_cuff_position_base64(base64_image, pose_model_path, segment_model_path, threshold=10)
#     print(f"Cuff position: {position}")

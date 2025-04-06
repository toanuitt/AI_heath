import numpy as np
import cv2
import torch
import base64
import io
import math
from PIL import Image
import face_recognition
from ultralytics import YOLO, YOLOE
import supervision as sv

class CuffDetectionResponse:
    def __init__(self, position="unknown", success=None, error_message="", 
                posture_correct=None, position_correct=None, is_talking=None):
        self.position = position
        self.success = success
        self.error_message = error_message
        self.posture_correct = posture_correct
        self.position_correct = position_correct
        self.is_talking = is_talking
    
    def __str__(self):
        return (f"CuffDetectionResponse:\n"
                f"  position: {self.position}\n"
                f"  success: {self.success}\n"
                f"  error_message: {self.error_message}\n"
                f"  posture_correct: {self.posture_correct}\n"
                f"  position_correct: {self.position_correct}\n"
                f"  is_talking: {self.is_talking}")

def detect_health_metrics(base64_image, pose_model_path, segment_model_path, 
                         cuff_class_name="sphygmomanometer cuff", threshold=10):
    """
    Comprehensive function that checks:
    1. Blood pressure cuff position (up/down/OK)
    2. User's posture correctness
    3. Whether the user is talking (mouth open)
    
    Args:
        base64_image (str): Base64 encoded image string
        pose_model_path (str): Path to the YOLO pose estimation model
        segment_model_path (str): Path to the YOLO segmentation model for cuff detection
        cuff_class_name (str): Class name for the blood pressure cuff
        threshold (int): Pixel threshold for considering the cuff centered (OK)
        
    Returns:
        CuffDetectionResponse: Object containing all detection results
    """
    response = CuffDetectionResponse()
    
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(base64_image)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        pil_image = Image.open(io.BytesIO(img_bytes))
        
        # Load models
        pose_model = YOLO(pose_model_path)
        try:
            segment_model = YOLOE(segment_model_path)
            segment_model.set_classes([cuff_class_name], segment_model.get_text_pe([cuff_class_name]))
        except AttributeError:
            segment_model = YOLO(segment_model_path)
        
        # Get pose keypoints
        pose_results = pose_model(image, conf=0.6)[0]
        keypoints = pose_results.keypoints.data.cpu().numpy() if pose_results.keypoints is not None else []
        
        if len(keypoints) == 0:
            response.error_message = "No person detected"
            return response
            
        # Check posture
        posture_ok, posture_details = check_posture(keypoints[0])
        response.posture_correct = posture_ok
        
        # Get left shoulder and left elbow coordinates
        left_shoulder = keypoints[0][5][:2]  # x, y coordinates of left shoulder
        left_elbow = keypoints[0][7][:2]     # x, y coordinates of left elbow
        
        # Detect cuff
        seg_results = segment_model(pil_image, conf=0.1, verbose=False)
        detections = sv.Detections.from_ultralytics(seg_results[0])
        
        if len(detections.xyxy) == 0:
            response.error_message = "No cuff detected"
            response.position_correct = False
        else:
            # Get cuff bounding box and calculate center
            cuff_box = detections.xyxy[0]
            cuff_center = np.array([
                (cuff_box[0] + cuff_box[2]) / 2,
                (cuff_box[1] + cuff_box[3]) / 2
            ])
            
            # Calculate arm line vector and position
            arm_vector = left_elbow - left_shoulder
            arm_length = np.linalg.norm(arm_vector)
            arm_unit_vector = arm_vector / arm_length
            perp_vector = np.array([-arm_unit_vector[1], arm_unit_vector[0]])
            cuff_vector = cuff_center - left_shoulder
            arm_projection = np.dot(cuff_vector, arm_unit_vector)
            projected_point = left_shoulder + arm_projection * arm_unit_vector
            perp_vector_to_cuff = cuff_center - projected_point
            perp_distance = np.linalg.norm(perp_vector_to_cuff)
            
            # Determine cuff position
            is_between = 0 <= arm_projection <= arm_length
            if is_between and perp_distance <= threshold:
                response.position = "OK"
                response.position_correct = True
            else:
                cross_product_z = arm_vector[0] * perp_vector_to_cuff[1] - arm_vector[1] * perp_vector_to_cuff[0]
                response.position = "up" if cross_product_z < 0 else "down"
                response.position_correct = False
        # Check if talking (mouth open)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        if face_locations:
            face_landmarks_list = face_recognition.face_landmarks(rgb_image, face_locations)
            if face_landmarks_list:
                response.is_talking = is_mouth_open(face_landmarks_list[0])
        
        response.success = True
        return response
        
    except Exception as e:
        response.success = False
        response.error_message = f"Error processing image: {str(e)}"
        return response

def get_lip_height(lip):
    """Calculate the average height of a lip"""
    total = 0
    for i in [2, 3, 4]:
        distance = math.sqrt(
            (lip[i][0] - lip[12 - i][0]) ** 2 +
            (lip[i][1] - lip[12 - i][1]) ** 2
        )
        total += distance
    return total / 3

def get_mouth_height(top_lip, bottom_lip):
    """Calculate the distance between top and bottom lip"""
    total = 0
    for i in [8, 9, 10]:
        distance = math.sqrt(
            (top_lip[i][0] - bottom_lip[18 - i][0]) ** 2 +
            (top_lip[i][1] - bottom_lip[18 - i][1]) ** 2
        )
        total += distance
    return total / 3

def is_mouth_open(face_landmarks):
    """Determine if mouth is open based on face landmarks"""
    if "top_lip" in face_landmarks and "bottom_lip" in face_landmarks:
        top_lip = face_landmarks["top_lip"]
        bottom_lip = face_landmarks["bottom_lip"]

        top_height = get_lip_height(top_lip)
        bottom_height = get_lip_height(bottom_lip)
        mouth_height = get_mouth_height(top_lip, bottom_lip)

        ratio = 0.5
        return mouth_height > min(top_height, bottom_height) * ratio
    return False

def check_posture(kpts):
    """
    Checks if posture is correct: straight back, relaxed arms, uncrossed legs.
    Skip criteria if keypoints are missing and count as True.

    Args:
        kpts: ndarray shape (17, 3) - x, y, conf for each keypoint

    Returns:
        (bool, list[str]) -> True if posture is correct, with description list
    """
    result = []
    ok_flags = []

    def visible(ids): 
        return all(kpts[i][2] > 0.5 for i in ids)

    # Check back straightness (shoulders and hips should form a near-vertical line)
    if visible([5, 6, 11, 12]):
        s_mid = np.mean([kpts[5][:2], kpts[6][:2]], axis=0)
        h_mid = np.mean([kpts[11][:2], kpts[12][:2]], axis=0)
        angle = np.degrees(np.arctan2(s_mid[1] - h_mid[1], s_mid[0] - h_mid[0]))
        straight_back = angle <= 45 or angle >= 135  # nearly vertical
        result.append("Straight back" if straight_back else "Curved back")
        ok_flags.append(straight_back)
    else:
        result.append("Straight back (assumed)")
        ok_flags.append(True)  # Mặc định là true nếu thiếu điểm

    # Check left arm relaxation (wrist lower than elbow, elbow close to shoulder)
    if visible([5, 7, 9]):
        s = kpts[5]  # shoulder
        e = kpts[7]  # elbow
        w = kpts[9]  # wrist

        relaxed = (w[1] > e[1]) and (e[1] > s[1])  # y increases downward

        result.append("Relaxed arm" if relaxed else "Tense arm")
        ok_flags.append(relaxed)
    else:
        result.append("Relaxed arm (assumed)")
        ok_flags.append(True)  # Mặc định là true nếu thiếu điểm

    # Check legs not crossed
    if visible([11, 12, 15, 16]):
        lh, rh = kpts[11][0], kpts[12][0]  # left and right hip
        la, ra = kpts[15][0], kpts[16][0]  # left and right ankle
        not_crossed = (la < ra and lh < rh) or (la > ra and lh > rh)
        result.append("Uncrossed legs" if not_crossed else "Crossed legs")
        ok_flags.append(not_crossed)
    else:
        result.append("Uncrossed legs (assumed)")
        ok_flags.append(True)  # Mặc định là true nếu thiếu điểm

    overall_ok = all(ok_flags)  # Bỏ "if ok_flags else False" vì ok_flags luôn có phần tử
    return overall_ok, result

# # Example usage
# if __name__ == "__main__":
#     # Load your base64 image here
#     with open("/home/ductoan/AI_health/tải xuống (7).jpeg", "rb") as image_file:
#          base64_image = base64.b64encode(image_file.read()).decode("utf-8")
#     pose_model_path = "/home/ductoan/AI_health/yolo11n-pose.pt"
#     segment_model_path = "/home/ductoan/AI_health/yoloe-11l-seg.pt"
    
#     response = detect_health_metrics(base64_image, pose_model_path, segment_model_path)
#     print(response)

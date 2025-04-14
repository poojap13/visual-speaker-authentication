import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
from tqdm import tqdm

# Path to the pre-trained Dlib 68-landmark predictor.
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Initialize Dlib's face detector and shape predictor.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def extract_lip_region(frame, target_size=(64, 64)):
    """
    Extracts and resizes the lip region from a video frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    mouth = shape[48:68]
    (x, y, w, h) = cv2.boundingRect(mouth)
    lip_region = frame[y:y + h, x:x + w]
    lip_region_resized = cv2.resize(lip_region, target_size)
    return lip_region_resized


def compute_optical_flow(video_path, output_folder, target_size=(64, 64)):
    """
    Computes dense optical flow between lip regions of consecutive frames in a video.
    Saves flow as a .npy file.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_folder, video_name + ".npy")
    if os.path.exists(output_path):
        print(f"Skipping {video_path}: output already exists.")
        return

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print(f"Error reading video: {video_path}")
        cap.release()
        return

    lip_prev = extract_lip_region(first_frame, target_size)
    if lip_prev is None:
        print(f"No lip region detected in first frame of: {video_path}")
        cap.release()
        return

    lip_prev_gray = cv2.cvtColor(lip_prev, cv2.COLOR_BGR2GRAY)
    flow_maps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lip_curr = extract_lip_region(frame, target_size)
        if lip_curr is None:
            continue
        lip_curr_gray = cv2.cvtColor(lip_curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(lip_prev_gray, lip_curr_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flow_maps.append(flow)
        lip_prev_gray = lip_curr_gray.copy()

    cap.release()

    if not flow_maps:
        print(f"No optical flow computed for video: {video_path}")
        return

    flow_seq = np.array(flow_maps)  # Shape: (T, H, W, 2)
    np.save(output_path, flow_seq)
    print(f"Saved optical flow for {video_path} to {output_path} with shape {flow_seq.shape}")


def process_real_dataset(base_video_dir, base_flow_dir, target_size=(64, 64)):
    """
    Processes all real videos in the dataset and extracts optical flow for each.
    """
    for user in os.listdir(base_video_dir):
        user_video_dir = os.path.join(base_video_dir, user)
        if not os.path.isdir(user_video_dir):
            continue
        user_flow_output_dir = os.path.join(base_flow_dir, user)
        os.makedirs(user_flow_output_dir, exist_ok=True)
        video_files = [f for f in os.listdir(user_video_dir) if f.lower().endswith(('.mp4', '.avi', '.mpg'))]
        for video_file in tqdm(video_files, desc=f"Processing user {user}"):
            video_path = os.path.join(user_video_dir, video_file)
            compute_optical_flow(video_path, user_flow_output_dir, target_size)


if __name__ == "__main__":
    # Example usage for real video dataset
    base_real_video_dir = "D:/visual_speaker_auth/data/gridcorpus/real_videos"
    base_real_flow_dir = "D:/visual_speaker_auth/data/gridcorpus/optical_flow/real"
    process_real_dataset(base_real_video_dir, base_real_flow_dir)
    print("Optical flow extraction for real dataset completed!")

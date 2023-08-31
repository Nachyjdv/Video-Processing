import cv2
import numpy as np
import os

def calculate_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_blurry(image, threshold=100):
    blur_val = calculate_blur(image)
    return blur_val < threshold


cap = cv2.VideoCapture(r"C:\Users\nachi\OneDrive\Desktop\machine learning\Studio_Project_V1.mp4")

n = 1000  
blurry_video_frames = []
clear_video_frames = []

while len(blurry_video_frames) + len(clear_video_frames) < n:
    ret, frame = cap.read()

    if not ret:
        break

    if is_blurry(frame):
        blurry_video_frames.append(frame)
    else:
        clear_video_frames.append(frame)

cap.release()


output_directory_blur = 'blur_video_frames'
output_directory_clear = 'clear_video_frames'
if not os.path.exists(output_directory_blur):
    os.makedirs(output_directory_blur)
if not os.path.exists(output_directory_clear):
    os.makedirs(output_directory_clear)

for idx, blurry_frame in enumerate(blurry_video_frames):
    output_path = os.path.join(output_directory_blur, f'blurry_frame_{idx}.jpg')
    cv2.imwrite(output_path, blurry_frame)

for idx, clear_frame in enumerate(clear_video_frames):
    output_path = os.path.join(output_directory_clear, f'clear_frame_{idx}.jpg')
    cv2.imwrite(output_path, clear_frame)

print(f"Extracted {len(blurry_video_frames)} blurry frames and {len(clear_video_frames)} clear frames. Saved them in the '{output_directory_blur}' and '{output_directory_clear}' folders.")

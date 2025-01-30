import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Define the correct path to the cyberpunk mask
mask_path = "/home/rootking/AR_Face_Filter/cyberpunk_mask.png"

# Load the cyberpunk mask image with alpha channel
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

if mask is None:
    print(f"❌ Error: Could not load image at {mask_path}. Check if the file exists and is readable.")
    exit()

# Ensure mask has an alpha channel (BGRA format)
if mask.shape[2] == 3:
    print("⚠️ Warning: Image does not have an alpha channel. Adding one.")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Open webcam
cap = cv2.VideoCapture(0)

# Define key facial landmark indices for mask positioning
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
CHIN = 152
FOREHEAD = 10
UPPER_LIP = 13
LOWER_LIP = 14

# Default glow color (Blue)
glow_color = (255, 0, 0)  # Default Blue

# Track time for pulsing effect
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Unable to capture video frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Key points for mask placement
            left_cheek = face_landmarks.landmark[LEFT_CHEEK]
            right_cheek = face_landmarks.landmark[RIGHT_CHEEK]
            chin = face_landmarks.landmark[CHIN]
            forehead = face_landmarks.landmark[FOREHEAD]
            upper_lip = face_landmarks.landmark[UPPER_LIP]
            lower_lip = face_landmarks.landmark[LOWER_LIP]

            # Convert to pixel coordinates
            x1, y1 = int(left_cheek.x * w), int(left_cheek.y * h)
            x2, y2 = int(right_cheek.x * w), int(right_cheek.y * h)
            y_chin = int(chin.y * h)
            y_forehead = int(forehead.y * h)
            y_upper_lip = int(upper_lip.y * h)
            y_lower_lip = int(lower_lip.y * h)

            # Calculate mouth opening
            mouth_opening = y_lower_lip - y_upper_lip
            scaling_factor = 1.0 + (mouth_opening / h)  # Dynamic resizing factor

            # Calculate mask dimensions
            face_width = x2 - x1
            face_height = y_chin - y_forehead
            mask_width = int(face_width * 1.45)
            mask_height = int(face_height * 1.5 * scaling_factor)

            # Ensure valid dimensions
            if mask_width > 0 and mask_height > 0:
                mask_resized = cv2.resize(mask, (mask_width, mask_height))
            else:
                continue

            # Adjust mask position
            y_start = int(y_forehead - (face_height * 0.3))
            y_end = y_start + mask_resized.shape[0]
            x_start = x1 - int(mask_width * 0.1)
            x_end = x_start + mask_resized.shape[1]

            # Ensure mask fits within frame
            if 0 <= x_start < w and 0 <= x_end < w and 0 <= y_start < h and y_end < h:
                mask_bgr = mask_resized[:, :, :3]
                mask_alpha = mask_resized[:, :, 3] / 255.0

                roi = frame[y_start:y_end, x_start:x_end]

                # **🔥 Pulsing Glow Effect 🔥**
                elapsed_time = time.time() - start_time
                glow_intensity = (math.sin(elapsed_time * 2) + 1) / 2  # Pulsing between 0-1

                # Create glow effect
                edges = cv2.Canny(mask_bgr, 50, 150)  # Detect edges
                edges = cv2.GaussianBlur(edges, (5, 5), 10)  # Apply blur for glow effect

                # Convert edges to colored glow
                glow_layer = np.zeros_like(mask_bgr)
                for c in range(3):  # Apply glow color with intensity
                    glow_layer[:, :, c] = edges * (glow_color[c] / 255.0) * glow_intensity * 2

                # Blend the mask with the frame using NumPy operations
                for c in range(3):
                    roi[:, :, c] = (mask_alpha * mask_bgr[:, :, c] +
                                    (1 - mask_alpha) * roi[:, :, c] +
                                    glow_layer[:, :, c])  # Add animated glow

                frame[y_start:y_end, x_start:x_end] = roi

    # Show the frame
    cv2.imshow("Cyberpunk AR Face Filter with Pulsing Glow", frame)

    # **🔹 Allow color change with key presses**
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Red Glow
        glow_color = (0, 0, 255)
    elif key == ord('g'):  # Green Glow
        glow_color = (0, 255, 0)
    elif key == ord('b'):  # Blue Glow (Default)
        glow_color = (255, 0, 0)

cap.release()
cv2.destroyAllWindows()

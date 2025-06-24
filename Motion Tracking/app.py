import cv2
import numpy as np
import streamlit as st
import tempfile
import os

def extract_motion_optical_flow(video_path, frame_skip=3, scale_percent=50):
    cap = cv2.VideoCapture(video_path)

    ret, first_frame = cap.read()
    if not ret:
        return None, "Cannot read video file"

    width = int(first_frame.shape[1] * scale_percent / 100)
    height = int(first_frame.shape[0] * scale_percent / 100)
    first_frame = cv2.resize(first_frame, (width, height))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # ORB for initial keypoints
    orb = cv2.ORB_create(nfeatures=200)
    kp = orb.detect(prev_gray, None)
    prev_pts = np.array([x.pt for x in kp], dtype=np.float32).reshape(-1, 1, 2)

    output_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (LK method)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

        # Select good points
        good_new = next_pts[status == 1]
        good_old = prev_pts[status == 1]

        # Draw tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
            cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)

        output_frames.append(frame)
        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

    cap.release()
    return output_frames, None

def save_video(frames, output_path, fps=15):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def main():
    st.title("🚁 Drone Video Motion Tracking (ORB + Optical Flow)")
    st.markdown("This app uses ORB keypoints + Lucas-Kanade Optical Flow for smooth motion tracking.")

    uploaded_file = st.file_uploader("Upload Drone Video", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        st.info("⏳ Processing video with Optical Flow...")
        frames, error = extract_motion_optical_flow(video_path)

        if error:
            st.error(error)
        else:
            output_path = os.path.join(tempfile.gettempdir(), "tracked_output.mp4")
            save_video(frames, output_path)

            st.success("✅ Tracking complete!")
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button("📥 Download Tracked Video", f, file_name="tracked_motion.mp4")

if __name__ == "__main__":
    main()

import os
import sys
import time
import cv2
from datetime import datetime

# Import AI pipeline and detection logic from app.py
from app import (
    BASE_DIR,
    detect_waste,
    detect_persons,
    filter_false_waste_detections,
    process_dumping,
    draw_persons,
    draw_wastes,
    draw_status,
    draw_dustbin_zone,
    get_effective_dustbin_zone,
    is_waste_inside_dustbin,
    create_evidence_image,
    save_detection,
    RESULT_FOLDER
)


def find_input_video(custom_path=None):
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    video_dir = os.path.join(BASE_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)

    supported = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.lower().endswith(supported)]

    if files:
        # Return most recently modified video
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files[0]

    return None


def run_video_detection(video_path=None, show_window=True, save_output=True):
    video_file = find_input_video(video_path)
    if not video_file or not os.path.isfile(video_file):
        print("\n" + "=" * 65)
        print("❌ KOI VIDEO FILE NAHI MILI!")
        print("=" * 65)
        print(f"Kripya apni video file ko is folder me daalein:")
        print(f"👉 {os.path.join(BASE_DIR, 'videos')}")
        print("Udaaharan ke liye: videos/test.mp4")
        print("Supported formats: .mp4, .avi, .mov, .mkv\n")
        return

    print("\n" + "=" * 65)
    print(f"🎬 Video File mili: {os.path.basename(video_file)}")
    print(f"📁 Path: {video_file}")
    print("=" * 65)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"❌ Video file open nahi ho saki: {video_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_sec = total_frames / fps if total_frames > 0 else 0

    print(f"📊 Resolution: {width}x{height} | FPS: {fps:.1f} | Total Frames: {total_frames} ({duration_sec:.1f} sec)")

    writer = None
    output_path = None
    if save_output:
        os.makedirs(RESULT_FOLDER, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_filename = f"tested_{base_name}_{timestamp}.mp4"
        output_path = os.path.join(RESULT_FOLDER, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Output Video yahan save hogi: {output_path}")

    frame_idx = 0
    legal_events = 0
    illegal_events = 0

    print("\n🚀 AI Detection start ho raha hai... (Press 'Q' to stop window)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1

            # Run detection every 2 frames for speed, or every frame
            wastes = detect_waste(frame)
            persons = detect_persons(frame)
            wastes = filter_false_waste_detections(wastes, persons, frame)

            # Evaluate dumping logic
            dumping_confirmed = process_dumping(persons, wastes, frame)

            # Draw visual feedback
            annotated = frame.copy()
            draw_dustbin_zone(annotated)
            draw_persons(annotated, persons)
            draw_wastes(annotated, wastes)
            draw_status(annotated, persons, wastes)

            # Check legal vs illegal events from app globals
            from app import current_illegal_dumping, current_legal_disposal

            if dumping_confirmed:
                illegal_events += 1
                sec = frame_idx / fps
                print(f"🚨 [{sec:05.1f}s | Frame {frame_idx}] ILLEGAL DUMPING DETECTED! (Waste dropped outside dustbin)")

            if current_legal_disposal and frame_idx % int(fps) == 0:
                sec = frame_idx / fps
                print(f"🟢 [{sec:05.1f}s | Frame {frame_idx}] LEGAL DISPOSAL: Person putting waste in Dustbin.")

            if writer is not None:
                writer.write(annotated)

            if show_window:
                cv2.imshow("Waste Detection System - Video Test", annotated)
                delay = max(1, int(1000 / fps))
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    print("⏹️ User ne playback stop kiya.")
                    break

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    print("\n" + "=" * 65)
    print("📋 VIDEO ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"Total Frames Processed : {frame_idx}/{total_frames}")
    print(f"🚨 Illegal Dumping Incidents : {illegal_events}")
    if output_path and os.path.exists(output_path):
        print(f"🎥 Processed Video Saved At : {output_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    custom = sys.argv[1] if len(sys.argv) > 1 else None
    run_video_detection(custom)


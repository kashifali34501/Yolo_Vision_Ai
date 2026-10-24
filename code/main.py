import cv2
import os
from ultralytics import YOLO

# --- Configuration ---
# To use a local webcam, set this to 0 (integer).
# To use an RTSP stream or file, set this to the URL/Path (string).
# Example: os.getenv("RTSP_URL", 0)
RTSP_URL = os.getenv("RTSP_URL", 0)
PRODUCT_CLASSES = ['bottle', 'cup', 'apple', 'orange', 'banana', 'mobile'] # COCO classes to count
MODEL_PATH = "notebooks/yolo26n.pt"

def main():
    # 1. Initialize YOLO model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize Camera Stream
    print(f"Connecting to camera source: {RTSP_URL}...")
    # Handle environment variable which always returns a string
    source = RTSP_URL
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video stream. Please check your RTSP URL.")
        return

    print("Stream connected. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Attempting to reconnect...")
            cap.open(RTSP_URL)
            continue

        # 3. Run Inference
        # stream=True is more memory efficient for videos
        results = model.predict(source=frame, stream=True, conf=0.25)

        # Current frame counting logic
        current_counts = {cls: 0 for cls in PRODUCT_CLASSES}

        # Iterate through results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                if cls_name in PRODUCT_CLASSES:
                    current_counts[cls_name] += 1
                    # Draw bounding box for detected products
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, cls_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Visualization: Overlay total counts
        y_offset = 30
        cv2.putText(frame, "Product Counts:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        total_products = 0
        for cls, count in current_counts.items():
            if count > 0:
                cv2.putText(frame, f"{cls}: {count}", (10, y_offset + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 30
                total_products += count

        cv2.putText(frame, f"Total: {total_products}", (10, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Real-Time Product Counting", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")

if __name__ == "__main__":
    main()

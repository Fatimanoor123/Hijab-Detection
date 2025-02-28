import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your trained model path

# Open a video file (replace 'video.mp4' with your actual video file path)
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get original video dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define max dimensions for display
max_width = 800   # Max width for display
max_height = 600  # Max height for display

# Compute scale factor while maintaining aspect ratio
scale_w = max_width / original_width
scale_h = max_height / original_height
scale = min(scale_w, scale_h)  # Pick the smaller scale to fit both dimensions

# Compute new dimensions
new_width = int(original_width * scale)
new_height = int(original_height * scale)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    # Resize frame while keeping aspect ratio within screen limits
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # Confidence score
            label = result.names[int(box.cls[0])]  # Class name (hijab or no-hijab)

            # Define color for bounding box
            color = (0, 255, 0) if label == "hijab" else (0, 0, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show the frame
    cv2.imshow("Hijab Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

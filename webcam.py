from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    # Show webcam
    cv2.imshow("AI Road Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
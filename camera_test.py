import cv2

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image.")
        break

    # Show the webcam feed
    cv2.imshow("Waste Detection System - Camera Test", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
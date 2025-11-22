"Author: Zhaozhe Zhang"

import cv2

# Initialize USB camera (0 represents the default USB camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open USB camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read from USB camera")
        continue

    # Display the captured frame
    cv2.imshow("USB Camera Output", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

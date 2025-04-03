import cv2
import numpy as np

# Define the color range for red
color_name = "Red"
lower_red1 = np.array([0, 120, 70])   # Lower red range
upper_red1 = np.array([10, 255, 255]) 
lower_red2 = np.array([170, 120, 70])  # Upper red range
upper_red2 = np.array([180, 255, 255])  

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for the red color range
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2  # Combine both masks
    
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw bounding box on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Display tracking color
        cv2.putText(frame, f"Tracking: {color_name}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display the result
    cv2.imshow("Red Object Tracking", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

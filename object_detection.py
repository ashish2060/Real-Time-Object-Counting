# Object Detection Project ----------------------------------
import cv2
import numpy as np

# A dictionary to store the tracked objects.
# Keys are object IDs, values are a list of their recent positions.
tracked_objects = {}
# A list to store the IDs of objects that have been counted.
counted_ids = []
# Global counter for objects.
object_counter = 0

# A simple class to track an object's position.
class ObjectTracker:
    def __init__(self, obj_id, centroid):
        self.id = obj_id
        self.centroids = [centroid]
        self.counted = False
    
    def add_centroid(self, centroid):
        self.centroids.append(centroid)
        # Keep a limited history of centroids for tracking.
        if len(self.centroids) > 5:
            self.centroids.pop(0)

# A function to find the center of a contour.
def get_centroid(x, y, w, h):
    """Calculates the centroid of a rectangle."""
    center_x = x + w // 2
    center_y = y + h // 2
    return (center_x, center_y)

# --- Main execution loop ---

video_path = 'car_counting.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# This line should be placed where you want to count the objects crossing it.
line_position_y = 350
offset = 20  # A small margin for checking if an object has crossed the line.

# Object ID counter.
next_object_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing.
    frame = cv2.resize(frame, (640, 480))
    # Apply the background subtractor to get the foreground mask.
    fgMask = backSub.apply(frame)

    # Apply morphological operations to remove noise.
    kernel = np.ones((5, 5), np.uint8)
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=2)
    
    # Finding contours of the objects.
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the counting line on the frame.
    cv2.line(frame, (0, line_position_y), (frame.shape[1], line_position_y), (0, 255, 0), 2)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        min_contour_area = 500
        if cv2.contourArea(contour) < min_contour_area:
            continue

        centroid = get_centroid(x, y, w, h)
        
        # Check if the object is already being tracked.
        is_tracked = False
        for obj_id, tracker in tracked_objects.items():
            last_centroid = tracker.centroids[-1]
            if np.linalg.norm(np.array(centroid) - np.array(last_centroid)) < 50:
                tracker.add_centroid(centroid)
                is_tracked = True
                # Check if the object has crossed the line and hasn't been counted yet.
                if (centroid[1] > line_position_y - offset and 
                    centroid[1] < line_position_y + offset and 
                    not tracker.counted):
                    object_counter += 1
                    tracker.counted = True # Mark object as counted.
                    print(f"Object {obj_id} has crossed the line. Total count: {object_counter}")
                break

        # If the contour is a new object, start tracking it.
        if not is_tracked:
            new_object = ObjectTracker(next_object_id, centroid)
            tracked_objects[next_object_id] = new_object
            next_object_id += 1

        # Draw the bounding box and centroid.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, centroid, 4, (0, 0, 255), -1)

    # Remove objects that have left the frame.
    ids_to_remove = []
    for obj_id, tracker in tracked_objects.items():
        if len(tracker.centroids) > 5 and np.linalg.norm(np.array(tracker.centroids[-1]) - np.array(tracker.centroids[0])) < 10:
            ids_to_remove.append(obj_id)
        
    for obj_id in ids_to_remove:
        del tracked_objects[obj_id]

    # Display the object count on the frame.
    cv2.putText(frame, f"Cars: {object_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Object Counting', frame)
    # Also show the foreground mask to see what the model is detecting.
    cv2.imshow('Foreground Mask', fgMask)

    # Break the loop when 'q' is pressed.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all windows.
cap.release()
cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np
#def main ():
#    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#    
#    # https://github.com/Itseez/opencv/blob/master
#    # /data/haarcascades/haarcascade_eye.xml
#    # Trained XML file for detecting eyes
#    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
#    
#    # capture frames from a camera
#    cap = cv2.VideoCapture(0)
#    
#    # loop runs if capturing has been initialized.
#    while 1: 
#        # reads frames from a camera
#        ret, img = cap.read() 
#    
#        # convert to gray scale of each frames
#        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    
#        # Detects faces of different sizes in the input image
#        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#    
#        for (x,y,w,h) in faces:
#            # To draw a rectangle in a face 
#            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
#            roi_gray = gray[y:y+h, x:x+w]
#            roi_color = img[y:y+h, x:x+w]
#    
#            # Detects eyes of different sizes in the input image
#            eyes = eye_cascade.detectMultiScale(roi_gray) 
#    
#            #To draw a rectangle in eyes
#            for (ex,ey,ew,eh) in eyes:
#                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
#    
#        # Display an image in a window
#        cv2.imshow('img',img)
#    
#        # Wait for Esc key to stop
#        k = cv2.waitKey(30) & 0xff
#        if k == 27:
#            break
#    
#    # Close the window
#    cap.release()
#    
#    # De-allocate any associated memory usage
#    cv2.destroyAllWindows()
#
#    print("done")

# def main():
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)

#         for face in faces:
#             landmarks = predictor(gray, face)

#             # Get eye regions
#             left_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
#             right_eye = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])

#             # Draw eyes
#             cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
#             cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
#             # Determine gaze direction
#             left_direction, left_pos = get_gaze_direction(left_eye, frame)
#             right_direction, right_pos = get_gaze_direction(right_eye, frame)
            
#             # Determine final gaze direction based on both eyes
#             gaze_direction = left_direction if left_direction == right_direction else "center"
            
#             # Draw arrows indicating gaze direction
#             if gaze_direction == "left":
#                 cv2.arrowedLine(frame, left_pos, (left_pos[0] - 50, left_pos[1]), (0, 0, 255), 3)
#                 cv2.arrowedLine(frame, right_pos, (right_pos[0] - 50, right_pos[1]), (0, 0, 255), 3)
#             elif gaze_direction == "right":
#                 cv2.arrowedLine(frame, left_pos, (left_pos[0] + 50, left_pos[1]), (0, 0, 255), 3)
#                 cv2.arrowedLine(frame, right_pos, (right_pos[0] + 50, right_pos[1]), (0, 0, 255), 3)
#             elif gaze_direction == "up":
#                 cv2.arrowedLine(frame, left_pos, (left_pos[0], left_pos[1] - 50), (0, 0, 255), 3)
#                 cv2.arrowedLine(frame, right_pos, (right_pos[0], right_pos[1] - 50), (0, 0, 255), 3)
#             elif gaze_direction == "down":
#                 cv2.arrowedLine(frame, left_pos, (left_pos[0], left_pos[1] + 50), (0, 0, 255), 3)
#                 cv2.arrowedLine(frame, right_pos, (right_pos[0], right_pos[1] + 50), (0, 0, 255), 3)
            

#         cv2.imshow("Eye Tracking", frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()



# # Function to determine gaze direction
# def get_gaze_direction(eye_points, frame):
#     eye_region = np.array(eye_points, dtype=np.int32)
#     min_x = np.min(eye_region[:, 0])
#     max_x = np.max(eye_region[:, 0])
#     min_y = np.min(eye_region[:, 1])
#     max_y = np.max(eye_region[:, 1])
    
#     eye_frame = frame[min_y:max_y, min_x:max_x]
#     gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
#     _, threshold_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
    
#     # Find the center of mass of the eye (approximates pupil position)
#     moments = cv2.moments(threshold_eye)
#     if moments["m00"] != 0:
#         cx = int(moments["m10"] / moments["m00"]) + min_x
#         cy = int(moments["m01"] / moments["m00"]) + min_y
#     else:
#         return "center", (min_x + (max_x - min_x) // 2, min_y + (max_y - min_y) // 2)
    
#     # Determine direction based on the pupil's position in the eye
#     eye_width = max_x - min_x
#     eye_height = max_y - min_y
    
#     if cx < min_x + eye_width * 0.3:
#         return "left", (cx, cy)
#     elif cx > min_x + eye_width * 0.7:
#         return "right", (cx, cy)
#     elif cy < min_y + eye_height * 0.3:
#         return "up", (cx, cy)
#     elif cy > min_y + eye_height * 0.7:
#         return "down", (cx, cy)
#     else:
#         return "center", (cx, cy)
        
        
def get_gaze_vector(eye_points, frame):
    eye_region = np.array(eye_points, dtype=np.int32)
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    
    eye_frame = frame[min_y:max_y, min_x:max_x]
    gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
    
    moments = cv2.moments(threshold_eye)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"]) + min_x
        cy = int(moments["m01"] / moments["m00"]) + min_y
    else:
        return (min_x + (max_x - min_x) // 2, min_y + (max_y - min_y) // 2)
    
    return (cx, cy)

def main():
    # Load the face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
            
            left_gaze = get_gaze_vector(left_eye, frame)
            right_gaze = get_gaze_vector(right_eye, frame)
            
            # Estimate gaze direction separately for each eye
            for eye_gaze, eye in [(left_gaze, left_eye), (right_gaze, right_eye)]:
                eye_center = np.mean(eye, axis=0).astype(int)
                dx = (eye_gaze[0] - eye_center[0]) * 4
                dy = (eye_gaze[1] - eye_center[1]) * 4
                end_point = (eye_center[0] + dx, eye_center[1] + dy)
                cv2.arrowedLine(frame, tuple(eye_center), end_point, (0, 255, 0), 3)
            
        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

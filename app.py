import cv2
import imutils
import time
from facial_detections import detectFace
from blink_detection import isBlinking
from mouth_tracking import mouthTrack
from object_detection import detectObject
from eye_tracker import gazeDetection
from head_pose_estimation import head_pose_detection
import winsound
from datetime import datetime

global data_record
data_record = []

# For Beeping
frequency = 2500
duration = 1000

# OpenCV videocapture for the webcam
cam = cv2.VideoCapture(0)

# If camera is already opened
if not cam.isOpened():
    cam.open()


# Face Count If-else conditions
def faceCount_detection(faceCount):
    if faceCount > 1:
        time.sleep(1)
        remark = "Multiple faces has been detected."
        winsound.Beep(frequency, duration)
    elif faceCount == 0:
        remark = "No face has been detected."
        time.sleep(1)
        winsound.Beep(frequency, duration)
    else:
        remark = "Face detecting properly."
    return remark


# Main function
def proctoringAlgo():
    blinkCount = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        overlay_frame = frame.copy()  # Used only for detection (to avoid drawing on frame)

        record = []
        y_offset = 20  # vertical text placement start

        # Time
        current_time = datetime.now().strftime("%H:%M:%S.%f")
        record.append(current_time)
        cv2.putText(frame, f"Time: {current_time}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

        # Face detection
        faceCount, faces = detectFace(overlay_frame)
        face_status = faceCount_detection(faceCount)
        record.append(face_status)
        cv2.putText(frame, face_status, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25

        if faceCount == 1:
            # Blink detection
            blinkStatus = isBlinking(faces, overlay_frame)
            blink_text = blinkStatus[2]
            if blink_text == "Blink":
                blinkCount += 1
                blink_text += f" (Count: {blinkCount})"
            record.append(blink_text)
            cv2.putText(frame, f"Blink: {blink_text}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 25

            # Gaze detection
            eyeStatus = gazeDetection(faces, overlay_frame)
            record.append(eyeStatus)
            cv2.putText(frame, f"Gaze: {eyeStatus}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

            # Mouth tracking
            mouth_status = mouthTrack(faces, overlay_frame)
            record.append(mouth_status)
            cv2.putText(frame, f"Mouth: {mouth_status}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

            # Object detection
            objectName = detectObject(overlay_frame)
            record.append(str(objectName))
            cv2.putText(frame, f"Objects: {str(objectName)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 100, 255), 1)
            y_offset += 20

            if len(objectName) > 1:
                time.sleep(2)
                winsound.Beep(frequency, duration)
                continue

            # Head pose estimation
            head_status = head_pose_detection(faces, overlay_frame)
            record.append(str(head_status))
            cv2.putText(frame, f"Head Pose: {str(head_status)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        data_record.append(record)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    proctoringAlgo()

    # Convert the list to a string with each element on a new line
    activityVal = "\n".join(map(str, data_record))

    with open('activity.txt', 'w') as file:
        file.write(str(activityVal))

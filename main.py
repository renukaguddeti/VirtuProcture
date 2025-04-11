import cv2
# import imutils
import time
import winsound
from facial_detections import detectFace
from blink_detection import isBlinking
from mouth_tracking import mouthTrack
from object_detection import detectObject
from eye_tracker import gazeDetection
from head_pose_estimation import head_pose_detection
from datetime import datetime

global data_record
data_record = []
running = True

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
        time.sleep(5)
        remark = "Multiple faces have been detected."
        winsound.Beep(frequency, duration)
    elif faceCount == 0:
        remark = "No face has been detected."
        time.sleep(5)
        winsound.Beep(frequency, duration)
    else:
        remark = "Face detecting properly."
    return remark


# Main function
def proctoringAlgo():
    blinkCount = 0

    while running:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        record = []

        # Reading the current time
        current_time = datetime.now().strftime("%H:%M:%S.%f")
        print("Current time is:", current_time)
        record.append(current_time)

        # Returns the face count and will detect the face.
        faceCount, faces = detectFace(frame)
        face_status = faceCount_detection(faceCount)
        print(face_status)
        record.append(face_status)

        if faceCount == 1:
            # Blink Detection
            blinkStatus = isBlinking(faces, frame)
            print(blinkStatus[2])

            if blinkStatus[2] == "Blink":
                blinkCount += 1
                record.append(blinkStatus[2] + " count: " + str(blinkCount))
            else:
                record.append(blinkStatus[2])

            # Gaze Detection
            eyeStatus = gazeDetection(faces, frame)
            print(eyeStatus)
            record.append(eyeStatus)

            # Mouth Position Detection
            mouth_status = mouthTrack(faces, frame)
            print(mouth_status)
            record.append(mouth_status)

            # Object detection using YOLO
            objectName = detectObject(frame)
            print(objectName)
            record.append(objectName)

            if len(objectName) > 1:
                time.sleep(4)
                winsound.Beep(frequency, duration)
                data_record.append(record)
                continue

            # Head Pose estimation
            head_pose_status = head_pose_detection(faces, frame)
            print(head_pose_status)
            record.append(head_pose_status)

        data_record.append(record)

        # Convert the frame to JPEG format (optional if using web interface)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()
    cv2.destroyAllWindows()


def main_app():
    activityVal = "\n".join(map(str, data_record))
    with open('activity.txt', 'w') as file:
        file.write(str(activityVal))


# Entry point
if __name__ == "__main__":
    try:
        for _ in proctoringAlgo():
            pass  # Let the generator keep running
    except KeyboardInterrupt:
        running = False
        main_app()
        print("\nProctoring stopped. Data saved to activity.txt")

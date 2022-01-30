import cv2 as cv
import functions as fun
import time


# Variables
COUNTER = 0
BLINKS = 0
CLOSED_EYES_FRAME = 3
cameraID = 0
# variables for frame rate.
FRAME_COUNTER = 0


# creating camera object
video = cv.VideoCapture(0)

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
	print("Something Is Wrong!!!")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.

result = cv.VideoWriter('Eye Blink Detection.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, size)

while True:
    FRAME_COUNTER += 1
    # getting frame from camera
    ret, frame = video.read()
    if ret == False:
        break

    # converting frame into Gray image.
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = grayFrame.shape
    circleCenter = (int(width/2), 50)
    # calling the face detector funciton
    image, face = fun.faceDetector(frame, grayFrame)
    if face is not None:
        # calling landmarks detector funciton.
        image, landmarks_points = fun.faceLandmakDetector(frame, grayFrame, face, False)
        #print(landmarks_points)

        LeftEye = landmarks_points[36:42]
        leftRatio, topMid, bottomMid = fun.blinkDetector(LeftEye)

        RightEye = landmarks_points[42:48]
        rightRatio, rTop, rBottom = fun.blinkDetector(RightEye)
        #cv.circle(image, topMid, 2, fun.YELLOW, -1)
        #cv.circle(image, bottomMid, 2, fun.YELLOW, -1)

        blinkRatio = (leftRatio + rightRatio)/2

        if blinkRatio > 4:
            COUNTER += 1
            cv.putText(image, f'Blinked', (100, 50), fun.fonts, 0.8, fun.LIGHT_BLUE, 2)
            # print("blink")
        else:
            if COUNTER > CLOSED_EYES_FRAME:
                BLINKS += 1
                COUNTER = 0
        cv.putText(image, f'Total Blinks: {BLINKS}', (230, 17), fun.fonts, 0.5, fun.LIGHT_BLUE, 2)

        #for p in RightEye:
            #cv.circle(image, p, 3, fun.MAGENTA, 1)
   
    if ret == True:
		    result.write(frame)
		# Display the frame
		# saved in the file
		    cv.imshow('Frame', frame)

    key = cv.waitKey(1)

    if key == ord(' '):
        break
# closing the camera
video.release()
# Recoder.release()
# closing  all the windows
cv.destroyAllWindows()

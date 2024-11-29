#!/usr/bin/env python3

#import cv2 as cv
import numpy as np
import ffmpeg
import cv2


# Need to specify the camera config....

# color camera setup
# rtsp://192.168.1.10/color
# pix_fmt = 'bgr24'
# w*h*3
# np.uint8

# depth camera setup
# rtsp://192.168.1.10/depth
# pix_fmt = 'gray16le'
# w*h*2
# np.uint16


url = "rtsp://192.168.1.10/depth"

process = (
        ffmpeg
        .input(url, rtsp_transport='tcp', t=1)
        .output('pipe:', format='rawvideo', pix_fmt='gray16le', vframes=1)
        .run_async(pipe_stdout=True)
        )

w = 1280
h = 720


in_bytes = process.stdout.read(w*h*2)

frame = np.frombuffer(in_bytes, np.uint16).reshape((h, w))
cv2.imwrite("./depth_single_image.jpg", frame)

normalized = cv2.normalize(frame, None, 0, 255, cv2.NMAX_MINMAX).astype(np.uint8)
cv2.imwrite("./normal_depth.jpg", normalized)

process.stdout.close()
process.wait()

#while True:
#    in_bytes = process.stdout.read(w*h*3)
#    if not in_bytes:
#        break

#    frame = (np.frombuffer(in_bytes, np.uint8)
#                .reshape((h, w, 3))
#            )
#    print(frame.shape)
#    cv2.imwrite("images/color_single_test%d.jpg" % count, frame)
#    break
    #cv2.imshow("RTS stream color", frame)
    
#    count += 1

#    if count > 4:
#        break

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#process.stdout.close()
#process.wait()
#cv2.destroyAllWindows()

#vcap = cv.VideoCapture("rtsp://192.168.1.10/color")

#count = 0
#while True:
    #ret, frame = vcap.read()
    #print(np.array(frame).shape)
    #cv.imshow("test_mode", frame)
    
    #cv.imwrite("images/depth_frame%d.jpg" % count, frame)
    
    #if count > 3:
    #    break

    #count += 1




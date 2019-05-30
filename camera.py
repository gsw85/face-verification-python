import cv2

class VideoCamera(object):
    def __init__(self, video_source):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(video_source)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        _, frame = self.video.read()

        # horizontal flip
        frame = cv2.flip(frame, 0)

        return frame

    def to_byte(self, frame):
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


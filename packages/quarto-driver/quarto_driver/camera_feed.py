import cv2


class CameraFeed:
    """CameraFeed class for USB cameras using OpenCV."""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start_feed(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Failed to open camera at index {self.camera_index}")
            self.cap = None
            return False
        print(f"Camera feed started on index {self.camera_index}")
        return True

    def stop_feed(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            print("Camera feed stopped.")

    def get_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                print("Failed to read frame from camera.")
        else:
            print("Camera feed is not started.")
        return None

    def live_detection_display(self):
        if not self.cap:
            print("Camera feed is not started.")
            return
        print("Press 'q' to quit live display.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break
            cv2.imshow(f"Live Camera Feed (index {self.camera_index})", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

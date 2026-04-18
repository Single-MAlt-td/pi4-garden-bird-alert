import numpy as np
import cv2

from bird_guard.camera.camera import Frame

class MotionDetector:
    def __init__(self, threshold: int = 400):
        self.prev_frame = None
        self.prev_gray = None
        self.threshold = threshold
        self.debug = True

    def __del__(self):
        self._hide_debug_window()

    def _show_debug_window(self):
        if self.debug:
            cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Debug", (1920, 1080))

    def _hide_debug_window(self):
        if self.debug:
            cv2.destroyWindow("Debug")

    def detect(self, frame: Frame) -> bool:
        # convert to gray image (if needed) and set kernel size
        match frame.type:
            case Frame.FrameType.COLOR:
                gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
                kernel_size = 21
            case Frame.FrameType.GRAY:
                gray = frame.data
                kernel_size = 21
            case Frame.FrameType.LORES:
                gray = frame.data[:frame.dim_y, :]
                kernel_size = 5
            case _:
                raise NotImplementedError("Frame type not yet implemented.")

        # blur image to reduce noise
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # store previous image (if not already set)
        if self.prev_gray is None:
            self.prev_frame = frame
            self.prev_gray = gray
            self._show_debug_window()
            return False

        # compute difference between current and previous image
        diff = cv2.absdiff(self.prev_gray, gray)
        #diff = cv2.subtract(self.prev_gray, gray)

        # detect significant differences via threshold
        _, thresh_raw = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # dilate to connect contours across small gaps
        thresh = cv2.dilate(thresh_raw, None, iterations=2)

        # count number of changed pixels
        changed_pixels = cv2.countNonZero(thresh)

        # detect contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug:
            #contour_areas = [cv2.contourArea(contour) for contour in contours]
            big_contours = [contour for contour in contours if cv2.contourArea(contour) > 500]

            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 2)
            cv2.drawContours(debug_img, big_contours, -1, (0, 255, 0), 2)
            for contour in big_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

            combined = np.vstack([#np.hstack([self.prev_frame, frame]),
                                  np.hstack([cv2.cvtColor(self.prev_gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)]),
                                  np.hstack([cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), debug_img])])
            cv2.imshow("Debug", cv2.resize(combined, (1920, 1080)))
            key = cv2.waitKey(0)

            if key == 27 or key == ord('q'):
                raise Exception("QUIT by user")


        # update previous image
        self.prev_frame = frame
        self.prev_gray = gray

        return changed_pixels > self.threshold
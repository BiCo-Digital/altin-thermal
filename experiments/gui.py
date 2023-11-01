# import tkinter and opencv
import tkinter as tk
import cv2
from PIL import Image, ImageTk

from video_mog4 import SoupClassifier

WIDTH = 320
HEIGHT = 480


# create a window with fixed size, show the image in the window so it fills the window, overlay 2 buttons over the image and show the window
class App:

    # initialize the app
    def __init__(self, window, window_title, video_source='./thermal.mp4'):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source
        self.vid = MyVideoFromFile(self.video_source)
        self.soup_classifier = SoupClassifier()

        # create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT)
        self.canvas.pack()

        # create 2 buttons
        self.btn_snapshot = tk.Button(window, text="+", command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        self.btn_snapshot.place(x=10, y=HEIGHT - 10 - 100, width=100, height=100)

        self.btn_snapshot = tk.Button(window, text="-", command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        self.btn_snapshot.place(x=WIDTH - 10 - 100, y=HEIGHT - 10 - 100, width=100, height=100)

        # create linem that acts like window border, its red
        self.canvas.create_line(0, 0, WIDTH, 0, fill="red", width=5)

        # after it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 40
        self.update()

        self.window.mainloop()

    # update method that updates the window
    def update(self):
        # get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:

            ## Classify the frame
            self.soup_classifier.classify(frame)
            dbg = self.soup_classifier.get_debug_image(WIDTH, HEIGHT)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(dbg))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            # create linem that acts like window border, its red
            self.canvas.create_line(0, 5, WIDTH, 5, fill="red", width=5)
            self.canvas.create_line(5, 0, 5, HEIGHT, fill="red", width=5)
            self.canvas.create_line(WIDTH - 2.5, 0, WIDTH - 2.5, HEIGHT, fill="red", width=5)
            self.canvas.create_line(0, HEIGHT - 2.5, WIDTH, HEIGHT - 2.5, fill="red", width=5)

            # every 50 frame switch the color of the border to green
            if self.soup_classifier.is_scene_ok():
                self.canvas.create_line(0, 5, WIDTH, 5, fill="green", width=5)
                self.canvas.create_line(5, 0, 5, HEIGHT, fill="green", width=5)
                self.canvas.create_line(WIDTH - 2.5, 0, WIDTH - 2.5, HEIGHT, fill="green", width=5)
                self.canvas.create_line(0, HEIGHT - 2.5, WIDTH, HEIGHT - 2.5, fill="green", width=5)


        self.window.after(self.delay, self.update)

    # method for taking a snapshot
    def snapshot(self):
        # get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# class for the video capture that gets the frames from the video source / or from a video file
class MyVideoCapture:
    def __init__(self, video_source=0):
        # open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # get the video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # method that gets the frame from the video source
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def get_frame_resized(self, width, height):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # resize the frame to the specified width and height
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                # return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # method that releases the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# create class for MyVideoFromFile that inherits from MyVideoCapture and gets the frames from a video file 'thermal.mp4' and loops the video indefinitely
class MyVideoFromFile(MyVideoCapture):
    def __init__(self, video_source='./thermal2.mp4'):
        MyVideoCapture.__init__(self, video_source)
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.total_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # return a boolean success flag and the current frame converted to BGR
                if self.vid.get(cv2.CAP_PROP_POS_FRAMES) == self.total_frames:
                    self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def get_frame_resized(self, width, height):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # resize the frame to the specified width and height
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                # return a boolean success flag and the current frame converted to BGR
                if self.vid.get(cv2.CAP_PROP_POS_FRAMES) == self.total_frames:
                    self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

# create a window and pass it to the Application object
# App(tk.Tk(), "Tkinter and OpenCV")



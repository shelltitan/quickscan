# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 20:32:06 2020

@author: Istvan Szalai
"""


import tkinter
import tkinter.filedialog
import cv2
import PIL.Image, PIL.ImageTk

class Applicaation:
    def __init__(self, master):
        self.master = master
        master.title("QuickScan")

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(master, width = 720, height = 480)
        self.canvas.pack()
        
        self.open_file_button = tkinter.Button(master, text="Open File", command=self.openVideoFile)
        self.open_file_button.pack()

        self.label = tkinter.Label(master, text="This is our first GUI!")
        self.label.pack()

        self.greet_button = tkinter.Button(master, text="Start Video", command=self.startVideo)
        self.greet_button.pack()
        
        # add bindings for clicking, dragging and releasing over
        # any object with the "token" tag
        self.canvas.tag_bind("token", "<ButtonPress-1>", self.dragStart)
        self.canvas.tag_bind("token", "<ButtonRelease-1>", self.dragStop)
        self.canvas.tag_bind("token", "<B1-Motion>", self.drag)
        
        self._drag_data = {"x": 0, "y": 0, "item": None}
        self.createRectangle(100, 100)
        
        self.master.mainloop()

    def startVideo(self):
        """open video source""" 
        if hasattr(self, 'filename'):
            self.vid = VideoManager(self.filename)
            self.master.after(34, self.update)
    
    def openVideoFile(self):
        """open any file"""
        self.filename =  tkinter.filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        
    def createRectangle(self, x, y):
        """Create a token at the given coordinate in the given color"""
        self.rectangle = self.canvas.create_rectangle(
            x ,
            y,
            x + 319,
            y + 239,
            outline="#05f",
#            fill="#f50", maybe transparent fill(?)
            tags=("token",),
        )

    def dragStart(self, event):
        """Begining drag of an object"""
        # record the item and its location
        self._drag_data["item"] = self.canvas.find_closest(event.x, event.y)[0]
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def dragStop(self, event):
        """End drag of an object"""
        # reset the drag information
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    def drag(self, event):
        """Handle dragging of an object"""
        # compute how much the mouse has moved
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        # move the object the appropriate amount
        self.canvas.move(self._drag_data["item"], delta_x, delta_y)
        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
    
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.getFrame()
        
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.canvas.tag_raise(self.rectangle)
        self.master.after(34, self.update)

class VideoManager:
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        
    def getFrame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

root = tkinter.Tk()
app = Applicaation(root)
root.mainloop()
import tkinter as tk
from tkinter import filedialog, Canvas, Frame, PhotoImage
from PIL import Image, ImageTk
import tkinter.font as font
from Detector import *
import os

class VideoDetectorApp:
    def __init__(self):
        self.master = tk.Tk()
        self.master.title("Video Detector App")
        self.master.geometry("1200x800")
        self.video_path = ""
        self.config_path = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
        self.model_path = os.path.join("model_data", "frozen_inference_graph.pb")
        self.classes_path = os.path.join("model_data", "coco.names")

        self.master.configure(bg="#690F40")
        self.myFont = font.Font(family='Helvetica')

        self.setup_gui_elements()

    def setup_gui_elements(self):
         # Load and display the image on Canvas
        image_path = 'model_data/img.jpg'
        original_image = Image.open(image_path)
        new_width, new_height = 800, 800
        resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
        image = ImageTk.PhotoImage(resized_image)
        
        canvas = Canvas(self.master, width=new_width, height=new_height, highlightthickness=0, bg="#690F40")
        canvas.pack(side="left", fill="y")
        canvas.create_image(0, 0, anchor="nw", image=image)

        self.label = tk.Label(self.master, text="!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n!\n", bg="#690F40", fg="white", font=('Roboto', 25))
        self.label.place(x=800, y=0)

        self.label = tk.Label(self.master, text="Unveiling the Unseen: \nEmpowering Video Surveillance\n:) with Deep Learning \n:) Precision-Packed Object Detection\n....................\n", bg="#690F40", fg="white", font=('Roboto', 25))
        self.label.place(x=900, y=150)

        button_frame = Frame(self.master, bg="#690F40")
        button_frame.pack(pady=350)

        self.select_button = tk.Button(button_frame, text="Select Video", command=self.select_video, bg="#EAD196", fg="black")
        self.select_button['font'] = self.myFont
        self.select_button.pack(pady=5)

        self.detect_button = tk.Button(button_frame, text="Detect", command=self.detect_video, bg="#EAD196", fg="black")
        self.detect_button['font'] = self.myFont
        self.detect_button.pack(pady=5)

        gif_image_path = "model_data/ani.gif"
        open_image = Image.open(gif_image_path)
        frames = open_image.n_frames
        image_object = [PhotoImage(file=gif_image_path, format=f"gif -index {i}") for i in range(frames)]
        count = 0

        def animation(count):
            new_image = image_object[count]
            self.gif_label.configure(image=new_image)
            count += 1
            if count == frames:
                count = 0
            self.master.after(40, lambda: animation(count))

        self.gif_label = tk.Label(self.master, image="")
        self.gif_label.place(x=1000, y=500, width=370, height=256)
        animation(count)

        self.master.protocol("WM_DELETE_WINDOW", self.master.destroy)  # Close window gracefully
        self.master.mainloop()

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if self.video_path:
            self.label.config(text=f"Selected Video:{self.video_path}", pady=30, font=('Roboto', 15))
            self.label.place(x=900, y=150)

    def detect_video(self):
        if self.video_path:
            detector = Detector(self.video_path, self.config_path, self.model_path, self.classes_path)
            result = detector.onVideo()
            result_label = tk.Label(self.master, text="Successfully Detected.", font=('Roboto', 17), bg="#186F65", fg="white")
            result_label.place(x=1050, y=450)

# calling the function
if __name__ == '__main__':
    app = VideoDetectorApp()

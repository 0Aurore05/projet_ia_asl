# install
# pip install pillow
#sudo apt install python3-tk
# pip install customtkinter
# pip install opencv-python
# pip install ultralytics

# Camera import
from PIL import Image, ImageTk  
import cv2

# GUI import
import tkinter
import tkinter.messagebox
import customtkinter

# Model import
from ultralytics import YOLO
import numpy
import random
import time

customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"

image_target_size = 300

class App(customtkinter.CTk):
  def __init__(self):
    super().__init__()

    # configure window
    self.title("ASL Recognition")
    self.geometry(f"{1300}x{650}")

    # configure grid layout (4x4)
    self.grid_columnconfigure(1, weight=1)
    self.grid_columnconfigure((1, 3), weight=0)
    self.grid_rowconfigure((0, 1, 2), weight=0)

    ## LEFT SIDEBAR
    self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
    self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
    self.sidebar_frame.grid_rowconfigure(4, weight=1)
    
      # title
    self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="YOLOv8 ASL detection", font=customtkinter.CTkFont(size=20, weight="bold"))
    self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
      # textbox
    self.textbox = customtkinter.CTkTextbox(self.sidebar_frame, width=250)
    self.textbox.grid(row=1, column=0, padx=20, pady=0)
    self.textbox.insert("0.0", "This program uses YOLOv8 (Objet detection AI) to detect alphabet and numbers in the American Sign Language\n\n" +
      "The base program was trained on a dataset of x images per class, y epochs,  z hyperparameters")



    ## CAMERA SETTINGS
    self.camera_settings_panel = customtkinter.CTkFrame(self)
    self.camera_settings_panel.grid(row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew")
    
    self.label_camera_settings_panel = customtkinter.CTkLabel(master=self.camera_settings_panel, text="Camera Settings", font=customtkinter.CTkFont(size=17, weight="bold"))
    self.label_camera_settings_panel.grid(row=0, column=0, columnspan=1, padx=10, pady=10)

      # camera output
    self.list_camera_index = customtkinter.CTkOptionMenu(self.camera_settings_panel, dynamic_resizing=False, command=self.open_camera)
    self.list_camera_index.grid(row=1, column=0, padx=20, pady=10, sticky="w")
      # close camera
    self.button_close_camera = customtkinter.CTkButton(self.camera_settings_panel, command=self.close_camera, text="Close camera")
    self.button_close_camera.grid(row=2, column=0, padx=20, pady=10, sticky="w")
    
      # colorspace
    self.camera_colorspace = customtkinter.CTkFrame(self.camera_settings_panel)
    self.camera_colorspace.grid(row=3, column=0, padx=20, pady=10, sticky="w")
    self.colorspace_var = tkinter.IntVar(value=0)
    self.label_radio_group = customtkinter.CTkLabel(master=self.camera_colorspace, text="Camera Colorpace:")
    self.label_radio_group.grid(row=0, column=0, columnspan=1, padx=20, pady=10, sticky="")
        # RGB
    self.camera_RGB = customtkinter.CTkRadioButton(master=self.camera_colorspace, text="RGB", variable=self.colorspace_var, value=0)
    self.camera_RGB.grid(row=1, column=0, padx=20, pady=10, sticky="n")
        # grayscale
    self.camera_grayscale = customtkinter.CTkRadioButton(master=self.camera_colorspace, text="Grayscale", variable=self.colorspace_var, value=1)
    self.camera_grayscale.grid(row=2, column=0, padx=20, pady=10, sticky="n")
        # HSV
    self.camera_HSV = customtkinter.CTkRadioButton(master=self.camera_colorspace, text="HSV", variable=self.colorspace_var, value=2)
    self.camera_HSV.grid(row=3, column=0, padx=20, pady=10, sticky="n")



    ## VIDEO PANEL
    self.video_feed_panel = customtkinter.CTkFrame(self)
    self.video_feed_panel.grid(row=0, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew")

    self.label_video_feed = customtkinter.CTkLabel(master=self.video_feed_panel, text="Video output", font=customtkinter.CTkFont(size=17, weight="bold"))
    self.label_video_feed.grid(row=0, column=0, columnspan=1, padx=10, pady=10)

      # toggle YOLO
    self.capture_on = customtkinter.IntVar(value=0)
    self.switch_predict = customtkinter.CTkSwitch(master=self.video_feed_panel, text=f"Start YOLO", command = self.toggle_predict, variable=self.capture_on, onvalue=1, offvalue=0, state="disabled")
    self.switch_predict.grid(row=1, column=0, padx=20, pady=5)
      # video feed
    self.tabview = customtkinter.CTkTabview(self.video_feed_panel, width=250, state="disabled")
    self.tabview.grid(row=2, column=0, padx=0, pady=0, sticky="w")
    self.tabview.add("Video")
    self.source_image = tkinter.Label(self.tabview.tab("Video"))
    self.source_image.grid(row=0, column=0)
    self.predict_image = tkinter.Label(self.tabview.tab("Video"))



    self.load_model()
    self.get_camera_indexes()


  def toggle_predict(self):
    if self.switch_predict.get():
      self.capture_on = True
      print("[-] Starting predictions...")
      self.predict_image.grid(row=0, column=0)
      self.get_predict_feed()
    else:
      self.capture_on = False
      self.predict_image.grid_forget()


  def load_model(self):
    self.model = YOLO('best.pt')
    
    self.class_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    self.class_colors = [] #create colors for boxes
    for i in range(26):
      r = random.randint(0, 255)
      g = random.randint(0, 255)
      b = random.randint(0, 255)
      self.class_colors.append((b,g,r))

    print("[-] YOLOv8 loaded")


  def get_camera_indexes(self):
    # checks the first 10 indexes for devices
    index = 0
    self.arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            self.arr.append(str(index))
            cap.release()
        index += 1
        i -= 1
    self.list_camera_index.configure(values = self.arr)
    self.list_camera_index.set("Select camera")

  
  def close_camera(self):
    # stop capture and clean up menu + image
    self.cap.release()
    self.switch_predict.deselect()
    self.switch_predict.configure(state="disabled")
    self.predict_image.configure(image='')
    self.source_image.configure(image='')
    self.get_camera_indexes()


  def open_camera(self, index):
    # open camera, removes index from dropdown menu to avoid user capturing twice the same camera
    self.cap = cv2.VideoCapture(int(index))
    self.switch_predict.configure(state="enabled")
    print("[-] Camera opened")
    self.arr.remove(index)
    self.list_camera_index.configure(values = self.arr)
    
    self.get_source_feed()


  def get_source_feed(self):
    ret, frame = self.cap.read()
    
    if ret: # grab image + detect colorspace option
      match self.colorspace_var.get():
        case 0:
          cv2image = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)
        case 1:
          cv2image = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
        case 2:
          cv2image = cv2.cvtColor(frame ,cv2.COLOR_BGR2HSV)

      img = Image.fromarray(cv2image)
      imgtk = ImageTk.PhotoImage(image = img)
      self.source_image.imgtk = imgtk
      self.source_image.configure(image=imgtk)
      
      self.source_image.after(10, self.get_source_feed)
    else:
      print("[-] Video stream ended")



  def get_predict_feed(self):
    if self.capture_on:
      ret, frame = self.cap.read()

      result_frame = self.predict(frame)
      
      if ret: # grab image + detect colorspace option
        match self.colorspace_var.get():
          case 0:
            cv2image = cv2.cvtColor(result_frame ,cv2.COLOR_BGR2RGB)
          case 1:
            cv2image = cv2.cvtColor(result_frame ,cv2.COLOR_BGR2GRAY)
          case 2:
            cv2image = cv2.cvtColor(result_frame ,cv2.COLOR_BGR2HSV)

        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image = img)
        self.predict_image.imgtk = imgtk
        self.predict_image.configure(image=imgtk)

        self.predict_image.after(20, self.get_predict_feed)
      else:
        print("[-] Video stream ended")
    else:
      self.predict_image.configure(image='')
      print("[-] Stopping predictions")


  def predict(self, frame):

    frame_resized = cv2.resize(frame, (image_target_size, image_target_size)) #resize frame

    output = self.model.predict(source=frame_resized)

    #keeping scales to resize again the frame when returning from the function
    scale_x = frame.shape[1] / image_target_size
    scale_y = frame.shape[0] / image_target_size

    for box in output[0].boxes:

      #label + confidence
      clsID = box.cls.numpy()[0]
      conf = box.conf.numpy()[0]
      
      #get boxes coordinates and draw them
      bb = box.xyxy.numpy()[0]
      xA = int(numpy.round(bb[0] * scale_x))
      yA = int(numpy.round(bb[1] * scale_y))
      xB = int(numpy.round(bb[2] * scale_x))
      yB = int(numpy.round(bb[3] * scale_y))

      cv2.rectangle(
      frame, (xA, yA), (xB, yB), self.class_colors[int(clsID)], 3)

      #display class name + confidence
      font = cv2.FONT_HERSHEY_COMPLEX
      cv2.putText(
        frame,
        self.class_list[int(clsID)]
        + " "
        + str(round(conf, 3))
        + "%",
        (xA, yA - 10),
        font,
        1,
        (255, 255, 255),
        2,
      )

    return frame


if __name__ == "__main__":
  app = App()
  app.mainloop()

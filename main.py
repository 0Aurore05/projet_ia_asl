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
    self.title("CustomTkinter complex_example.py")
    self.geometry(f"{1100}x{580}")

    # configure grid layout (4x4)
    self.grid_columnconfigure(1, weight=1)
    self.grid_columnconfigure((2, 3), weight=0)
    self.grid_rowconfigure((0, 1, 2), weight=1)

    # create sidebar frame with widgets
    self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
    self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
    self.sidebar_frame.grid_rowconfigure(4, weight=1)

    self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="CustomTkinter", font=customtkinter.CTkFont(size=20, weight="bold"))
    self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))


    self.tabview = customtkinter.CTkTabview(self, width=250)
    self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
    self.tabview.add("Video")
    self.source_image = tkinter.Label(self.tabview.tab("Video"))
    self.source_image.grid(row=0, column=0)
    self.predict_image = tkinter.Label(self.tabview.tab("Video"))

    #OLD TABS FOR 'SOURCE' AND 'PREDICT'
    """ # create tabview
    self.tabview = customtkinter.CTkTabview(self, width=250)
    self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
    self.tabview.add("Source")
    self.tabview.add("Predict")
    self.tabview.tab("Predict").grid_columnconfigure(0, weight=1)

    # Create source & predict video feed
    self.source_image = tkinter.Label(self.tabview.tab("Source"), text="source")
    self.source_image.pack()
    self.predict_image = tkinter.Label(self.tabview.tab("Predict"), text="predict")
    self.predict_image.pack() """

    self.capture_on = customtkinter.IntVar(value=0)
    self.switch_predict = customtkinter.CTkSwitch(master=self, text=f"Toggle prediction", command = self.toggle_predict, variable=self.capture_on, onvalue=1, offvalue=0)
    self.switch_predict.grid(row=0, column=0, padx=10, pady=(0, 20))

    self.load_model()
    self.open_camera()

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


  def open_camera(self):
    self.cap = cv2.VideoCapture(0)
    
    if not self.cap.isOpened():
      print("[X] Error while opening camera\nExiting...")
      exit()
    else:
      print("[-] Camera opened")
      self.get_source_feed()

  def get_source_feed(self):
    ret, frame = self.cap.read()
    
    if not ret:
      print("Not receiving frames (stream ended ?)\nExiting...")
      exit()

    cv2image = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB) #grab image + colorspace

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = img)
    self.source_image.imgtk = imgtk
    self.source_image.configure(image=imgtk)

    self.source_image.after(20, self.get_source_feed)


  def get_predict_feed(self):
    if self.capture_on:
      ret, frame = self.cap.read()

      result_frame = self.predict(frame)
    
      cv2image = cv2.cvtColor(result_frame ,cv2.COLOR_BGR2RGB) #grab image + colorspace

      img = Image.fromarray(cv2image)
      imgtk = ImageTk.PhotoImage(image = img)
      self.predict_image.imgtk = imgtk
      self.predict_image.configure(image=imgtk)

      self.predict_image.after(20, self.get_predict_feed)
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

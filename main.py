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
from tkinter import filedialog

# Model import
from ultralytics import YOLO
import numpy
import random
import time

customtkinter.set_default_color_theme("blue")
customtkinter.set_appearance_mode("System")

image_target_size = 256

class App(customtkinter.CTk):
  def __init__(self):
    super().__init__()

    # configure window
    self.title("YOLO: Detection ASL")
    self.geometry(f"{1300}x{650}")

    # configure grid layout (4x4)
    self.grid_columnconfigure(1, weight=0)
    self.grid_columnconfigure((1, 3), weight=0)
    self.grid_rowconfigure((0, 1, 2), weight=0)

    ## LEFT SIDEBAR
    self.sidebar_frame = customtkinter.CTkFrame(self, width=300, corner_radius=0)
    self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
    self.sidebar_frame.grid_rowconfigure(4, weight=1)
    
      # title
    self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="YOLOv8 Detection ASL", font=customtkinter.CTkFont(size=20, weight="bold"))
    self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
      # textbox
    self.textbox = customtkinter.CTkTextbox(self.sidebar_frame, width=250, height=250, wrap="word")
    self.textbox.grid(row=1, column=0, padx=20, pady=0)
    self.textbox.insert("0.0", """
Ce programme utilise YOLOv8 (IA de détection d'objets) pour détecter les lettres et les chiffres de la Langue des Signes Américaine.\n\n
Les poids utilisés pour l'inférence du modèle sont le résultat de nos recherches sur l'entraînement d'un modèle IA.\n\n\n

OPTIONS:
Changer l'espace chromatique n'influence pas le résultat de l'inférence, YOLO utilise déjà différents espaces chromatiques par défaut pour l'augmentation de ses données.\n
Ce sont des démonstrations des différentes approches de traitement d'images qui ont chacunes leurs avantages et leurs inconvénients.\n\n

Le modèle actuel a été entraîné pour recevoir des images de 256 pixels de hauteur et de large.\n
Changer cette option donnera au modèle des images de tailles différentes, pour étudier en temps réel comment il réagit et interprète ces gains / pertes d'informations.
      """)


    ## SETTINGS
    self.settings_panel = customtkinter.CTkFrame(self)
    self.settings_panel.grid(row=0, column=1, padx=(20, 20), pady=(20, 0), sticky="nsew")
    
    self.label_settings_panel = customtkinter.CTkLabel(master=self.settings_panel, text="Paramètres", font=customtkinter.CTkFont(size=17, weight="bold"))
    self.label_settings_panel.grid(row=0, column=0, columnspan=1, padx=10, pady=10)

      # camera output
    self.list_camera_index = customtkinter.CTkOptionMenu(self.settings_panel, dynamic_resizing=False, command=self.open_camera)
    self.list_camera_index.grid(row=1, column=0, padx=20, pady=10)
      # close camera
    self.button_close_camera = customtkinter.CTkButton(self.settings_panel, command=self.close_camera, text="Fermer la caméra")
    self.button_close_camera.grid(row=2, column=0, padx=20, pady=10)

      # colorspace
    self.camera_colorspace = customtkinter.CTkFrame(self.settings_panel)
    self.camera_colorspace.grid(row=3, column=0, padx=20, pady=10)
    
    self.colorspace_var = tkinter.IntVar(value=0)
    self.label_radio_group = customtkinter.CTkLabel(master=self.camera_colorspace, font=customtkinter.CTkFont(size=12, weight="bold"), text="Espace chromatique:")
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

      # image size
    self.label_imgsize = customtkinter.CTkLabel(master=self.camera_colorspace, font=customtkinter.CTkFont(size=12, weight="bold"), text="Résolution d'image:")
    self.label_imgsize.grid(row=4, column=0, columnspan=1, padx=20, pady=10, sticky="")

    self.label_imgsize_value = customtkinter.CTkLabel(master=self.camera_colorspace, text="256 x 256")
    self.label_imgsize_value.grid(row=5, column=0, columnspan=1, padx=20, pady=0, sticky="")

    self.slider_imgsize = customtkinter.CTkSlider(self.camera_colorspace, from_=1, to=1024, number_of_steps=32, command=self.update_slider_res)
    self.slider_imgsize.grid(row=6, column=0, padx=20, pady=5, sticky="w")

      # inference image
    self.inference_image = customtkinter.CTkFrame(self.settings_panel)
    self.inference_image.grid(row=4, column=0, padx=20, pady=10)
    
    self.label_imgsize = customtkinter.CTkLabel(master=self.inference_image, font=customtkinter.CTkFont(size=12, weight="bold"), text="Prédiction d'image")
    self.label_imgsize.grid(row=0, column=0, columnspan=1, padx=20, pady=10, sticky="")

    self.label_imgsize = customtkinter.CTkLabel(master=self.inference_image, font=customtkinter.CTkFont(size=11), text="Pas de caméra ?\nUtilisez le modèle sur une image")
    self.label_imgsize.grid(row=1, column=0, columnspan=1, padx=20, pady=0, sticky="")
        # image browse button
    self.button_browse_image = customtkinter.CTkButton(self.inference_image, command=self.get_image_inference, text="Image...")
    self.button_browse_image.grid(row=2, column=0, padx=20, pady=10)


    ## VIDEO PANEL
    self.video_feed_panel = customtkinter.CTkFrame(self)
    self.video_feed_panel.grid(row=0, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew")

    self.label_video_feed = customtkinter.CTkLabel(master=self.video_feed_panel, text="Flux vidéo", font=customtkinter.CTkFont(size=17, weight="bold"))
    self.label_video_feed.grid(row=0, column=0, columnspan=1, padx=10, pady=10)

      # toggle YOLO
    self.switch_predict = customtkinter.CTkSwitch(master=self.video_feed_panel, text="Activer YOLO", onvalue=1, offvalue=0, state="disabled")
    self.switch_predict.grid(row=1, column=0, padx=20, pady=5)
      # video feed
    self.tabview = customtkinter.CTkTabview(self.video_feed_panel, width=250, state="disabled")
    self.tabview.grid(row=2, column=0, padx=0, pady=0, sticky="w")
    self.tabview.add("Video")
    self.video_feed = tkinter.Label(self.tabview.tab("Video"))
    self.video_feed.grid(row=0, column=0)


    self.load_model()
    self.get_camera_indexes()


  def load_model(self):
    self.model = YOLO('best.pt')
    
    self.class_list = ["1", "10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    self.class_colors = [] #create colors for boxes
    for i in range(36):
      r = random.randint(0, 255)
      g = random.randint(0, 255)
      b = random.randint(0, 255)
      self.class_colors.append((b,g,r))

    print("[-] YOLOv8 loaded")


  def update_slider_res(self, value):
    self.label_imgsize_value.configure(text=str(int(value)) + " x " + str(int(value)))
    global image_target_size
    image_target_size = int(value)


  def get_image_inference(self):
    filename = filedialog.askopenfilename()
    if filename and not self.switch_predict.get():

      pil_image = Image.open(filename)
      opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)

      frame = self.predict_frame(opencvImage)
      cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      img = Image.fromarray(cv2image)
      imgtk = ImageTk.PhotoImage(image = img)
      self.video_feed.imgtk = imgtk
      self.video_feed.configure(image=imgtk)

    
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
    self.list_camera_index.set("Caméra...")

  
  def close_camera(self):
    # stop capture and clean up menu + image
    self.cap.release()
    self.switch_predict.deselect()
    self.switch_predict.configure(state="disabled")
    self.button_browse_image.configure(state="normal")
    self.video_feed.configure(image='')
    
    self.get_camera_indexes()


  def open_camera(self, index):
    # open camera, removes index from dropdown menu to avoid user capturing twice the same camera
    self.cap = cv2.VideoCapture(int(index))
    self.switch_predict.configure(state="enabled")
    self.button_browse_image.configure(state="disabled")
    print("[-] Camera opened")
    self.arr.remove(index)
    self.list_camera_index.configure(values = self.arr)
    
    self.get_video_feed()


  def get_video_feed(self):
    ret, frame = self.cap.read()

    if ret: # grab image + detect colorspace option
      if self.switch_predict.get(): # if in predition mode: replace source image by YOLO image
        frame = self.predict_frame(frame)

      match self.colorspace_var.get():
        case 0:
          cv2image = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)
        case 1:
          cv2image = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
        case 2:
          cv2image = cv2.cvtColor(frame ,cv2.COLOR_BGR2HSV)

      img = Image.fromarray(cv2image)
      imgtk = ImageTk.PhotoImage(image = img)
      self.video_feed.imgtk = imgtk
      self.video_feed.configure(image=imgtk)
      
      self.video_feed.after(25, self.get_video_feed)
 
    else:
      print("[-] Video stream ended")


  def predict_frame(self, frame):
    frame_resized = cv2.resize(frame, (image_target_size, image_target_size)) #resize frame

    output = self.model.predict(source=frame_resized, imgsz=image_target_size, show=False)

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

# Camera import
from PIL import Image, ImageTk  
import cv2

# GUI import
import tkinter
import tkinter.messagebox
import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


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

    # create tabview
    self.tabview = customtkinter.CTkTabview(self, width=250)
    self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
    self.tabview.add("Tab 1")
    self.tabview.add("Tab 2")
    self.tabview.tab("Tab 2").grid_columnconfigure(0, weight=1)

    # Create a button to open the camera in GUI app
    self.label = tkinter.Label(self.tabview.tab("Tab 1"), text="oui")
    self.label.pack()


    self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="CustomTkinter", font=customtkinter.CTkFont(size=20, weight="bold"))
    self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))    
    
    self.cap = cv2.VideoCapture(0)

  def show_frames(self):
    cv2image = cv2.cvtColor(self.cap.read()[1],cv2.COLOR_BGR2RGB) #grab image + colorspace
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = img)
    self.label.imgtk = imgtk
    self.label.configure(image=imgtk)
    self.label.after(20, self.show_frames)

if __name__ == "__main__":
  app = App()
  app.show_frames()
  app.mainloop()

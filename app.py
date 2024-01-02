from tkinter import *
import customtkinter as ctk
from tkinter.colorchooser import askcolor
from tkinter import messagebox, filedialog
from PIL import ImageGrab
from PIL import Image
import os


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'White'

    def __init__(self):
        ctk.set_default_color_theme("green")
        self.root = ctk.CTk()
        self.root.title("Pix2Pix Maps Application")

        self.pen_button = ctk.CTkButton(self.root, text='fill', command=self.fill)
        self.pen_button.grid(row=0, column=0)

        self.color_button = ctk.CTkButton(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=1)

        self.eraser_button = ctk.CTkButton(self.root, text='save to png', command=self.export_canvas)
        self.eraser_button.grid(row=0, column=2)

        self.lowend = 5
        self.highend = 40
        self.choose_size_button = ctk.CTkSlider(self.root, from_=self.lowend, to=self.highend, orientation=HORIZONTAL, command=lambda x: self.brushsizeupdate(x),)
        self.choose_size_button.set(0)
        self.choose_size_button.grid(row=1, columnspan=3)

        self.slider_label = ctk.CTkLabel(self.root, text=f"Brush Size: {int(self.lowend)}")
        self.slider_label.grid(row=2, column=0)

        self.clearbutton = ctk.CTkButton(self.root, text='clear', command=self.clear)
        self.clearbutton.grid(row=3, column=1)

        self.linebutton = ctk.CTkButton(self.root, text='line', command=self.setupline)
        self.linebutton.grid(row=3, column=2)

        self.c = ctk.CTkCanvas(self.root, bg='white', width=600, height=600)
        self.c.grid(rowspan = 4, columnspan = 3)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_start_x = None
        self.line_start_y = None
        self.line_end_x = None
        self.line_end_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
    
    def mousebindreset(self):
        self.c.unbind_all('<Button-1>')
        self.c.unbind_all('<ButtonRelease-1>')
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        

    def use_pen(self):
        self.activate_button(self.pen_button)

    def brushsizeupdate(self, x):
        self.slider_label.configure(text=f"Brush Size: {int(x)}")

    def choose_color(self):
        self.eraser_on = False

        # Create a new window
        self.color_window = Toplevel()
        self.color_window.title("Choose Color")
        
        #Vars
        self.boundline = None
        self.boundlinedraw = None

        # Define the available colors
        colors = ["#ff0000", "#0000ff", "#00ff00", "#ffffff", "#ffff00", "#00ffff"]
        label = ["Urban", "Sea", "Green", "Road", "Highway", "Houses"]

        # Create buttons for each color
        for i, color in enumerate(colors):
            color_label = Label(self.color_window, text=label[i])
            color_label.grid(row=i, column=0, padx=5, pady=2)
            color_button = Button(self.color_window, bg=color, width=10, height=2, command=lambda c=color: self.set_color(c))
            color_button.grid(row=i, column=1, padx=5, pady=5)

    def fill(self):
        self.c.create_rectangle(0, 0, 600, 600, fill=self.color)

    def set_color(self, color):
        self.color = color
        self.color_window.destroy()
    
    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def start_line(self, event):
        if event.x > 0 and event.x < 600 and event.y > 0 and event.y < 600:
            self.line_start_x = event.x
            self.line_start_y = event.y
            self.c.unbind_all('<Button-1>')
            self.c.unbind('<Button-1>', self.boundline)
            self.boundlinedraw = self.c.bind('<Button-1>', self.draw_line)

    def draw_line(self, event):
        self.line_end_x = event.x
        self.line_end_y = event.y
        self.line_width = self.choose_size_button.get()
        self.c.create_line(self.line_end_x, self.line_end_y, self.line_start_x, self.line_start_y,
                               width=self.line_width, fill=self.color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.c.unbind('<Button-1>', self.boundlinedraw)

    def setupline(self):
        self.c.unbind_all('<Button-1>')
        self.boundline = self.c.bind('<Button-1>', self.start_line)

    def endofstartline(self, event):
        self.c.unbind_all('<Button-1>')
        self.c.unbind_all('<ButtonRelease-1>')
        self.c.bind('<B1-Motion>', self.draw_line)
        self.c.bind('<ButtonRelease-1>', self.resetline)

    def resetline(self):
        self.old_x, self.old_y = None, None
        self.line_start_x, self.line_start_y = None, None
        self.line_end_x, self.line_end_y = None, None
        self.c.unbind_all('<Button-1>')
        self.c.unbind_all('<ButtonRelease-1>')
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear(self):
        self.c.delete(ALL)

    def export_canvas(self):
        try:
            # Get the coordinates of the canvas
            x = self.c.winfo_rootx()
            y = self.c.winfo_rooty()
            width = self.c.winfo_width()
            height = self.c.winfo_height()

            # Capture the content of the canvas as an image
            image = ImageGrab.grab((x, y, x + width, y + height))
            print(type(image))
            # Ask the user to choose a file name and location
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPG Files", "*.jpg")])

            # Save the image as a PNG file
            if file_path:
                image.save(file_path)
                messagebox.showinfo("Export Successful", "Canvas exported as JPG successfully.")
        except Exception as e:
            messagebox.showerror("Export Failed", f"An error occurred while exporting the canvas: {str(e)}")

        self.outputimage = Toplevel(height=600, width=600)
        self.outputimage.title("Output Image")
        #load in tt.png
        image_file = "tt.png" #TODO -> CHANGE TO ACTUAL OUTPUT IMAGE :D
        #with Image.open(image_file) as image_location:
        #NOW IT JUST USES THE INPUT IMAGE    
        with image as image_location:
            output_image = ctk.CTkImage(light_image=image_location, dark_image=image_location, size=(600, 600))
            output_image_button = ctk.CTkButton(self.outputimage, image=output_image, text="", state="disabled",anchor="center", height=600, width=600, fg_color="transparent", bg_color="transparent")
            output_image_button.grid_configure(column=0, row=0, sticky="ew")


if __name__ == '__main__':
    Paint()
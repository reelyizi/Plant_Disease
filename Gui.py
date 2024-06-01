from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog, Label
from PIL import Image, ImageTk, ImageDraw
import os
import re
import numpy as np
import cv2
import PredictImage as PI

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/Users/aycaaydin/Desktop/build/assets/frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def normalize_text(text):
    return re.sub(r'[_\d\s]', '', text).lower()

window = Tk()
window.geometry("600x500")
window.configure(bg = "#FFFFFF")
window.title("Plant Disease Detection")  # Pencere başlığını ayarlayın

canvas = Canvas(
    window,
    bg = "#98A68A",
    height = 500,
    width = 600,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
canvas.place(x = 0, y = 0)

# Actual text
canvas.create_text(
    257.0,
    26.0,
    anchor="nw",
    text="Actual",
    fill="#FFFFFF",
    font=("Timmana", 20 * -1)
)

# Predict text
canvas.create_text(
    452.0,
    26.0,
    anchor="nw",
    text="Predict",
    fill="#FFFFFF",
    font=("Timmana", 20 * -1)
)

# Grey Scale text
canvas.create_text(
    240.0,
    283.0,
    anchor="nw",
    text="Grayscale",
    fill="#FFFFFF",
    font=("Timmana", 20 * -1)
)

# High Pass Filter text
canvas.create_text(
    418.0,
    283.0,
    anchor="nw",
    text="High Pass Filter",
    fill="#FFFFFF",
    font=("Timmana", 20 * -1)
)

# Accuracy text
canvas.create_text(
    58.0,
    283.0,
    anchor="nw",
    text="Accuracy",
    fill="#FFFFFF",
    font=("Timmana", 18 * -1)
)
# Result text
canvas.create_text(
    68.0,
    355.0,
    anchor="nw",
    text="Result",
    fill="#FFFFFF",
    font=("Timmana", 18 * -1)
)
# Correct text
canvas.create_text(
    72.0,
    432.0,
    anchor="nw",
    text="Correct",
    fill="#FFFFFF",
    font=("Timmana", 18 * -1)
)
# Wrong text
canvas.create_text(
    72.0,
    461.0,
    anchor="nw",
    text="Wrong",
    fill="#FFFFFF",
    font=("Timmana", 18 * -1)
)

# actual image name
actual_image_name_label = canvas.create_text(
    281.0, 245.0,  # Centered within the rectangle
    anchor="center",
    text="",
    fill="#FFFFFF",
    font=("Timmana", 12 * -1)
)

# predict image name
predict_image_name_label = canvas.create_text(
    480.0, 245.0,  # Centered within the rectangle
    anchor="center",
    text="",
    fill="#FFFFFF",
    font=("Timmana", 12 * -1)
)

# Creating image labels
actual_image_label = Label(window, bg="#67735C")
actual_image_label.place(x=196.0, y=53.0, width=170, height=170)

predicted_image_label = Label(window, bg="#67735C")
predicted_image_label.place(x=395.0, y=53.0, width=170, height=170)

grayscale_image_label = Label(window, bg="#67735C")
grayscale_image_label.place(x=196.0, y=311.0, width=170, height=170)

other_image_label = Label(window, bg="#67735C")
other_image_label.place(x=395.0, y=311.0, width=170, height=170)

accuracy_value_label = Label(window, text="", bg="#67735C")
accuracy_value_label.place(x=44.0, y=311.0, width=100, height=30)

result_label = Label(window, text="", bg="#67735C")
result_label.place(x=44.0, y=383.0, width=100, height=30)

correct_label = Label(window, text="", bg="#4AC90F")
correct_label.place(x=44.0, y=432.0, width=20, height=20)

wrong_label = Label(window, text="", bg="#EB1212")
wrong_label.place(x=44.0, y=461.0, width=20, height=20)



# Function to load an image
def load_image():
    global loaded_image, file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        loaded_image = Image.open(file_path)
        img = ImageTk.PhotoImage(loaded_image.resize((170, 170)))
        actual_image_label.config(image=img)
        actual_image_label.image = img
        dir_name = os.path.basename(os.path.dirname(file_path))
        clean_dir_name = normalize_text(dir_name)  # Normalize the text
        canvas.itemconfig(actual_image_name_label, text=clean_dir_name)


# Function to predict the image
def predict_image():
    # Assuming the predict_image function and model are already defined
    if loaded_image:
        predict_label, probability = PI.PredictGivenImage(file_path)
        clean_predict_label = normalize_text(predict_label)  # Normalize the text
        predicted_img = Image.open(file_path)
        img = ImageTk.PhotoImage(predicted_img.resize((170, 170)))
        predicted_image_label.config(image=img)
        predicted_image_label.image = img
        canvas.itemconfig(predict_image_name_label, text=clean_predict_label)
        accuracy_value_label.config(text=f"{probability * 100:.2f}%", fg="white")  # Set text color to white

        # Check if actual and predicted labels are the same
        actual_label = canvas.itemcget(actual_image_name_label, 'text')
        clean_actual_label = normalize_text(actual_label)  # Normalize the text

        if clean_predict_label == clean_actual_label:
            result_label.config(bg="#4AC90F")  # Green
        else:
            result_label.config(bg="#EB1212")  # Red


# Function to convert to grayscale
def convert_to_grayscale():
    if loaded_image:
        grayscale_img = loaded_image.convert("L")
        img = ImageTk.PhotoImage(grayscale_img.resize((170, 170)))
        grayscale_image_label.config(image=img)
        grayscale_image_label.image = img

# Function to apply high pass filter
def apply_high_pass_filter(image):
    image = np.array(image)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    high_pass_filtered_image = cv2.filter2D(image, -1, kernel)
    return Image.fromarray(high_pass_filtered_image)

# Function to apply high pass filter
def high_pass_filter():
    if loaded_image:
        high_pass_filtered_img = apply_high_pass_filter(loaded_image)
        img = ImageTk.PhotoImage(high_pass_filtered_img.resize((170, 170)))
        other_image_label.config(image=img)
        other_image_label.image = img

# Buttons
load_button = Button(window, text="Load", command=load_image, bg="#000000", fg="#4E5946")
load_button.place(x=53.0, y=53.0, width=83, height=40)

predict_button = Button(window, text="Predict", command=predict_image, bg="#000000", fg="#4E5946")
predict_button.place(x=53.0, y=110.0, width=83, height=40)

grayscale_button = Button(window, text="Grayscale", command=convert_to_grayscale, bg="#000000", fg="#4E5946")
grayscale_button.place(x=53.0, y=167.0, width=83, height=40)

highPassFilter_button = Button(window, text="High Pass\nFilter", command=high_pass_filter, bg="#000000", fg="#4E5946")
highPassFilter_button.place(x=53.0, y=224.0, width=83, height=40)

window.resizable(False, False)
window.mainloop()

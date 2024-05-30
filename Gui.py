from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog, Label
from PIL import Image, ImageTk, ImageDraw

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/Users/aycaaydin/Desktop/build/assets/frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()
window.geometry("600x500")
window.configure(bg = "#FFFFFF")

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
    text="Grey Scale",
    fill="#FFFFFF",
    font=("Timmana", 20 * -1)
)

# Bla bla text
canvas.create_text(
    452.0,
    283.0,
    anchor="nw",
    text="Bla bla",
    fill="#FFFFFF",
    font=("Timmana", 20 * -1)
)

# Accuracy text
canvas.create_text(
    57.0,
    355.0,
    anchor="nw",
    text="Accuracy:",
    fill="#FFFFFF",
    font=("Timmana", 20 * -1)
)
#actual image name
canvas.create_rectangle(
    196.0,
    229.0,
    366.0,
    262.0,
    fill="#D9D9D9",
    outline="")

#predict image name
canvas.create_rectangle(
    395.0,
    229.0,
    565.0,
    262.0,
    fill="#D9D9D9",
    outline="")

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
accuracy_value_label.place(x=51.0, y=382.0, width=100, height=30)

# Function to load an image
def load_image():
    global loaded_image
    file_path = filedialog.askopenfilename()
    if file_path:
        loaded_image = Image.open(file_path)
        img = ImageTk.PhotoImage(loaded_image.resize((135, 135)))
        actual_image_label.config(image=img)
        actual_image_label.image = img

# Function to predict the image
def predict_image():
    # Assuming the predict_image function and model are already defined
    if loaded_image:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        img = transform(loaded_image)
        predicted_label = predict_image_function(img, model)  # Use the actual predict function
        img = ImageTk.PhotoImage(loaded_image.resize((135, 135)))
        predicted_image_label.config(image=img)
        predicted_image_label.image = img
        accuracy_value_label.config(text=f"Predicted: {predicted_label}")

# Function to convert to grayscale
def convert_to_grayscale():
    if loaded_image:
        grayscale_img = loaded_image.convert("L")
        img = ImageTk.PhotoImage(grayscale_img.resize((135, 135)))
        grayscale_image_label.config(image=img)
        grayscale_image_label.image = img

# Placeholder function for other tasks
def other_function():
    pass

# Buttons
load_button = Button(window, text="Load", command=load_image, bg="#000000", fg="#4E5946")
load_button.place(x=53.0, y=79.0, width=83, height=40)

predict_button = Button(window, text="Predict", command=predict_image, bg="#000000", fg="#4E5946")
predict_button.place(x=53.0, y=136.0, width=83, height=40)

grayscale_button = Button(window, text="Grayscale", command=convert_to_grayscale, bg="#000000", fg="#4E5946")
grayscale_button.place(x=53.0, y=193.0, width=83, height=40)

other_button = Button(window, text="Other", command=other_function, bg="#000000", fg="#4E5946")
other_button.place(x=53.0, y=250.0, width=83, height=40)

window.resizable(False, False)
window.mainloop()

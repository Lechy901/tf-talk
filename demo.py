from tkinter import *
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np

width, height = 560, 560
model = tf.keras.models.load_model("trained_model.h5")
offset = 25

def savePosn(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    canvas.create_line((lastx, lasty, event.x, event.y), fill='white', width=2 * offset)
    draw.line([lastx, lasty, event.x, event.y], fill=(255), width=2 * offset)
    canvas.create_oval((event.x - offset, event.y - offset, event.x + offset, event.y + offset), fill='white', outline='white')
    draw.ellipse([event.x - offset, event.y - offset, event.x + offset, event.y + offset], fill=(255), outline=(255))
    image_r =  image.resize((28, 28), resample=Image.BILINEAR)
    image_np = np.array(image_r) / 255
    predictions = model.predict(np.resize(image_np, (1, 28, 28)))
    print([round(x, 3) for x in predictions[0]])
    savePosn(event)

def reset(event):
    canvas.delete("all")
    draw.rectangle([0, 0, width, height], fill=(0))
    

root = Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

canvas = Canvas(root, width=width, height=height, bg='black')
canvas.grid(column=0, row=0, sticky=(N, W, E, S))
canvas.bind("<Button-1>", savePosn)
canvas.bind("<B1-Motion>", addLine)
root.bind('r', reset)

image = Image.new("L", (width, height), (0))
draw = ImageDraw.Draw(image)

root.mainloop()

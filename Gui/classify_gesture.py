import numpy as np

from skimage.io import imread
from skimage.transform import resize
from tkinter import Label, Tk, Button
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import pkg_resources
import joblib
from Phase2.transformers import RGB2GrayTransformer, HogTransformer

clf = joblib.load(pkg_resources.resource_filename('Phase2', 'image_classification_model.joblib'))
grayify = joblib.load(pkg_resources.resource_filename('Phase2', 'grayify.joblib'))
hogify = joblib.load(pkg_resources.resource_filename('Phase2', 'hogify.joblib'))
scalify = joblib.load(pkg_resources.resource_filename('Phase2', 'scalify.joblib'))

output_shape = (102, 136, 3)

root = Tk()

values = {"image_rgb": [], "label": Label(root, cnf={'height': 600, 'width': 800})}
lbl_classification = Label(master=root, text="", fg='black')


def classify_gesture():
    """Classify a text as profane or not-profane
    """
    image_rgb_value = values["image_rgb"]
    resized = resize(image_rgb_value, output_shape=output_shape)

    x_input = np.array([resized])

    result = clf.predict(scalify.transform(hogify.transform(grayify.transform(x_input))))

    print(result)

    if result[0] == 'skull' or result[0] == 'middlefinger':
        lbl_classification["text"] = 'Result : Profane.'
    else:
        lbl_classification["text"] = 'Result : Not profane.'


def pick_image():
    path = askopenfilename(filetypes=[("Image Files", '.jpg;*.jpeg;*.png')])

    if not path:
        return

    im = Image.open(path)

    values["image_rgb"] = imread(path)
    lbl_classification["text"] = ''

    tk_image = ImageTk.PhotoImage(im)

    label = values["label"]

    label.config(image=tk_image)
    label.image = tk_image
    label.pack()


pick_image()

btn_convert = Button(
    master=root,
    text="Classify    \N{RIGHTWARDS BLACK ARROW}",
    command=classify_gesture
)

btn_choose = Button(
    master=root,
    text="Choose Another",
    command=pick_image
)

btn_convert.pack()
btn_choose.pack()

lbl_classification.config(font=("Courier", 40))
lbl_classification.pack()

root.mainloop()

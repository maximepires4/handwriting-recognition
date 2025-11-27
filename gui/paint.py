# Credits to nikhilkumarsingh
# https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06

from tkinter import *

from gui.imagehandler import ImageHandler
from gui.neuralnethandler import NeuralNetHandler


class Paint(object):
    DEFAULT_PEN_SIZE = 50.0
    DEFAULT_COLOR = "white"
    DEFAULT_BACKGROUND = "black"

    def __init__(self, model_path):
        self.model_path = model_path
        self.root = Tk()
        self.root.title("Handwriting recognition")

        self.pen_button = Button(self.root, text="Pen", command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.eraser_button = Button(self.root, text="Eraser", command=self.use_eraser)
        self.eraser_button.grid(row=0, column=1)

        self.erase_all_button = Button(
            self.root, text="Erase all", command=self.erase_all
        )
        self.erase_all_button.grid(row=0, column=2)

        self.export_image_button = Button(
            self.root, text="Export image", command=self.export_image
        )
        self.export_image_button.grid(row=0, column=3)

        self.choose_size_button = Scale(
            self.root,
            from_=self.DEFAULT_PEN_SIZE - 10,
            to=self.DEFAULT_PEN_SIZE + 10,
            orient=HORIZONTAL,
        )
        self.choose_size_button.grid(row=0, column=4)
        self.choose_size_button.set(self.DEFAULT_PEN_SIZE)

        self.c = Canvas(
            self.root,
            bg=self.DEFAULT_BACKGROUND,
            width=600,
            height=600,
            highlightthickness=0,
        )
        self.c.grid(row=1, rowspan=10, columnspan=5)

        # Create a fixed-size frame for the prediction labels
        prediction_frame = Frame(self.root, width=250, height=600)
        prediction_frame.grid(row=1, column=5, rowspan=10, columnspan=2, padx=10)
        prediction_frame.grid_propagate(False)

        self.labels = []
        self.textvars = []

        for i in range(10):
            self.textvars.append([StringVar(), StringVar()])
            self.labels.append(
                (
                    Label(
                        prediction_frame,
                        font=("TkDefaultFont", 30),
                        fg="#888" if i != 0 else "#000",
                        textvariable=self.textvars[i][0],
                    ),
                    Label(
                        prediction_frame,
                        font=("TkDefaultFont", 30),
                        fg="#888" if i != 0 else "#000",
                        textvariable=self.textvars[i][1],
                    ),
                )
            )
            self.labels[i][0].grid(row=i, column=0, padx=10)
            self.labels[i][1].grid(row=i, column=1, padx=10)

        self._throttle_flag = False
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind("<B1-Motion>", self.paint)
        self.c.bind("<ButtonRelease-1>", self.reset)

        # Ensure canvas is fully rendered to get correct size for PIL image
        self.root.update_idletasks()

        self.image_handler = ImageHandler(self.root, self.c)
        self.neuralnet_handler = NeuralNetHandler(self.model_path)
        self._clear_prediction_labels()

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def export_image(self):
        self.image_handler.save()

    def erase_all(self):
        self.c.delete("all")
        self.image_handler.clear()
        self._clear_prediction_labels()

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        # Use white for drawing on PIL image, black for erasing
        paint_color = "black" if self.eraser_on else "white"

        if self.old_x and self.old_y:
            # Draw on the screen canvas
            self.c.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=self.line_width,
                fill=self.DEFAULT_BACKGROUND if self.eraser_on else self.DEFAULT_COLOR,
                capstyle=ROUND,
                smooth=TRUE,
                splinesteps=36,
            )
            # Draw on the in-memory PIL image
            self.image_handler.add_line(
                self.old_x, self.old_y, event.x, event.y, self.line_width, paint_color
            )

        self.old_x = event.x
        self.old_y = event.y

        # Throttle the prediction
        if not self._throttle_flag:
            self._throttle_flag = True
            self._trigger_prediction()
            self.root.after(100, self._release_throttle)

    def _release_throttle(self):
        self._throttle_flag = False

    def _trigger_prediction(self):
        self.image_handler.update()
        prediction = self.neuralnet_handler.predict(self.image_handler.image)

        prediction_textvars = [[i, p] for i, p in zip(range(10), prediction[0])]

        count = 0
        # Sort by probability and update labels
        for item in reversed(sorted(prediction_textvars, key=lambda item: item[1])):
            self.textvars[count][0].set(item[0])
            self.textvars[count][1].set("{:,.2%}".format(item[1]))
            # Dim non-top predictions
            color = "#000" if count == 0 else "#888"
            self.labels[count][0].config(fg=color)
            self.labels[count][1].config(fg=color)
            count += 1

    def _clear_prediction_labels(self):
        for i in range(10):
            self.textvars[i][0].set("")
            self.textvars[i][1].set("")

    def reset(self, event):
        self.old_x, self.old_y = None, None
        # Trigger one final prediction on mouse release
        self._trigger_prediction()

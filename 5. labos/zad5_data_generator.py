from tkinter import *
from copy import deepcopy
import numpy as np
import math

class GUI(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title("Press left click to start and to end with drawing.")
        self.leftclick = False
        self.shape_coordinates = []
        self.saved_shapes = []
        self.canvas = Canvas(self, width=600, height=400, bg="white", cursor="dot")
        self.button_display = Button(self, text="Display points", command=self.print_points)
        self.button_save = Button(self, text="Save", command=self.save_letter)
        self.button_clear = Button(self, text="Clear", command=self.clear_all)

        self.canvas.pack(side="top", fill="both", expand=True)
        self.button_clear.pack(side="bottom", fill="both", expand=True)
        self.button_save.pack(side="bottom", fill="both", expand=True)
        self.button_display.pack(side="bottom", fill="both", expand=True)


        self.canvas.bind("<Motion>", self.get_coordinates)
        self.canvas.bind("<Button-1>", self.leftClick)

    def reset(self):
        self.shape_coordinates.clear()
        self.saved_shapes.clear()

    # saves coordinates only if leftclick is TRUE
    def leftClick(self, event):
        self.leftclick = not self.leftclick

    def clear_all(self):
        self.canvas.delete("all")
        self.shape_coordinates.clear()

    def save_letter(self):
        self.saved_shapes.append(deepcopy(self.shape_coordinates))

    def print_points(self):
        self.canvas.create_line(self.shape_coordinates, fill="yellow")

    def get_coordinates(self, event):
        if self.leftclick:
            self.shape_coordinates.append(event.x)
            self.shape_coordinates.append(event.y)

    def get_dataset(self, M):
        dataset = []
        for shape in self.saved_shapes:
            Xs, Ys = shape[::2], shape[1::2]
            xc, yc = (sum(Xs) / len(Xs)), (sum(Ys) / len(Ys))
            shape = [shape[i] - xc if (i % 2 == 0) else shape[i] - yc for i in range(len(shape))]  # centriranje

            shape_abs = [abs(x) for x in shape]
            max_XY = max(shape_abs)
            shape = [shape[i] / max_XY for i in range(len(shape))]  # skaliranje

            # SAMPLING
            D = shape_length(shape)  # duljina geste
            sample_points = sampling(D, M, shape)
            dataset.append(sample_points)

        return dataset


# returns distance between two points
def find_distance(two_dots):
    x1, y1, x2, y2 = two_dots
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# returns length of a shape
def shape_length(shape):
    distances = list()  # udaljenost 1. do 2., 2. do 3., ... n-1. do n.-te
    for i in range(0, len(shape) - 3, 2):
        distances.append(find_distance(shape[i:i + 4]))

    return sum(distances)

# returns index of a sample point
def find_sample_point(start_point, radius, shape):
    last_point_index = len(shape) - 2
    for index in range(0, len(shape), 2):
        d = find_distance(start_point + shape[index:index+2])
        if abs(radius) > d:
            continue
        else:
            return index

    return last_point_index

# returns M sample points
def sampling(D, M, shape):
    shape_copy = deepcopy(shape)
    sample_points = []
    radius = D / (M - 1)
    start_point = shape[:2]
    sample_points += start_point
    shape = shape[2:]
    again = 0
    while len(sample_points) < 2*M:
        index = find_sample_point(start_point, radius, shape)
        start_point = shape[index:index + 2]
        shape = shape[index + 2:]
        sample_points += start_point
        if len(shape) == 0:
            again += 1
            shape = deepcopy(shape_copy)
            start_point = shape[again * 2:again * 2 + 2]
            shape = shape[again * 2 + 2:]

    return sample_points

def label_data(X_train, labels, examples_per_class):
    for j in range(len(labels)):
        for i in range(j * examples_per_class, (j + 1) * examples_per_class):
            X_train[i] += labels[j]

    return X_train


if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()

    M = int(input("Unesite M\n"))
    X_train = gui.get_dataset(M)

    # labeling dataset
    examples_per_class = 20  # defines number of examples for every class
    labels = [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]

    train_data = label_data(X_train, labels, examples_per_class)


    # write to file
    f = open("train2.txt", 'w')
    for example in train_data:
        example = list(map(str, example))
        example = " ".join(example)
        f.write(example + '\n')

    f.close()
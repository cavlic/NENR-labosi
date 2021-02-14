import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_dataset(location):
    dataset = []

    f = open(location, 'r')
    for line in f.readlines():
        line = line.split()
        inputs, labels = list(map(float, line[:2])), list(map(int, line[2:]))
        example = [inputs, labels]
        dataset.append(example)

    f.close()

    return dataset


### loading dataset test ###
dataset = load_dataset("zad7-dataset.txt")
for d in dataset:
    print(d)

### ZAD1 ###
plt.figure(1)
plt.title('Zadatak 1.')
plt.xlabel('x')
plt.ylabel('y')

apscis = [i for i in range(-8, 11)]
ss = [4, 1, 0.25]
w = 2
f = lambda x, w, s: 1 / (1 + abs(x - w) / abs(s))

for s in ss:
    y = [f(x, w, s) for x in apscis]
    plt.plot(apscis, y, label='s =' + str(s))
    plt.legend()


### ZAD2 ###
f = open("zad7-dataset.txt", 'r')
cnt_A, cnt_B, cnt_C = 0, 0, 0
x_A, y_A = [], []
x_B, y_B = [], []
x_C, y_C = [], []


for line in f.readlines():
    line = line.split()
    x, y, A, B, C = float(line[0]), float(line[1]), int(line[2]), int(line[3]), int(line[4])

    if A:
        cnt_A += 1
        x_A.append(x)
        y_A.append(y)

    elif B:
        cnt_B += 1
        x_B.append(x)
        y_B.append(y)

    else:
        cnt_C += 1
        x_C.append(x)
        y_C.append(y)

f.close()

fig = plt.figure(2)
plt.title("Zadatak 2.")
ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.scatter(x_A, y_A, c='b', marker='^')
ax.scatter(x_B, y_B, c='r', marker='P')
ax.scatter(x_C, y_C, c='g', marker='o')

plt.show()

# razredi nisu linearno odvojivi. svaki klaster sadrži ulazne primjere koji pripadaju istom razredu.
# primjeri unutar klasera tvore eliptičan oblik. klaster može biti sačinjen od primjera koji pripadaju
# jednoj od tri rezreda.
import numpy as np


def exponential_pacing(data_len, batch_size, increase_amount = 1.5, starting_percent=0.04):
    pacing_len_list = []
    starting_percent *= data_len
    multiplier = starting_percent
    while multiplier < data_len:
        if int(multiplier) < data_len:
            if int(multiplier) >= batch_size:
                pacing_len_list.append(int(multiplier))
            else:
                pacing_len_list.append(batch_size)
        else:
            pacing_len_list.append(data_len)
        multiplier = multiplier * increase_amount
    return pacing_len_list, batch_size


def parabolic_pacing(data_len, batch_size, power = 1.5, starting_percent=0.04):
    pacing_len_list = []
    starting_percent *= data_len
    for i in range(int(data_len / starting_percent + 1)):
        if int(starting_percent * i ** power) < data_len:
            if int(starting_percent * i ** power) >= batch_size:
                pacing_len_list.append(int(starting_percent * i ** power))
            else:
                pacing_len_list.append(batch_size)
        else:
            pacing_len_list.append(data_len)
            break

    return pacing_len_list[1:], batch_size


def exponential_pacing_normalized(
    data_len, batch_size, increase_amount=2, starting_percent=0.04
):
    pacing_len_list = []
    starting_percent *= data_len
    multiplier = starting_percent
    while multiplier < data_len:
        multiplier = multiplier * increase_amount
        if int(multiplier) < data_len:
            if int(multiplier) >= batch_size:
                pacing_len_list.append(int(multiplier))
            else:
                pacing_len_list.append(batch_size)
        else:
            pacing_len_list.append(data_len)
    return normalize(pacing_len_list, batch_size)


def parabolic_pacing_normalized(data_len, batch_size, power=2, starting_percent=0.04):
    pacing_len_list = []
    starting_percent *= data_len
    for i in range(int(data_len / starting_percent + 1)):
        if int(starting_percent * i ** power) < data_len:
            if int(starting_percent * i ** power) >= batch_size:
                pacing_len_list.append(int(starting_percent * i ** power))
            else:
                pacing_len_list.append(batch_size)
        else:
            pacing_len_list.append(data_len)
            break

    return normalize(pacing_len_list, batch_size)[1:]


def normalize(liste, batch_size):
    liste = np.array(liste) / batch_size

    k = []

    for i in range(len(liste)):
        k.append(liste[i] - sum(liste[i - 1 : i]))
    return k


a1 = parabolic_pacing(100, 1,  starting_percent = 0.10)[0]

print(a1)

a1_2 = []

for i in a1:
    for _ in range(8):
        a1_2.append(i)

a2 = exponential_pacing(100, 1,  starting_percent = 0.10)[0]

a2.append(100)

a2_2 = []

for i in a2:
    for _ in range(5):
        a2_2.append(i)

print(a2)

import matplotlib.pyplot as plt
import numpy as np

# xpoints = np.array([1, 8])
# ypoints = np.array([3, 10])

plt.plot(a1_2)
plt.plot(a2_2)
plt.show()

# a3 = exponential_pacing_normalized(10000, 32, starting_percent = 0.04)

# print(a3)

# a4 = parabolic_pacing_normalized(10000, 32, starting_percent = 0.04)

# print(a4)

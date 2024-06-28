import numpy as np
import time
import matplotlib.pyplot as plt

# x = np.linspace(0, 10, 100)
# y = np.cos(x)

# plt.ion()

# figure, ax = plt.subplots(figsize=(8,6))
# line1, = ax.plot(x, y)

# plt.title("Dynamic Plot of sinx",fontsize=25)

# plt.xlabel("X",fontsize=18)
# plt.ylabel("sinX",fontsize=18)

# for p in range(100):
#     updated_y = np.cos(x-0.05*p)
    
#     line1.set_xdata(x)
#     line1.set_ydata(updated_y)
    
#     figure.canvas.draw()
    
#     figure.canvas.flush_events()
#     time.sleep(0.1)

# l1 = [1,2,3,4]
# l2 = [1,2,3,4]

l1 = []
l2 = []

# # plt.plot(l1,l2)
# # plt.show()

# for i in range(10):
#   l1.append(i)
#   l2.append(i)
#   print(l1,l2)
#   plt.plot(l1,l2)
#   plt.show()
#   time.sleep(1)

import matplotlib.pyplot as plt
import numpy as np

plt.ion()
for i in range(50):
    l1.append(i)
    l2.append(i)
    # print(l1,l2)
    plt.plot(l1,l2)
    plt.draw()
    plt.pause(1)
    plt.clf()
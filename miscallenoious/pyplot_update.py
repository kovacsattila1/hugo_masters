# # # # importing libraries
# # # import numpy as np
# # # import time
# # # import matplotlib.pyplot as plt
 
# # # # creating initial data values
# # # # of x and y
# # # x = np.linspace(0, 10, 100)
# # # y = np.sin(x)
 
# # # # to run GUI event loop
# # # plt.ion()
 
# # # # here we are creating sub plots
# # # figure, ax = plt.subplots(figsize=(10, 8))
# # # line1, = ax.plot(x, y)
 
# # # # setting title
# # # plt.title("Geeks For Geeks", fontsize=20)
 
# # # # setting x-axis label and y-axis label
# # # plt.xlabel("X-axis")
# # # plt.ylabel("Y-axis")
 
# # # # Loop
# # # for _ in range(50):
# # #     # creating new Y values
# # #     new_y = np.sin(x-0.5*_)
 
# # #     # updating data values
# # #     line1.set_xdata(x)
# # #     line1.set_ydata(new_y)
 
# # #     # drawing updated values
# # #     figure.canvas.draw()
 
# # #     # This will run the GUI event
# # #     # loop until all UI events
# # #     # currently waiting have been processed
# # #     figure.canvas.flush_events()
 
# # #     time.sleep(0.1)



# # from matplotlib.animation import FuncAnimation
# # import matplotlib.pyplot as plt
# # import random
 
# # # initial data
# # x = [1]
# # y = [random.randint(1,10)]
 
# # # creating the first plot and frame
# # fig, ax = plt.subplots(2,2)
# # graph1 = ax[0,0].plot(x,y,color = 'g')[0]
# # graph2 = ax[0,1].plot(x,y,color = 'g')[0]
# # graph3 = ax[1,0].plot(x,y,color = 'g')[0]
# # graph4 = ax[1,1].plot(x,y,color = 'g')[0]
# # plt.ylim(0,10)
 
 
# # # updates the data and graph
# # def update(frame):
# #     # print(frame)
# #     global graph
 
# #     # updating the data
# #     x.append(x[-1] + 1)
# #     y.append(random.randint(1,10))
 
# #     # creating a new graph or updating the graph
# #     graph1.set_xdata(x)
# #     graph1.set_ydata(y)

# #     graph2.set_xdata(x)
# #     graph2.set_ydata(y)

# #     graph3.set_xdata(x)
# #     graph3.set_ydata(y)

# #     graph4.set_xdata(x)
# #     graph4.set_ydata(y)

# #     plt.xlim(x[0], x[-1])
 
# # anim = FuncAnimation(fig, update, frames = None, interval=100)
# # plt.show()





# # import matplotlib.pyplot as plt
# # import numpy as np
# # from matplotlib.animation import FuncAnimation

# # # Create figure and subplots
# # fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# # # Initialize data
# # x = np.linspace(0, 2 * np.pi, 100)
# # y1 = np.sin(x)
# # y2 = np.cos(x)
# # y3 = np.sin(x + np.pi / 2)
# # y4 = np.cos(x + np.pi / 2)

# # # Create initial plots
# # lines = []
# # lines.append(axs[0, 0].plot(x, y1, color='b')[0])
# # lines.append(axs[0, 1].plot(x, y2, color='r')[0])
# # lines.append(axs[1, 0].plot(x, y3, color='g')[0])
# # lines.append(axs[1, 1].plot(x, y4, color='m')[0])

# # # Set titles for subplots
# # axs[0, 0].set_title('Sine Wave')
# # axs[0, 1].set_title('Cosine Wave')
# # axs[1, 0].set_title('Sine Wave (Phase Shift)')
# # axs[1, 1].set_title('Cosine Wave (Phase Shift)')

# # # Set labels for subplots
# # for ax in axs.flat:
# #     ax.set_xlabel('x')
# #     ax.set_ylabel('y')

# # # Animation update function
# # def update(frame):
# #     y1 = np.sin(x + 0.1 * frame)
# #     y2 = np.cos(x + 0.1 * frame)
# #     y3 = np.sin(x + 0.1 * frame + np.pi / 2)
# #     y4 = np.cos(x + 0.1 * frame + np.pi / 2)

# #     lines[0].set_ydata(y1)
# #     lines[1].set_ydata(y2)
# #     lines[2].set_ydata(y3)
# #     lines[3].set_ydata(y4)

# #     return lines

# # # Create animation
# # ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

# # # Show plot
# # plt.tight_layout()
# # plt.show()




# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.animation import FuncAnimation

# # Create figure
# fig = plt.figure(figsize=(12, 12))

# # Define grid spec
# gs = fig.add_gridspec(3, 3)

# # Create subplots
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[0, 2])
# ax4 = fig.add_subplot(gs[1, 0])
# ax5 = fig.add_subplot(gs[1, 1])
# ax6 = fig.add_subplot(gs[1, 2])
# ax7 = fig.add_subplot(gs[2, :])

# # Initialize data
# x = np.linspace(0, 2 * np.pi, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.sin(x + np.pi / 2)
# y4 = np.cos(x + np.pi / 2)
# y5 = np.sin(x + np.pi)
# y6 = np.cos(x + np.pi)
# y7 = np.sin(x) * np.cos(x)

# # Create initial plots
# lines = []
# lines.append(ax1.plot(x, y1, color='b')[0])
# lines.append(ax2.plot(x, y2, color='r')[0])
# lines.append(ax3.plot(x, y3, color='g')[0])
# lines.append(ax4.plot(x, y4, color='m')[0])
# lines.append(ax5.plot(x, y5, color='c')[0])
# lines.append(ax6.plot(x, y6, color='y')[0])
# lines.append(ax7.plot(x, y7, color='k')[0])

# # Set titles for subplots
# ax1.set_title('Sine Wave')
# ax2.set_title('Cosine Wave')
# ax3.set_title('Sine Wave (Phase Shift)')
# ax4.set_title('Cosine Wave (Phase Shift)')
# ax5.set_title('Sine Wave (Phase Shift 2)')
# ax6.set_title('Cosine Wave (Phase Shift 2)')
# ax7.set_title('Sine * Cosine Wave')

# # Set labels for subplots
# for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')

# # Animation update function
# def update(frame):
#     y1 = np.sin(x + 0.1 * frame)
#     y2 = np.cos(x + 0.1 * frame)
#     y3 = np.sin(x + 0.1 * frame + np.pi / 2)
#     y4 = np.cos(x + 0.1 * frame + np.pi / 2)
#     y5 = np.sin(x + 0.1 * frame + np.pi)
#     y6 = np.cos(x + 0.1 * frame + np.pi)
#     y7 = np.sin(x + 0.1 * frame) * np.cos(x + 0.1 * frame)

#     lines[0].set_ydata(y1)
#     lines[1].set_ydata(y2)
#     lines[2].set_ydata(y3)
#     lines[3].set_ydata(y4)
#     lines[4].set_ydata(y5)
#     lines[5].set_ydata(y6)
#     lines[6].set_ydata(y7)

#     return lines
# print("mydebug1")

# # Create animation
# ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)

# print("mydebug2")

# # Show plot
# plt.tight_layout()
# plt.show()

# print("mydebug3")





# import matplotlib.pyplot as plt 
# import numpy as np 
  
# x = np.linspace(0, 10*np.pi, 100) 
# y = np.sin(x) 
  
# plt.ion() 
# fig = plt.figure() 
# ax = fig.add_subplot(111) 
# line1, = ax.plot(x, y, 'b-') 
  
# for phase in np.linspace(0, 10*np.pi, 100): 
#     line1.set_ydata(np.sin(0.5 * x + phase)) 
#     fig.canvas.draw() 
#     fig.canvas.flush_events() 


import matplotlib.pyplot as plt 
import numpy as np 
  
x =[0]
y =[0]
  
plt.ion() 

fig = plt.figure() 
ax = fig.add_subplot(111) 
line1, = ax.plot(x, y, 'b-') 
  
for phase in np.linspace(0, 10*np.pi, 500): 
    line1.set_ydata(y)
    line1.set_xdata(x)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 100)
    fig.canvas.draw() 
    fig.canvas.flush_events() 
    # plt.cla()
    x.append(phase)
    y.append(phase)
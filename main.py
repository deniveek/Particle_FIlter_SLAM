import sys
import time
import numpy as np
import RDP
import matplotlib.pyplot as plt
import sim

print('Программа начала работу')
sim.simxFinish(-1) # если имелась кака-либо незавершенная сессия, она будет завершена
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) # соединение с портом
if clientID != -1:
    print('Server is connected!')
else:
    print('Server is unreachable!')
    sys.exit(0)

errorCode, ranges = sim.simxGetStringSignal(clientID, 'scan ranges', sim.simx_opmode_streaming)
time.sleep(0.5)

errorCode, ranges = sim.simxGetStringSignal(clientID, 'scan ranges', sim.simx_opmode_buffer)

ranges = sim.simxUnpackFloats(ranges)
ranges = np.array(ranges).reshape(len(ranges)//2, 2)
objects = []

#new = np.array([[pair[1]*np.sin(pair[0])*-1, pair[1]*np.cos(pair[0])] for pair in ranges])
print(ranges)
new_ = RDP.DouglasPeucker(ranges)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_aspect('equal','datalim', 'C')
tmp = new_.transpose()
ax1.plot(-tmp[1], tmp[0], 'o')

plt.show()


print('done')

from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import sys
import sim
import time
import RDP



class LIDAR(QtWidgets.QWidget):
    default_speed = 3.0
    def __init__(self):
        super(LIDAR, self).__init__()
        self.setGeometry(300, 300,300,100)
        self.setWindowTitle('rangefinder')
        self.left_speed = 0
        self.right_speed = 0
        sim.simxFinish(-1)  # если имелась кака-либо незавершенная сессия, она будет завершена
        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # соединение с портом
        if self.clientID != -1:
            print('Server is connected!')
        else:
            print('Server is unreachable!')
            sys.exit(0)

        self.errorCode, self.points = sim.simxGetStringSignal(self.clientID, 'scan ranges', sim.simx_opmode_streaming)
        time.sleep(0.5)
        code, self.left_wheel = sim.simxGetObjectHandle(self.clientID,'Left_motor',sim.simx_opmode_blocking)
        print(f'code: {code}, left: {self.left_wheel}')
        code, self.right_wheel = sim.simxGetObjectHandle(self.clientID,'Right_motor',sim.simx_opmode_blocking)
        print(f'code: {code}, right: {self.right_wheel}')
        self.clusters = []

        self.startTimer(200)
        print('init_done')


    def send_speed(self):
        if abs(self.right_speed)<0.0001:
            self.right_speed = 0.0001
        if abs(self.left_speed)<0.0001:
            self.left_speed = 0.0001

        sim.simxSetJointTargetVelocity(self.clientID, self.left_wheel, self.left_speed, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(self.clientID, self.right_wheel, self.right_speed, sim.simx_opmode_oneshot)

    def keyReleaseEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key_Up:
            self.left_speed -= self.default_speed
            self.right_speed -= self.default_speed
        elif a0.key() == QtCore.Qt.Key_Down:
            self.left_speed += self.default_speed
            self.right_speed += self.default_speed
        elif a0.key() == QtCore.Qt.Key_Left:
            self.left_speed += 1
            self.right_speed -= 1
        elif a0.key() == QtCore.Qt.Key_Right:
            self.left_speed -= 1
            self.right_speed += 1

        self.send_speed()

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key_Up:
            self.left_speed = self.default_speed
            self.right_speed = self.default_speed
        elif a0.key() == QtCore.Qt.Key_Down:
            self.left_speed = -self.default_speed
            self.right_speed = -self.default_speed
        elif a0.key() == QtCore.Qt.Key_Left:
            self.left_speed -= 1
            self.right_speed += 1
        elif a0.key() == QtCore.Qt.Key_Right:
            self.left_speed += 1
            self.right_speed -= 1

        self.send_speed()

    def timerEvent(self, a0: 'QTimerEvent') -> None:
        self.errorCode, self.points = sim.simxGetStringSignal(self.clientID, 'scan ranges', sim.simx_opmode_buffer)

        points_ = sim.simxUnpackFloats(self.points)
        points_ = np.array(points_).reshape(len(points_) // 3,3 )
        points_ = np.transpose(points_.transpose()[:-1])


        self.clusters = RDP.clust_then_RDP(points_)

        self.update()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        qp = QtGui.QPainter(self)
        qp.translate(self.width()/2, self.height()/2)
        cx = (self.width() - 20)/7
        cy = (self.height() - 20)/7
        if cy > cx:
            cy = cx
        else:
            cx = cy
        for cluster in self.clusters:
            if len(cluster) < 2:
                print('no polyline')
            points = [QtCore.QPoint(int(-pair[1]*cx), int(-pair[0]*cy)) for pair in cluster]
            qp.drawPolyline(QtGui.QPolygon(points))
        qp.setPen(QtGui.QPen(QtCore.Qt.blue, 5, QtCore.Qt.SolidLine))
        qp.drawRect(-5,-2, 10,15)
        qp.end()


def window():
    app = QtWidgets.QApplication(sys.argv)
    win = LIDAR()
    win.show()
    sys.exit(app.exec_())


window()

from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import sys
import sim
import time
import RDP


class Connection:
    clientID = 0
    @classmethod
    def connect(cls):
        sim.simxFinish(-1)  # если имелась кака-либо незавершенная сессия, она будет завершена
        cls.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # соединение с портом
        if cls.clientID != -1:
            print('Server is connected!')
        else:
            print('Server is unreachable!')
            sys.exit(0)

class Odometry_view(QtWidgets.QWidget):
    def __init__(self):
        self.path  = []
        self.path.append((0,0))
        self.theta=0
        super(Odometry_view, self).__init__()
        self.setGeometry(300, 300,300,100)
        self.setWindowTitle('odometry')
        self.startTimer(300)

    def timerEvent(self, a0: 'QTimerEvent') -> None:
        code, t, tt, ttt, tttt = sim.simxCallScriptFunction(Connection.clientID, "Chassis", sim.sim_scripttype_childscript,'getBuffer',[],[],[],bytearray(0), sim.simx_opmode_blocking)
        self.update_path(tt)
        self.update()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        qp = QtGui.QPainter(self)
        qp.translate(self.width()/2, self.height()/2)
        cx = (self.width() - 20) / 20
        cy = (self.height() - 20) / 20
        if cy > cx:
            cy = cx
        else:
            cx = cy
        points = [QtCore.QPoint(int(pair[0] * cx), int(-pair[1] * cy)) for pair in self.path]
        qp.drawPolyline(QtGui.QPolygon(points))
        qp.end()

    def update_path(self, array, width=0.095):
        array_ = np.array(array).reshape(len(array)//3, 3)
        for timestamp in array_:
            if abs(timestamp[1]) > 0.001 and abs(timestamp[2])>0.001:
                time = timestamp[0]
                d_left = timestamp[1] * 0.04
                d_right = timestamp[2] * 0.04
                length = (d_right + d_left) / 2
                alpha = (d_left - d_right) / width if abs(d_right - d_left)>0.01 else 0
                self.theta = self.theta+alpha%(2*np.pi)
                next_pt = np.matmul(np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]]), [0, length])
                new_pt = (self.path[-1][0] + next_pt[0], self.path[-1][1] + next_pt[1])
                self.path.append(new_pt)



class LIDAR_view(QtWidgets.QWidget):
    default_speed = 3.0
    def __init__(self):
        super(LIDAR_view, self).__init__()
        self.setGeometry(300, 300,300,100)
        self.setWindowTitle('rangefinder')
        self.left_speed = 0
        self.right_speed = 0


        self.errorCode, self.points = sim.simxGetStringSignal(Connection.clientID, 'scan ranges', sim.simx_opmode_streaming)
        time.sleep(0.5)
        code, self.left_wheel = sim.simxGetObjectHandle(Connection.clientID,'Left_motor',sim.simx_opmode_blocking)
        print(f'code: {code}, left: {self.left_wheel}')
        code, self.right_wheel = sim.simxGetObjectHandle(Connection.clientID,'Right_motor',sim.simx_opmode_blocking)
        print(f'code: {code}, right: {self.right_wheel}')
        self.clusters = []

        self.startTimer(200)
        print('init_done')


    def send_speed(self):
        if abs(self.right_speed)<0.0001:
            self.right_speed = 0.0001
        if abs(self.left_speed)<0.0001:
            self.left_speed = 0.0001

        sim.simxSetJointTargetVelocity(Connection.clientID, self.left_wheel, self.left_speed, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(Connection.clientID, self.right_wheel, self.right_speed, sim.simx_opmode_oneshot)

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
        self.errorCode, self.points = sim.simxGetStringSignal(Connection.clientID, 'scan ranges', sim.simx_opmode_buffer)

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
    win1 = LIDAR_view()
    win1.show()
    win2 = Odometry_view()
    win2.show()
    sys.exit(app.exec_())


Connection.connect()
window()

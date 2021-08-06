import numpy as np
from scipy import stats
import RDP
import random
from copy import copy

import cv2 as cv


class Landmark:
    def __init__(self, pos: np.ndarray, dist: float, times_observed=0):
        self.pos = pos
        self.dist = dist
        self.neighbors = []
        self.times_observed = times_observed

    def __getitem__(self, item):
        return self.pos[item]

    def __lt__(self, other):
        return self.dist < other.dist


class Features(dict):
    def __init__(self, precision=1):
        super(Features, self).__init__()
        self.prc = precision

    def __getitem__(self, item):
        if isinstance(item, Landmark):
            return super().__getitem__((round(item.pos[0]*self.prc, 1)/self.prc, round(item.pos[1]*self.prc, 1)/self.prc))
        else:
            return super().__getitem__((round(item[0] * self.prc, 1) / self.prc, round(item[1] * self.prc, 1) / self.prc))

    def __contains__(self, item):
        if isinstance(item, Landmark):
            return super().__contains__((round(item.pos[0] * self.prc, 1) / self.prc, round(item.pos[1] * self.prc, 1) / self.prc))
        else:
            return super().__contains__((round(item[0] * self.prc, 1) / self.prc, round(item[1] * self.prc, 1) / self.prc))

    def append(self, item):
        super().__setitem__((round(item.pos[0]*self.prc, 1)/self.prc, round(item.pos[1]*self.prc, 1)/self.prc), item)


class Map:
    def __init__(self):
        self.edges = np.array([])  # геометрия на карте
        self.features = Features()
        self.features_prov = Features()
        self.pos = [0, 0]
        self.once = True

    def update(self, edges=None, landmarks=None):
        if edges:
            self.edges = edges
        if landmarks and self.once:
            print('got_features')
            self.features = landmarks
            self.once = False

    def update_features(self, features):
        for f in features:
            if f in self.features_prov:
                self.features_prov[f].times_observed += 1
            else:
                self.features_prov.append(f)

        for f in self.features_prov.values():
            if f not in self.features and f.times_observed >= 10:
                self.features.append(f)

    def paint_map(self, img):
        x_size, y_size = (np.size(img, axis=0), np.size(img, axis=1))
        for edge in self.edges:
            pt_0 = (x_size // 2 + int(-y_size * edge[0][0] / 8), y_size // 2 + int(-y_size * edge[0][1] / 8))
            pt_1 = (x_size // 2 + int(-y_size * edge[1][0] / 8), y_size // 2 + int(-y_size * edge[1][1] / 8))
            cv.line(img, pt_0, pt_1, 100, 2)

    def paint_landmarks(self, img):
        x_size, y_size = (np.size(img, axis=0), np.size(img, axis=1))
        center = (x_size // 2 + int(-y_size * 0 / 8), y_size // 2 + int(-y_size * 0 / 8))
        cv.circle(img, center, 5, 255, 1)
        for pt in self.features:
            center = (x_size // 2 + int(-y_size * pt[0] / 8), y_size // 2 + int(-y_size * pt[1] / 8))
            cv.circle(img, center, 5, 255, 1)


class Robot:
    def __init__(self, zero=(0,0), theta=0, scan=None, scan_angle=4/3*np.pi, vector=np.array([0.,1.])):
        self.pos = np.array(zero)
        self.theta = theta
        self.vector = Landmark(vector, 1.0)
        self.landmarks = []
        self.scan_angle = scan_angle
        self.Map = Map()
        self.particles = self.spawn_particles()
        self.vision = []
        self.estimation = 0
        self.ctr = 0

    def spawn_particles(self, p_num=50):
        rng = np.random.default_rng(11111)
        return [Particle(pos=np.array([self.pos[0]+random.uniform(-0.1, 0.1), self.pos[0]+random.uniform(-0.1, 0.1)]),
                         theta=random.uniform(-0.1, 0.1), weight=1/p_num) for i in range(p_num)]

    def update(self, pt, theta, scan):
        points = [Landmark(Robot.pol2cart(scan[i], -self.scan_angle/2 + self.scan_angle * i / len(scan)), scan[i]) for i in range(len(scan))]
        pts = copy(points)
        pts = RDP.DouglasPeucker(pts)
        self.landmarks = self.filter_landmarks(pts)
        if np.linalg.norm(pt) > 0.005 or abs(theta) > 0.01:
            self.ctr += 1
            for prtcl in self.particles:
                prtcl.move(pt, theta)
                if len(self.Map.features) > 0:
                    prtcl.look(self.landmarks, self.Map.features)
            if len(self.Map.features) > 0:
                if self.ctr > 0:
                    self.particles = Particle.resample(self.particles)
                    self.ctr = 0
                self.estimation = Particle.estimate(self.particles)
                self.pos = self.estimation.pos
                self.theta = self.estimation.theta
            else:
                self.theta += theta
                self.pos = Robot.trans_mat(self.pos, pt, self.theta)
        #edges_ = map(lambda edge: (Robot.trans_mat(self.pos, edge[0].pos, self.theta), Robot.trans_mat(self.pos, edge[1].pos, self.theta)), edges)
        for lm_ in self.landmarks:
            lm_.pos = Robot.trans_mat(self.pos, lm_.pos, self.theta)
        self.Map.update_features(self.landmarks)
        self.vision = [Robot.trans_mat(self.pos, lm.pos, self.theta) for lm in points]

    def update_path(self, array, width=0.175):
        array_ = np.array(array).reshape(len(array)//3, 3)
        theta = 0
        pt = np.array([0, 0])
        for timestamp in array_:
            if abs(timestamp[1]) > 0.001 and abs(timestamp[2])>0.001:
                time = timestamp[0]
                d_left = timestamp[1] * 0.04
                d_right = timestamp[2] * 0.04
                length_ = (d_right + d_left) / 2
                theta = theta + (d_left - d_right) / width if abs(d_right - d_left)>0.001 else 0
                next_pt = np.matmul(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), [0, length_])
                pt = pt + next_pt
        return pt, theta

    def filter_landmarks(self, points_):
        points = [pt for pt in points_ if pt.dist > 0.05]
        points.sort(key=lambda pt: pt.dist)
        thresh_0, thresh_1 = 0.49 * self.scan_angle, self.scan_angle / 300
        res = []
        for i in range(np.size(points, axis=0)):
            not_shadow = True
            if Robot.lm_angle(points[i], self.vector) < thresh_0:
                for point in points[:i]:
                    if Robot.lm_angle(points[i], point) < thresh_1 or np.linalg.norm(point.pos - points[i].pos) < points[i].dist * 0.05:
                        not_shadow = False
                        break
                if not_shadow:
                    res.append(points[i])
        return res

    @staticmethod
    def pol2cart(r, theta):
        return np.array([r * np.sin(theta), r * np.cos(theta)])

    @staticmethod
    def lm_angle(pt_1: Landmark, pt_0: Landmark):
        return np.arccos((pt_0.pos @ pt_1.pos) / (pt_0.dist * pt_1.dist))

    @staticmethod
    def trans_mat(origin, point, theta):
        return origin + np.matmul([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], point)


class Particle:
    def __init__(self, pos=np.array([0, 0]), theta=0.0, weight=1.0):
        self.pos = pos
        self.theta = theta
        self.weight = weight

    def move(self, pt, alpha):
        self.theta += np.random.normal(alpha, 0.05 + abs(alpha) / 5)
        self.pos = Robot.trans_mat(self.pos, [pt[0]+np.random.normal(0, 0.05), pt[1]+np.random.normal(0, 0.05)], self.theta)

    def look(self, landmarks, features, m_err=0.05):
        #print(f'{landmarks}')
        vision = map(lambda lm: Robot.trans_mat(self.pos, lm.pos, self.theta), landmarks)
        p=1
        for lm in vision:
            if lm in features:
                err = np.linalg.norm(features[lm].pos - lm)
                p = p * stats.norm(scale=m_err).pdf(err)
        self.weight = self.weight*p

    @classmethod
    def resample(cls, particles):
        w_sum = sum([particle.weight for particle in particles])
        for particle in particles:
            particle.weight = particle.weight/w_sum

        w_max = max([particle.weight for particle in particles])

        N = len(particles)
        index = random.randint(0, N-1)
        betta = 0
        res = []
        for i in range(N):
            betta = betta + random.uniform(0, 2 * w_max)
            while betta > particles[index].weight:
                betta = betta - particles[index].weight
                index = (index + 1) % N  # индекс изменяется в цикле от 0 до N
            res.append(copy(particles[index]))
        return res

    @classmethod
    def estimate(cls, particles):
        w_sum = sum([particle.weight for particle in particles])
        res = Particle()
        for particle in particles:
            particle.weight = particle.weight/w_sum
            res.theta += particle.theta*particle.weight
            res.pos = res.pos + particle.pos*particle.weight

        return res


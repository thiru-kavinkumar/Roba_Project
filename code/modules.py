import pygame
import numpy as np
from numpy.random import randn, random, uniform
import scipy.stats
import scipy.stats
from PIL import Image

# Class :
# -------


class Environment:
    def __init__(self):
        # Map initialization
        self.importMap()
        self.setMaster()
        self.setRobots()

    def setRobots(self):
        # Robot initialization
        self.main_robot_color = (0, 255, 0)
        self.sim_robot_color = (0, 0, 255)

        self.transValue = 10  # Forward robot step
        self.rotValue = 10  # Robot turning angle (deg)
        self.motion_std = (2, 2)  # (rot, trans)

        self.robot_x_range = (0, self.mapArr.shape[1])  # Map x range
        self.robot_y_range = (0, self.mapArr.shape[0])  # Map y range
        self.robot_hdg_range = (0, 360)
        self.nb_simulated_robot = 500

        self.robot_step = 0
        self.robot_max_step = 10  # Max step to reach to resample

        # Sensor initialization
        self.sim_robot_sensor_color = (255, 128, 128)
        self.main_robot_sensor_color = (255, 0, 0)
        self.nb_sensor_ray = 5
        self.sensor_length = 200
        self.measure_std_error = 10

        # Main robot
        self.main_robot = Robot(
            measure_std_error=self.measure_std_error, env=self, N=1, nb_sensor_ray=self.nb_sensor_ray, sensor_length=self.sensor_length)
        self.main_robot.create_uniform_particles(x_range=self.robot_x_range, y_range=self.robot_y_range,
                                                 hdg_range=self.robot_hdg_range)

        # Simulated robot
        self.sim_robot = Robot(
            measure_std_error=self.measure_std_error, env=self, N=self.nb_simulated_robot, nb_sensor_ray=self.nb_sensor_ray, sensor_length=self.sensor_length)
        self.sim_robot.create_uniform_particles(x_range=self.robot_x_range, y_range=self.robot_y_range,
                                                hdg_range=self.robot_hdg_range)

    def importMap(self, path="map/default.jpg"):
        """Import map image and map array"""
        image = Image.open(path)
        self.mapArr = np.array(image)[:, :, 0]
        self.masterBg = pygame.image.load(path)

    def setMaster(self):
        """Set main window"""
        self.masterRunning = True
        self.masterTitle = "Robot Map"
        self.masterDim = (self.mapArr.shape[1], self.mapArr.shape[0])
        self.masterFPS = 5000

        self.master = pygame.display.set_mode(
            self.masterDim)  # Set master's dimension
        pygame.display.set_caption(self.masterTitle)  # Set master's title
        self.clock = pygame.time.Clock()
        self.clock.tick(self.masterFPS)  # Set master's FPS

        self.master.blit(self.masterBg, (0, 0))  # Dispkay background image

    def move(self):
        """Robots motion"""
        event = pygame.key.get_pressed()
        update = False

        # Go forward
        if event[pygame.K_UP] == True:
            self.main_robot.predictTrans(self.transValue, self.motion_std[1])
            self.sim_robot.predictTrans(self.transValue, self.motion_std[1])
            update = True
        # Turn anticlokwise
        elif event[pygame.K_RIGHT] == True:
            self.main_robot.predictRot(self.rotValue, self.motion_std[0])
            self.sim_robot.predictRot(self.rotValue, self.motion_std[0])
            update = True
        # Turn clockwise
        elif event[pygame.K_LEFT] == True:
            self.main_robot.predictRot(-self.rotValue, self.motion_std[0])
            self.sim_robot.predictRot(-self.rotValue, self.motion_std[0])
            update = True

        if update:
            self.robot_step += 1

            # Get main robot sensor collision point
            dist_sensor = np.empty((self.nb_sensor_ray, 2))
            for i in range(self.nb_sensor_ray):
                dist_sensor[i, 0] = self.main_robot.sensor_collision[0, 0, i]
                dist_sensor[i, 1] = self.main_robot.sensor_collision[0, 1, i]

            # Compute main robot sensor range with noise
            zs = np.linalg.norm(dist_sensor - self.main_robot.particles[0, :2]) + \
                randn(self.nb_sensor_ray)*self.measure_std_error

            # Update weight
            self.sim_robot.update(z=zs)

            # Reample particles
            if self.robot_step > self.robot_max_step:
                self.robot_step = 0
                self.sim_robot.resample()

    def draw(self):
        """Draw main robot simulated robot and sensor"""
        # Set background
        self.master.blit(self.masterBg, (0, 0))

        # Main robot's sensor
        for ray in range(self.main_robot.nb_sensor_ray):
            pygame.draw.line(self.master, self.main_robot_sensor_color,
                             self.main_robot.particles[0, :2], self.main_robot.sensor_collision[0, :2, ray])

        # # Simulated robot
        for i in range(self.nb_simulated_robot):
            pygame.draw.circle(self.master, self.sim_robot_color,
                               self.sim_robot.particles[i, :2], self.sim_robot.radius)

        # Main robot
        pygame.draw.circle(self.master, self.main_robot_color,
                           self.main_robot.particles[0, :2], self.main_robot.radius)

        # Draw Mean
        mean = self.sim_robot.estimate()
        pygame.draw.circle(self.master, (255, 0, 0),
                           mean, self.sim_robot.radius)


class Robot:
    def __init__(self, measure_std_error, env, N, nb_sensor_ray, sensor_length):
        self.env = env
        self.radius = 6
        self.particles_size = (N, 3)

        self.nb_sensor_ray = nb_sensor_ray
        self.sensor_coef_step = 10  # ray casting
        self.sensor_range = [45, 45]  # left side range, right side rage
        self.R = measure_std_error
        self.sensor_length = sensor_length

        self.particles = np.empty(self.particles_size)  # x, y, heading

        self.weights = np.empty(self.particles_size[0])
        self.weights.fill(1./N)

    def create_uniform_particles(self, x_range, y_range, hdg_range):
        """Initiale position"""
        particles = np.empty(self.particles_size)
        particles[:, 0] = uniform(
            x_range[0], x_range[1], size=self.particles_size[0])
        particles[:, 1] = uniform(
            y_range[0], y_range[1], size=self.particles_size[0])
        particles[:, 2] = uniform(
            hdg_range[0], hdg_range[1], size=self.particles_size[0])
        particles[:, 2] %= 360

        # Chek collision with walls
        isCollisionArr = self.checkCollision(particles)

        for i in range(self.particles_size[0]):
            if isCollisionArr[i]:
                isCollided = True
                while isCollided:
                    particles[i, 0] = uniform(x_range[0], x_range[1], size=1)
                    particles[i, 1] = uniform(y_range[0], y_range[1], size=1)
                    isCollided = self.isCollision(particles[i, :2])
        self.particles = particles

        # Sensor values
        self.getSensorValue()

    def checkCollision(self, particles):
        """ Return array that tell if is collision or not"""
        isCollionArray = np.empty((self.particles_size[0]))
        for i in range(self.particles_size[0]):
            pos = particles[i, :2]
            isCollionArray[i] = self.isCollision(pos)
        return isCollionArray

    def isCollision(self, pos):
        """Retur collision state for one position"""
        top = [int(pos[0]), int(pos[1] - self.radius)]
        right = [int(pos[0] + self.radius), int(pos[1])]
        bot = [int(pos[0]), int(pos[1] + self.radius)]
        left = [int(pos[0] - self.radius), int(pos[1])]

        if self.env.mapArr[top[1], top[0]] == 0 or self.env.mapArr[right[1], right[0]] == 0 or \
                self.env.mapArr[bot[1], bot[0]] == 0 or self.env.mapArr[left[1], left[0]] == 0:
            return True

        return False

    def predictTrans(self, u, std):
        """ move according to control input u (velocity)
        with noise std"""
        particles = np.copy(self.particles)
        d = u + randn(1) * std
        particles[:, 0] += np.cos(np.deg2rad(self.particles[:, 2])) * d
        particles[:, 1] += np.sin(np.deg2rad(self.particles[:, 2])) * d

        # Chek collision with walls
        isCollisionArr = self.checkCollision(particles)

        for i in range(self.particles_size[0]):
            if isCollisionArr[i]:
                isCollided = True
                while isCollided:
                    particles[i, 0] = self.particles[i, 0]
                    particles[i, 1] = self.particles[i, 1]
                    isCollided = self.isCollision(particles[i, :2])
        self.particles = particles

        # Sensor values
        self.getSensorValue()

    def predictRot(self, u, std):
        """ move according to control input u (heading change)
        with noise std"""
        self.particles[:, 2] += u + randn(self.particles_size[0]) * std
        self.particles[:, 2] %= 360

        # Sensor values
        self.getSensorValue()

    def getSensorValue(self):
        """ Get sensor ray collision position"""
        sensor = np.empty((self.particles_size[0], 1))  # [starting range]
        sensor[:, 0] = self.particles[:, 2] - \
            self.sensor_range[0]  # Starting range

        #  Angle between 2 ray
        sensor_delta_angle = (
            self.sensor_range[0] + self.sensor_range[1])/self.nb_sensor_ray

        self.sensor_collision = np.empty(
            (self.particles_size[0], 2, self.nb_sensor_ray)
        )

        for part_index in range(self.particles_size[0]):
            pos = (self.particles[part_index, 0],
                   self.particles[part_index, 1])
            for ray_index in range(self.nb_sensor_ray):
                angle = sensor[part_index] + ray_index * sensor_delta_angle
                x_coef = np.cos(np.deg2rad(angle)) * self.sensor_coef_step
                y_coef = np.sin(np.deg2rad(angle)) * self.sensor_coef_step
                isCollided = False
                x = pos[0]
                y = pos[1]
                dist = 0
                while not isCollided:
                    x += x_coef
                    y += y_coef
                    X = int(x)
                    X = min(X, self.env.mapArr.shape[1]-1)
                    X = max(X, 0)
                    Y = int(y)
                    Y = min(Y, self.env.mapArr.shape[0]-1)
                    Y = max(Y, 0)
                    dist = np.linalg.norm(np.array([X, Y])-np.array(pos))
                    if self.env.mapArr[Y, X] == 0:
                        isCollided = True
                    if dist > self.sensor_length:
                        isCollided = True
                X = int(X-x_coef)
                Y = int(Y-y_coef)
                self.sensor_collision[part_index, 0, ray_index] = X
                self.sensor_collision[part_index, 1, ray_index] = Y

    def update(self, z):
        """Update robot's weight """
        for i in range(self.nb_sensor_ray):
            distance = np.linalg.norm(
                self.sensor_collision[:, :2, i] - self.particles[:, :2], axis=1)
            self.weights *= scipy.stats.norm(distance, self.R).cdf(z[i])

        # Avoid Division by Zero
        self.weights += 1.e-300
        self.weights /= sum(self.weights)  # normalization

    def resample(self):
        """Resample particles"""
        cumulative_sum = np.cumsum(self.weights)

        cumulative_sum[-1] = 1.  # avoid round-off error
        indexes = np.searchsorted(
            cumulative_sum, uniform(0, 1, self.particles_size[0]))

        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1.0/self.particles_size[0])
        # self.weights = self.weights[indexes]
        # self.weights /= np.sum(self.weights)  # normalization

    def estimate(self):
        """ returns mean and variance """
        pos = self.particles[:, 0:2]
        mu = np.average(pos, weights=self.weights, axis=0)

        return mu

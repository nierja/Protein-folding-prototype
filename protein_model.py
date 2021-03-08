from math import pi, sin, cos, sqrt
import numpy as np
import random
import subprocess
import statistics


""" ============================================= different potentials =================================================
I.      U(r) = e * (1 - (r / s)) * ((r / s) - (c / s)) ^ 2 / ((r / s) + (r / s) ^ 9)
        dU(r)/dr = (e * s**6 * (r - c) * (6 * r**10 + (-7 * s - 8 * c) * r**9 + 9 * c * s * r**8 -
                    2 * s**8 * r**2 + s**9 * r + c * s**9)) / (r**2 * (r**8 + s**8)**2)

II.     U(r) = e * ((r - c) / s) ^ 2 * ((r + c / 2) / s)
        dU(r)/dr = (3 * e * r * (r - c)) / s**3

III.    LJ_U(r) = 4 * e * (r ** -12 - r ** -6)
        dU(r)/dr = - 24 *(e / r) * (2 * (s / r)**12-(s / r)**6)
"""


""" =============================================== vector operations ============================================== """


def subtract_vectors(vector_A, vector_B):
    return [vector_B[0] - vector_A[0], vector_B[1] - vector_A[1], vector_B[2] - vector_A[2]]


def dot_product(vector_A, vector_B):
    return vector_A[0] * vector_B[0] + vector_A[1] * vector_B[1] + vector_A[2] * vector_B[2]


def cross_product(vector_A, vector_B):
    product = [vector_A[1] * vector_B[2] - vector_A[2] * vector_B[1],  # x
               vector_A[2] * vector_B[0] - vector_A[0] * vector_B[2],  # y
               vector_A[0] * vector_B[1] - vector_A[1] * vector_B[0]]  # z
    return product


def norm_vector(vector_A):
    norma = sqrt(vector_A[0] ** 2 + vector_A[1] ** 2 + vector_A[2] ** 2)
    for i in range(3):
        vector_A[i] /= norma
    return vector_A


""" ==================================================== functions ================================================= """
def block_method(E):
    # computes yerr of list E, using block method
    yerr = statistics.variance(E)
    yerr = yerr / len(E)

    while len(E) > 2:
        new_length = len(E) // 2
        new_E = []
        for i in range(new_length):  # blocking operation
            new_E.append((E[2 * i] + E[2 * i + 1]) / 2)

        new_yerr = statistics.variance(new_E) / (new_length - 1)
        if new_yerr - yerr < sqrt(2 / (new_length - 1)) * new_yerr:
            break
        E = new_E
        yerr = new_yerr

    return yerr




""" ===================================================== classes ================================================== """
class grain:
    """class for storing position, velocity and acceleration of an individual grain"""

    def __init__(self):
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
        self.acc_x = 0
        self.acc_y = 0
        self.acc_z = 0

    def __str__(self):
        return "xyz: {} {} {}, vel_x,y,z: {} {} {}, acc_x,y,z: {} {} {} ".format(self.pos_x, self.pos_y, self.pos_z,
                                                                                 self.vel_x, self.vel_y, self.vel.z,
                                                                                 self.acc_x, self.acc_y, self.acc_z)


class Protein:

    def __init__(self, sequence):
        self.N = len(sequence)
        self.sequence = sequence

        grains = [grain() for i in range(self.N)]
        self.grains = np.array(grains)

        angles = []
        for i in range(2 * self.N - 5):
            angles.append((2 * pi) / 3)

        #x,y,z coordinates are calculated for each grain in grains
        self.xyz_coordinates(angles)


        for i in range(self.N-1):
            print(i, ": x,y,z->", self.grains[i].pos_x, self.grains[i].pos_y, self.grains[i].pos_z, "dist i -> i+1: ",
                  self.distance(i, i+1))
        return


    def distance(self, grain_i, grain_j):
        #returns distance between grain_i, grain_j
        return sqrt((self.grains[grain_i].pos_x - self.grains[grain_j].pos_x) ** 2
                  + (self.grains[grain_i].pos_y - self.grains[grain_j].pos_y) ** 2
                  + (self.grains[grain_i].pos_z - self.grains[grain_j].pos_z) ** 2)


    def generate_velocities(self):
        # initial velocities are generated from normal distribution
        for i in range(self.N):
            self.grains[i].vel_x = sqrt(kB_T / m_grain) * random.gauss(0, 1)
            self.grains[i].vel_y = sqrt(kB_T / m_grain) * random.gauss(0, 1)
            self.grains[i].vel_z = sqrt(kB_T / m_grain) * random.gauss(0, 1)
        return


    def Andersen(self):
        if random.random() < 0.5:
            i = random.randint(0, self.N-1)
            self.grains[i].vel_x = sqrt(kB_T / m_grain) * random.gauss(0, 1)
            self.grains[i].vel_y = sqrt(kB_T / m_grain) * random.gauss(0, 1)
            self.grains[i].vel_z = sqrt(kB_T / m_grain) * random.gauss(0, 1)
        return

    def xyz_coordinates(self, angle_array):
        # generate xyz protein coordinates, # updates self.grains[all_grains],pos_x/y/z with values from
        # arrays x, y, z and returns those arrays

        x, y, z = [0, 0, cos(angle_array[0])], [0, 1, 1+sin(angle_array[0])], [0, 0, 0]
        for i in range(self.N - 3):
            a = [x[i], y[i], z[i]]
            b = [x[i + 1], y[i + 1], z[i + 1]]
            c = [x[i + 2], y[i + 2], z[i + 2]]

            ab = subtract_vectors(b, a)
            ab = norm_vector(ab)
            bc = subtract_vectors(c, a)
            bc = norm_vector(bc)
            nr = cross_product(ab, bc)
            nr = norm_vector(nr)
            ns = cross_product(nr, bc)

            X = -cos(angle_array[i + 1])
            Y = sin(angle_array[i + 1]) * cos(angle_array[self.N - 2 + i])
            Z = sin(angle_array[i + 1]) * sin(angle_array[self.N - 2 + i])

            x.append((bc[0] * X + ns[0] * Y + nr[0] * Z) + x[i + 2])
            y.append((bc[1] * X + ns[1] * Y + nr[1] * Z) + y[i + 2])
            z.append((bc[2] * X + ns[2] * Y + nr[2] * Z) + z[i + 2])
            # print("i: {}, \ti+1: {}, \tN-2+i: {}, \tlen_x: {}".format(i, i+1, self.N - 2 + i, len(x)))

        for i in range(self.N):
            self.grains[i].pos_x = x[i]
            self.grains[i].pos_y = y[i]
            self.grains[i].pos_z = z[i]

        return x, y, z


    def create_hashtable(self):
        # cuts space into cubes (side length = c), puts all grains in coresponding cubes and returns a dictionary
        # key -> value == cube's key(tuple) -> list of containing grains
        hashtable = {}

        for i in range(self.N):
            #computes key of the cube, which contains grain N
            key = (self.grains[i].pos_x // c, self.grains[i].pos_y // c, self.grains[i].pos_z // c)
            value = [i]
            if key in hashtable:
                reminder = hashtable[key]
                for thing in reminder: value.append(thing)
            hashtable[key] = value

        return hashtable


    def get_neighbours(self, hashtable, grain_i):
        # finds out all neighbours of grain_i within current cube and all neighbouring cubes
        # returns their positions in self.grains as an array
        key = (self.grains[grain_i].pos_x // c, self.grains[grain_i].pos_y // c, self.grains[grain_i].pos_z // c)
        neighbours = []

        # finds all adjacent cubes
        for i in [key[0] - 1, key[0], key[0] + 1]:
            for j in [key[1] - 1, key[1], key[1] + 1]:
                for k in [key[2] - 1, key[2], key[2] + 1]:
                    # if key (i, j, k) is in hashtable, append grains in that cube to the neighbours array
                    if (i, j, k) in hashtable: neighbours.append(hashtable[(i, j, k)])

        # ensures, that a simple list is returned
        result = []
        for i in neighbours:
            for j in i:
                result.append(j)
        neighbours = result
        # exclude grain_i
        neighbours.remove(grain_i)

        return neighbours


    def H(self):
        """ H = P + V = sum(E_kinetic(grain_i)) + sum(improved potential energy) + sum(bond energy)"""
        V = 0
        #"""
        # V = sum(improved potential energy) + sum(bond energy)
        # computes potential energy between grains, whose distance is shorter than cutoff
        x, y, z = self.get_coordinates()
        hashtable = self.create_hashtable()
        for i in range(self.N):
            # get list of neighbours
            neighbours = self.get_neighbours(hashtable, i)
            for neighbour in neighbours:
                E_cut = 4 * e * ((s / c) ** 12 - (s / c) ** 6)
                if i < neighbour:
                    r = self.distance(i, neighbour)
                    if r > c:
                        continue

                    if self.sequence[i] == "A" and self.sequence[neighbour] == "A":
                        V += 4 * e * ((s / r)**12 - (s / r)**6) - E_cut # interaction A-A
                    else:
                        V += 2*e*((s / r)**12 - (s / r)**6) - 0.5 * E_cut # interaction A-B, B-A, B-B
        #"""
        # harmonic bond potential
        # default bond length == 1
        # V += V_bonding(grain_i) = bond_rigidity * (r - r_0)^2
        for i in range(self.N - 1):
            V += bond_rigidity * (self.distance(i, i+1) - bond_length) ** 2
        #"""

        T = 0
        # P = sum(E_kin(grain_i))
        for i in range(self.N):
            T += 0.5 * m_grain * ((self.grains[i].vel_x)**2 + (self.grains[i].vel_y)**2 + (self.grains[i].vel_z)**2)

        return V + T


    def to_xyz(self, name):
        """creates name.xyz file with configuration coresponding to angle_array; B residue called Z"""
        x, y, z = self.get_coordinates()
        with open("{}.xyz".format(name), "a") as xyz:
            xyz.write("{}\n".format(self.N))
            xyz.write("comment line\n")
            for i in range(self.N):
                if self.sequence[i] == "A":
                    xyz.write("N {} {} {}\n".format(x[i], y[i], z[i]))
                else:
                    xyz.write("O {} {} {}\n".format(x[i], y[i], z[i]))


    def run_xyz_in_vmd(self, filename):
        # opens structure in vmd
        path_to_visualiser = "C:\\Program Files (x86)\\University of Illinois\\VMD\\vmd.exe"
        path_to_file = "C:\\Users\\Administrator\\.PyCharmCE2019.3\\config\\scratches\\best.xyz"
        subprocess.call([path_to_visualiser, path_to_file])


#============================================== constants and parameters ===============================================
k_1 = -1                                                                # local interaction constant
k_2 = 0.5                                                               # local interaction constant
num_steps = 1                                                           # number of velocity-Verlet steps
T = 1                                                                   # temperature                           [e/kB]
dt = 0.0001                                                             # velocity-Verlet step lenght  [s*(m/e)**(1/2)]
m_grain = 1                                                             # weight of single grain                [m]
kB = 1                                                                  # Bolzman constant
e = 1                                                                   # LJ potential constant                 [e]
s = 1                                                                   # LJ potential constant                 [s]
c = 2 * s                                                               # cutoff distance                       [s]
steps = 1000                                                            # number of velocity-Verlet steps
accepted = 0                                                            # number of accepted configurations
kB_T = 1                                                                # kB_T == k_B * T                       [e]
bond_length = 1                                                         #                                       [s]
bond_rigidity = 0.5
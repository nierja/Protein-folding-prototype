from math import sqrt
import numpy as np
import random
import matplotlib.pyplot as plt
from protein_model import *
import os


class HMC_simulation(Protein):
    """class for HMC simulation of protein model class defined in protein.py alongside with some vector operations"""

    def __init__(self, sequence):
        super().__init__(sequence)


    def dist(self, grain1, grain2):
        # calculates distance between 2 grains
        x_1, y_1, z_1 = self.grains[grain1].pos_x, self.grains[grain1].pos_y, self.grains[grain1].pos_z
        x_2, y_2, z_2 = self.grains[grain2].pos_x, self.grains[grain2].pos_y, self.grains[grain2].pos_z
        dx = x_1 - x_2
        dy = y_1 - y_2
        dz = z_1 - z_2

        dist = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return dx, dy, dz, dist


    def update_coordinates(self):
        """updates beads positions"""
        for grain_i in range(self.N):
            self.grains[grain_i].pos_x += self.grains[grain_i].vel_x * dt + 0.5 * self.grains[grain_i].acc_x * dt ** 2
            self.grains[grain_i].pos_y += self.grains[grain_i].vel_y * dt + 0.5 * self.grains[grain_i].acc_y * dt ** 2
            self.grains[grain_i].pos_z += self.grains[grain_i].vel_z * dt + 0.5 * self.grains[grain_i].acc_z * dt ** 2
        return


    def update_velocity(self):
        """updates beads velocities"""
        for grain_i in range(self.N):
            self.grains[grain_i].vel_x += 0.5 * self.grains[grain_i].acc_x * dt
            self.grains[grain_i].vel_y += 0.5 * self.grains[grain_i].acc_y * dt
            self.grains[grain_i].vel_z += 0.5 * self.grains[grain_i].acc_z * dt
        return


    def get_coordinates(self):
        # stores self.grains[all_grains],pos_x/y/z in arrays x, y, z
        x, y, z = [], [], []
        for i in range(self.N):
            x.append(self.grains[i].pos_x)
            y.append(self.grains[i].pos_y)
            z.append(self.grains[i].pos_z)

        return x, y, z


    def load_coordinates(self, x_array, y_array, z_array):
        # updates self.grains[all_grains],pos_x/y/z with values from x_array, y_array, z_array
        for i in range(self.N):
            self.grains[i].pos_x = x_array[i]
            self.grains[i].pos_y = y_array[i]
            self.grains[i].pos_z = z_array[i]
        return


    def deriv_of_nonbonding_potential(self, grain_i, grain_j, r):
        # returns -d(nonbonding potential)/dr between grain_i and grain_j
        if self.sequence[grain_i] == "A" and self.sequence[grain_j] == "A":
            force = -24 * (e / r) * (2 * (s / r)**12 - (s  / r)**6) # interaction A-A
        else:
            force = -12 * (e / r) * (2 * (s / r)**12 - (s  / r)**6) # interaction A-B, B-A, B-B
        return force


    def update_acceleration(self):
        #"""
        for grain_i in range(self.N):
            self.grains[grain_i].acc_x = 0
            self.grains[grain_i].acc_y = 0
            self.grains[grain_i].acc_z = 0
        #"""
        #"""
        # computes nonbondig forces, updates acceleration
        hashtable = self.create_hashtable()
        for i in range(self.N):
            # get list of neighbours
            neighbours = self.get_neighbours(hashtable, i)
            for neighbour in neighbours:
                if i < neighbour:
                    r = self.distance(i, neighbour)
                    if r > c:
                        continue
                    force = self.deriv_of_nonbonding_potential(i, neighbour, r)

                    a_x = force * (self.grains[neighbour].pos_x - self.grains[i].pos_x) / (r * m_grain)
                    a_y = force * (self.grains[neighbour].pos_y - self.grains[i].pos_y) / (r * m_grain)
                    a_z = force * (self.grains[neighbour].pos_z - self.grains[i].pos_z) / (r * m_grain)

                    self.grains[i].acc_x += a_x
                    self.grains[i].acc_y += a_y
                    self.grains[i].acc_z += a_z

                    self.grains[neighbour].acc_x -= a_x
                    self.grains[neighbour].acc_y -= a_y
                    self.grains[neighbour].acc_z -= a_z
        """

        #"""
        # computes bondig forces, updates acceleration
        #print("forces: ",forces)
        for i in range(self.N):
            a_x = a_y = a_z = 0
            if i > 0:
                dx, dy, dz, r = self.dist(i, i-1)
                force = -2 * bond_rigidity * (r - bond_length)

                a_x += force * dx / (r * m_grain)
                a_y += force * dy / (r * m_grain)
                a_z += force * dz / (r * m_grain)

            if i < self.N - 1:
                dx, dy, dz, r = self.dist(i, i+1)
                force = -2 * bond_rigidity * (r - bond_length)

                a_x += force * dx / (r * m_grain)
                a_y += force * dy / (r * m_grain)
                a_z += force * dz / (r * m_grain)

            self.grains[i].acc_x += a_x
            self.grains[i].acc_y += a_y
            self.grains[i].acc_z += a_z
        return
        #"""

    def Verlet(self):
        for i in range(num_steps):
            self.update_coordinates()
            self.update_velocity()
            self.update_acceleration()
            self.update_velocity()
        return

    def HMC(self):
        global accepted
        E = []
        positions = []
        E_best = self.H()

        for i in range(steps):
            self.generate_velocities()
            #self.Andersen()
            H1 = self.H()
            x_i, y_i, z_i = self.get_coordinates()
            self.Verlet()
            H2 = self.H()

            if np.exp(-(H2 - H1) / kB_T) > random.random():
                accepted += 1
                E.append(H2)
                if H2 < E_best: E_best = H2

            else:
                self.load_coordinates(x_i, y_i, z_i)
                E.append(H1)

            self.to_xyz(name="best")

            k = 10
            positions.append(self.grains[k].pos_x)
            r = self.distance(0, k)
            print("HMC step n. {}: \tx: {:+.8f}, y: {:+.8f}, \tvel_x: {:+.8f}, vel_y: {:+.8f}, acc_y: {:+.8f}, "
                  "\tr: {:+.8f}, E_best: {:+.8f}, E_total: {:+.8f}".format(i + 1, self.grains[k].pos_x,
                                                                           self.grains[k].pos_y,
                                                                           self.grains[k].vel_x, self.grains[k].vel_y,
                                                                           self.grains[
                                                                               k].acc_y, r, E_best, self.H()))
            #self.to_xyz("best")
        #self.run_xyz_in_vmd("best.xyz")
        print("yerr: ", block_method(E))

        return E

    def trace_E(self):
        # deletes best.xyz from previous simulation
        try:
            os.remove("best.xyz")
        except:
            FileNotFoundError

        E = self.HMC()
        print("P_acc:", accepted / steps)

        x = range(len(E))
        plt.xlim(min(x), max(x))
        plt.ylim(min(E) - 1, max(E) + 1)

        plt.plot(x, E)
        plt.title("trace E: HMC steps={}, chain lenght={}, T={}, dt={}".format(steps, self.N, T, dt))
        plt.xlabel("steps")
        plt.ylabel("E")
        plt.show()
        plt.close()
        return


# =========================================== Real protein sequences from PDB ===========================================
short = "ABBBBABAAABABBABBABABB"  # shortened protein p_4RXN        #N=22
p_4RXN = "ABBBBABAAABABBABBABABBAABAABBBBBAABBBAAAAAAAABBBBBBABB"  # N=54
p_1FCA = "ABAABBAAABAAAABABAAABAABBAABBBAABABBAABAAAAAAAAAABAAABA"  # N=55
p_2GB1 = "ABBBAAABABBABABBBBBAABAABABBABBBBABBBAABABBBBBBABBBBBABB"  # N=56
p_2OVO = "AAAABABABBBABAAABABBBAAAABBBBBBABBABBABAAABBBABABABBBABA"  # N=56
p_2YGS = "ABABABBAAABBBBAABBBABBBBAABBAABBABABABBBBBABBBABBBBBAAAAABAAABBBBBBBABBBBAAABBABBBAAAAABBAAA"  # N=92

# ============================================== constants and parameters ===============================================
k_1 = -1                                                                # local interaction constant
k_2 = 0.5                                                               # local interaction constant
num_steps = 10                                                          # number of velocity-Verlet steps
T = 1                                                                   # temperature                           [e/kB]
dt = 0.0001                                                             # velocity-Verlet step lenght  [s*(m/e)**(1/2)]
m_grain = 1                                                             # weight of single grain                [m]
kB = 1                                                                  # Bolzman constant
e = 1                                                                   # LJ potential constant                 [e]
s = 1                                                                   # LJ potential constant                 [s]
c = 2 * s                                                               # cutoff distance                       [s]
steps = 200                                                            # number of HMC steps
accepted = 0                                                            # number of accepted configurations
kB_T = 1                                                                # kB_T == k_B * T                       [e]
bond_length = 1                                                         #                                       [s]
bond_rigidity = 0.5


def main():
    simulation = HMC_simulation(short)
    simulation.trace_E()

if __name__ == '__main__':
    main()

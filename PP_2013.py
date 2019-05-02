import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit
from progressbar import Percentage, ProgressBar, Bar, ETA

"""
Initialise function and nearest neigbours boundary conditions check function

"""


def init(n):
	down = -1

	i_matrix = np.random.choice([0, 1, 2], size=(n, n), p=[1/3, 1/3, 1/3])

	return i_matrix

def nnb(config, i, j):
	i_matrix = config
	up = n - 1

	if i == 0:
		i1 = up
	else:
		i1 = i - 1

	if i == up:
		i2 = 0
	else:
		i2 = i + 1

	if j == up:
		j1 = 0
	else:
		j1 = j + 1

	if j == 0:
		j2 = up
	else:
		j2 = j - 1

	nb1 = np.array([i_matrix[i2, j], i_matrix[i, j1], i_matrix[i1, j], i_matrix[i, j2]])

	# nns =  [nn1,nn2,nn3,nn4]
	# nns = np.sum(nns)
	return nb1


"""
mc and mc_k functions each performing glauber and kawaski dynamics repectively
"""

def kc(x,y):
	if x == y:
		return 1
	else:
		return 0

def mc(config, te, n):
	# print('g')
	for i in range(n*n):
		cost = 0
		cost_1 = 0
		a = int(np.random.uniform(0, n))
		b = int(np.random.uniform(0, n))
		s = config[a, b]
		nn = nnb(config, a, b)
		for k in range(len(nn)):
			cost_1 =+ kc(s,nn[k])
		choice = np.random.choice([0,1,2])
		for m in range(len(nn)):
			cost =+ kc(choice,nn[m])
		de = -1*s*cost - (-1*s*cost_1)
		if de <= 0:
			s = choice
		elif np.random.rand() < np.exp(-(1 * de) / te):
			s = choice
		config[a, b] = s

	return config

def energy(config,n):
	et = 0
	for i in range(n):
		for j in range(n):
			nn = nnb(config,i,j)
			e = 0
			for k in range(len(nn)):
				e  =+ kc(config[i,j],nn[k])
			et += e
	et = -1*et/4
	norm_e = et/(n**2)
	return norm_e

def frac(config,n):
	s1 = 0
	s2 = 0
	s3 = 0
	for i in range(n):
		for j in range(n):
			if config[i,j] == 0:
				s1 += 1
			elif config[i,j] == 1:
				s2 += 1
			elif config[i,j] == 2:
				s3 =+ 1
	s1 = s1/(n*n)
	s2 = s2/(n*n)
	s3 = s3/(n*n)

	return s1,s2,s3

def updatefig_1(*args):
    im.set_data(mc(ani_config,t,n))
    return im,

n = 50
# config = init(n)
t = 0.01
# sweep = 1000
#
# for i in range(sweep):
# 	mc(config,t,n)
# 	e = energy(config,n)
# 	st1,st2,st3 = frac(config,n)
# 	print(st1,st2,st3)
# 	print(e)

ani_config = init(n)
fig = plt.figure()
im = plt.imshow(mc(ani_config,t,n), animated=True)
plt.title(t)

for i in range(5):
    updatefig_1()

ani = animation.FuncAnimation(fig, updatefig_1 ,interval=25, blit=True
                              ,frames = 500,repeat = True)
plt.show()



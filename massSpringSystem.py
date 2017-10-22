import numpy as np 
from scipy.spatial import Delaunay
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt 
import mesh
import utils

class MassSpring:
	""" generating,processing and visualizing the mass spring system for physics simulation """
	def __init__(self):
		# the coordinates of all the particles,(n,3) array
		# ..note: the points on the surface will ordered first, then the points in the volume
		self.x = []
		# the number of the surface particles
		self.surface_x_num = 0
		# the neighbor list of each particle
		self.x_neighbor = []
		# the velocity of each particle; numpy array (n,3)
		self.v = []
		# the mass of each particle
		self.m = []
		# the stiffness coefficient (assigned uniformly at the current state)
		self.stiff_coef = 0
		# the dampling coefficient (assigned uniformly at the current state)
		self.damp_coef = 0

	def generate_from_tetmesh(self,tmesh,density = 1,stiffness = 1, dampling = 1):
		""" 
		use a tetrahedral mesh to generate a mass spring system
		based on the coursenote of real time physics,3.6 mesh creation
		..note: the tmesh here is specially the TetMesh Class in mesh.py
		--------------------------------------------------------------
		process:
		create the mass spring system from a tetrahedral mesh by turning each vertex into a particle
		and each tetrahedral edge into a spring
		as for the masses of the particles and the stiffness and dampling coefficients of the springs,
		given a user specified density, the mass of each tetrahedron can be computed as its volume * density,
		then each tetrahedron distributes its mass evenly among its four adjacent vertices
		Finding reasonable spring coefficients is not straight forward
		since there is no correct way, we might as well assign a common stiffness and dampling coefficients to all springs
		"""
		self.x = tmesh.vertices
		self.surface_x_num = tmesh.surnum 
		self.x_neighbor = tmesh.get_vertex_neighbor_list()
		self.v = np.zeros(self.x.shape)		
		# compute the mass
		tet_volume = tmesh.get_tet_volume_list()
		self.m = np.array([[0.0] for k in range(self.x.shape[0])])
		for k in range (len(tet_volume)):
			temp_mass = tet_volume[k] * density / 4.0
			for j in range(4):
				self.m[tmesh.tets[k][j]] += temp_mass
		# assign the stiffness and dampling coefficients uniformly
		self.stiff_coef = stiffness
		self.damp_coef = dampling

	def generate_from_points(self,points,sur_num,density = 1,stiffness = 1, dampling = 1):
		""" 
		use a given point set to generate a mass spring system
		para::points: (n*3) array (the point coordinate)
		para::sur_num: the number to the surface particles
		(and the first sur_num points will be regarded as surface particles)
		----------------------------------------------------------------------------------------------------------------
		process:
		will first construct a tetrohedral mesh at first by Delaunay triangulation, then generate the mass spring system
		"""
		# to be finished
		"""
		assert points.shape[0] > sur_num
		# construct the tetrahedral mesh
		tet = Delaunay(points)
		self.x = tet.points
		self.surface_x_num = sur_num
		self.v = np.zeros(self.x.shape)
		"""
		pass

	def generate_from_trimesh(self,tmesh,density = 1,stiffness = 1, dampling = 1):
		""" 
		generate the mass spring using a triangle mesh
		process: get sample points --> generate from points
		"""
		pass


# for test
tm = np.random.random((50,3))
tetm = mesh.TetMesh()
tetm.generate_from_points(tm,15)
tmss = MassSpring()
tmss.generate_from_tetmesh(tetm)
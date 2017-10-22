import numpy as np 
from scipy.spatial import Delaunay
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt 
import trimesh

# for simpling generate,process(animation),visualization of triangle mesh, tetrahedron mesh
# and esp. mass spring system

class TriMesh:
	""" 
	wrapped some data from trimesh data structure at current state
	in the future the function of directly import the data from the file will be added
	----------------------------------------------------------------------------------
	at current state only the vertice coordinate and the face information are recorded,
	since now the program is mainly for generating the mass spring system
	"""
	def __init__(self):
		# the coordinates of all the vertices of the mesh (n*3 array)
		self.vertices = []
		# all the triangle face in the mesh(m*3 array, each row is the vertice index)
		self.tris = []


	def set_vf_trimesh(self,tmesh):
		""" to set the vertices and triangle face information by a given mesh of trimesh class"""
		self.vertices = np.array([tmesh.vertices[k] for k in range(tmesh.vertices.shape[0])])
		self.tris = np.array([tmesh.faces[k] for k in range(tmesh.faces.shape[0])])





class TetMesh:
	""" for gerating, processing and visulizing the tetrahedral mesh """
	def __init__(self):
		# the coordinates of all the vertices of the mesh (n*3 array)
		self.vertices = []
		# all the tetrahedrons in the mesh(m*4 array,each row is the vertice index)
		self.tets = []
		# indices of neighbors of each tetrahedrons(m*4 array: the k-th neighbor is opposite to the k-th vertex)
		# (for tetrahedrons at the boundary, -1 denotes no neighbor)
		self.tet_neighbors = []
		# the number of the vertices that on the surface
		self.surnum = 0

	def generate_from_points(self,points,snum):
		""" 
		generate the mesh from given points using Delaunay method 
		para::points: numpy array (n,3)
		para::snum: the number of the vertices that are on the surface
		"""
		tet = Delaunay(points)
		self.vertices = tet.points
		self.tets = tet.simplices
		self.tet_neighbors = tet.neighbors
		self.surnum = snum

	def get_vertex_tet_list(self):
		""" i-th row of the list is the indices of the tetrahedrons where the i-th vertex is in """
		vt_list = [[] for k in range(self.vertices.shape[0])]
		for k in range(self.tets.shape[0]):
			for j in range(4):
				vt_list[self.tets[k,j]].append(k)
		return vt_list

		#>>> d = [list(set(c[k])) for k in range(10)]


	def get_vertex_neighbor_list(self):
		""" get the neighbors of each vertex """
		vt_list = self.get_vertex_tet_list()
		vn_list = [[] for k in range(len(vt_list))]
		# add the neighbors
		for k in range (len(vt_list)):
			for j in range(len(vt_list[k])):
				vn_list[k] = vn_list[k] + self.tets[vt_list[k][j]].tolist()
		# remove the repeat elements
		vn_list = [list(set(vn_list[k])) for k in range(len(vn_list))]
		return vn_list

	def get_tet_volume(self,tet_id):
		""" get the volume of the given tetrahedron """
		tet = self.tets[tet_id]
		e1 = self.vertices[tet[1]] - self.vertices[tet[0]]
		e2 = self.vertices[tet[2]] - self.vertices[tet[0]]
		e3 = self.vertices[tet[3]] - self.vertices[tet[0]]
		volume = np.abs(np.dot(np.cross(e1,e2),e3)) * 1./6
		return volume

	def get_tet_volume_list(self):
		""" get the volume of all of the tetrahedrons """
		volum_list = np.array([self.get_tet_volume(k) for k in range(self.tets.shape[0])])
		return volum_list

	def get_triangle_list(self):
		""" 
		get the list of all the triangle faces in the mesh
		output: (m*4,3) array
		each row is the index of the vertices forming the triangle
		----------------------------------------------------------------------------------------
		the order issue:
		for each tetrahedron, the corresponding triangle face is orderer in this way:
			[v1,v2,v3]
			[v0,v2,v3]
			[v0,v1,v3]
			[v0,v1,v2]
		the triangle faces belonging to the same tetrahedron will be ordered together, 
		and the total order is consist with the order of the tetradron in self.tets
		"""
		tri_id = [[
		[self.tets[k,1],self.tets[k,2],self.tets[k,3]],
		[self.tets[k,0],self.tets[k,2],self.tets[k,3]],
		[self.tets[k,0],self.tets[k,1],self.tets[k,3]],
		[self.tets[k,0],self.tets[k,1],self.tets[k,2]]
		] for k in range(self.tets.shape[0])]
		tri_id = [item for sublist in tri_id for item in sublist]
		tri_id = np.array(tri_id)
		return tri_id

	def show(self):
		""" for visualizing the tetradron mesh """
		# get the triangles in the mesh
		tri_id = self.get_triangle_list()
		axes = a3.Axes3D(plt.figure())
		vts = self.vertices[tri_id,:]
		tri = a3.art3d.Poly3DCollection(vts)
		tri.set_alpha(0.2)
		tri.set_color('grey')
		axes.add_collection3d(tri)
		axes.plot(self.vertices[:,0], self.vertices[:,1], self.vertices[:,2], 'ko')
		axes.set_axis_off()
		axes.set_aspect('equal')
		plt.show()



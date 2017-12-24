import numpy as np 
from scipy.spatial import cKDTree as kdtree

"""
providing some tools for geometry process
including:
area, coordinate convert, remove close points, bounding box
..note: if the codes are too long, may seperate the functions into different .py in the future
"""

def get_area(vertices):
	"""
	compute the area of the given polygon formed by the vertices(already in order)
	para::vertices: (n,m) numpy array, where n is the number of the vertices and m is the dimension
	output::area: the area of the given polygon
	"""
	assert vertices.shape[0] > 2
	area = 0
	for k in range(1,vertices.shape[0]-1):
		v1 = vertices[k] - vertices[0]
		v2 = vertices[k+1] - vertices[0]
		area += np.linalg.norm(np.cross(v1,v2)) * 0.5
	return area

def get_average_area(vertices,facet):
	"""
	compute the average area of each facet
	..each row of facet is the id of the vertices of the face
	"""
	area = np.array([get_area(vertices[facet[k]]) for k in range(len(facet))])
	ave = area.mean()
	return ave

def get_average_edge_length(vertices,facet):
	"""
	compute the average length of the triangle mesh
	"""
	# loop each triangle 
	edge_length = np.zeros(len(facet))
	for k in range(len(facet)):
		v1 = vertices[facet[k,1]] - vertices[facet[k,0]]
		v2 = vertices[facet[k,2]] - vertices[facet[k,0]]
		v3 = vertices[facet[k,2]] - vertices[facet[k,1]]
		edge_length[k] = (np.linalg.norm(v1) + np.linalg.norm(v2) + np.linalg.norm(v3)) / 3.0
	ave = edge_length.mean()
	return ave

def xyz2sp(xyz):
	"""
	to convert the cartesian coordinate system to spherical system
	para::xyz: numpy array (n,3)
	output::sp: numpy array (n,3)
	.. note: each row is (r,u,v) where r is radius, u in (-pi,pi), v in (-0.5pi,0.5pi)
	"""
	sp = np.zeros(xyz.shape)
	xy = xyz[:,0]**2 + xyz[:,1]**2
	sp[:,0] = np.sqrt(xy + xyz[:,2]**2)
	sp[:,1] = np.arctan2(xyz[:,1], xyz[:,0])
	sp[:,2] = np.arctan2(xyz[:,2], np.sqrt(xy))
	return sp

def sp2xyz(sp):
	"""
	to convert the spherical coordinate to the cartesian system
	para::sp: numpy array (n,3)
	.. note: each row is (r,u,v) where r is radius
	output::xyz: numpy array (n,3) the cartesian coordinate
	"""
	xyz = np.zeros(sp.shape)
	xyz[:,0] = sp[:,0] * np.cos(sp[:,1]) * np.cos(sp[:,2])
	xyz[:,1] = sp[:,0] * np.sin(sp[:,1]) * np.cos(sp[:,2])
	xyz[:,2] = sp[:,0] * np.sin(sp[:,2])
	return xyz

def xyz2bary(tri_vert,point):
	"""
	to convert the cartesian coordinates to barycentric coordinates
	using the 3 vertices of a given triangle
	if the triangle vertices are a,b,c and point is denoted as p
	p = alpha*a + beta*b + gamma*c,
	where alpha+beta+gamma=1 
	..note if alpha,beta,gamma will be in [0,1], then p is in the triangle
	reference:https://math.stackexchange.com/questions/4322/check-whether-a-point-is-within-a-3d-triangle(answer2)
	------------------------------------------------------------------------------------------------------------------
	para::tri_vert: numpy array(3,3), the cartesian coordinates of 3 vertices of the given triangle(each row is 1 vertex)
	para::point: the point for computing the barycentric coordinates
	return::bary: the barycentric coordinate of the point numpy array (alpha,beta,gamma)
	"""
	area_abc = get_area(tri_vert)
	area_a = get_area(np.vstack((tri_vert[1],tri_vert[2],point)))
	area_b = get_area(np.vstack((tri_vert[0],tri_vert[2],point)))
	area_c = get_area(np.vstack((tri_vert[0],tri_vert[1],point)))
	alpha = area_a / area_abc
	beta = area_b / area_abc
	#gamma = area_c / area_abc
	gamma = 1.0 - alpha - beta 
	return np.array([alpha,beta,gamma])

def bary2xyz(tri_vert,bary):
	"""
	to convert the barcentric coordinates of the given triangle to cartesian coordinate
	para::tri_vert: numpy array(3,3), the cartesian coordinates of 3 vertices of the given triangle mesh(each row is 1 vertex)
	para::bary: the barycentric coordinate of the point with respect to the given triangle
	return::xyz: the cartesian coordinates of the given point
	"""
	xyz = tri_vert[0] * bary[0] + tri_vert[1] * bary[1] + tri_vert[2] * bary[2]
	return xyz

def remove_close(points,radius):
	"""
	given an (n,m) set of points where n = 2 or n =3 return a list of points where no point is closer than radius
	para::points: numpy array(m,2) or (m,3), coordinates of the points
	para::radius: the least distance between the points
	return: the unique points and its index 
	"""
	tree = kdtree(points)
	consumed = np.zeros(len(points),dtype=np.bool)
	unique = np.zeros(len(points),dtype=np.bool)
	for k in range(len(points)):
		if consumed[k]:
			continue
		neighbors = tree.query_ball_point(points[k],r=radius)
		consumed[neighbors] = True
		unique[k] = True
	index = np.array([k for k in range(len(points))])
	return points[unique],index[unique]

def get_bounding_box(points):
	"""
	for input 3d points,find a bounding box for them
	..note:currently the box's edge are parallel with x,y,z axis
	para::points: numpy array(n,3)  (only works for 3d points)
	return::corner: the corner of the bounding box with the min value of x,y,z coordinate
	return::info: numpy array(3,) with information of length,width,height
	"""
	xmin,ymin,zmin = [np.min(points[:,k]) for k in range(3)]
	xmax,ymax,zmax = [np.max(points[:,k]) for k in range(3)]
	corner = np.array([xmin,ymin,zmin])
	info = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
	return corner,info

def get_trimesh_centroid(vertices,facet):
	"""
	get the centroid of the triangle mesh
	http://wwwf.imperial.ac.uk/~rn/centroid.pdf
	"""
	"""
	area = np.array([get_area(vertices[facet[k]]) for k in range(facet.shape[0])])
	tri_center = np.array([np.mean(vertices[facet[k]],axis=0) for k in range(facet.shape[0])])
	centroid = np.average(tri_center,axis=0,weights=area)
	"""
	basis = np.identity(3)
	volume = get_trimesh_volume(vertices,facet)
	# compute the intergral
	intergral = np.zeros((len(facet),3))
	for k in range(len(facet)):
		for j in range(3):

			normal = np.cross((vertices[facet[k,1]]-vertices[facet[k,0]]), 
				(vertices[facet[k,2]]-vertices[facet[k,0]]))
			intergral[k,j] = np.dot(normal, basis[j]) * (
				np.square(np.dot((vertices[facet[k,0]] + vertices[facet[k,1]]), basis[j])) +
				np.square(np.dot((vertices[facet[k,1]] + vertices[facet[k,2]]), basis[j])) +
				np.square(np.dot((vertices[facet[k,2]] + vertices[facet[k,0]]), basis[j]))
				) / 24.0
	centroid = np.sum(intergral,axis=0) / (2.0 * volume)
	return centroid

def get_trimesh_volume(vertices,facet):
	"""
	http://wwwf.imperial.ac.uk/~rn/centroid.pdf
	compute the volume of the given triangle mesh
	"""
	volume = 0
	for k in range(len(facet)):
		normal = np.cross((vertices[facet[k,1]]-vertices[facet[k,0]]), 
			(vertices[facet[k,2]]-vertices[facet[k,0]]))
		volume += np.dot(vertices[facet[k,0]], normal) / 6.0
	volume = np.abs(volume)
	return volume

def get_edge_list(vertices,facet):
	"""get the list of the edge in the triangle mesh"""
	edge_list = []
	for k in range(len(facet)):
		edge_list.append([facet[k,0],facet[k,1]])
		edge_list.append([facet[k,1],facet[k,2]])
		edge_list.append([facet[k,2],facet[k,0]])
	edge_list = np.array(edge_list)
	return edge_list

def get_vertex_normal_list(vertices,facet):
	""" get the normal vector of each vertice of the triangle mesh """
	normal = np.zeros((len(vertices),3))
	centroid = get_trimesh_centroid(vertices,facet)
	for k in range(len(facet)):
		v1 = vertices[facet[k,1]] - vertices[facet[k,0]]
		v2 = vertices[facet[k,2]] - vertices[facet[k,0]]
		temp = np.cross(v1,v2)
		# normalized 
		temp = temp / np.linalg.norm(temp)
		# check the direction
		v = vertices[facet[k,0]] - centroid
		if np.dot(temp,v) < 0:
			temp = -temp
		# assigned to each vertex
		for j in range(3):
			normal[facet[k,j]] += temp
	# normalize
	for k in range(len(normal)):
		normal[k] = normal[k] / np.linalg.norm(normal[k])
	return normal

def get_vertex_outer_normal(vertices,facet,id):
	""" to get the normal vector of the given vertice (outer)"""
	centroid = get_trimesh_centroid(vertices,facet)
	v = vertices[id] - centroid
	outer_normal = np.zeros(3)
	# find the triangle; row is the face index
	row, col = np.where(facet==id)
	for k in range(len(row)):
		v1 = vertices[facet[row[k],1]] - vertices[facet[row[k],0]]
		v2 = vertices[facet[row[k],2]] - vertices[facet[row[k],0]]
		temp = np.cross(v1,v2)
		temp = temp / np.linalg.norm(temp)
		if np.dot(v,temp) < 0:
			temp = -temp
		outer_normal = outer_normal + temp
	outer_normal = outer_normal / np.linalg.norm(outer_normal)
	return outer_normal

def get_gauss_curvature(vertices,facet):
	"""
	http://libigl.github.io/libigl/tutorial/tutorial.html#gaussiancurvature
	whose reference is 
	Discrete Differential-Geometry Operators for Triangulated 2-Manifolds
	see: http://multires.caltech.edu/pubs/diffGeoOps.pdf
	"""
	pass

def get_facet_normal_list(vertices, facet):
	""" return the outer surface normal vectors for each face """
	normal = np.zeros(facet.shape)
	# help to judge the normal vector is outer or not
	centroid = get_trimesh_centroid(vertices, facet)
	for k in range(len(facet)):
		v1 = vertices[facet[k,1]] - vertices[facet[k,0]]
		v2 = vertices[facet[k,2]] - vertices[facet[k,0]]
		temp = np.cross(v1, v2)
		vert = vertices[facet[k,0]] - centroid
		# make sure that the normal vector is outer
		if np.dot(temp, vert) < 0.:
			temp = -temp
		normal[k] = (temp / np.linalg.norm(temp))
	return normal

def get_face_outer_normal(vertices, facet, face_id):
	""" given the id of the face, return the outer normal vector of the triangle """
	centroid = get_trimesh_centroid(vertices, facet)
	v1 = vertices[facet[face_id,1]] - vertices[facet[face_id,0]]
	v2 = vertices[facet[face_id,2]] - vertices[facet[face_id,0]]
	temp = np.cross(v1, v2)
	vert = vertices[facet[face_id,0]] - centroid
	if np.dot(temp, vert) < 0:
		temp = -temp
	normal = temp / np.linalg.norm(temp)
	return normal
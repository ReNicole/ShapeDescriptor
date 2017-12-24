import numpy as np 

"""
python codes for ray testing algorithm
reference:Fast, Minimum Storage Ray Triangle Intersection
http://www.graphics.cornell.edu/pubs/1997/MT97.pdf
------------------------------------------------------------
implementation
ray function: R(t) = O + tD
a point T(u,v) in a triangle: T(u,v) = (1-u-v)V0 + uV1 + vV2
if ray intersects with the triangle, then by R(t) = T(u,v), we get
[-D V1-V0 V2-V0] [t u v].transpose = O-V0
denote E1=V1-V0 E2=V2-V0 T=O-V0
[-D E1 E2] [t u v].transpose = T
denote det = [-D E1 E2]
by Cramer's rule
t = det(T E1 E2) / det
u = det(-D T E2) / det
v = det(-D E1 T) / det
denote P = cross(D,E2),Q = cross(T,E1)
then det = dot(P,E1),
and t = dot(Q,E2) / det, u = dot(P,T) / det, v = dot(Q,D) / det
"""

def ray_triangle_intersect(triangle,origin,direction):
	"""
	to test whether the given ray intersect with the given triangle
	if no intersect,only return false
	if intersect, return true t(see the ray function)
	para::triangle: the coordinates of the vertices of the triangle
	..numpy array(3,3), each row is a vertex
	para::origin: the origin of the ray
	para::direction: the direction of the ray
	return::False and -1 if no intersection or True and t if intersect
	..note:origin+t*direction=intersection
	"""
	# E1 = V1-V0, E2 = V2-V0
	E1 = triangle[1] - triangle[0]
	E2 = triangle[2] - triangle[0]
	# D = direction, P = cross(D,E2) ,det= dot(E1,P)
	D = direction
	P = np.cross(D,E2)
	det = np.dot(E1,P)
	# keep determine(det) > 0, modify T accordingly
	if det > 0:
		T = origin - triangle[0]
	else:
		T = triangle[0] - origin
		det = -det
	# if the determine is near zero, the ray is parallel with the plane of the triangle
	if det < 1e-04:
		return False,-1
	# compute u(actually u*det) and see whether it satisfies u in [0,1]
	u = np.dot(T,P)
	if u<0 or u>det:
		return False,-1
	Q = np.cross(T,E1)
	# compute v(actually v*det) and see whether v>=0 and u+v in [0,1]
	v = np.dot(Q,D)
	if v<0 or (u+v)>det:
		return False,-1
	# compute t(actually t*det) and see whether t>=0 (note det >0)
	t = np.dot(Q,E2)
	if t<0:
		return False,-1
	# compute the final value of t
	t = t / det
	return True,t

def ray_trimesh_intersect_first(vertices,facet,origin,direction):
	"""
	loop the given triangle facet and return the first intersection
	if no intersection at all, return false instead
	..note: here first means t value is smallest(i.e. the intersection is closest to the origin), 
	  so the given order of the facet doesn't matter 
	para::vertices: the coordinates of the vertices of the triangle mesh (numpy array (n,3))
	para::facet: the index of the vertices of each face (numpy array(m,3),dtype=int)
	para::origin: the origin of the ray
	para::direction: the direction vector of the ray
	return:True ,t, and face id of the nearest intersection if the ray intersect with the mesh 
	  or False and -1,-1 if no intersection at all
	"""
	judge = np.zeros(len(facet),dtype=np.bool)
	tvalue = np.zeros(len(facet))
	face_id = np.array([k for k in range(len(facet))])
	for k in range(len(facet)):
		triangle = vertices[facet[k]]
		judge[k],tvalue[k] = ray_triangle_intersect(triangle,origin,direction)
	if tvalue[judge].shape == (0,):
		return False,-1,-1
	# get the id of the minimum t value in the list of intersections
	first_id = np.argmin(tvalue[judge])
	# convert the face id to the whole faces
	first_id = face_id[judge][first_id]
	t = tvalue[first_id]
	return True,t,first_id

def ray_trimesh_intersect_last(vertices, facet, origin, direction):
	"""
	loop the given triangle facet and return the last intersection
	if no intersection at all, return false instead
	..note: here last means t value is largest(i.e. the intersection is furthest to the origin),
		so the given order of the facet doesn't matter
	para::vertices, facet: triangle mesh
	para::origin: the origin of the ray
	para::direction: the direction of the ray
	return::True, t and face id of the nearest intersection if the ray intersect with the mesh
		or False and -1, -1 if no intersection at all
	"""
	judge = np.zeros(len(facet), dtype=np.bool)
	tvalue = np.zeros(len(facet))
	face_id = np.array([k for k in range(len(facet))])
	for k in range(len(facet)):
		triangle = vertices[facet[k]]
		judge[k], tvalue[k] = ray_triangle_intersect(triangle, origin, direction)
	if tvalue[judge].shape == (0,):
		return False, -1, -1
	# get the id of the maximum t value in the list of intersections
	last_id = np.argmax(tvalue[judge])
	# convert the face id to the whole faces
	last_id = face_id[judge][last_id]
	t = tvalue[last_id]
	return True, t, last_id

def ray_trimesh_intersect_all(vertices,facet,origin,direction):
	"""
	give all the intersections of the ray and the given mesh
	para::vertices: the coordinates of the vertices of the triangle mesh (numpy array (n,3))
	para::facet: the index of the vertices of each face (numpy array(m,3),dtype=int)
	para::origin: the coordinate of the ray
	para::direction: the direction vector of the ray
	return:True, t(single value or sorted array) and corresponding index if any intersection
	  or False,-1,-1 if no intersection at all
	"""
	judge = np.zeros(len(facet),dtype=np.bool)
	# the second column stores the face id
	tvalue = np.zeros((len(facet),2))
	for k in range(len(facet)):
		triangle = vertices[facet[k]]
		judge[k],tvalue[k,0] = ray_triangle_intersect(triangle,origin,direction)
		tvalue[k,1] = k
	if len(tvalue[judge]) == 0:
		return False,-1,-1
	intersections = tvalue[judge]
	intersections = intersections[intersections[:,0].argsort()]
	# for remove the duplicates(same point with same t)
	judge = np.ones(len(intersections), dtype=np.bool)
	for k in range(1,len(intersections)):
		if np.allclose(intersections[k,0],intersections[k-1,0]):
			judge[k] = False
	intersections = intersections[judge]
	return True,intersections[:,0],intersections[:,1]


def ray_trimesh_intersect_any(vertices,facet,origin,direction):
	""" 
	judge whether the ray has any intersections with the mesh
	return::True if any intersections or False if no intersections
	"""
	for k in range(len(facet)):
		triangle = vertices[facet[k]]
		if ray_triangle_intersect(triangle,origin,direction)[0] == True:
			return True
	return False


def mesh_contains(vertices,facet,point):
	"""
	to test whether the point is in the mesh
	algorithm: emit a ray from the point in a direction(the direction can be any valid direction)
	and count the number of intersections of the mesh
	if the number is odd, then the point is in the mesh
	"""
	direction = np.array([1.,0.,0.])
	info = ray_trimesh_intersect_all(vertices,facet,point,direction)
	if info[0] == False:
		return False
	else:
		number = len(info[1])
		if (number%2) == 1:
			return True
		else:
			return False
	
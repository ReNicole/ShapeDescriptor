import numpy as np 
import geometry
import ray

# providing tools for sampling
def triangle_face_sample(vertices,facet,number=10):
	"""
	for uniformly sampling points on the surface of the triangle mesh
	para::vertices: the vertices of the mesh(numpy array(n,3))
	para::facet: the facet of the mesh(numpy array(m,3))
	para::number: the number of the samples
	return::samples: the coordinates of the samples
	return::face_index: the index of the faces where the samples are located on 
	..note: for the sampling method,please refer to
	http://mathworld.wolfram.com/TrianglePointPicking.html
	"""
	# get the area of each face 
	area = np.array([geometry.get_area(vertices[facet[k]]) for k in range(facet.shape[0])])
	area_sum = np.sum(area)
	area_cum = np.cumsum(area)
	# get the face index of the sample points
	face_pick = np.random.random(number) * area_sum
	face_index = np.searchsorted(area_cum, face_pick)
	# get (v2-v0),(v1-v0) of each triangle (pull triangles into the form of an origin + 2 vectors)
	triangles = vertices[facet]
	tri_origins = triangles[:,0]
	tri_vectors = triangles[:,1:].copy()
	tri_vectors -= np.tile(tri_origins,(1,2)).reshape((-1,2,3))
	# pull the vectors for the facet we are going to sample from
	tri_origins = tri_origins[face_index]
	tri_vectors = tri_vectors[face_index]
	# randomly generate two 0-1 scalar components to multiply edge vectors by
	random_lengths = np.random.random((len(tri_vectors),2,1))
	# points will be distributed on a quadrilateral if we use 2 0-1 samples
	# if the two scalar components sum less than 1.0 the point will be in the triangle
	# so we find vectors longer than 1.0 and transform them to be inside the triangle
	random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
	random_lengths[random_test] -= 1.0
	random_lengths = np.abs(random_lengths)
	# multiply triangle edge vectors by the random lengths and sum
	# then offset by the origin to generate sample points in space on the triangle
	sample_vector = (tri_vectors * random_lengths).sum(axis=1)
	samples = sample_vector + tri_origins
	return samples,face_index

def triangle_face_sample_even(vertices,facet,radius):
	"""
	for as even as possible to sample the surface
	any pair of points in the result with distance >=radius
	basic idea: compute the approximate number of the samples,then sample twice and remove the close
	..note: may be much slower than the trivial sample method (so only for specific use)
	para::vertices: the vertices of the mesh(numpy array(n,3))
	para::facet: the facet of the mesh(numpy array(m,3))
	para::radius: desired distance lower bound for the samples
	return::samples: the coordinates of the samples
	return::face_index[index]: the index of the faces where the samples(after remove the close points) are located on 
	"""
	area = np.array([geometry.get_area(vertices[facet[k]]) for k in range(facet.shape[0])])
	area_sum = np.sum(area)
	num = (int(area_sum / (radius**2 * np.pi) ) + 1) * 2
	samples,face_index = triangle_face_sample(vertices,facet,number=num)
	samples,index = geometry.remove_close(samples,radius)
	return samples,face_index[index]

def triangle_volume_sample_even(vertices,facet,radius):
	"""
	for as even as possible to sample the volume 
	basic idea is first get the bounding box of the mesh
	and then emit parallel ray from the bottom of the bounding box(the direction is parallel with z positive axis)
	and get the intersections of the ray and mesh
	get the points in the mesh is equal to get the points in the each two intersections
	the intersections will be regarded as the surface samples, and the points in the mesh will be regarded as the volume sumples
	this sample method is specificly useful for constructing a mass spring sysytem from a triangle mesh
	return: surface samples,volume samples
	"""
	bdy_box = geometry.get_bounding_box(vertices)
	# get the origins of the rays
	origin = np.mgrid[bdy_box[0][0]:(bdy_box[0][0]+bdy_box[1][0]):radius, bdy_box[0][1]:(bdy_box[0][1]+bdy_box[1][1]):radius,
	bdy_box[0][2]:(bdy_box[0][2]+0.1):1.].reshape(3,-1).T 
	direction = np.array([0.,0.,1.])
	sect_info = [ray.ray_trimesh_intersect_all(vertices,facet,origin[k],direction) for k in range(len(origin))]
	ssamples = []
	vsamples = []
	for k in range(len(origin)):
		if sect_info[k][0] == False:
			continue
		tvalue = sect_info[k][1]
		pair_num = len(tvalue) / 2
		for j in range(pair_num):
			ssamples.append((origin[k] + tvalue[2*j] * direction))
			ssamples.append((origin[k] + tvalue[2*j+1] * direction))
			sample_num = int((tvalue[2*j+1]-tvalue[2*j]) / radius)
			if sample_num < 2:
				continue
			for i in range(1,sample_num):
				vsamples.append((origin[k] + (tvalue[2*j] + radius*i) * direction))
	return ssamples,vsamples
	
def triangle_in_sample_even(vertices,facet,radius):
	"""
	for as even as possible only sample the points in the triangle return the samples
	"""
	bdy_box = geometry.get_bounding_box(vertices)
	# get the origins of the rays
	origin = np.mgrid[bdy_box[0][0]:(bdy_box[0][0]+bdy_box[1][0]):radius, bdy_box[0][1]:(bdy_box[0][1]+bdy_box[1][1]):radius,
	bdy_box[0][2]:(bdy_box[0][2]+0.1):1.].reshape(3,-1).T 
	direction = np.array([0.,0.,1.])	
	sect_info = [ray.ray_trimesh_intersect_all(vertices,facet,origin[k],direction) for k in range(len(origin))]
	samples = []
	for k in range(len(origin)):
		if sect_info[k][0] == False:
			continue
		tvalue = sect_info[k][1]
		pair_num = len(tvalue) / 2
		for j in range(pair_num):
			sample_num = int((tvalue[2*j+1]-tvalue[2*j]) / radius)
			if np.rint(sample_num) < 2.0:
				continue
			for i in range(1,sample_num):
				samples.append((origin[k] + (tvalue[2*j] + radius*i) * direction))
	return samples
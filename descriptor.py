import os 
import numpy as np 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt 
from pyshtools.expand import SHExpandDH,MakeGridDH,SHExpandDHC
from pyshtools.spectralanalysis import spectrum
from mpl_toolkits.mplot3d import Axes3D
import sample,geometry

"""
for providing a descriptor of the surface mesh(esp. the triangle mesh)
refer to:
	Description of 3D-shape using a complex function on the sphere
	D.V. Vranic, D. Saupe, 2002
"""

def uniform_volume(vertices,facet,reset_volume=1.):
	"""
	reset the volume uniformly so that the volume of the different mesh will be the same
	"""
	src_volume = geometry.get_trimesh_volume(vertices,facet)
	scale = (src_volume / reset_volume) ** (1./3)
	vertices = vertices / scale
	return vertices,facet

def gridSampleXU(vertices,facet):
	"""
	get samples of x(u) of the given mesh and make a grid form of the sample
	for the convenience to apply discrete spherical harmonics transform
	.. note: the definition of x(u) can be seen in func::descriptorRS
	output::grid_xu: numpy array (180,360)
	
	"""
	# to generate the random samples for each degree in (-180,180),(-90,90)
	# the result is (n,3) numpy array, each row is the cartesian coordinate of sample point
	#samples = trimesh.sample.sample_surface(mesh,360*180*3)
	vertices,facet = uniform_volume(vertices,facet)
	samples = sample.triangle_face_sample(vertices,facet,number=(360*180*3))[0]
	# get aligned
	#samples = samples - np.array([mesh.center_mass]*samples.shape[0])
	centroid = geometry.get_trimesh_centroid(vertices,facet)
	samples = samples - np.array([centroid]*samples.shape[0])
	# convert the cartesian coordinate to spherical coordinate (and the first coordinate radius is the desired x(u))
	spsam = geometry.xyz2sp(samples)
	# use nearest point on each grid position as the value and generate the grid form of the sample value
	xi = np.linspace(-np.pi,np.pi,360)
	yi = np.linspace(-0.5*np.pi,0.5*np.pi,180)
	grid_xu = griddata(spsam[:,1:3], spsam[:,0], (xi[None,:],yi[:,None]), method='nearest')
	#xi,yi = np.mgrid[-np.pi:np.pi:360j, -0.5*np.pi:0.5*np.pi:180j]
	#grid_xu = griddata(spsam[:,1:3], spsam[:,0], (xi,yi), method='nearest')
	return grid_xu

def gridSampleYU(vertices,facet):
	""" 
	get samples of y(u) of the given mesh and make a grid form of the sample
	for the convenience to apply discrete spherical harmonics transform
	.. note: the definition of y(u) can be seen in func::descriptorSS
	-------------------------------------------------------------------------------------------
	output::grid_yu: numpy array (180,360)
	"""
	# sample
	vertices,facet = uniform_volume(vertices,facet)
	samples,face_index = sample.triangle_face_sample(vertices,facet,number=(360*180*3))
	# align
	centroid = geometry.get_trimesh_centroid(vertices,facet)
	samples = samples - np.array([centroid]*samples.shape[0])
	# get the cross product of (v2-v0),(v1-v0) of each face corresponding to the sample points
	tri_cross = np.array([np.cross((vertices[facet[k,2]]-vertices[facet[k,0]]),(vertices[facet[k,1]]-vertices[facet[k,0]])) for 
		k in range(len(facet))])
	sample_cross = tri_cross[face_index]
	# get the norm of each cross product
	sample_corssnorm = np.linalg.norm(sample_cross,axis=1)
	# get y(u) for each sample point (without normalization)
	yu = np.array([np.dot(samples[k],sample_cross[k]) for k in range(len(samples))])
	# normalize
	yu = np.divide(yu,sample_corssnorm)
	# make sure y(u) is non-negative value(since we suppose dot(u,n(u)) should non-negtive)
	yu = np.abs(yu)
	# convert the cartesian coordinate to spherical coordinate
	spsam = geometry.xyz2sp(samples)
	# use nearest point on each grid position as the value and generate the grid form of the sample value y(u)
	xi = np.linspace(-np.pi,np.pi,360)
	yi = np.linspace(-0.5*np.pi,0.5*np.pi,180)
	grid_yu = griddata(spsam[:,1:3], yu, (xi[None,:],yi[:,None]), method='nearest')
	#xi,yi = np.mgrid[-np.pi:np.pi:360j, -0.5*np.pi:0.5*np.pi:180j]
	#grid_yu = griddata(spsam[:,1:3], yu, (xi,yi), method='nearest')
	return grid_yu


def gridSampleRU(vertices,facet):
	""" 
	get samples of r(u) of the given mesh and make a grid form of the sample
	for the convenience to apply discrete spherical harmonics transform
	.. note: the definition of r(u) can be seen in func::descriptorCS
	-------------------------------------------------------------------------------------------
	output::grid_ru: numpy array (180,360)
	"""
	# get enough sample points to cover the sphere S^2 (-180,180)*(-90,90) (degree based)
	# sample process: multiply triangle edge vectors by the random lengths and sum 
	# then offset by the origin to generate sample points in space on the triangle
	vertices,facet = uniform_volume(vertices,facet)
	samples,face_index = sample.triangle_face_sample(vertices,facet,number=(360*180*3))
	# align
	centroid = geometry.get_trimesh_centroid(vertices,facet)
	samples = samples - np.array([centroid]*samples.shape[0])
	# get the cross product of (v2-v0),(v1-v0) of each face corresponding to the sample points
	tri_cross = np.array([np.cross((vertices[facet[k,2]]-vertices[facet[k,0]]),(vertices[facet[k,1]]-vertices[facet[k,0]])) for 
		k in range(len(facet))])
	sample_cross = tri_cross[face_index]
	# get the norm of each cross product
	sample_corssnorm = np.linalg.norm(sample_cross,axis=1)
	# get y(u) for each sample point (without normalization)
	yu = np.array([np.dot(samples[k],sample_cross[k]) for k in range(len(samples))])
	# normalize
	yu = np.divide(yu,sample_corssnorm)
	# make sure y(u) is non-negative value(since we suppose dot(u,n(u)) should non-negtive)
	yu = np.abs(yu)
	# convert the cartesian coordinate to spherical coordinate (and the first coordinate radius is the desired x(u))
	spsam = geometry.xyz2sp(samples)
	# compute r(u) for each sample points
	ru = spsam[:,0] + 1j * yu
	# use nearest point on each grid position as the value and generate the grid form of the sample value r(u)
	xi = np.linspace(-np.pi,np.pi,360)
	yi = np.linspace(-0.5*np.pi,0.5*np.pi,180)
	grid_ru = griddata(spsam[:,1:3], ru, (xi[None,:],yi[:,None]), method='nearest')
	#xi,yi = np.mgrid[-np.pi:np.pi:360j, -0.5*np.pi:0.5*np.pi:180j]
	#grid_ru = griddata(spsam[:,1:3], ru, (xi,yi), method='nearest')
	return grid_ru	

def showSampleXU(vertices,facet):
	"""
	visualize the samples of x(u) generated by the func::gridSampleXU
	"""
	fig,ax = plt.subplots(1,1,figsize=(10,5))
	ax.imshow(gridSampleXU(vertices,facet),extent=(-180,180,-90,90),cmap='viridis')
	plt.show()

def showSampleYU(vertices,facet):
	"""
	visulize the samples of y(u) generated by the func::gridSampleYU
	"""
	fig,ax = plt.subplots(1,1,figsize=(10,5))
	ax.imshow(gridSampleYU(vertices,facet),extent=(-180,180,-90,90),cmap='viridis')
	plt.show()	

def descriptorRS(vertices,facet, coef_num_sqrt=13):
	""" 
	apply spherical harmonics transform on the real function on the sphere S^2
	and use the truncated coefficients as the descriptor of the given mesh
	refer to:
		Description of 3D-shape using a complex function on the sphere
		D.V. Vranic, D. Saupe, 2002
		(this is the ray-based method mentioned in the paper)
	basic idea:
		x: S^2 --> [0,inf) in R, u |--> max{x>=0| xu in mesh-surface or {0}}
		get enough sample of x(u) and apply spherical harmonics transform on it
	---------------------------------------------------------------------------------
	para::coef_num_sqrt: the square root of desired number of the dimensions of the 
	    shape of the mesh which is also the number of the truncated coefficients
	output:: coeffs_trunc: list with size coef_num_sqrt^2
	..the desired shape descriptor
	"""
	# get the sample value of x(u)
	zi = gridSampleXU(vertices,facet)
	# generate the sherical harmonics coefficients
	coeffs = SHExpandDH(zi,sampling=2)
	coeffs_trunc=[[coeffs[0,k,:(k+1)].tolist(),coeffs[1,k,1:(k+1)].tolist()] for k in range(coef_num_sqrt)]
	coeffs_trunc = [var for sublist in coeffs_trunc for subsublist in sublist for var in subsublist]
	coeffs_trunc = np.array(coeffs_trunc)
	return coeffs_trunc

def descriptorSS(vertices,facet, coef_num_sqrt=13):
	"""
	apply spherical harmonics transform on the imaginary part of complex function on the sphere S^2
	and use the truncated coefficients as the descriptor of the given mesh
	refer to:
		Description of 3D-shape using a complex function on the sphere
		D.V. Vranic, D. Saupe, 2002
		(this is the shading-based method mentioned in the paper)	
	basic idea:
		y: S^2 --> [0,inf) in R, u |-->  0  if x(u) = 0 ; dot(u,n(u)), otherwise
		where n(u) is the normal vector of the mesh at the point x(u)*u 
		(the fast intersection point on the surface with ray in direction u)
		get enough sample of y(u) and apply spherical harmonics transform on it
	---------------------------------------------------------------------------------
	para::coef_num_sqrt: the square root of desired number of the dimensions of the 
	    shape of the mesh which is also the number of the truncated coefficients
	output:: coeffs_trunc: list with size coef_num_sqrt^2
	..the desired shape descriptor
	"""
	# get the sample value of y(u)
	zi = gridSampleYU(vertices,facet)
	# generate the sherical harmonics coefficients
	coeffs = SHExpandDH(zi,sampling=2)
	coeffs_trunc=[[coeffs[0,k,:(k+1)].tolist(),coeffs[1,k,1:(k+1)].tolist()] for k in range(coef_num_sqrt)]
	coeffs_trunc = [var for sublist in coeffs_trunc for subsublist in sublist for var in subsublist]
	coeffs_trunc = np.array(coeffs_trunc)
	return coeffs_trunc	

def descriptorCS(vertices,facet, coef_num_sqrt=13):
	"""
	apply spherical harmonics transform on the complex function on the sphere S^2
	and use the truncated coefficients as the descriptor of the given mesh
	refer to:
		Description of 3D-shape using a complex function on the sphere
		D.V. Vranic, D. Saupe, 2002
		(this is the complex feature vector method mentioned in the paper)	
	basic idea:
		r: S^2 --> C, u |--> x + y, with
		x: S^2 --> [0,inf) in R, u |--> max{x>=0| xu in mesh-surface or {0}}
		y: S^2 --> [0,inf) in R, u |-->  0  if x(u) = 0 ; dot(u,n(u)), otherwise
		where n(u) is the normal vector of the mesh at the point x(u)*u 
		(the fast intersection point on the surface with ray in direction u)
		get enough sample of r(u) and apply spherical harmonics transform (complex form) on it
	---------------------------------------------------------------------------------
	para::coef_num_sqrt: the square root of desired number of the dimensions of the 
	    shape of the mesh which is also the number of the truncated coefficients
	output:: coeffs_trunc: list with size coef_num_sqrt^2 (use the absolute value)
	..the desired shape descriptor
	"""
	# get the sample value of r(u)
	zi = gridSampleRU(vertices,facet)
	# generate the sherical harmonics coefficients
	coeffs = np.abs(SHExpandDHC(zi,sampling=2)) 
	coeffs_trunc=[[coeffs[0,k,:(k+1)].tolist(),coeffs[1,k,1:(k+1)].tolist()] for k in range(coef_num_sqrt)]
	coeffs_trunc = [var for sublist in coeffs_trunc for subsublist in sublist for var in subsublist]
	coeffs_trunc = np.array(coeffs_trunc)
	return coeffs_trunc	

def showReXU2d(vertices,facet, coef_num_sqrt=13):
	"""
	visualize the reconstruction of x(u) with truncated spherical system 
	para::coef_num_sqrt: the square root of desired number of the dimensions of the 
	    shape of the mesh which is also the number of the truncated coefficients
	"""
	# get the sample value of x(u)
	zi = gridSampleXU(vertices,facet)
	# generate the spherical harmonics coefficients
	coeffs = SHExpandDH(zi,sampling=2)
	# generate the filter of the coefficients
	for k in range(coef_num_sqrt):
		coeffs[:,k,(k+1):] = 0
	# reconstruct x(u) with truncated coefficients
	zi_filtered = MakeGridDH(coeffs,sampling=2)
	fig,ax = plt.subplots(1,1,figsize=(10,10))
	ax.imshow(zi_filtered,extent=(-180,180,-90,90),cmap='viridis')
	plt.show()

def showReYU2d(vertices,facet, coef_num_sqrt=13):
	"""
	visualize the reconstruction of y(u) with truncated spherical system 
	para::coef_num_sqrt: the square root of desired number of the dimensions of the 
	    shape of the mesh which is also the number of the truncated coefficients
	"""
	# get the sample value of y(u)
	zi = gridSampleYU(vertices,facet)
	# generate the spherical harmonics coefficients
	coeffs = SHExpandDH(zi,sampling=2)
	# generate the filter of the coefficients
	for k in range(coef_num_sqrt):
		coeffs[:,k,(k+1):] = 0
	# reconstruct y(u) with truncated coefficients
	zi_filtered = MakeGridDH(coeffs,sampling=2)
	fig,ax = plt.subplots(1,1,figsize=(10,10))
	ax.imshow(zi_filtered,extent=(-180,180,-90,90),cmap='viridis')
	plt.show()

def showScatterMesh(vertices,facet):
	"""
	to see the scatter mesh (i.e. plot the vertices of the mesh in 3d space)
	for visual comparison with other reconstruction of the mesh
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2])
	ax.set_xlabel('X label')
	ax.set_ylabel('Y label')
	ax.set_zlabel('Z label')
	plt.show()

def showScatterReXU3d(vertices,facet,coef_num_sqrt=13):
	"""
	visualize the reconstruction mesh use the truncated spherical harmonics coefficients
	generated by real function x(u) on the sphere
	"""
	# pre: generate the truncated coefficients and reconstruct x(u)
	zi = gridSampleXU(vertices,facet)
	coeffs = SHExpandDH(zi,sampling=2)
	for k in range(coef_num_sqrt):
		coeffs[:,k,(k+1):] = 0
	zi_filtered = MakeGridDH(coeffs,sampling=2)
	# reconstruct the scatter point of the mesh
	sp = np.array([[zi_filtered[v][u],(u-180.0)/180.0*np.pi,(v-90.0)/180.0*np.pi] for u in range(360) for v in range(180)])
	xyz = geometry.sp2xyz(sp)
	# plot the scatter point
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2])
	ax.set_xlabel('X label')
	ax.set_ylabel('Y label')
	ax.set_zlabel('Z label')
	plt.show()

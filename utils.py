import numpy as np 
import os
import trimesh
import quickmodel as qm

# this file is for providing some basic tool funtions for editing



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
	.. note: each row is (r,u,v) where r is radius, u in (-pi,pi), v in (-0.5pi,0.5pi)
	output::xyz: numpy array (n,3) the cartesian coordinate
	"""
	xyz = np.zeros(sp.shape)
	xyz[:,0] = sp[:,0] * np.cos(sp[:,1]) * np.cos(sp[:,2])
	xyz[:,1] = sp[:,0] * np.sin(sp[:,1]) * np.cos(sp[:,2])
	xyz[:,2] = sp[:,0] * np.sin(sp[:,2])
	return xyz

def sample_volume(tmesh,num = 100, bytrimesh = True, evenly = False):
	"""
	for sampling the points in the volume of the given mesh (usually triangle mesh)
	para::mesh: mesh to be sampled
	para::num: the amount of the sample points
	para::bytrimesh: using the trimesh interface or not
	para::evenly: distribute the points evenly or not
	output::samples: the sample result,(n,3) array
	..note: n <= num as samples are produces by rejecting points
	"""
	if bytrimesh:
		if evenly:
			radius = (tmesh.volume/num) ** (1./3)
			temp = trimesh.sample.volume_mesh(tmesh, num*5)
			samples = trimesh.points.remove_close(temp,radius)
		else:
			samples = trimesh.sample.volume_mesh(tmesh,num)
	return samples

def sample_for_mss(tmesh,output_surnum = True):
	"""
	sample points on the surface and in the volume of the given triangle mesh for generate a mass spring system
	..note: this is a temporary version
	..still use trimesh for convenience
	..the vertices of the original mesh will be all regarded as the surface particles
	..and the radius of the points will computed according to the surface area and surface particles number
	para::output_surnum: whether to output the number of the surface particle number
	..note: surface particles will be arranged at first,then the particles in the volume
	"""
	# the number of surface particles is equal to the number of the vertices of the triangle mesh
	surnum = tmesh.vertices.shape[0]
	# get the surface particles(vertices of the triangle mesh actually)
	samples = np.array([tmesh.vertices[k] for k in range(tmesh.vertices.shape[0])])
	# compute the radius
	radius = np.sqrt(tmesh.area / (2*surnum))
	# compute the number of the particles to be sampled in the volume of the mesh
	innum = int(tmesh.volume / (radius ** 3))
	# approximately evenly sample the particles in the volume
	temp = trimesh.sample.volume_mesh(tmesh,innum*5)
	samples2 = trimesh.points.remove_close(temp,radius)
	# concatenate the surface samples and the in volume samples
	samples = np.concatenate((samples,samples2),axis=0)
	if output_surnum:
		return samples,surnum
	else:
		return samples

#ms = qm.sphere
#samples = sample_for_mss(ms)

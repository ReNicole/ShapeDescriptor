import os
import numpy as np 
from scipy import sparse
from scipy.sparse.linalg import lsqr, cg, eigsh,spsolve
import trimesh 



#for laplace mesh edit
class LapMeshEdit:
	""" intended for the laplacian mesh edit """

	def __init__(self,src_mesh):
		""" initial the laplacian mesh edit process """
		# amount of the vertices of the processing mesh
		self.vertices_num_=len(src_mesh.vertices)
		# laplacian coordinates
		self.lx_array=[]
		self.ly_array=[]
		self.lz_array=[]
		# the id of the fixed point and handle point
		self.fixed_id_array=[]
		self.handle_id_array=[]
		# handle change record
		self.delta_handle_x=[]
		self.delta_handle_y=[]
		self.delta_handle_z=[]

	def setFixedId(self,fixId):
		""" reset the id of fixed points """
		# clear the history record
		self.fixed_id_array=[]
		# assign new id of the fixed points
		for i in range(len(fixId)):
			self.fixed_id_array.append(fixId[i])

	def getFixedId(self):
		""" return the id of current fixed points """
		return self.fixed_id_array

	def setHandleId(self,handleId):
		""" reset the id of handle points """
		# clear the history record
		self.handle_id_array=[]
		# assign new id of the handle point
		for i in range(len(handleId)):
			self.handle_id_array.append(handleId[i])

	def getHandleId(self):
		""" return the id of current handle points """
		return self.handle_id_array

	def setLaplacianCoord(self,src_mesh):
		""" 
		set the laplacian coordinates for the given mesh (x,y,z coord seperately) 
		note that the laplacian coordinates for a mesh can be seen as invariant 
		during the deformation for convenience of computation
		if large artifacts occur, please reset the laplacian coordinate
		"""
		# clear the history record in case of resetting
		self.lx_array=[]
		self.ly_array=[]
		self.lz_array=[]
		for i in range(self.vertices_num_):
			# get the degree of the current vertex, i.e. the number of the neighbors
			degree_i_=len(src_mesh.vertex_neighbors[i])
			# initial the sum of the coordinate of the neighbors
			temp_x_=temp_y_=temp_z_=0.0
			# compute the sum of coordinates of neighbors
			for j in range(degree_i_):
				temp_x_+=src_mesh.vertices[src_mesh.vertex_neighbors[i][j]][0]
				temp_y_+=src_mesh.vertices[src_mesh.vertex_neighbors[i][j]][1]
				temp_z_+=src_mesh.vertices[src_mesh.vertex_neighbors[i][j]][2]
			# set the laplacian coordiante value
			temp_x_=src_mesh.vertices[i][0]-1.0/degree_i_*temp_x_
			temp_y_=src_mesh.vertices[i][1]-1.0/degree_i_*temp_y_
			temp_z_=src_mesh.vertices[i][2]-1.0/degree_i_*temp_z_
			self.lx_array.append(temp_x_)
			self.ly_array.append(temp_y_)
			self.lz_array.append(temp_z_)

	def setBasicCoefMat(self,src_mesh):
		""" 
		this is intended to set the coefficient matrix for the laplace mesh deformation
		note that the part without handle points and fixed points are invariant
		(unless the laplacian coordinate is reset)
		so we split the coefficient matrix into two parts--
		one is only related to the mesh property
		the other is related to the information of handle & fixed points
		"""
		# serves as the triplets for constructing the sparse matrix
		# the row index
		basic_coef_mat_r=[]
		# the column index
		basic_coef_mat_c=[]
		# the data
		basic_coef_mat_d=[]
		for i in range(self.vertices_num_):
			tmpt_mat_i_=np.array([
				[self.lx_array[i],0,self.lz_array[i],-self.ly_array[i],0,0,0],
				[self.ly_array[i],-self.lz_array[i],0,self.lx_array[i],0,0,0],
				[self.lz_array[i],self.ly_array[i],-self.lx_array[i],0,0,0,0]])
			degree_i_=len(src_mesh.vertex_neighbors[i])
			A_i=np.zeros((3*degree_i_+3,7))
			A_i[0:3]=np.array([
				[src_mesh.vertices[i][0],0,src_mesh.vertices[i][2],-src_mesh.vertices[i][1],1,0,0],
				[src_mesh.vertices[i][1],-src_mesh.vertices[i][2],0,src_mesh.vertices[i][0],0,1,0],
				[src_mesh.vertices[i][2],src_mesh.vertices[i][1],-src_mesh.vertices[i][0],0,0,0,1]])
			for j in range(degree_i_):
				# short reference for the id of the j-th neighbor vertex
				t_id=src_mesh.vertex_neighbors[i][j]
				A_i[3*(j+1):(3*(j+2))]=np.array([
					[src_mesh.vertices[t_id][0],0,src_mesh.vertices[t_id][2],-src_mesh.vertices[t_id][1],1,0,0],
					[src_mesh.vertices[t_id][1],-src_mesh.vertices[t_id][2],0,src_mesh.vertices[t_id][0],0,1,0],
					[src_mesh.vertices[t_id][2],src_mesh.vertices[t_id][1],-src_mesh.vertices[t_id][0],0,0,0,1]])
			coef_mat_i_ =(tmpt_mat_i_.dot(np.linalg.inv((A_i.T).dot(A_i)))).dot(A_i.T)
			# set the triplet for the basic coefficient matrix: part 1 for the current vertex
			# the row index
			basic_coef_mat_r+=[
			i,i,i,
			i+self.vertices_num_,i+self.vertices_num_,i+self.vertices_num_,
			i+2*self.vertices_num_,i+2*self.vertices_num_,i+2*self.vertices_num_]
			# the column index
			basic_coef_mat_c+=[
			i,i+self.vertices_num_,i+2*self.vertices_num_,
			i,i+self.vertices_num_,i+2*self.vertices_num_,
			i,i+self.vertices_num_,i+2*self.vertices_num_]
			# the corresponding data
			basic_coef_mat_d+=[
			1 - coef_mat_i_[0,0],-coef_mat_i_[0,1],-coef_mat_i_[0,2],
			-coef_mat_i_[1,0],1 - coef_mat_i_[1,1],-coef_mat_i_[1,2],
			-coef_mat_i_[2,0],-coef_mat_i_[2,1],1 - coef_mat_i_[2,2]]
			for j in range(degree_i_):
				# short reference for the id of the j-th neighbor vertex
				t_id=src_mesh.vertex_neighbors[i][j]
				# the row index
				basic_coef_mat_r+=[
				i,i,i,
				i+self.vertices_num_,i+self.vertices_num_,i+self.vertices_num_,
				i+2*self.vertices_num_,i+2*self.vertices_num_,i+2*self.vertices_num_]
				# the column index
				basic_coef_mat_c+=[
				t_id,t_id+self.vertices_num_,t_id+2*self.vertices_num_,
				t_id,t_id+self.vertices_num_,t_id+2*self.vertices_num_,
				t_id,t_id+self.vertices_num_,t_id+2*self.vertices_num_]
				# the corresponding data
				basic_coef_mat_d+=[
				-1.0/degree_i_ - coef_mat_i_[0,3*(j+1)], -coef_mat_i_[0,3*(j+1)+1], -coef_mat_i_[0,3*(j+1)+2],
				-coef_mat_i_[1,3*(j+1)], -1.0/degree_i_ - coef_mat_i_[1,3*(j+1)+1], -coef_mat_i_[1,3*(j+1)+2],
				-coef_mat_i_[2,3*(j+1)], -coef_mat_i_[2,3*(j+1)+1], -1.0/degree_i_ - coef_mat_i_[2,3*(j+1)+2]]
		basic_coef_mat = sparse.coo_matrix((basic_coef_mat_d,(basic_coef_mat_r,basic_coef_mat_c)),
			shape=(3*self.vertices_num_,3*self.vertices_num_))
		# at current stage just return the matrix
		return basic_coef_mat

	def setExtraCoefMat(self):
		"""
		this part is to set the other part of the coefficient matrix
		which is related to the fixed & handle points part
		"""
		# set the weight for the fixed & handle points
		extra_weight=20.0
		# row index,column index and data seperately
		extra_coef_mat_r=[]
		extra_coef_mat_c=[]
		extra_coef_mat_d=[]
		fixed_num=len(self.fixed_id_array)
		handle_num=len(self.handle_id_array)
		for i in range(fixed_num):
			extra_coef_mat_r+=[i,i+fixed_num,i+2*fixed_num]
			extra_coef_mat_c+=[self.fixed_id_array[i],self.fixed_id_array[i]+self.vertices_num_,self.fixed_id_array[i]+self.vertices_num_*2]
			extra_coef_mat_d+=[extra_weight,extra_weight,extra_weight]
		for i in range(handle_num):
			extra_coef_mat_r+=[i+3*fixed_num,i+3*fixed_num+handle_num,i+3*fixed_num+2*handle_num]
			extra_coef_mat_c+=[self.handle_id_array[i],self.handle_id_array[i]+self.vertices_num_,self.handle_id_array[i]+self.vertices_num_*2]
			extra_coef_mat_d+=[extra_weight,extra_weight,extra_weight]
		extra_coef_mat=sparse.coo_matrix((extra_coef_mat_d,(extra_coef_mat_r,extra_coef_mat_c)),
			shape=(3*(fixed_num+handle_num),3*self.vertices_num_))
		return extra_coef_mat

	def setCoefMat(self,src_mesh):
		""" concatenate the above matrix to give the real coefficient matrix in computation """
		temp_mat_a=self.setBasicCoefMat(src_mesh)
		temp_mat_b=self.setExtraCoefMat()
		coef_mat_=sparse.vstack([temp_mat_a,temp_mat_b])
		return coef_mat_

	def setRHS(self,src_mesh):
		""" set the right hand side of the linear equation """
		# set extra weight for the handle points and the fixed points
		extra_weight = 20.0
		fixed_num = len(self.fixed_id_array)
		handle_num = len(self.handle_id_array)
		rhs = np.zeros(3*(self.vertices_num_+fixed_num+handle_num))
		for i in range(fixed_num):
			rhs[i+3*self.vertices_num_] = extra_weight*src_mesh.vertices[self.fixed_id_array[i]][0]
			rhs[i+3*self.vertices_num_+fixed_num] = extra_weight*src_mesh.vertices[self.fixed_id_array[i]][1]
			rhs[i+3*self.vertices_num_+2*fixed_num] = extra_weight*src_mesh.vertices[self.fixed_id_array[i]][2]
		for i in range(handle_num):
			rhs[i+3*(self.vertices_num_+fixed_num)]=extra_weight*src_mesh.vertices[self.handle_id_array[i]][0]
			rhs[i+3*(self.vertices_num_+fixed_num)+handle_num]=extra_weight*src_mesh.vertices[self.handle_id_array[i]][1]
			rhs[i+3*(self.vertices_num_+fixed_num)+2*handle_num]=extra_weight*src_mesh.vertices[self.handle_id_array[i]][2]
		return rhs

	def solveEquation(self,coef,rhs):
		""" 
		the result here is the new x,y,z position of the mesh
		the order: xxxxxxxxx....yyyyyyyyyy.....zzzzzzzzzzzzzzzz
		"""
		mat_aat=coef.T.dot(coef)
		result=spsolve(mat_aat,coef.T.dot(rhs))
		return result

	def transition(self,src_mesh,pos,force):
		"""
		update the mesh when external force is applied
		input
			src_mesh: the mesh to deform
			pos: the (u,v) pos of the point where the force is applied
			force: use the deform distance as the representation of the force
		output
			new mesh
		..note:
			since the current laplace deform framework is applied on the vertices of the mesh
			so even the (u,v) refers a point on the surface of the mesh not exactly the vertice
			for convenience we adopt vertices of the triangle the point is located as the handle point 
			and the vertices of the triangle where (-u,-v) lies as the fixed point
		..note:
			here u and v are radius based!!
			also: u-2pi, v-pi
		"""
		# set the handle point and the fixed point
		ray_origin = np.array([src_mesh.centroid]*2)
		u = pos[0]
		v = pos[1]
		ray_direction = np.array([
			[np.cos(v)*np.cos(u),np.cos(v)*np.sin(u),np.sin(v)],
			[-np.cos(v)*np.cos(u),-np.cos(v)*np.sin(u),-np.sin(v)]])
		action_data = src_mesh.ray.intersects_location(ray_origin,ray_direction)
		callable_id = [0,0,0]
		for k in range(3):
			callable_id[k]=(src_mesh.faces[action_data[2][0]][k])
		self.setHandleId(callable_id)
		for k in range(3):
			callable_id[k] = src_mesh.faces[action_data[2][1]][k]
		self.setFixedId(callable_id)
		self.setLaplacianCoord(src_mesh)
		#self.setHandleId(src_mesh.faces(action_data[2][0]))
		#self.setFixedId(src_mesh.faces(action_data[2][1]))
		# set the new pos for the handle point; first compute the deformation direction for them
		dir1 = np.cross((src_mesh.vertices[self.handle_id_array[2]]-src_mesh.vertices[self.handle_id_array[0]]),
			(src_mesh.vertices[self.handle_id_array[1]]-src_mesh.vertices[self.handle_id_array[0]]))
		if(np.dot(dir1,ray_direction[0])>0):			# the direction is outwards
			dir1 = -dir1
		for k in range(3):
			src_mesh.vertices[self.handle_id_array[k]] = src_mesh.vertices[self.handle_id_array[k]] + force*dir1
		# construct the equation to get the new position
		coef_mat = self.setCoefMat(src_mesh)
		rhs = self.setRHS(src_mesh)
		new_pos = self.solveEquation(coef_mat,rhs)
		for k in range(self.vertices_num_):
			src_mesh.vertices[k]=[new_pos[k],new_pos[k+self.vertices_num_],new_pos[k+2*self.vertices_num_]]
		return src_mesh
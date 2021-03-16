import numpy as np
import os, sys, pathlib, math, itertools, bz2, io

from concurrent.futures import ThreadPoolExecutor

script_path = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/").strip()
sys.path.append(script_path+os.path.sep+".."+os.path.sep)

import lib.trig

# known limitation: cannot process .obj files with mixed polygonal faces (as in quads AND triangles f.i.)


def read_obj(obj_file):
	obj = Wavefront()
	obj.read(obj_file)
	return obj

def write_obj(obj_file, obj):
	obj.write(obj_file)
	
	
def read_compressed_obj(file):
	obj = Wavefront()
	with bz2.open(file, "rt") as f:
		obj.read(f)
	return obj
		
def write_compressed_obj(file, obj):
	with bz2.open(file, "wt") as f:
		obj.write(f)


class Wavefront():
	
	def __init__(self):
		
		self.vertices = None
		self.texturecoords = None
		self.normals = None
		self.colors = None
		self.facevertices = None
		self.facetexcoords = None
		self.facevertnormals = None
		
		self.extmaterials = []
		
		self.texturesize = None
		
		
	def read(self, obj_file):
		
		vertices = []
		vertcolors = []
		vertnorms = []
		texturecoords = []
		facevertinds = []
		facetextureinds = []
		facevertnorminds = []
		extmaterials = []
		
		texturesize = None
		facesize = None
		vertsize = None
		
		f = obj_file
		if not isinstance(obj_file, io.IOBase):
			f = open(obj_file, "r")
		
		for line in f.readlines():
			if line.startswith("v "):
				verts = [float(x) for x in line.split(" ")[1:]]
				if len(verts) == 3:
					vertices += verts
					if vertsize is None:
						vertsize = 3
				elif len(verts) == 6:
					vertices += verts[:3]
					vertcolors += verts[3:]
					if vertsize is None:
						vertsize = 3
				elif len(verts) == 4:
					vertices += verts
					if vertsize is None:
						vertsize = 4
			elif line.startswith("f "):
				idstrs = line.split(" ")[1:]
				if "/" in line:
					for s in idstrs:
						idvalstrs = s.split("/")
						facevertinds.append(int(idvalstrs[0]))
						if idvalstrs[1]:
							facetextureinds.append(int(idvalstrs[1]))
						if len(idvalstrs) >= 3 and idvalstrs[2]:
							facevertnorminds.append(int(idvalstrs[2]))
				else:
					facevertinds += [int(x) for x in idstrs]
				if facesize is None:
					facesize = len(idstrs)
			elif line.startswith("vt "):
				ts = [float(x) for x in line.split(" ")[1:]]
				texturecoords += ts
				if texturesize is None:
					texturesize = len(ts)
			elif line.startswith("vn "):
				vns = [float(x) for x in line.split(" ")[1:]]
				vertnorms += vns
			elif line.startswith("mtllib "):
				mtl = line.split(" ")[-1]
				extmaterials.append(mtl)
				
		if not isinstance(obj_file, io.IOBase):
			f.close()
		
		if vertices:
			self.vertices = np.reshape(np.array(vertices, dtype=np.float32), (-1, vertsize))
		if texturecoords:
			self.texturecoords = np.reshape(np.array(texturecoords, dtype=np.float32), (-1, texturesize))
			self.texturesize = texturesize
		if vertcolors:
			self.colors = np.reshape(np.array(vertcolors, dtype=np.float32), (-1, vertsize))
		if vertnorms:
			self.normals = np.reshape(np.array(vertnorms, dtype=np.float32), (-1, vertsize))
		if facevertinds:
			self.facevertices = np.reshape(np.array(facevertinds, dtype=np.int32), (-1, facesize))
		if facetextureinds:
			self.facetexcoords = np.reshape(np.array(facetextureinds, dtype=np.int32), (-1, facesize))
		if facevertnorminds:
			self.facevertnormals = np.reshape(np.array(facevertnorminds, dtype=np.int32), (-1, facesize))
		if extmaterials:
			self.extmaterials = extmaterials
			
			
	def write(self, obj_file):
		
		f = obj_file
		if not isinstance(obj_file, io.IOBase):
			f = open(obj_file, "w+")
		
		for mtl in self.extmaterials:
			f.write("mtllib "+mtl+"\n")
		
		f.write("\n")
		
		if not self.vertices is None:
			if self.colors is None:
				s = "v" + (" %f" * self.vertices.shape[1]) + "\n"
				for i in range(self.vertices.shape[0]):
					f.write(s % tuple(self.vertices[i].tolist()))
			else:
				s = "v" + (" %f" * (self.vertices.shape[1] + self.colors.shape[1])) + "\n"
				for i in range(self.vertices.shape[0]):
					f.write(s % tuple(self.vertices[i].tolist() + self.colors[i].tolist()))
					
		if not self.texturecoords is None:
			s = "vt" + (" %f" * self.texturesize) + "\n"
			for i in range(self.texturecoords.shape[0]):
				f.write(s % tuple(self.texturecoords[i].tolist()))
				
		if not self.normals is None:
			s = "vn" + (" %f" * self.normals.shape[1]) + "\n"
			for i in range(self.normals.shape[0]):
				f.write(s % tuple(self.normals[i].tolist()))
				
		if not self.facevertices is None:
			if self.facetexcoords is None:
				if self.facevertnormals is None:
					s = "f" + (" %d" * self.facevertices.shape[1]) + "\n"
					for i in range(self.facevertices.shape[0]):
						f.write(s % tuple(self.facevertices[i].tolist()))
				else:
					s = "f" + (" %d//%d" * self.facevertices.shape[1]) + "\n"
					for i in range(self.facevertices.shape[0]):
						f.write(s % tuple(itertools.chain.from_iterable(zip(self.facevertices[i].tolist(), self.facevertnormals[i].tolist()))))
			else:
				if self.facevertnormals is None:
					s = "f" + (" %d/%d" * self.facevertices.shape[1]) + "\n"
					for i in range(self.facevertices.shape[0]):
						f.write(s % tuple(itertools.chain.from_iterable(zip(self.facevertices[i].tolist(), self.facetexcoords[i].tolist()))))
				else:
					s = "f" + (" %d/%d/%d" * self.facevertices.shape[1]) + "\n"
					for i in range(self.facevertices.shape[0]):
						f.write(s % tuple(itertools.chain.from_iterable(zip(self.facevertices[i].tolist(), self.facetexcoords[i].tolist(), self.facevertnormals[i].tolist()))))
						
		if not isinstance(obj_file, io.IOBase):
			f.close()
							
							
	def subdivided(self, multithread=True):
		
		# Simple subdivision, currently only vertices and faces are supported (no vertex colors, texcoords etc.)
		
		if self.facevertices.shape[1] != 3:
			print("Error: Only triangle meshes supported for subdivision!")
			return
		
		dst_faces = np.empty((self.facevertices.shape[0] * 4, 3), self.facevertices.dtype)
		dst_vertices = np.empty((self.vertices.shape[0] + self.facevertices.shape[0] * 3, 3), self.vertices.dtype)
		
		dst_vertices[:self.vertices.shape[0]] = self.vertices
		
		if not multithread or self.facevertices.shape[0] < 40000:
			self._subd_step(0, self.facevertices.shape[0], dst_faces, dst_vertices)
		else:
			cores = os.cpu_count()
			with ThreadPoolExecutor(max_workers=cores) as xec:
				slice_count = int(math.ceil(self.facevertices.shape[0] / cores))
				for i in range(cores):
					start = i * slice_count
					end = min((i+1) * slice_count, self.facevertices.shape[0])
					xec.submit(Wavefront._subd_step, self, start, end, dst_faces, dst_vertices)
		
		subdivided = Wavefront()
		subdivided.vertices = dst_vertices
		subdivided.facevertices = dst_faces
		
		return subdivided
		
		
	def _subd_step(self, start, end, dst_faces, dst_vertices):
		
		for ind in range(start, end):
			
			face = self.facevertices[ind]
			
			v0 = self.vertices[face[0]-1]
			v1 = self.vertices[face[1]-1]
			v2 = self.vertices[face[2]-1]
			
			n_v0 = (v0 + v1) / 2
			n_v1 = (v1 + v2) / 2
			n_v2 = (v0 + v2) / 2
			
			n_v0_ind = self.vertices.shape[0] + ind * 3
			n_v1_ind = self.vertices.shape[0] + ind * 3 + 1
			n_v2_ind = self.vertices.shape[0] + ind * 3 + 2
			
			dst_faces[ind*4]   = np.array([face[0],    n_v0_ind+1, n_v2_ind+1], dtype=self.facevertices.dtype)
			dst_faces[ind*4+1] = np.array([n_v0_ind+1, face[1],    n_v1_ind+1], dtype=self.facevertices.dtype)
			dst_faces[ind*4+2] = np.array([n_v0_ind+1, n_v1_ind+1, n_v2_ind+1], dtype=self.facevertices.dtype)
			dst_faces[ind*4+3] = np.array([n_v2_ind+1, n_v1_ind+1, face[2]],    dtype=self.facevertices.dtype)
			
			dst_vertices[n_v0_ind] = n_v0
			dst_vertices[n_v1_ind] = n_v1
			dst_vertices[n_v2_ind] = n_v2
			
			
	def cull_vertices(self):
		
		if not self.colors is None:
			self.facevertices, self.vertices, self.colors = cull_arrays(self.facevertices, [self.vertices, self.colors])
		else:
			self.facevertices, self.vertices = cull_arrays(self.facevertices, self.vertices)
		
		if not self.facevertnormals is None:
			self.facevertnormals, self.normals = cull_arrays(self.facevertnormals, self.normals)
		
		if not self.facetexcoords is None:
			self.facetexcoords, self.texturecoords = cull_arrays(self.facetexcoords, self.texturecoords)
			
			
	def apply_mask(self, mask, cull_vertices=False):
		
		if mask is None:
			raise ValueError("Argument \"mask\" is None.")
		elif isinstance(mask, list):
			mask = set(mask)
		elif isinstance(mask, np.ndarray):
			mask = set(mask.tolist())
		
		keep_faces = []
		
		for i in range(self.facevertices.shape[0]):
			keep = True
			for j in range(self.facevertices.shape[1]):
				if not self.facevertices[i,j] in mask:
					keep = False
					break
			if keep:
				keep_faces.append(i)
				
		self.facevertices = np.take(self.facevertices, keep_faces, axis=0)
		
		if cull_vertices:
			self.cull_vertices()



def calc_normal_triangle(face, obj):
	
	p0 = obj.vertices[face[0]-1]
	p1 = obj.vertices[face[1]-1]
	p2 = obj.vertices[face[2]-1]
	
	u = p1 - p0
	v = p2 - p0
	
	normal_x = (u[1] * v[2]) - (u[2] * v[1])
	normal_y = (u[2] * v[0]) - (u[0] * v[2])
	normal_z = (u[0] * v[1]) - (u[1] * v[0])
	
	return lib.trig.norm(np.array([normal_x, normal_y, normal_z], dtype=obj.vertices.dtype))
	
	
def cull_arrays(ind_array, lists):
	
	if isinstance(lists, np.ndarray):
		lists = [lists]
	elif not type(lists) == list:
		lists = list(lists)
	
	indices = sorted([x-1 for x in set(np.squeeze(np.reshape(ind_array, (-1, 1))).tolist())])
	
	out_lists = []
	for lst in lists:
		out_lists.append(np.take(lst, indices, axis=0))
		
	new_indices = list(range(1, len(indices)+1))
	
	mapping = dict()
	for i in range(len(indices)):
		mapping[indices[i]+1] = new_indices[i]
	
	#for i in range(len(lists)):
	#	print(str(lists[i].shape)+" -> "+str(out_lists[i].shape))
		
	out_ind_array = np.zeros(ind_array.shape, dtype=ind_array.dtype)
	
	for i, val in np.ndenumerate(ind_array):
		out_ind_array[i] = mapping[val]
		
	if len(out_lists) == 1:
		return out_ind_array, out_lists[0]
		
	return out_ind_array, tuple(out_lists)
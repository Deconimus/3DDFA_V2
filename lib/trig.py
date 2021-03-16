import numpy as np
import math


def yaw(v):
	return angleBetween(np.array([0.0, 1.0]), norm(np.array([v[0], v[2]]))) * np.sign(v[0])

def pitch(v):
	return angleBetween(norm(np.array([v[0], 0.0, v[2]])), norm(v)) * np.sign(v[1])

def norm(v):
	length = np.linalg.norm(v)
	if length == 0:
		return v
	return v / length

def angleBetween(u, v):
	return np.arccos(np.dot(u, v))
	
	
def euler_angles_yxz(Rt):
	
	if Rt[1,2] < 1:
		if Rt[1,2] > -1:
			angle_x = math.asin(-Rt[1,2])
			angle_y = math.atan2(Rt[0,2], Rt[2,2])
			angle_z = math.atan2(Rt[1,0], Rt[1,1])
		else:
			angle_x = math.PI / 2.0
			angle_y = -math.atan2(-Rt[0,1], Rt[0,0])
			angle_z = 0.0
	else:
		angle_x = -math.pi / 2.0
		angle_y = math.atan2(-Rt[0,1], Rt[0,0])
		angle_z = 0.0
		
	return angle_x, angle_y, angle_z
	
	
def euler_angle_xyz(Rt):
	
	angle_x = math.atan2(Rt[2,1], Rt[2,2])
	angle_y = math.atan2(-Rt[2,0], math.sqrt(Rt[2,1]**2 + Rt[2,2]**2))
	angle_z = math.atan2(Rt[1,0], Rt[0,0])
	
	return angle_x, angle_y, angle_z
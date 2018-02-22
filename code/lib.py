import matplotlib.pyplot as plt 
import networkx as nx
from scipy import io
import pandas as pd
import numpy as np
import scipy
import json



def preprocess(A, xy, y):
	Px=xy[:,0]
	Py=xy[:,1]
	y=np.array([i[0] for i in y])
	A[A!=0] = 1
	
	graph_A=nx.Graph(A)
	
	D=np.diag(A.sum(axis=1))
	I = np.diag(np.ones(len(y)))
	L = D - A
	
	lambdas, U = np.linalg.eigh(L)
	
	return L, U, lambdas, Px, Py, y, graph_A



def clustering_generator(U):
	
	clusters = U[:,1].copy()
	m=np.median(clusters)
	clusters[clusters<m] = 0
	clusters[clusters>m] = 1
	
	clusters2 = U[:,2].copy()
	m2 = np.median(clusters2)
	
	ix0=set(np.where(clusters == 0)[0])
	ix00=set(np.where(clusters2 < m2)[0])
	i0=list(ix0.intersection(ix00))
	clusters2[i0] = 0
	
	ix1=set(np.where(clusters == 0)[0])
	ix11=set(np.where(clusters2 > m2)[0])
	i1=list(ix1.intersection(ix11))
	clusters2[i1] = 1
	
	ix2=set(np.where(clusters == 1)[0])
	ix22=set(np.where(clusters2 < m2)[0])
	i2=list(ix2.intersection(ix22))
	clusters2[i2] = 2
	
	ix3=set(np.where(clusters == 1)[0])
	ix33=set(np.where(clusters2 > m2)[0])
	i3=list(ix3.intersection(ix33))
	clusters2[i3] = 3
	
	return clusters, clusters2


def p(a, b, c):
	res = np.dot(a, np.dot(b, c))
	return res


def greedy_e_opt(Uf, y, S):
	
	index_set = set()
	index_list=[]
	logic=True
	n = len(y) - 1
	k = 0
	I = np.diag(np.ones(len(y)))
	while len(index_set) < S:
		i = -1
		i_best = -1
		old_list = []
		sigma_best = np.inf
		while i < n:
			i = i + 1
			if i in index_set:
				continue
			else:
				Ds_list = np.zeros(len(y))
				ix = index_list + [i]
				Ds_list[ix] = 1

				Ds = np.diag(Ds_list)
				Ds_bar = I - Ds
				DU = np.dot(Ds_bar, Uf)
				s = np.linalg.svd(DU, compute_uv=False)
				sigma_max = max(s)

				if sigma_max < sigma_best and sigma_max != -np.inf:
					sigma_best = sigma_max
					i_best = i
		k = k + 1
		#print(k)        
		index_set.add(i_best)
		index_list.append(i_best)
	return index_list


def greedy_a_opt(Uf, y, S, v):
	
	Rv = np.dot(v, v.T)
	Uft = Uf.conj().T
	d = np.ones(len(v))
	Rv_inv = np.diag(d) 
	#Rv_inv = np.linalg.inv(Rv)
	Rv_inv_Uf = np.dot(Rv_inv, Uf)
	index_set = set()
	index_list=[]
	logic=True
	n = len(y) - 1
	k = 0
	while len(index_set) < S:
		i = -1
		i_best = -1
		t_best = np.inf
		while i < n:
			i = i + 1
			if i in index_set:
				continue
			else:
				Ds_list = np.zeros(len(y))
				ix = index_list + [i]
				Ds_list[ix] = 1
				Ds = np.diag(Ds_list)
				try:
					t = np.trace(np.linalg.pinv(p(Uft, Ds, Rv_inv_Uf)))
					if t < t_best and t != -np.inf:
						t_best = t
						i_best = i
				except np.linalg.LinAlgError:
					#print("error")
					continue
					
		k = k + 1
		if i_best != -1:
			index_set.add(i_best)
			index_list.append(i_best)
		else:
			continue
	return index_list


def greedy_d_opt(Uf, y, S, v):
	
	Rv = np.dot(v, v.T)
	Uft = Uf.conj().T
	d = np.ones(len(v))
	Rv_inv = np.diag(d) 
	#Rv_inv = np.linalg.inv(Rv)
	Rv_inv_Uf = np.dot(Rv_inv, Uf)
	index_set = set()
	index_list=[]
	logic=True
	n = len(y) - 1
	k = 0
	while len(index_set) < S:
		i = -1
		i_best = -1
		t_best = -np.inf
		while i < n:
			i = i + 1
			if i in index_set:
				continue
			else:
				Ds_list = np.zeros(len(y))
				ix = index_list + [i]
				Ds_list[ix] = 1
				Ds = np.diag(Ds_list)
				try:
					m = p(Uft, Ds, Rv_inv_Uf)
					#lambdas, _ = np.linalg.eig(m)
					#lambdas = lambdas[lambdas>1e-14]
					_, t = np.linalg.slogdet(m)
					if t > t_best and t != np.inf:
						t_best = t
						i_best = i
				except np.linalg.LinAlgError:
					#print("error")
					continue
		k = k + 1
		#print(k)
		if i_best != -1:
			index_set.add(i_best)
			#print(index_set)
			index_list.append(i_best)
		else:
			continue
	return index_list

def compute_sample(U, Ps, y, index_list, F_list):
	
	s = np.dot(U.conj().T, y)
	Uf = U[:,F_list]
	s_f = s[F_list]
	y_sampled = p(Ps.T, Uf, s_f)
	y_sampled[y_sampled < 0] = 0

	y_sample = np.zeros(len(y))
	y_sample[:] = np.NaN
	y_sample[index_list] = y_sampled[:]
	
	return y_sampled, y_sample

def compute_sample_noise(U, Ps, y, index_list, F_list, v):
	
	s = np.dot(U.conj().T, y)
	Uf = U[:,F_list]
	s_f = s[F_list]
	y_sampled = p(Ps.T, Uf, s_f)
	
	y_noise = np.dot(Ps.T, v)
	y_noise = y_noise.squeeze()
	y_sampled = y_sampled + y_noise
	y_sampled[y_sampled < 0] = 0

	y_sample = np.zeros(len(y))
	y_sample[:] = np.NaN
	y_sample[index_list] = y_sampled[:]
	
	return y_sampled, y_sample

def compute_reconstruction_matrix(U, index_list, F_list):
	
	Uf = U[:,F_list]
	n = Uf.shape[0]
	Ds = np.zeros(n)
	Ds[index_list] = 1
	Ds = np.diag(Ds)
	
	Uft = Uf.conj().T
	
	Ps = Ds[:, index_list]
	I = np.diag(np.ones(n))

	matrix = p(Uft, Ds, Uf)
	matrix_inv = scipy.linalg.inv(matrix)
	Q = p(matrix_inv, Uft, Ps)
	
	matrix_r = np.dot(Uf, Q)
	
	Ds_bar = I - Ds
	s = np.linalg.svd(np.dot(Ds_bar, Uf), compute_uv=False)
	check = max(s) < 1
	if check:
		print("Ok")
	
	return matrix_r, Ds, Ps

def compute_reconstruction_matrix_noise(U, index_list, F_list, v):
	
	Uf = U[:,F_list]
	d = np.ones(len(v))
	Rv = np.diag(d)
	#Rv = np.dot(v, v.T)
	n = Uf.shape[0]
	Ds = np.zeros(n)
	Ds[index_list] = 1
	Ds = np.diag(Ds)
	
	Uft = Uf.conj().T
	
	Ps = Ds[:, index_list]
	#print(Ps.shape)

	I = np.diag(np.ones(n))
	matrix_inv=np.linalg.inv(p(Ps.T, Rv, Ps))
	
	matrix_inv2 = p(matrix_inv, Ps.T, Uf)
	matrix_inv2 = p(Uft, Ps, matrix_inv2)
	matrix_inv2 = np.linalg.inv(matrix_inv2)
	
	matrix3 = p(Uft, Ps, matrix_inv)
	matrix_r = p(Uf, matrix_inv2, matrix3)
	
	Ds_bar = I - Ds
	s = np.linalg.svd(np.dot(Ds_bar, Uf), compute_uv=False)
	check = max(s) < 1
	if check:
		print("Ok")
	
	return matrix_r, Ds, Ps, Rv

def y_reconstruction_routine(U, y, index_list, F_list):
	matrix_rec, Ds, Ps = compute_reconstruction_matrix(U, index_list, F_list)
	y_sampled, y_sample = compute_sample(U, Ps, y, index_list, F_list)
	y_r = np.dot(matrix_rec, y_sampled)
	return y_r, y_sample, matrix_rec, y_sampled

def y_reconstruction_routine_noise(U, y, index_list, F_list, v):
	matrix_rec, Ds, Ps, Rv = compute_reconstruction_matrix_noise(U, index_list, F_list, v)
	y_sampled, y_sample = compute_sample_noise(U, Ps, y, index_list, F_list, v)
	y_r = np.dot(matrix_rec, y_sampled)
	return y_r, y_sample, matrix_rec, y_sampled

def compute_error(y_reconstructed, y):
	error = ((y_reconstructed - y)**2).sum() / (y**2).sum()
	error = 10 * np.log10(error)
	return error
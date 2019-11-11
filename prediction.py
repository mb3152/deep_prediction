#!/cbica/home/bertolem/anaconda2/bin/python
import sys
import matplotlib
import numpy.linalg as npl
import numpy as np
import os
import nibabel as nib
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from scipy.stats import pearsonr, spearmanr,linregress
from scipy.spatial import distance
from scipy import stats, linalg
from scipy.spatial.distance import pdist
import scipy.io
import scipy
import statsmodels.api as sm
import math
import copy
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge, RidgeClassifier, LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import glob
from itertools import combinations
import operator
import seaborn as sns
import matplotlib.pylab as plt
import brain_graphs
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
from sklearn.cluster import KMeans
import igraph
import louvain
from matplotlib.gridspec import GridSpec

from memory_profiler import profile

import matlab.engine
sns.plt = plt # this is super annoying
import matplotlib as mpl
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import patches
import matplotlib.animation as animation

plt.rcParams['font.sans-serif'] = "Palatino"
plt.rcParams['font.serif'] = "Palatino"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Palatino:italic'
plt.rcParams['mathtext.bf'] = 'Palatino:bold'
plt.rcParams['mathtext.cal'] = 'Palatino'
plt.rcParams['pdf.fonttype'] = 42
path = '/cbica/home/bertolem/Palatino/Palatino.ttf'
mpl.font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = 'serif'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-homedir',action='store',dest='homedir',default='/home/mbmbertolero/')
parser.add_argument('-ctype',action='store',dest='ctype',default='fc')
parser.add_argument('-ptype',action='store',dest='ptype',default='nn')
parser.add_argument('-task',action='store',dest='task',default='all_p')
parser.add_argument('-node',action='store',dest='node',default=0,type=int)
parser.add_argument('-r',action='store',dest='runtype',default=None)
parser.add_argument('-neurons',action='store',dest='neurons',type=int,default=15)
parser.add_argument('-layers',action='store',dest='layers',type=int,default=3)
parser.add_argument('-matrix',action='store',dest='matrix',type=str,default='all')
parser.add_argument('-components',action='store',dest='components',type=str,default='None')
r = parser.parse_args()
locals().update(r.__dict__)
# 1/0
global edges
global loop_n
global b_array
global neurons
global layers
global neurons_array
global ptype
global atlas_path
global homedir
global matrix
global this_task
global components
global matrices
global models


global pc_mod_array
global mod_acc_array
global pc_acc_array
global med_array
global med_data
global pc



global c
global grouped
global donotmove


homedir = '/cbica/home/bertolem/deep_prediction/'
atlas_path = '/%s/Schaefer2018_400Parcels_17Networks_order.dlabel.nii'%(homedir)
pc = np.load('/%s/results/pc.npy'%(homedir))

def load_subjects(matrices=True):
	df = pd.read_csv('/%s/S1200.csv'%(homedir))
	subjects = df.Subject[df['3T_Full_MR_Compl'].values==True].values
	if matrices == True:
		for s in subjects:
			if os.path.exists('/%s/all_matrices/%s_matrix.npy'%(homedir,s)) == False:
				subjects = subjects[subjects!=s]
	return subjects

def load_tasks():
	scanner = ['WM_Task_2bk_Acc','Social_Task_Random_Perc_Random','Social_Task_TOM_Perc_TOM','Language_Task_Story_Avg_Difficulty_Level','Emotion_Task_Face_Acc','Language_Task_Math_Avg_Difficulty_Level','Relational_Task_Rel_Acc']
	public = ['PicSeq_AgeAdj','CardSort_AgeAdj','Flanker_AgeAdj','PMAT24_A_CR','ReadEng_AgeAdj','PicVocab_AgeAdj','ProcSpeed_AgeAdj',\
	'DDisc_AUC_200','DDisc_AUC_40K','VSPLOT_TC','SCPT_TP','IWRD_TOT','ListSort_AgeAdj','ER40_CR','AngAffect_Unadj','AngHostil_Unadj','AngAggr_Unadj','Sadness_Unadj','LifeSatisf_Unadj',\
	'MeanPurp_Unadj','PosAffect_Unadj','PSQI_Score','MMSE_Score','Friendship_Unadj','Loneliness_Unadj','PercHostil_Unadj','PercReject_Unadj','EmotSupp_Unadj',\
	'InstruSupp_Unadj','PercStress_Unadj','SelfEff_Unadj','Endurance_AgeAdj','GaitSpeed_Comp','Dexterity_AgeAdj','Strength_AgeAdj',\
	'NEOFAC_A','NEOFAC_O','NEOFAC_C','NEOFAC_N','NEOFAC_E','Noise_Comp','Odor_AgeAdj','PainInterf_Tscore','Taste_AgeAdj','Mars_Final']

	restricted = ['ASR_Anxd_Pct','ASR_Witd_T','ASR_Soma_T','ASR_Thot_T','ASR_Attn_T','ASR_Aggr_T','ASR_Rule_T','ASR_Intr_T',\
	'ASR_Intn_T','ASR_Oth_Raw','ASR_Crit_Raw','ASR_Extn_T','ASR_Totp_T','DSM_Depr_T','DSM_Anxi_T','DSM_Somp_T','DSM_Avoid_T',\
	'DSM_Adh_T','DSM_Inat_Raw','DSM_Hype_Raw','DSM_Antis_Raw','DSM_Antis_T','SSAGA_ChildhoodConduct','SSAGA_PanicDisorder','SSAGA_Agoraphobia',\
	'SSAGA_Depressive_Ep','SSAGA_Depressive_Sx']
	return scanner,public,restricted

def load_matrices(subjects,matrix='all',components=None):
	if components != 'None':
		if components != None:
			components = int(components)
	matrices = []
	for s in subjects: 
		if matrix == 'all': matrices.append(np.load('/%s/all_matrices/%s_matrix.npy' %(homedir,s)))
		if matrix == 'WM': matrices.append(np.load('/%s/WM_matrices/%s_matrix.npy' %(homedir,s)))
	matrices = np.array(matrices)
	if components == 7 or components == 17:
		big_matrices = np.array(matrices).copy()
		labels = yeo_partition(components)[1]
		if components == 17: components = 16
		matrices = np.zeros((big_matrices.shape[0],int(components),int(components)))
		for i,j in combinations(range(components),2):
			cval =  np.nanmean(big_matrices[:,np.where(labels==i)[0]][:,:,np.where(labels==j)[0]].reshape(len(subjects),-1),axis=1)
			matrices[:,i,j] = cval
			matrices[:,j,i] = cval
	return matrices

def yeo_partition(n_networks=7):
	if n_networks == 17:
		full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,
		'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	if n_networks==7:
		full_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,
		'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':6}
		name_dict = {0:'Visual',1:'Sensory\nMotor',2:'Dorsal\nAttention',3:'Ventral\nAttention',4:'Limbic',5:'Control',6:'Default'}
	membership = np.zeros((400)).astype(str)
	membership_ints = np.zeros((400)).astype(int)
	yeo_df = pd.read_csv('/%s/Schaefer2018_400Parcels_17Networks_order.txt'%(homedir),sep='\t',header=None,names=['name','R','G','B','0'])['name']
	yeo_colors = pd.read_csv('/%s/Schaefer2018_400Parcels_17Networks_order.txt'%(homedir),sep='\t',header=None,names=['name','r','g','b','0'])
	yeo_colors = np.array([yeo_colors['r'],yeo_colors['g'],yeo_colors['b']]).transpose() /256.
	for i,n in enumerate(yeo_df):
		if n_networks == 17:
			membership[i] = n.split('_')[2]
			membership_ints[i] = int(full_dict[n.split('_')[2]])
		if n_networks == 7:
			membership_ints[i] = int(full_dict[n.split('_')[2]])
			membership[i] = name_dict[membership_ints[i]]
	return membership,membership_ints,yeo_colors

def mean_metrics():
	subjects = load_subjects(True)
	m = load_matrices(subjects).mean(axis=0)
	m = m + m.transpose()
	m = np.tril(m,-1)
	m = m + m.transpose()
	pc = []
	wcd = []
	degree = []
	between = []
	for cost in np.linspace(0.05,0.1):
		g = brain_graphs.matrix_to_igraph(m.copy(),cost=cost,mst=True)
		g = brain_graphs.brain_graph(VertexClustering(g,yeo_partition(7)[1],params={'weight':'weight'}))
		pc.append(g.pc)
		wcd.append(g.wmd)
		degree.append(g.community.graph.strength(weights='weight'))
		between.append(g.community.graph.betweenness())
	pc = np.nanmean(pc,axis=0)
	wcd = np.nanmean(wcd,axis=0)
	degree = np.nanmean(degree,axis=0)
	between = np.nanmean(between,axis=0)
	np.save('/%s/results/pc.npy'%(homedir),pc)
	np.save('/%s/results/strength.npy'%(homedir),degree)
	np.save('/%s/results/wcd.npy'%(homedir),wcd)
	np.save('/%s/results/between.npy'%(homedir),between)

	colors = np.array(make_heatmap(cut_data(pc,1),sns.diverging_palette(220, 10,n=1001)))
	write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/pc'%(homedir))

def nan_pearsonr(x,y):
	x,y = np.array(x),np.array(y)
	mask = ~np.logical_or(np.isnan(x), np.isnan(y))
	return pearsonr(x[mask],y[mask])

def log_p_value(p):
	if p > 0.001: 
		p = np.around(p,3)
		p = "$\it{p}$=%s"%(p)
	else: 
		p = (-1) * np.log10(p)
		p = "-log10($\it{p}$)=%s"%(np.around(p,0).astype(int))
	return p

def convert_r_p(r,p):
	return "$\it{r}$=%s\n%s"%(np.around(r,3),log_p_value(p))

def cut_data(data,min_cut=1.5,max_cut=1.5):
	d = data.copy()
	max_v = np.mean(d) + np.std(d)*max_cut
	min_v = np.mean(d) - np.std(d)*min_cut
	d[d>max_v] = max_v
	d[d<min_v] = min_v
	return d

def make_heatmap(data,cmap):
	if cmap == 'nicegreen': orig_colors = sns.cubehelix_palette(1001, rot=-.5, dark=.3)
	elif cmap == 'nicepurp': orig_colors = sns.cubehelix_palette(1001, rot=.5, dark=.3)
	elif cmap == 'stock': orig_colors = sns.color_palette("RdBu_r",n_colors=1001)
	elif cmap == 'Reds': orig_colors = sns.color_palette("Reds",n_colors=1001)
	else: orig_colors = cmap
	norm_data = copy.copy(data)
	if np.nanmin(data) < 0.0: norm_data = norm_data + (np.nanmin(norm_data)*-1)
	elif np.nanmin(data) > 0.0: norm_data = norm_data - (np.nanmin(norm_data))
	norm_data = norm_data / float(np.nanmax(norm_data))
	norm_data = norm_data * 1000
	norm_data = norm_data.astype(int)
	colors = []
	for d in norm_data:
		colors.append(orig_colors[d])
	return colors

def write_cifti(atlas_path,out_path,colors):
	os.system('wb_command -cifti-label-export-table %s 1 temp.txt'%(atlas_path))
	df = pd.read_csv('temp.txt',header=None)
	for i in range(df.shape[0]):
		try:
			d = np.array(df[0][i].split(' ')).astype(int)
		except:
			continue
		real_idx = d[0] -1
		try: df[0][i] = str(d[0]) + ' ' + str(int(colors[real_idx][0]*255)) + ' ' + str(int(colors[real_idx][1]*255)) + ' ' + str(int(colors[real_idx][2]*255)) + ' ' + str(int(colors[real_idx][3]*255))
		except: df[0][i] = str(d[0]) + ' ' + str(int(colors[real_idx][0]*255)) + ' ' + str(int(colors[real_idx][1]*255)) + ' ' + str(int(colors[real_idx][2]*255)) + ' ' + str(255)
	df.to_csv('temp.txt',index=False,header=False)
	os.system('wb_command -cifti-label-import %s temp.txt %s.dlabel.nii'%(atlas_path,out_path))
	os.system('rm temp.txt')

def behavior():
	subjects = load_subjects(matrices=True)
	df = pd.read_csv('/%s/S1200.csv'%(homedir))
	rdf = pd.read_csv('/%s/restricted.csv'%(homedir))
	df = df.merge(rdf,on='Subject')
	for df_sub in df.Subject.values:
		if df_sub not in subjects:
			df = df[df.Subject.values!=df_sub]
	
	scanner,public,restricted = load_tasks()

	for c in df.columns.values:
		if c not in public:
			if c not in restricted:
				if c not in scanner:
					if c !='Subject':
						df = df.drop(c,1)
	return df

def make_matrices(matrix):
	"""
	only works on CFN! 
	"""
	subjects = load_subjects(False)
	for s in subjects:
		print s
		files = []
		for m in glob.glob('/home/mbmbertolero/hcp/CompCor_matrices/yeo_400*%s**'%(s)):
			if 'ts' in m:continue
			files.append(m)
		if len(files) != 18: continue
		matrix = []
		for m in files:
			m = np.load(m)
			np.fill_diagonal(m,0.0)
			m[np.isnan(m)] = 0.0
			m = np.arctanh(m)
			matrix.append(m)
		matrix = np.nanmean(matrix,axis=0)
		np.save('/home/mbmbertolero/prediction/all_matrices/%s_matrix.npy'%(s),matrix)

def run(task='PicSeq_AgeAdj',matrix='all',components=None):
	global edges
	global b_array
	
	ptype = 'nn'
	if layers == 0: ptype = 'ols'
	df = behavior()
	subjects = df.Subject.values
	
	matrices = load_matrices(subjects,matrix,components)

	b_array = df[task].values
	b_array[np.isnan(b_array)] = np.nanmean(b_array)
	kf = KFold(5,shuffle=True,random_state=315)
	
	loop_n = 400
	try:
		if int(components) == 7 or int(components) == 17: 
			loop_n = int(components)
			if loop_n == 17: loop_n = 16
	except: pass 
	full_prediction_array = np.zeros((loop_n,len(b_array)))
	acc = np.zeros((loop_n))

	for node in range(loop_n):
		edges = matrices[:,node,:].reshape(len(subjects),-1)
		idx = 0
		prediction_array = np.zeros((len(subjects)))
		for train, test in kf.split(subjects):
			if ptype == 'nn': model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons_array)
			if ptype == 'ols': model = LinearRegression()
			
			model.fit(edges[train],b_array[train])
			prediction_array[test] = model.predict(edges[test])
			del model
			idx = idx + 1
		acc[node] = pearsonr(prediction_array,b_array)[0]
		full_prediction_array[node] = prediction_array

	# if ptype == 'nn': np.save('%s//results/%s_%s_%s_%s_%s_%s_prediction.npy'%(homedir,ctype,neurons,layers,task,matrix,components),full_prediction_array.astype('float16'))
	# else: np.save('%s//results/%s_%s_%s_%s_%s_prediction.npy'%(homedir,ctype,ptype,task,matrix,components),full_prediction_array.astype('float16'))
	# if ptype == 'nn': np.save('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,task,matrix,components),acc.astype('float16'))
	# else: np.save('%s//results/%s_%s_%s_%s_%s.npy'%(homedir,ctype,ptype,task,matrix,components),acc.astype('float16'))

def modularity_und_sign(W, ci, qtype='sta'):
    '''
    This function simply calculates the signed modularity for a given
    partition. It does not do automatic partition generation right now.
    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted/binary connection matrix with positive and
        negative weights
    ci : Nx1 np.ndarray
        community partition
    qtype : str
        modularity type. Can be 'sta' (default), 'pos', 'smp', 'gja', 'neg'.
        See Rubinov and Sporns (2011) for a description.
    Returns
    -------
    ci : Nx1 np.ndarray
        the partition which was input (for consistency of the API)
    Q : float
        maximized modularity metric
    Notes
    -----
    uses a deterministic algorithm
    '''
    n = len(W)
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    W0 = W * (W > 0)  # positive weights matrix
    W1 = -W * (W < 0)  # negative weights matrix
    s0 = np.sum(W0)  # positive sum of weights
    s1 = np.sum(W1)  # negative sum of weights
    Knm0 = np.zeros((n, n))  # positive node-to-module degree
    Knm1 = np.zeros((n, n))  # negative node-to-module degree

    for m in range(int(np.max(ci))):  # loop over initial modules
        Knm0[:, m] = np.sum(W0[:, ci == m + 1], axis=1)
        Knm1[:, m] = np.sum(W1[:, ci == m + 1], axis=1)

    Kn0 = np.sum(Knm0, axis=1)  # positive node degree
    Kn1 = np.sum(Knm1, axis=1)  # negative node degree
    Km0 = np.sum(Knm0, axis=0)  # positive module degree
    Km1 = np.sum(Knm1, axis=0)  # negaitve module degree

    if qtype == 'smp':
        d0 = 1 / s0
        d1 = 1 / s1  # dQ=dQ0/s0-dQ1/s1
    elif qtype == 'gja':
        d0 = 1 / (s0 + s1)
        d1 = 1 / (s0 + s1)  # dQ=(dQ0-dQ1)/(s0+s1)
    elif qtype == 'sta':
        d0 = 1 / s0
        d1 = 1 / (s0 + s1)  # dQ=dQ0/s0-dQ1/(s0+s1)
    elif qtype == 'pos':
        d0 = 1 / s0
        d1 = 0  # dQ=dQ0/s0
    elif qtype == 'neg':
        d0 = 0
        d1 = 1 / s1  # dQ=-dQ1/s1
    else:
        raise KeyError('modularity type unknown')

    if not s0:  # adjust for absent positive weights
        s0 = 1
        d0 = 0
    if not s1:  # adjust for absent negative weights
        s1 = 1
        d1 = 0

    m = np.tile(ci, (n, 1))

    q0 = (W0 - np.outer(Kn0, Kn0) / s0) * (m == m.T)
    q1 = (W1 - np.outer(Kn1, Kn1) / s1) * (m == m.T)
    q = d0 * np.sum(q0) - d1 * np.sum(q1)

    return ci, q

def nn_multi(node):
	node = int(node)
	print node
	edges = matrices[:,node,:].reshape(len(b_array),-1)
	model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons_array)
	model.fit(edges,b_array)
	return model

def multi_find3(rseed):
	global c
	global grouped
	global donotmove
	already_in = np.unique(grouped).shape[0]
	new_idxs = np.random.RandomState(rseed).choice(range(grouped.shape[0]),grouped.shape[0],replace=False)
	# new_idx = np.random.choice(range(grouped.shape[0]),1)
	tempg = grouped.copy()
	final_idxs = []
	for nidx in new_idxs:
		if nidx in donotmove:continue
		final_idxs.append(nidx)

	for nidx in donotmove:
		final_idxs.append(nidx)
	for new_idx in final_idxs:
		new_val = grouped[new_idx]
		already_in = np.unique(tempg)
		if len(already_in) == grouped.shape[0]:break
		# print len(already_in)
		if new_val not in already_in:continue
		diff = 1
		while True:
			if new_val + diff < grouped.shape[0]:
				if new_val + diff not in already_in:
					tempg[new_idx] = new_val + diff 
					# print 'put %s in at %s'%(new_val + diff,new_idx)
					break
			if new_val - diff >= 0:
				if new_val - diff not in already_in:
					tempg[new_idx] = new_val - diff 
					# print 'put %s in at %s'%(new_val - diff,new_idx)
					break
			diff = diff + 1
	mthisd = 0
	sorted_c = np.zeros((c.shape))
	for l in range(c.shape[0]): sorted_c[l,tempg.astype(int)] = c[l,:]
	sorted_c = sorted_c.astype(int)
	thisd = 0
	for l in range(c.shape[0]):
		bd = np.diff([sorted_c[l]]) == 0
		thisd = thisd +(bd[bd==True]).shape[0]
	for n in range(c.shape[1]):
		bd = np.diff(sorted_c[:,n]) == 0
		thisd = thisd +(bd[bd==True]).shape[0]
	return thisd,tempg

def nn_structure(task='PicSeq_AgeAdj',matrix='all',components=None,write_graph=True):
	global edges
	global b_array
	global matrices
	global grouped
	global c
	global donotmove
	
	if write_graph:	task = 'WM_Task_2bk_Acc'	
	ptype = 'nn'
	pc = np.load('/%s/results/pc.npy'%(homedir))
	prediction_accs = np.load('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,task,matrix,components))
	df = behavior()
	subjects = df.Subject.values
	
	matrices = load_matrices(subjects,matrix,components)

	b_array = df[task].values
	b_array[np.isnan(b_array)] = np.nanmean(b_array)

	loop_n = 400
	try:
		if int(components) == 7 or int(components) == 17: 
			loop_n = int(components)
			if loop_n == 17: loop_n = 16
	except: pass 
	model_arrays = []
	if components != 'None':
		if components != None:
			loop_n = int(components) 
	posmod = np.zeros((loop_n))
	posmod[:] = np.nan
	com = posmod.copy()
	negmod = posmod.copy()
	posmod_o = posmod.copy()
	from oct2py import octave
	octave.addpath('/%s/GenLouvain-master/'%(homedir),nout=0)
	for node in range(loop_n):
		node = int(node)
		if write_graph: 
			node = np.argmax(pc)

		# if write_graph: node = np.argmin(prediction_accs)
		print node
		edges = matrices[:,node,:].reshape(len(subjects),-1)
		model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons_array)
		model.fit(edges,b_array)
		model_array = np.array(model.coefs_[0:layers])
		1/0


		pos_graph_arrays = []
		for l in range(1,layers):
			graph_array = model.coefs_[l].copy()
			graph_array = graph_array + graph_array.transpose()
			pos_graph_array = graph_array.copy()
			pos_graph_array[pos_graph_array!=0.0] = pos_graph_array[pos_graph_array!=0.0] + abs(np.nanmin(pos_graph_array))
			pos_graph_array =  brain_graphs.threshold(pos_graph_array,0.1,mst=True)
			pos_graph_arrays.append(pos_graph_array)
		pos_graph_arrays = np.array(pos_graph_arrays)
		c,q=octave.fcn_genlouv(pos_graph_arrays.swapaxes(0,2),2.,.5,nout=2)
		c = np.array(c).transpose()
		
		posmod[node] = q

		if write_graph:
			import time
			while True:
				c,q=octave.fcn_genlouv(pos_graph_arrays.swapaxes(0,2),np.random.randint(1090,1200,1)/1000.,np.random.randint(1070,1080,1)/1000.,nout=2)# for plotting
				c = np.array(c).transpose()

				same = np.abs(np.diff(c,axis=0)).sum(axis=0)
				print len(same[same==0]) 
				if len(same[same==0]) > 200:
					if len(same[same==0]) < 300:
						break
			# 1/0
			vcount = loop_n
			for l in range(layers):
				vcount = vcount + neurons
			
			big_graph_array = np.zeros((vcount,vcount))

			#first layer weights
			for i in range(loop_n):
				connections = model_array[0][i]
				for j in range(neurons):
					# big_graph_array[i,j+loop_n] = connections[j]
					big_graph_array[i,j+loop_n] = 0.00000 #fill later so you don't put too many in
			#all other layers
			for layer in range(1,layers):
				connections = model_array[layer]
				i_offset = loop_n + (abs(layer-1)*neurons)
				j_offset = loop_n + (abs(layer-1)*neurons) + neurons
				for i in range(neurons):
					for j in range(neurons):
						big_graph_array[i+i_offset,j+j_offset] = connections[i][j]
			
			g = brain_graphs.matrix_to_igraph(big_graph_array.copy(),cost=.02,mst=False)
			# the final layer, add after making the graph
			connections = model.coefs_[-1]
			connections = np.array(connections).flatten()
			connections = abs(np.min(connections)) + connections + 0.01
			output_node = g.vcount()
			print g.ecount()
			g.add_vertex(output_node)
			for c_idx,connect in enumerate(connections):
				g.add_edge(vcount-neurons+c_idx,output_node,**{'weight':abs(connect)})
			# add that first layer in
			for i in range(loop_n):
				connections = model_array[0][i]
				g.add_edges(zip(np.tile(i,3),np.random.choice(range(400,800),3,replace=False)))
			w = np.array(g.es['weight'])
			non_zero = np.where(w==None)[0]
			new_w = np.random.choice(model_array[0].flatten(),len(non_zero),replace=False)
			new_w = new_w + abs(np.min(new_w))
			w[non_zero] = new_w
			g.es['weight'] = list(w)

			grouped = np.zeros((neurons))
			findbest = []
			for i in range(layers-1):
				gidx = 0
				for com in np.unique(c[i]):
					for n in np.where(c[i]== com)[0]:
						grouped[n] = gidx
						gidx = gidx +1
				findbest.append(grouped.copy())

			if cluster:
				grouped = scipy.stats.mode(findbest,axis=0)[0][0]
				donotmove = scipy.stats.mode(findbest,axis=0)[1][0]
				donotmove = np.where(donotmove==donotmove.max())[0]
				how_long = 30
				
				# del pool
				pool = multiprocessing.Pool(40)
				t1 = time.time()
				best = np.array(pool.map(multi_find3,range(100)))
				t2 = time.time()

				perrun=(t2-t1)/100/ 60.
				runs = how_long / perrun
				
				best = np.array(pool.map(multi_find3,range(int(runs))))
				grouped = best[:,1][best[:,0].argmax()]	
			else:
				grouped = findbest[-1]



			glayers = []
			gpositions = []
			xs = []
			ys = []

			#only want right lateral and left medial
			brain = nib.load('Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii').get_data()
			for lpn in range(400):
				x,y,z = np.round(np.mean(np.where(brain==lpn),axis=1)).astype(int)
				xs.append(x)
				ys.append(y)
				#if left hemi
				if lpn <200:
					#if medial
					if x <=53:
						gpositions.append(y+303)
						glayers.append(z-100)
					# if lateral
					else: 
						gpositions.append(y-14)
						glayers.append(z-100)
				# if right hemi
				if lpn >=200:
					if x >=40:#if medial
						gpositions.append(y+303)
						glayers.append(z-100)
					else: #if lateral
						gpositions.append(y-14)
						glayers.append(z-100)
			
			for n in range(neurons):
				glayers.append(1)
				gpositions.append(n)	

			for l in range(layers-1):
				for n in grouped:
					glayers.append((l+2)*8)
					gpositions.append(n)	

			glayers.append((l+10)*8)
			gpositions.append(neurons/2.)		
		
			final_colors = np.zeros((7,3))
			final_colors[0] = [120,18,134]
			final_colors[1] = [70,130,180]
			final_colors[2] = [0,118,14]
			final_colors[3] = [196,58,250]
			final_colors[4] = [220,248,164]
			final_colors[5] = [230,148,34]
			final_colors[6] = [205,62,78]
			final_colors = final_colors /255.

			membership = np.zeros((loop_n))
			membership = np.append(membership,np.ones((neurons)))
			membership = np.append(membership,np.array(c).astype('float16').flatten()+1)
			from matplotlib.colors import rgb2hex
			colors = sns.color_palette('cubehelix',n_colors=len(np.unique(membership)))
			color_array = np.zeros((g.vcount())).astype(str)
			for n in range(g.vcount()-1):
				if n < 400: node_c = final_colors[yeo_partition(7)[1][n]]
				if n>= 400: 
					if n < 800:node_c = final_colors[yeo_partition(7)[1][n-400]]
					else:node_c = np.array(colors[membership[n].astype(int)])
				color_array[n] = str(rgb2hex(node_c))
			color_array[-1] = rgb2hex([0,0,0])
			color_array[node] = rgb2hex([0,0,0])
			membership = np.append(membership,-1)
			sizes = np.ones(membership.shape)
			sizes[node] = 3
			sizes[-1] = 3
			g.vs['membership'] = membership
			g.vs['size'] = sizes
			g.vs['color'] = color_array.astype('str')
			g.vs['layer'] = np.array(glayers).astype('float16')
			g.vs['neuron'] = np.array(gpositions).astype('float16')
			g.write_gml('/%s/neural_network_%s_%s_%s.gml'%(homedir,layers,neurons,node))
			break

	if write_graph==False:np.save('%s//results/%s_%s_%s_%s_%s_%s_network_structure.npy'%(homedir,ctype,neurons,layers,task,matrix,components),posmod)

def flipup():

	g = igraph.read('neural_network_10_400_93.gml')
	l = np.array(g.vs['layer'])
	nn_move = l[l>0] * -1
	nn_move[nn_move == -1] = -8
	nn_move = nn_move - abs(np.unique(l[l<0]).min()) - 8
	l[l>0] = nn_move
	g.vs['layer'] = l
	g.write_gml('neural_network_10_400_93_swap.gml')

def write_networks():
	
	write_graph=True
	if write_graph:	task = 'WM_Task_2bk_Acc'
	ptype = 'nn'
	pc = np.load('/%s/results/pc.npy'%(homedir))
	prediction_accs = np.load('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,task,matrix,components))
	df = behavior()
	subjects = df.Subject.values
	
	matrices = load_matrices(subjects,matrix,components)

	b_array = df[task].values
	b_array[np.isnan(b_array)] = np.nanmean(b_array)

	loop_n = 400
	try:
		if int(components) == 7 or int(components) == 17: 
			loop_n = int(components)
			if loop_n == 17: loop_n = 16
	except: pass 
	model_arrays = []
	if components != 'None':
		if components != None:
			loop_n = int(components) 
	posmod = np.zeros((loop_n))
	posmod[:] = np.nan
	com = posmod.copy()
	negmod = posmod.copy()
	posmod_o = posmod.copy()
	from oct2py import octave
	octave.addpath('/%s/GenLouvain-master/'%(homedir),nout=0)
	node = np.argmax(pc)

	# if write_graph: node = np.argmin(prediction_accs)
	print node
	edges = matrices[:,node,:].reshape(len(subjects),-1)
	model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons_array)
	model.fit(edges,b_array)
	model_array = np.array(model.coefs_[0:layers])


	pos_graph_arrays = []
	for l in range(1,layers):
		graph_array = model.coefs_[l].copy()
		graph_array = graph_array + graph_array.transpose()
		pos_graph_array = graph_array.copy()
		pos_graph_array[pos_graph_array!=0.0] = pos_graph_array[pos_graph_array!=0.0] + abs(np.nanmin(pos_graph_array))
		pos_graph_array =  brain_graphs.threshold(pos_graph_array,0.1,mst=True)
		pos_graph_arrays.append(pos_graph_array)
	pos_graph_arrays = np.array(pos_graph_arrays)	
	posmod[node] = q
	if layers == 5 and neurons == 10:
		c,q=octave.fcn_genlouv(pos_graph_arrays.swapaxes(0,2),2,.5,nout=2)
		c = np.array(c).transpose()
	if layers == 10 and neurons == 10:
		c,q=octave.fcn_genlouv(pos_graph_arrays.swapaxes(0,2),2,1.5,nout=2)
		c = np.array(c).transpose()
		same = np.abs(np.diff(c,axis=0)).sum(axis=0) 
	if layers == 5 and neurons == 50:
		c,q=octave.fcn_genlouv(pos_graph_arrays.swapaxes(0,2),2,.9,nout=2)
		c = np.array(c).transpose()
		same = np.abs(np.diff(c,axis=0)).sum(axis=0) 
		print float(len(same[same==0.0]))/neurons
	# eincrease = 0
	# while True:
	# 	c,q=octave.fcn_genlouv(pos_graph_arrays.swapaxes(0,2),np.random.randint(1090,1200,1)/1000.,np.random.randint(1100+increase,1200+increase,1)/1000.,nout=2)# for plotting
	# 	c = np.array(c).transpose()

	# 	same = np.abs(np.diff(c,axis=0)).sum(axis=0)
	# 	print len(same[same==0.0])
	# 	if len(same[same==0.0]) > len(same) /2.:
	# 		if len(same[same==0.0]) < (len(same) /4.)*3:
	# 			break
	# 	eincrease = eincrease + 50


	# 1/0
	loop_n = 1
	vcount = loop_n
	for l in range(layers):
		vcount = vcount + neurons
	
	big_graph_array = np.zeros((vcount,vcount))

	#first layer weights
	for i in range(400):
		connections = abs(model_array[0][i])
		for j in range(neurons):
			big_graph_array[0,j+loop_n] = connections[j] + big_graph_array[0,j+loop_n]
			# big_graph_array[i,j+loop_n] = 0.00000 #fill later so you don't put too many in
	big_graph_array = big_graph_array / 200.
	#all other layers
	for layer in range(1,layers):
		connections = abs(model_array[layer])
		i_offset = loop_n + (abs(layer-1)*neurons)
		j_offset = loop_n + (abs(layer-1)*neurons) + neurons
		for i in range(neurons):
			for j in range(neurons):
				big_graph_array[i+i_offset,j+j_offset] = connections[i][j]
	
	g = brain_graphs.matrix_to_igraph(abs(big_graph_array.copy()),cost=1,mst=False)
	# the final layer, add after making the graph
	connections = model.coefs_[-1]
	connections = np.array(connections).flatten()
	connections = abs(np.min(connections)) + connections + 0.01
	output_node = g.vcount()
	print g.ecount()
	g.add_vertex()
	for c_idx,connect in enumerate(connections):
		g.add_edge(output_node,vcount-neurons+c_idx,**{'weight':abs(connect)})

	grouped = np.zeros((neurons))
	findbest = []
	for i in range(layers-1):
		gidx = 0
		for com in np.unique(c[i]):
			for n in np.where(c[i]== com)[0]:
				grouped[n] = gidx
				gidx = gidx +1
		findbest.append(grouped.copy())
	grouped = findbest[-1]

	grouped = scipy.stats.mode(findbest,axis=0)[0][0]
	donotmove = scipy.stats.mode(findbest,axis=0)[1][0]
	donotmove = np.where(donotmove==donotmove.max())[0]
	how_long = 30
	
	# del pool
	pool = multiprocessing.Pool(40)
	t1 = time.time()
	best = np.array(pool.map(multi_find3,range(1000000)))
	t2 = time.time()

	grouped = best[:,1][best[:,0].argmax()]	

	glayers = [-1]
	gpositions = [np.around(np.median(np.arange(neurons)))]

	# for d1 in range(20):
	# 	for d2 in range(20):
	# 		gpositions.append(d1)
	# 		glayers.append(d2*-1)


	# for n in np.linspace(0,20,neurons).astype(int):
	# 	glayers.append(1)
	# 	gpositions.append(n)	

	for l in range(layers):
		for n in grouped:
		# for n in np.linspace(0,20,neurons).astype(int)[grouped.astype(int)]:
			glayers.append((l))
			gpositions.append(n)

	glayers.append((l+2))
	gpositions.append(neurons/2.)		

	final_colors = np.zeros((7,3))
	final_colors[0] = [120,18,134]
	final_colors[1] = [70,130,180]
	final_colors[2] = [0,118,14]
	final_colors[3] = [196,58,250]
	final_colors[4] = [220,248,164]
	final_colors[5] = [230,148,34]
	final_colors[6] = [205,62,78]
	final_colors = final_colors /255.

	membership = np.zeros((loop_n))
	membership = np.append(membership,np.ones((neurons)))
	membership = np.append(membership,np.array(c).astype('float16').flatten())
	from matplotlib.colors import rgb2hex
	colors = sns.color_palette('cubehelix',n_colors=len(np.unique(membership)))
	color_array = np.zeros((g.vcount())).astype(str)
	for n in range(g.vcount()-1):
		if n <= neurons: node_c = rgb2hex([0,0,0])
		else:node_c = np.array(colors[membership[n].astype(int)])
		color_array[n] = str(rgb2hex(node_c))
	color_array[-1] = rgb2hex([0,0,0])
	membership = np.append(membership,-1)
	# sizes = np.ones(g.vcount())
	# # sizes[node] = 3
	# sizes[-1] = 3
	g.vs['membership'] = membership
	# g.vs['size'] = sizes
	g.vs['color'] = color_array.astype('str')
	g.vs['layer'] = np.array(glayers).astype('float16')
	g.vs['neuron'] = np.array(gpositions).astype('float16')
	g.write_gml('/%s/neural_network_%s_%s_%s.gml'%(homedir,layers,neurons,node))

def write_tasks():
	plt.close()
	plt.figure(figsize=(5,5))
	sns.despine()
	tasks = []
	tg = load_tasks()[1]
	for t in tg: tasks.append(t)
	tg = load_tasks()[0]
	for t in tg: tasks.append(t)
	colors = sns.color_palette('cubehelix',n_colors=len(tasks))

	text = '   |   '.join(word for word in tasks)
	plt.text(.5, .5, text, ha='center',va='center',wrap=True)
	sns.despine(left=True,bottom=True)
	plt.xticks([])
	plt.yticks([])
	columns = 4
	t = 0
	# for row in np.linspace(0,1,4):
	for c in np.linspace(0,1,13):

		text = ' | '.join(word for word in tasks[t:t+columns])
		t = t + columns
		plt.text(0,c,text,horizontalalignment='left',**{'fontsize':8})
		
		if row == 0: plt.text(row,c,tasks[t],horizontalalignment='left',**{'fontsize':8})
		elif row == 4:plt.text(row,c,tasks[t],horizontalalignment='right',**{'fontsize':8})
		else:plt.text(row,c,tasks[t],horizontalalignment='center',**{'fontsize':8})
	plt.savefig('/%s/figures/tasks.pdf'%(homedir))

def cv_fig():
	pc = np.load('/%s/results/pc.npy'%(homedir))
	df = behavior()
	task = 'WM_Task_2bk_Acc'
	subjects = df.Subject.values
	matrices = load_matrices(subjects,matrix,components)
	b_array = df[task].values
	b_array[np.isnan(b_array)] = np.nanmean(b_array)
	pc = np.load('/%s/results/pc.npy'%(homedir))
	node = np.argmax(pc)
	
	c1=sns.diverging_palette(220, 10,n=7)[-2]
	c2=sns.diverging_palette(220, 10,n=7)[2]
	cvs = np.zeros((5,len(subjects)))
	kf = KFold(5,shuffle=True,random_state=315)
	icvs = 0
	for train, test in kf.split(subjects):
		cvs[icvs,train] = 1
		icvs = icvs + 1
	fig = plt.figure(figsize=(11.5,5))
	gs = GridSpec(41, 40, left=0.1, right=.9,top=.9,bottom=0.1,wspace=0.05,hspace=0.01)
	ax1 = fig.add_subplot(gs[0:10,:39])
	cbar1 = fig.add_subplot(gs[0:10,39])
	plt.sca(ax1)
	sns.heatmap(cvs,cmap=[c1,c2],cbar_ax=cbar1,cbar_kws={'label': 'Train Test'})
	cbar = ax1.collections[0].colorbar
	cbar.set_label('Train    Test',rotation=270,labelpad=-5)
	cbar.set_ticks([])
	plt.xticks([])
	plt.ylabel('Validation Folds')

	ax2 = fig.add_subplot(gs[10:38,:39])
	cbar2 = fig.add_subplot(gs[10:,39])
	plt.sca(ax2)
	sns.heatmap(matrices[:,node,np.arange(28)].transpose(),cbar_ax=cbar2) 
	plt.ylabel('FC stength to each node')
	plt.xticks([])
	cbar = ax2.collections[0].colorbar
	cbar.set_label('Functional Connectivity Strength',rotation=270,labelpad=-30)

	ax3 = fig.add_subplot(gs[39:40,:39])

	plt.sca(ax3)
	sns.heatmap(matrices[:,node,399].transpose().reshape(1,-1),cbar=False) 
	plt.xlabel('Subjects')
	plt.yticks([0],['399 ...'],rotation=90)
	plt.savefig('/%s/cvfig.pdf'%(homedir))

def final_prediction(task='PicSeq_AgeAdj'):
	b = behavior()
	b_array = b[task]
	b_array[np.isnan(b_array)] = np.nanmean(b_array)
	layers_array = np.array([0,1,2,3,4,5,6,7,8,9,10])
	neurons_array = np.array([10,15,25,50,75,100,150,200,300,400])

	acc = np.zeros((len(layers_array),len(neurons_array)))

	for lidx,layers in enumerate(layers_array):
		for nidx,neurons in enumerate(neurons_array):
			prediction = np.zeros((400,len(b_array)))

			if layers > 0: prediction = np.load('%s/results/%s_%s_%s_%s_prediction.npy'%(homedir,ctype,neurons,layers,task))
			if layers == 0: 
				prediction = np.load('%s//results/%s_%s_%s_prediction.npy'%(homedir,ctype,'ols',task))

			kf = KFold(30,shuffle=True)

			prediction_array = np.zeros((len(b)))
			features = np.zeros(prediction.shape)
			features[:] = prediction
			features = features.transpose()
			p = np.zeros(b_array.shape)
			p[:] = b_array

			for train, test in kf.split(b.Subject.values):
				model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(400,))
				model.fit(features[train],p[train])
				prediction_array[test] = model.predict(features[test])
				
			
			acc[lidx,nidx] = pearsonr(prediction_array,b_array)[0]
			print layers,neurons, acc[lidx,nidx]

def m_plot_structure(data):
	# try:
	global pc
	global neurons_array
	global layers_array

	lidx,nidx,this_task,task_idx = data
	neurons = neurons_array[nidx]
	layers = layers_array[lidx]
	# prediction = np.zeros((400,b_array.shape[0]))
	if os.path.exists('%s//results/%s_%s_%s_%s_%s_%s_network_structure.npy'%(homedir,'fc',neurons,layers,this_task,matrix,components)) and os.path.exists('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,'fc',neurons,layers,this_task,matrix,components)):
		prediction = np.load('%s//results/%s_%s_%s_%s_%s_%s_prediction.npy'%(homedir,'fc',neurons,layers,this_task,matrix,components))
		mod = np.load('%s//results/%s_%s_%s_%s_%s_%s_network_structure.npy'%(homedir,'fc',neurons,layers,this_task,matrix,components))
		prediction_accs = np.load('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,'fc',neurons,layers,this_task,matrix,components))
		pc_acc = nan_pearsonr(pc,prediction_accs)[0]
		pc_mod = nan_pearsonr(mod,pc)[0]
		mod_acc = nan_pearsonr(mod,prediction_accs)[0]

		# meddata = pd.DataFrame(data={'pc':pc,'acc':prediction_accs,'q':mod},index=range(len(pc)))
		# outcome_model = sm.OLS.from_formula("acc ~ q + pc", meddata)
		# mediator_model = sm.OLS.from_formula("q ~ pc", meddata)
		# med = np.mean(sm.stats.Mediation(outcome_model, mediator_model, "pc", "q").fit(n_rep=10).ACME_avg)
		return pc_acc,pc_mod,mod_acc,np.nanmean(prediction_accs),prediction_accs,mod,pearsonr(b_array[this_task],np.nanmean(prediction,axis=0))[0]
	else:
		print layers,neurons,this_task
		return np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

def participation_coef_sign(W, ci):
    '''
    Participation coefficient is a measure of diversity of intermodular
    connections of individual nodes.
    Parameters
    ----------
    W : NxN np.ndarray
        undirected connection matrix with positive and negative weights
    ci : Nx1 np.ndarray
        community affiliation vector
    Returns
    -------
    Ppos : Nx1 np.ndarray
        participation coefficient from positive weights
    Pneg : Nx1 np.ndarray
        participation coefficient from negative weights
    '''
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices

    def pcoef(W_):
        S = np.sum(W_, axis=1)  # strength
        # neighbor community affil.
        Gc = np.dot(np.logical_not(W_ == 0), np.diag(ci))
        Sc2 = np.zeros((n,))

        for i in range(1, int(np.max(ci) + 1)):
            Sc2 += np.square(np.sum(W_ * (Gc == i), axis=1))

        P = np.ones((n,)) - Sc2 / np.square(S)
        P[np.where(np.isnan(P))] = 0
        P[np.where(np.logical_not(P))] = 0  # p_ind=0 if no (out)neighbors
        return P

    #explicitly ignore compiler warning for division by zero
    with np.errstate(invalid='ignore'):
        Ppos = pcoef(W * (W > 0))
        Pneg = pcoef(-W * (W < 0))

    return Ppos, Pneg

def cluster_nodes_by_nn():
	"""
	find communities of nodes, where edges are if they best predicted from same NN
	"""
	df = behavior()
	subjects = df.Subject.values
	if task == 'ptasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = ptasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'all':
		stasks,ptasks,rtasks = load_tasks()
		# tasks = ptasks
		tasks = np.zeros((len(stasks) + len(ptasks) + len(rtasks))).astype('S64')
		tasks[:len(stasks)] = stasks
		tasks[len(stasks):len(stasks)+len(ptasks)] = ptasks
		tasks[len(ptasks) + len(stasks):] = rtasks
		# tasks = tasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)

	layers_array = np.array([1,2,3,4,5,6,7,8,9,10,])
	neurons_array = np.array([10,15,25,50,75,100,150,200,300,400])
	prediction_accs = np.zeros((400,b_array.shape[1],len(layers_array)*len(neurons_array)))
	idx = 0
	for lidx,layers in enumerate(layers_array):
		for nidx,neurons in enumerate(neurons_array):
			for task_idx,this_task in enumerate(tasks):
				try: prediction_accs[:,task_idx,idx] = np.load('%s//results/%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,this_task))
				except: prediction_accs[:,task_idx,idx] = np.nan
			idx = idx + 1

	matrix = np.zeros((400,400))
	prediction_accs = prediction_accs.reshape(400,-1)
	for i,j in combinations(np.arange(400),2):
		r = nan_pearsonr(prediction_accs[i],prediction_accs[j])[0]
		matrix[i,j] = r
		matrix[j,i] = r

	for clusters in range(2,10):
		kmeans = KMeans(n_clusters=clusters, random_state=0).fit(matrix)
		colors = sns.color_palette("Paired",7)
		write_colors = np.zeros((400,3))
		for i in range(clusters):
			write_colors[np.argwhere(kmeans.labels_==i)] = colors[i]
		write_cifti(colors=write_colors,atlas_path=atlas_path,out_path='/%s/brains/nn_clusters_%s'%(homedir,clusters))

	# for i in range(10,15):
	# 	bgraph = brain_graphs.matrix_to_igraph(bmatrix.copy(),i*0.01,mst=True)
	# 	# bgraph = brain_graphs.brain_graph(VertexClustering(bgraph,yeo_partition(17)[1],params={'weight':'weight'}))
	# 	bgraph = brain_graphs.brain_graph(bgraph.community_infomap(edge_weights='weight'))
	# 	1/0

def brain_to_behavior_pc():
	"""
	calculate PC in network where nodes are connected if they predict the same tasks, communitites are yeo 7
	"""

	df = behavior()
	subjects = df.Subject.values
	if task == 'ptasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = ptasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'all':
		stasks,ptasks,rtasks = load_tasks()
		# tasks = ptasks
		tasks = np.zeros((len(stasks) + len(ptasks) + len(rtasks))).astype('S64')
		tasks[:len(stasks)] = stasks
		tasks[len(stasks):len(stasks)+len(ptasks)] = ptasks
		tasks[len(ptasks) + len(stasks):] = rtasks
		# tasks = tasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)

	pc = np.load('/%s/results/pc.npy'%(homedir))
	

	layers_array = np.array([1,2,3,4,5,6,7,8,9,10,])
	neurons_array = np.array([10,15,25,50,75,100,150,200,300,400])
	prediction_accs = np.zeros((400,b_array.shape[1],len(layers_array)*len(neurons_array)))
	idx = 0
	for lidx,layers in enumerate(layers_array):
		for nidx,neurons in enumerate(neurons_array):
			for task_idx,this_task in enumerate(tasks):
				try: prediction_accs[:,task_idx,idx] = np.load('%s//results/%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,this_task))
				except: prediction_accs[:,task_idx,idx] = np.nan
			idx = idx + 1
	raw_prediction_acc = np.nanmean(prediction_accs,axis=1).mean(axis=1)
	task_prediction_acc = np.nanmean(prediction_accs,axis=2)

	"""
	make a matrix where nodes are connected if they predict same behaviors
	make graph, community is yeo. is PC here same as brain?
	"""
	bmatrix = np.zeros((400,400))
	full_matrix = task_prediction_acc.copy()
	for i,j in combinations(np.arange(400),2):
		r = nan_pearsonr(full_matrix[i],full_matrix[j])[0]
		bmatrix[i,j] = r
		bmatrix[j,i] = r
	bpc = []
	for i in range(10,15):
		bgraph = brain_graphs.matrix_to_igraph(bmatrix.copy(),i*0.01,mst=True)
		bgraph = brain_graphs.brain_graph(VertexClustering(bgraph,yeo_partition(7)[1],params={'weight':'weight'}))
		bpc.append(bgraph.pc)

	colors = np.array(make_heatmap(cut_data(np.nanmean(bpc,axis=0),0,.7),sns.diverging_palette(220, 10,n=1001)))
	write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/b_pc'%(homedir))

def loop_sge(components=None,matrix='all'):
	scanner,ptasks,rtasks = load_tasks()
	for task in ptasks:
		# sge_structure(task,components,matrix)
		sge(task,components,matrix)
	# for task in scanner: 
		# sge_structure(task,components,matrix)
		# sge(task,components,matrix)

def sge(task,components=None,matrix='all'):
	if os.path.exists('%s//results/%s_%s_%s_prediction.npy'%(homedir,ctype,'ols',task)) == False:
		os.system('qsub -l h_vmem=4G,s_vmem=4G -N p_%s_%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
			/%s/prediction.py -r run -neurons %s -layers %s -task %s -matrix %s -components %s' %(0,0,homedir,0,0,task,matrix,components))
	for layers in [1,2,3,4,5,6,7,8,9,10,15,20,25,30]:
	# for layers in [25]:
		for neurons in [5,10,15,25,50,75,100,150,200,300,400,500,750,1000]:
			if os.path.exists('%s//results/%s_%s_%s_%s_%s_%s_prediction.npy'%(homedir,ctype,neurons,layers,task,matrix,components)) == True:
				if os.path.exists('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,task,matrix,components)) == True:
					continue
			if neurons <= 500:
				os.system('qsub -l h_vmem=4G,s_vmem=4G -N p%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
					/%s/prediction.py -r run -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))
				continue
			elif layers <=10:
				os.system('qsub -l h_vmem=6G,s_vmem=6G -N p%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
					/%s/prediction.py -r run -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))
				continue
			elif layers ==20:
				if neurons == 750:
					os.system('qsub -l h_vmem=6G,s_vmem=6G -N p%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
						/%s/prediction.py -r run -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))
					continue
				if neurons == 1000:
					os.system('qsub -l h_vmem=10G,s_vmem=10G -N p%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
						/%s/prediction.py -r run -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))
					continue
			elif layers ==25:
				if neurons == 750:
					os.system('qsub -l h_vmem=8G,s_vmem=8G -N p%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
						/%s/prediction.py -r run -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))
					continue
				if neurons == 1000:
					os.system('qsub -l h_vmem=12G,s_vmem=12G -N p%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
						/%s/prediction.py -r run -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))
					continue
			elif layers ==30:
				if neurons == 750:
					os.system('qsub -l h_vmem=8G,s_vmem=8G -N p%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
						/%s/prediction.py -r run -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))
					continue
				if neurons == 1000:
					os.system('qsub -l h_vmem=12G,s_vmem=12G -N p%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
						/%s/prediction.py -r run -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))
					continue

def sge_structure(task,components=None,matrix='all'):
	for layers in [2,3,4,5,6,7,8,9,10]:
		for neurons in [5,10,15,25,50,75,100,150,200,300,400]:
			# if os.path.exists('%s//results/%s_%s_%s_%s_%s_%s_network_structure.npy'%(homedir,ctype,neurons,layers,task,matrix,components)) == True: continue
			if layers <=5 and neurons < 100:
				os.system('qsub -l h_vmem=16G,s_vmem=16G -N s%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
					/%s/prediction.py -r structure -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))
			else:
				os.system('qsub -l h_vmem=60G,s_vmem=60G -N s%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/\
					/%s/prediction.py -r structure -neurons %s -layers %s -task %s -matrix %s -components %s' %(layers,neurons,task,homedir,neurons,layers,task,matrix,components))

def convert():
	files = glob.glob('/cbica/home/bertolem/deep_prediction/results/**')
	for f in files:
		data = np.load(f)
		if data.dtype != 'float16':
			np.save(f,data.astype('float16'))

def figure2(task='all_p',ctype=ctype,matrix='all',components='None'):
	df = behavior()
	subjects = df.Subject.values
	if task == 'ptasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = ptasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'all':
		stasks,ptasks,rtasks = load_tasks()
		# tasks = ptasks
		tasks = np.zeros((len(stasks) + len(ptasks) + len(rtasks))).astype('S64')
		tasks[:len(stasks)] = stasks
		tasks[len(stasks):len(stasks)+len(ptasks)] = ptasks
		tasks[len(ptasks) + len(stasks):] = rtasks
		# tasks = tasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'all_p':
		stasks,ptasks,rtasks = load_tasks()
		# tasks = ptasks
		tasks = np.zeros((len(stasks) + len(ptasks))).astype('S64')
		tasks[:len(stasks)] = stasks
		tasks[len(stasks):] = ptasks
		# tasks = tasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'rtasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = rtasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'stasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = stasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task not in ['ptasks','stasks','rtasks','all','all_p']:
		b_array = df[task].values
		b_array[np.isnan(b_array)] = np.nanmean(b_array)
		tasks = [task]

	loop_n = 400
	try:
		if int(components) == 7 or int(components) == 17: 
			loop_n = int(components)
	except: pass

	print loop_n
	if loop_n == 400: pc = np.load('/%s/results/pc.npy'%(homedir))
	else:
		yeo_labels = yeo_partition(int(components))[1]
		if loop_n == 17: loop_n = 16
		pc = np.zeros(int(loop_n))
		pc_raw = np.load('/%s/results/pc.npy'%(homedir))
		for i in range(int(loop_n)):
			pc[i] = np.mean(pc_raw[np.where(yeo_labels==i)])


	full_prediction_array = np.zeros((loop_n,len(b_array)))
	acc = np.zeros((loop_n))

	pcthresh= np.where(pc>np.percentile(pc,80))
	

	l = [] #layers to plot
	n = [] #neurons to plot
	r = [] #mean(accuracy)
	p = [] #pearsonr(pc,accuracy)
	
	# layers_array = np.array([1,2,3,4,5,6,7,8,9,10,15,20,25,30])
	# neurons_array = np.array([25,50,75,100,150,200,300,400,500,750,1000])

	layers_array = np.array([1,2,3,4,5,6,7,8,9,10,])
	neurons_array = np.array([10,15,25,50,75,100,150,200,300,400,])


	plot_layers_array = np.arange(len(layers_array))
	plot_neurons_array = np.arange(len(neurons_array))
	r_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
	p_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
	all_prediction_accs = np.zeros((len(layers_array),len(neurons_array),loop_n))
	try:full_prediction_accs = np.zeros((len(layers_array),len(neurons_array),loop_n,b_array.shape[1]))
	except:full_prediction_accs = np.zeros((len(layers_array),len(neurons_array),loop_n))
	# if loop_n == 400: 
	c_prediction_accs = np.zeros((len(layers_array),len(neurons_array),b_array.shape[1]))
	pc_prediction_accs = np.zeros((len(layers_array),len(neurons_array),b_array.shape[1]))
	try:full_prediction = np.zeros((len(layers_array),len(neurons_array),loop_n,b_array.shape[0],b_array.shape[1]))
	except:full_prediction = np.zeros((len(layers_array),len(neurons_array),loop_n))
	all_nn_pc_r = []
	idx = 0

	all_mean_prediction_acc = np.zeros((len(layers_array),len(neurons_array),b_array.shape[1]))
	pc_all_mean_prediction_acc =  np.zeros((len(layers_array),len(neurons_array),b_array.shape[1]))

	if loop_n == 7: names = ['Visual','Sensory\nMotor','Dorsal\nAttention','Ventral\nAttention','Limbic','Control','Default']
	if loop_n == 16: names = ['Visual','Visual','Sensory\nMotor','Sensory\nMotor','Dorsal\nAttention','Dorsal\nAttention','Ventral\nAttention','Ventral\nAttention','Limbic','Control','Control','Control','Default','Default','Default','Default']
	
	membership,membership_ints,colors = yeo_partition(7)
	if loop_n != 400: membership = names

	df_membership = []
	df_pc = []
	df_layers = []
	df_neurons = []
	for lidx,layers in enumerate(layers_array):
		for nidx,neurons in enumerate(neurons_array):
			for ri in range(loop_n):
				df_neurons.append(neurons)
				df_layers.append(layers)
			df_membership.append(membership)
			df_pc.append(pc)
			if task not in ['ptasks','stasks','rtasks','all','all_p']:
				prediction_accs = np.zeros((loop_n))
				prediction = np.zeros((loop_n,b_array.shape[0]))			
				try:
					prediction_accs = np.load('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,task,matrix,components))
					prediction = np.load('%s//results/%s_%s_%s_%s_%s_%s_prediction.npy'%(homedir,ctype,neurons,layers,task,matrix,components))
				except:
					prediction[:] = np.nan
					prediction_accs[:] = np.nan
					print layers,neurons,task
				mean_prediction_acc = np.nanmean(prediction_accs)
				all_prediction_accs[lidx,nidx] = prediction_accs
				full_prediction_accs[lidx,nidx] = prediction_accs
				hub_corr = nan_pearsonr(prediction_accs,pc)[0]
				l.append(lidx)
				n.append(nidx)
				r.append(mean_prediction_acc)
				p.append(hub_corr)
				print layers,neurons,mean_prediction_acc, hub_corr
				r_array[lidx,nidx] = mean_prediction_acc
				p_array[lidx,nidx] = hub_corr
			else:
				prediction_accs = np.zeros((loop_n,b_array.shape[1]))
				prediction = np.zeros((loop_n,b_array.shape[0],b_array.shape[1]))
				for task_idx,this_task in enumerate(tasks):
					try:
						if layers > 0: 
							prediction_accs[:,task_idx] = np.load('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,this_task,matrix,components))
							prediction[:,:,task_idx] = np.load('%s//results/%s_%s_%s_%s_%s_%s_prediction.npy'%(homedir,ctype,neurons,layers,this_task,matrix,components))
						if layers == 0: 
							prediction_accs[:,task_idx] = np.load('%s//results/%s_%s_%s_%s_%s.npy'%(homedir,ctype,'ols',this_task,matrix,components))
							prediction[:,:,task_idx] = np.load('%s//results/%s_%s_%s_%s_%s_prediction.npy'%(homedir,ctype,'ols',this_task,matrix,components))
					except:
						prediction_accs[:,task_idx] = np.nan
					all_nn_pc_r.append(nan_pearsonr(prediction_accs[:,task_idx],pc)[0])
				full_prediction_accs[lidx,nidx,:,:] = prediction_accs
				full_prediction[lidx,nidx] = prediction
				# 1/0

				for task_idx,this_task in enumerate(tasks):
					all_mean_prediction_acc[lidx,nidx,task_idx] = nan_pearsonr(np.nanmean(prediction[:,:,task_idx],axis=0),b_array[this_task])[0]
					pc_all_mean_prediction_acc[lidx,nidx,task_idx] = nan_pearsonr(np.nanmean(prediction[pc>.549,:,task_idx],axis=0),b_array[this_task])[0]
				mean_prediction_acc = np.nanmean(prediction_accs)
				# print scipy.stats.ttest_rel(pc_mean_prediction_acc,mean_prediction_acc)
				c_prediction_accs[lidx,nidx,:] = mean_prediction_acc
				# pc_prediction_accs[lidx,nidx,:] = pc_mean_prediction_acc
				all_prediction_accs[lidx,nidx] = np.nanmean(prediction_accs,axis=1)
				## don't do it on collapsed
				
				# 1/0
				hub_corr = nan_pearsonr(pc,np.nanmean(prediction_accs,axis=1))[0]
				l.append(lidx)
				n.append(nidx)
				r.append(np.nanmean(np.nanmean(mean_prediction_acc)))
				p.append(hub_corr)
				print layers,neurons,np.nanmean(mean_prediction_acc),hub_corr
				r_array[lidx,nidx] = np.nanmean(mean_prediction_acc)
				p_array[lidx,nidx] = hub_corr
			idx = idx + 1	

	np.save('/%s/results/hub_corr.npy'%(homedir),p_array)

	# fps = []
	# full_prediction = full_prediction.reshape(100,400,607,52)
	# fp = full_prediction.mean(axis=0).mean(axis=0)
	# for i in range(52):
	# 	fps.append(nan_pearsonr(fp[:,i],b_array[tasks[i]])[0])


	all_nn_pc_r = np.array(all_nn_pc_r)
	mean_nodal_nn_acc = np.nanmean(np.nanmean(all_prediction_accs,axis=0),axis=0)
	
	if loop_n == 400:
		colors = np.array(make_heatmap(cut_data(mean_nodal_nn_acc,1.5,1.5),sns.diverging_palette(220, 10,n=1001)))
		write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/pc_r_nn_%s'%(homedir,task))

		colors = np.array(make_heatmap(cut_data(mean_nodal_nn_acc,1.5,1.5),sns.diverging_palette(220, 10,n=1001)))
		write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/pc_r_nn_%s'%(homedir,task))

	nn_df = pd.DataFrame(columns=['system','prediction accuracy','task','model'])
	for task_idx,this_task in enumerate(tasks):
		df = pd.DataFrame(columns=['system','prediction accuracy','task','model'])
		df['system'] = np.array(df_membership).flatten()
		df['task'] = this_task
		if len(tasks) > 1: df['prediction accuracy'] = full_prediction_accs[:,:,:,task_idx].flatten()
		else:df['prediction accuracy'] = full_prediction_accs.flatten()
		df['model'] = 'neural network'
		df['pc'] = np.array(df_pc).flatten()
		df['neurons'] = np.array(df_neurons).flatten()
		df['layers'] = np.array(df_layers).flatten()
		nn_df = pd.concat([df,nn_df])

	if task in ['ptasks','stasks','rtasks','all','all_p']:
		ols_df_membership = []
		ols_pc_r = np.zeros((len(tasks)))
		ols_prediction_accs = np.zeros((loop_n,b_array.shape[1]))
		mean_ols_prediction_accs = np.zeros((b_array.shape[1]))
		pc_mean_ols_prediction_accs = np.zeros((b_array.shape[1]))
		ols_pred = np.zeros((loop_n,b_array.shape[0],b_array.shape[1]))


		for task_idx,this_task in enumerate(tasks):
			try:
				ols_pred[:,:,task_idx] = np.load('%s/results/%s_%s_%s_%s_%s_prediction.npy'%(homedir,ctype,'ols',this_task,matrix,components))
				ols_prediction_accs[:,task_idx] = np.load('%s/results/%s_%s_%s_%s_%s.npy'%(homedir,ctype,'ols',this_task,matrix,components))
				ols_pc_r[task_idx]= pearsonr(pc,ols_prediction_accs[:,task_idx])[0]
				
				mean_ols_prediction_accs[task_idx] = pearsonr(np.nanmean(ols_pred[:,:,task_idx],axis=0),b_array[this_task])[0]
				pc_mean_ols_prediction_accs[task_idx] = pearsonr(np.nanmean(ols_pred[pcthresh,:,task_idx],axis=1)[0],b_array[this_task])[0]
			except:
				ols_pred[:,:,task_idx] = np.nan
				ols_prediction_accs[:,task_idx] = np.nan

		ols_pred = np.nanmean(ols_pred,axis=2)
		# ols_prediction_accs = np.nanmean(ols_prediction_accs,axis=1)
	else:
		ols_pred = np.load('%s/results/%s_%s_%s_%s_%s_prediction.npy'%(homedir,ctype,'ols',task,matrix,components))
		ols_prediction_accs= np.load('%s/results/%s_%s_%s_%s_%s.npy'%(homedir,ctype,'ols',task,matrix,components))
		ols_pc_r = nan_pearsonr(ols_prediction_accs,pc)[0]

	if loop_n == 400:
		colors = np.array(make_heatmap(cut_data(np.nanmean(ols_prediction_accs,axis=1),1.5,1.5),sns.diverging_palette(220, 10,n=1001)))
		write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/pc_r_ols_%s'%(homedir,task))

		nn_high = scipy.stats.zscore(mean_nodal_nn_acc) - scipy.stats.zscore(np.nanmean(ols_prediction_accs,axis=1))
		colors = np.array(make_heatmap(cut_data(nn_high,1,1),sns.diverging_palette(220, 10,n=1001)))
		write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/nn_higher_%s'%(homedir,task))

		ols_high =  scipy.stats.zscore(np.nanmean(ols_prediction_accs,axis=1)) - scipy.stats.zscore(mean_nodal_nn_acc)
		colors = np.array(make_heatmap(cut_data(ols_high,1,1),sns.diverging_palette(220, 10,n=1001)))
		write_cifti(colors=colors,atlas_path=atlas_path,out_path='/%s/brains/ols_higher_%s'%(homedir,task))
	
	ols_df = pd.DataFrame(columns=['system','prediction accuracy','task','model'])
	for task_idx,this_task in enumerate(tasks):
		df = pd.DataFrame(columns=['system','prediction accuracy','task','model'])
		df['system'] = membership
		if len(tasks) > 1: df['prediction accuracy'] = ols_prediction_accs[:,task_idx].flatten()
		else: df['prediction accuracy'] = ols_prediction_accs.flatten()
		df['task'] = this_task
		df['model'] = 'linear'
		df['pc'] = pc
		df['layers'] = 0
		df['neurons'] = 0
		ols_df = pd.concat([ols_df,df])

	df = nn_df.copy()
	# for i in range(len(layers_array)*len(neurons_array)):
	df = pd.concat([df,ols_df])
	df = df.dropna()
	stat_d = {}
	for system in df.system.unique():
		x = df['prediction accuracy'][(df.model == 'neural network')&(df.system == system)]
		y = df['prediction accuracy'][(df.model == 'linear')&(df.system == system)]
		stat_d[system] = scipy.stats.ttest_ind(x,y)

	df.to_csv('/%s/results/%s_%s_%s_df.csv'%(homedir,task,matrix,components))


	cdf = pd.DataFrame(columns=['accuracy','task','model'])
	cdf['task'] = np.tile(tasks,101)
	cdf['model'][:] = 'neural network'
	cdf['model'][-len(tasks):] = 'linear'
	cdf['accuracy'][:-len(tasks)] = c_prediction_accs.flatten()
	cdf['accuracy'][-len(tasks):] = mean_ols_prediction_accs

	x = c_prediction_accs.reshape(100,52).mean(axis=0)
	y = mean_ols_prediction_accs
	scipy.stats.ttest_rel(x,y)	
	cdf.to_csv('/%s/results/%s_%s_%s_cdf.csv'%(homedir,task,matrix,components))

	cdf = pd.DataFrame(columns=['accuracy','task','model'])
	cdf['task'] = np.array([tasks,tasks]).flatten()
	cdf['model'][:len(tasks)] = 'neural network'
	cdf['model'][len(tasks):] = 'linear'
	cdf['accuracy'][:len(tasks)] = c_prediction_accs.reshape(100,52).mean(axis=0)
	cdf['accuracy'][len(tasks):] = mean_ols_prediction_accs

	x = c_prediction_accs.reshape(100,52).mean(axis=0)
	y = mean_ols_prediction_accs
	scipy.stats.ttest_rel(x,y)	
	cdf.to_csv('/%s/results/%s_%s_%s_cdfm.csv'%(homedir,task,matrix,components))


	plt.close()
	sns.set(context="paper",font_scale=1,font='Palatino',style='darkgrid',palette="pastel",color_codes=True)
	f = plt.figure(figsize=(4.2,6))
	with sns.axes_style("whitegrid"):
		sns.set_style("whitegrid")
		sns.set_style({'font.family':'serif'})
		sns.set_style({"axes.labelcolor": "black"})
		sns.set_style({"patch.edgecolor": "black"})
		sns.set_style({"grid.color": "black"})
		c1=sns.diverging_palette(220, 10, sep=80, n=7)[-2]
		c2=sns.diverging_palette(220, 10, sep=80, n=7)[1]
		ax1 = f.add_subplot(211)
	f.sca(ax1)
	sns.boxenplot(data=df,y='prediction accuracy',x='system',hue='model',outlier_prop=0,k_depth="trustworthy",palette=[c1,c2])
	names = ['Visual','Sensory\nMotor','Dorsal\nAttention','Ventral\nAttention','Limbic','Control','Default']	
	label_min = np.min([ols_df['prediction accuracy'].min(),nn_df['prediction accuracy'].min()])
	for idx,system in enumerate(names):
		t_val,p_val = stat_d[system]
		if p_val == 0.0: p_val = "-log10($\it{p}$)>100"
		else: p_val = log_p_value(p_val)
		if t_val > 0: plt.text(idx,label_min-0.01,'$\it{t}$=%s\n%s' %(np.around(abs(t_val),1),p_val),{'fontsize':5},horizontalalignment='center',verticalalignment='top',color=c1)
		if t_val <= 0: plt.text(idx,label_min-0.01,'$\it{t}$=%s\n%s' %(np.around(abs(t_val),1),p_val),{'fontsize':5},horizontalalignment='center',verticalalignment='top',color=c2)
	plt.legend(loc='upper left')
	ax1.set_xticklabels(names,{'fontsize':7})
	
	plt.ylim(-0.27,0.5)
	with sns.axes_style("white"):
		sns.set_style({'font.family':'serif'})
		ax2 = f.add_subplot(212)
		f.sca(ax2)
		ax2.remove()
		ax2=plt.subplot(2,1,2,projection='3d')
		ax2 = f.gca(projection='3d')
		x,y	= np.meshgrid(plot_neurons_array,plot_layers_array)
		ax2.plot_wireframe(x, y, r_array, 1, color ='black')
		ax2.set_xticks(np.arange(len(neurons_array)))
		ax2.set_xticklabels(neurons_array,fontdict={'fontsize':'x-small'},rotation=-15)
		ax2.set_yticks(np.arange(len(layers_array)))
		ax2.set_yticklabels(layers_array,fontdict={'fontsize':'x-small'},rotation=15)
		ax2.set_ylabel('layers')
		ax2.set_xlabel('neurons')
		ax2.set_zlabel('accuracy')
		plotp = np.array(p)
		plotp[plotp<(plotp.mean()-plotp.std()*1)] = plotp.mean()-plotp.std()*1
		plotp[plotp>(plotp.mean()+plotp.std()*1)] = plotp.mean()+plotp.std()*1
		ax2.view_init(40,-135)
		scatter = ax2.scatter(n,l, r, c=plotp,cmap=sns.diverging_palette(220, 10, sep=80, n=100,as_cmap=True),s=30,depthshade=False,edgecolors="black")
		cbar = f.colorbar(scatter, shrink=0.5, aspect=30,pad = -0.05)
		plt.tight_layout()
	plt.savefig('prediction_figure_%s_%s_%s.pdf'%(task,matrix,components),bbox_inches="tight")
	plt.close()

	if task in ['ptasks','stasks','rtasks','all','all_p']:ols_prediction_accs = ols_prediction_accs.mean(axis=1)

	pc_c_df = pd.DataFrame(columns=['model','accuracy hub pearsonr'])
	for val in all_nn_pc_r:
		pc_c_df = pc_c_df.append({'model':'neural network','accuracy hub pearsonr':val},ignore_index=True)
	for val in ols_pc_r:
		pc_c_df= pc_c_df.append({'model':'linear','accuracy hub pearsonr':val},ignore_index=True)

	full_corr_df = pd.DataFrame(columns=['model','accuracy','participation coef'])
	for p,val in zip(pc,mean_nodal_nn_acc):
		full_corr_df = full_corr_df.append({'model':'neural network','accuracy':val,'participation coef':p},ignore_index=True)
	for p,val in zip(pc,ols_prediction_accs):
		full_corr_df = full_corr_df.append({'model':'linear','accuracy':val,'participation coef':p},ignore_index=True)

	pc_c_df = pc_c_df.dropna()

	plt.close()
	sns.set(context="paper",font_scale=1,font='Palatino',style='white',palette="pastel",color_codes=True)
	sns.set_style({"axes.labelcolor": "black"})
	sns.set_style({"patch.edgecolor": "black"})
	sns.set_style({"grid.color": "black"})
	f, axes = plt.subplots(1, 4,figsize=(11,2.5),tight_layout=True)
	sns.boxenplot(data=pc_c_df,y='accuracy hub pearsonr',x='model',hue_order=['neural network','linear'],ax=axes[0],palette=[c1,c2])
	t,p = scipy.stats.ttest_ind(pc_c_df[pc_c_df.model=='neural network']['accuracy hub pearsonr'],pc_c_df[pc_c_df.model=='linear']['accuracy hub pearsonr'])
	f.sca(axes[0])
	plt.ylabel("Pearson's $\it{r}$ (PC,accuracy)")
	plt.text(.5,0,'$\it{t}$=%s\n%s,df=50' %(np.around(t,1),log_p_value(p)),{'fontsize':6},horizontalalignment='center',transform=axes[0].transAxes)

	sns.regplot(data=full_corr_df[full_corr_df.model=='neural network'],y='accuracy',x='participation coef',ax=axes[1],color=c1,scatter_kws={'alpha':0.5})
	sns.regplot(data=full_corr_df[full_corr_df.model=='linear'],y='accuracy',x='participation coef',ax=axes[2],color=c2,scatter_kws={'alpha':0.5})
	sns.regplot(y=nn_high,x=pc,ax=axes[3],color=np.nanmean([c1,c2],axis=0),scatter_kws={'alpha':0.5})

	t,p = nan_pearsonr(mean_nodal_nn_acc,pc)
	f.sca(axes[1])
	plt.text(.5,0,'$\it{r}$=%s\n%s,df=399' %(np.around(t,3),log_p_value(p)),{'fontsize':6},horizontalalignment='center',transform=axes[1].transAxes)
	d = full_corr_df[full_corr_df.model=='neural network']['participation coef'].values
	ymin,ymax = d.min(),d.max()
	axes[1].set_xticks([ymin,ymax])
	axes[1].set_xticklabels([np.around(ymin,2),np.around(ymax,2)])
	axes[1].set_ylabel('neural network accuracy')
	
	t,p = nan_pearsonr(ols_prediction_accs,pc)
	f.sca(axes[2])
	plt.text(.5,0,'$\it{r}$=%s\n%s,df=399' %(np.around(t,3),log_p_value(p)),{'fontsize':6},horizontalalignment='center',transform=axes[2].transAxes)
	d = full_corr_df[full_corr_df.model=='linear']['participation coef'].values
	ymin,ymax = d.min(),d.max()
	axes[2].set_xticks([ymin,ymax])
	axes[2].set_xticklabels([np.around(ymin,2),np.around(ymax,2)])
	axes[2].set_ylabel('linear accuracy')

	t,p = nan_pearsonr(pc,nn_high)
	f.sca(axes[3])
	plt.text(.5,0,'$\it{r}$=%s\n%s,df=399' %(np.around(t,3),log_p_value(p)),{'fontsize':6},horizontalalignment='center',transform=axes[3].transAxes)
	d = pc
	ymin,ymax = d.min(),d.max()
	axes[3].set_xticks([ymin,ymax])
	axes[3].set_xticklabels([np.around(ymin,2),np.around(ymax,2)])
	axes[3].set_ylabel('neural network accuracy\n > linear accuracy')
	axes[3].set_xlabel('participation coef')
	plt.savefig('prediction_bar_corr_%s_%s_%s.pdf'%(task,matrix,components),bbox_inches="tight")
	plt.close()

def figure3(task='all_p',ctype='fc',matrix='all',components='None'):
	global pc
	global neurons_array
	global layers_array
	global pc_mod_array
	global pc_acc_array
	global mod_acc_array
	global med_array
	sns.set(context="paper",font_scale=1,font='Palatino')
	df = behavior()
	subjects = df.Subject.values
	names = ['Visual','Sensory\nMotor','Dorsal\nAttention','Ventral\nAttention','Control','Default']	
	matrices = load_matrices(subjects,matrix,components)
	mean_matrix = np.nanmean(matrices,axis=0)
	if task == 'ptasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = ptasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'stasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = stasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'all':
		stasks,ptasks,rtasks = load_tasks()
		tasks = np.zeros((len(stasks) + len(ptasks) + len(rtasks))).astype('S64')
		tasks[:len(stasks)] = stasks
		tasks[len(stasks):len(stasks)+len(ptasks)] = ptasks
		tasks[len(ptasks) + len(stasks):] = rtasks
		# tasks = tasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'all_p':
		stasks,ptasks,rtasks = load_tasks()
		# tasks = ptasks
		tasks = np.zeros((len(stasks) + len(ptasks))).astype('S64')
		tasks[:len(stasks)] = stasks
		tasks[len(stasks):] = ptasks
		# tasks = tasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'rtasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = rtasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task not in ['ptasks','stasks','rtasks','all','all_p']:
		b_array = df[task].values
		b_array[np.isnan(b_array)] = np.nanmean(b_array)
		tasks = [task]

	layers_array = np.array([2,3,4,5,6,7,8,9,10])
	neurons_array = np.array([10,15,25,50,75,100,150,200,300,400])

	loop_n = 400
	try:
		if int(components) == 7 or int(components) == 17: 
			loop_n = int(components)
	except: pass
	pc = np.load('/%s/results/pc.npy'%(homedir))
	
	if task not in ['ptasks','stasks','rtasks','all','all_p']:
		pc_mod_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],))
		mod_acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
		pc_acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
		med_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
		acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
		mod_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],loop_n))
	else:
		pc_mod_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],b_array.shape[1]))
		mod_acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],b_array.shape[1]))
		pc_acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],b_array.shape[1]))
		med_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],b_array.shape[1]))
		acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],b_array.shape[1]))
		mod_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],b_array.shape[1],loop_n))
		full_acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],b_array.shape[1],loop_n))
		mean_acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],b_array.shape[1]))
	
	pc_mod_array[:] = np.nan
	mod_acc_array[:] = np.nan
	pc_acc_array[:] = np.nan
	med_array[:] = np.nan
	acc_array[:] = np.nan
	mod_array[:] = np.nan

	
	pool = Pool(39)
	for lidx,layers in enumerate(layers_array):
		for nidx,neurons in enumerate(neurons_array):
			print layers,neurons
			if task not in ['ptasks','stasks','rtasks','all','all_p']:
				if layers == 1: continue
				prediction_accs = np.load('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,task,matrix,components))
				mod = np.load('%s//results/%s_%s_%s_%s_%s_%s_network_structure.npy'%(homedir,ctype,neurons,layers,task,matrix,components))
				pc_acc_array[lidx,nidx] = nan_pearsonr(pc,prediction_accs)[0]
				pc_mod_array[lidx,nidx] = nan_pearsonr(mod,pc)[0]
				mod_acc_array[lidx,nidx] = nan_pearsonr(mod,prediction_accs)[0]
				acc_array[lidx,nidx] = np.nanmean(prediction_accs)
				mod_array[lidx,nidx] = mod
			else:
				tosubmit= []
				for task_idx,this_task in enumerate(tasks):
					tosubmit.append([lidx,nidx,this_task,task_idx])
				result = pool.map(m_plot_structure,tosubmit)
				ridx = 0
				for r in result:
					pc_acc_array[lidx,nidx,ridx],pc_mod_array[lidx,nidx,ridx],mod_acc_array[lidx,nidx,ridx],acc_array[lidx,nidx,ridx],full_acc_array[lidx,nidx,ridx],mod_array[lidx,nidx,ridx],mean_acc_array[lidx,nidx,ridx] = r
					ridx = ridx +1
	pc_acc_array = []
	for i in range(52):pc_acc_array.append(np.load('/%s/results/hub_corr.npy'%(homedir))[1:,:])
	pc_acc_array = np.array(pc_acc_array).swapaxes(0,1).swapaxes(1,2)
	f, axes = plt.subplots(2, 4,figsize=(14.333,6),tight_layout=True)
	#architecture Q
	vals = np.nanmean(mod_array,axis=3).mean(axis=2)/10000.
	ax = axes[0][0]
	sns.heatmap(vals,cmap='coolwarm',vmax=.25,vmin=0.05,ax=ax,cbar_kws={'label': 'DNN Q'})
	cbar = ax.collections[0].colorbar
	cbar.set_ticks([0.05,0.25])
	cbar.set_ticklabels([0.05,.25])
	cbar.ax.locator_params(nbins=2)
	cbar.set_label('architecture $\it{Q}$',rotation=270,labelpad=-10)
	ax.set_yticklabels(layers_array,fontsize=8)
	ax.set_xticklabels(neurons_array,fontsize=8)
	ax.set_ylabel('layers')
	ax.set_xlabel('neurons')
	#architecture accuracy
	vals = np.nanmean(acc_array[:,:,:],axis=2)
	ax = axes[0][1]
	sns.heatmap(vals,cmap='coolwarm',ax=ax,cbar_kws={'label': 'DNN Performance'})
	cbar = ax.collections[0].colorbar
	cmin,cmax = vals.min(),vals.max()
	cbar.set_ticks([cmin,cmax])
	cmin,cmax = np.around(vals.min(),3),np.around(vals.max(),3)
	cbar.set_ticklabels([cmin,cmax])
	cbar.set_label('architecture accuracy',rotation=270,labelpad=-10)
	ax.set_yticklabels(layers_array,fontsize=8)
	ax.set_xticklabels(neurons_array,fontsize=8)
	ax.set_ylabel('layers')
	ax.set_xlabel('neurons')
	#architecture,pc-accuracy
	ax = axes[0][2]
	vals = np.nanmean(pc_acc_array,axis=2)
	sns.heatmap(vals,cmap='coolwarm',ax=ax)
	cbar = ax.collections[0].colorbar
	cmin,cmax = vals.min(),vals.max()
	cbar.set_ticks([cmin,cmax])
	cmin,cmax = np.around(vals.min(),2),np.around(vals.max(),2)
	cbar.set_ticklabels([cmin,cmax])
	cbar.set_label("Pearson's $\it{r}$ (PC,accuracy",rotation=270,labelpad=-8)
	ax.set_yticklabels(layers_array,fontsize=8)
	ax.set_xticklabels(neurons_array,fontsize=8)
	ax.set_ylabel('layers')
	ax.set_xlabel('neurons')
	#q,acc,pc polyfit
	ax = axes[0][3]
	plt.sca(ax)
	x,y=full_acc_array.mean(axis=3).mean(axis=2).flatten(),mod_array.mean(axis=3).mean(axis=2).flatten()/10000
	# for i in range(2,10):
	# 	polynomial_features= PolynomialFeatures(degree=i)
	# 	xp = polynomial_features.fit_transform(x.reshape(-1,1))
	# 	model = sm.OLS(y, xp).fit()
	# 	print i,model.rsquared,pearsonr(model.resid,pc_acc_array.mean(axis=2).flatten())
	polynomial_features= PolynomialFeatures(degree=4)
	xp = polynomial_features.fit_transform(x.reshape(-1,1))
	model = sm.OLS(y, xp).fit()
	# print i,model.rsquared,pearsonr(model.resid,pc_acc_array.mean(axis=2).flatten())
	ypred = model.predict(xp)
	scat = plt.scatter(x,y,c=pc_acc_array.mean(axis=2).flatten(),cmap='coolwarm',alpha=0.75)
	plt.colorbar(scat,ax=ax)
	ax.set_xlim(0.0525,0.061)
	cbar = ax.collections[0].colorbar
	cmin,cmax=pc_acc_array.mean(axis=2).flatten().min(),pc_acc_array.mean(axis=2).flatten().max()
	cbar.set_ticks([cmin,cmax])
	cbar.set_ticklabels([np.around(cmin,2),np.around(cmax,2)])
	cbar.set_label("Pearson's $\it{r}$ (PC,accuracy)",rotation=270,labelpad=-10)
	ax.set_ylabel('architecture $\it{Q}$')
	ax.set_xlabel('architecture accuracy')
	sns.lineplot(x,ypred)
	#residuals from q,acc,pc polyfit
	ax = axes[1][0]
	x,y = model.resid,pc_acc_array.mean(axis=2).flatten()
	sns.regplot(x,y,ax=ax,color=sns.color_palette('coolwarm',20)[-2])
	r_val,p_val = nan_pearsonr(x,y)
	ax.text(.85,.1,'$\it{r}$=%s\n%s,df=98' %(np.around(r_val,2),log_p_value(p_val)),{'fontsize':6},horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	ax.set_xlim(x.min(),x.max())
	ax.set_ylim(y.min(),y.max())
	ax.set_xlabel("Residuals of $\\bf{d}$")
	ax.set_ylabel("architecture models hubs\nPearson's $\it{r}$ (PC,accuracy)")
	#if a task is captured by high q arch, high q arch captures hubs
	ax = axes[1][1]
	x,y= mod_acc_array.mean(axis=0).mean(axis=0).flatten(),pc_mod_array.mean(axis=0).mean(axis=0).flatten()
	sns.regplot(x,y,ax=ax,color=sns.color_palette('coolwarm',20)[-2])
	r_val,p_val = nan_pearsonr(x,y)
	ax.text(.85,.1,'$\it{r}$=%s\n%s,df=50' %(np.around(r_val,2),log_p_value(p_val)),{'fontsize':6},horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	ax.set_xlim(x.min(),x.max())
	ax.set_ylim(y.min(),y.max())
	ax.set_xlabel("High $\it{Q}$ architectures predict measure\nPearson's $\it{r}$ ($\it{Q}$,accuracy)")
	ax.set_ylabel("High $\it{Q}$ architectures predict hubs\nPearson's $\it{r}$ ($\it{Q}$,PC)")
	# what DNN do better when in isolation, in that there is a lot of information in each node?
	# We might think that these DNN do the best for connector hubs, which integrate information across the other nodes
	# so the information is already contained in single nodes, connector hubs
	# if so, we'd expect these DNN to do especially well at predicting connector hubs, we do! 
	vals = full_acc_array.max(axis=3).mean(axis=2)-mean_acc_array.mean(axis=2)
	ax = axes[1][2]
	sns.heatmap(vals,cmap='coolwarm',ax=ax)
	cbar = ax.collections[0].colorbar
	cmin,cmax = vals.min(),vals.max()
	cbar.set_ticks([cmin,cmax])
	cmin,cmax = np.around(vals.min(),2),np.around(vals.max(),2)
	cbar.set_ticklabels([cmin,cmax])
	cbar.set_label('single node accuracy\n- combined accuracy',rotation=270,labelpad=-8)
	ax.set_yticklabels(layers_array,fontsize=8)
	ax.set_xticklabels(neurons_array,fontsize=8)
	ax.set_ylabel('layers')
	ax.set_xlabel('neurons')

	ax = axes[1][3]
	x,y =  (full_acc_array.max(axis=3).mean(axis=2)-mean_acc_array.mean(axis=2)).flatten(),pc_acc_array.mean(axis=2).flatten()
	sns.regplot(x,y,color=sns.color_palette('coolwarm',20)[2],ax=ax)
	r_val,p_val = nan_pearsonr(x,y)
	ax.text(.85,.1,'$\it{r}$=%s\n%s,df=98' %(np.around(r_val,2),log_p_value(p_val)),{'fontsize':6},horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	ax.set_xlim(x.min(),x.max())
	ax.set_ylim(y.min(),y.max())
	ax.set_ylabel("Architecture models hubs\nPearson's $\it{r}$ (PC,accuracy)")
	ax.set_xlabel("Architecture's single node accuracy\n- combined accuracy")
	axes[0,0].set_title('a',loc='left',weight="bold")
	axes[0,1].set_title('b',loc='left',weight="bold")
	axes[0,2].set_title('c',loc='left',weight="bold")
	axes[0,3].set_title('d',loc='left',weight="bold")
	axes[1,0].set_title('e',loc='left',weight="bold")
	axes[1,1].set_title('f',loc='left',weight="bold")
	axes[1,2].set_title('g',loc='left',weight="bold")
	axes[1,3].set_title('h',loc='left',weight="bold")
	plt.tight_layout()
	plt.show()
	# plt.savefig('/%s/figures/archs.pdf'%(homedir))
	# plt.close()
	# plt.savefig('/%s/figures/archs.png'%(homedir),dpi=900)


	# t_full_acc_array = full_acc_array.reshape(90,52,400).mean(axis=0)
	# n_full_acc_array = np.zeros((52))
	# for t in range(52):
	# 	# n_full_acc_array[t] = scipy.stats.entropy(t_full_acc_array[t]+abs(np.min(t_full_acc_array[t])))
	# 	n_full_acc_array[t] = len(t_full_acc_array[t][t_full_acc_array[t]>0])
		
	# t_full_acc_array=scipy.stats.zscore(full_acc_array.reshape(90,52,400).mean(axis=0),axis=1)
	# sns.regplot(pc_acc_array.mean(axis=0).mean(axis=0),t_full_acc_array.max(axis=1))
	# plt.xlabel('predictive nodes are connector hubs')
	# plt.ylabel('predictive nodes are outliers')
	# plt.savefig('pc_outlier_%s.pdf'%(task))

	# sns.regplot(pc_acc_array.mean(axis=0).mean(axis=0),n_full_acc_array)
	# plt.xlabel('predictive nodes are connector hubs')
	# plt.ylabel('task is predicted well by many nodes')
	# plt.savefig('pc_lots_of_nodes_%s.pdf'%(task))

def figure3(task='all_p',ctype='fc',matrix='all',components='None'):
	global pc
	global neurons_array
	global layers_array
	global pc_mod_array
	global pc_acc_array
	global mod_acc_array
	global med_array
	sns.set(context="paper",font_scale=1,font='Palatino')
	df = behavior()
	subjects = df.Subject.values
	names = ['Visual','Sensory\nMotor','Dorsal\nAttention','Ventral\nAttention','Control','Default']	
	matrices = load_matrices(subjects,matrix,components)
	mean_matrix = np.nanmean(matrices,axis=0)
	if task == 'ptasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = ptasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'stasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = stasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'all':
		stasks,ptasks,rtasks = load_tasks()
		tasks = np.zeros((len(stasks) + len(ptasks) + len(rtasks))).astype('S64')
		tasks[:len(stasks)] = stasks
		tasks[len(stasks):len(stasks)+len(ptasks)] = ptasks
		tasks[len(ptasks) + len(stasks):] = rtasks
		# tasks = tasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'all_p':
		stasks,ptasks,rtasks = load_tasks()
		# tasks = ptasks
		tasks = np.zeros((len(stasks) + len(ptasks))).astype('S64')
		tasks[:len(stasks)] = stasks
		tasks[len(stasks):] = ptasks
		# tasks = tasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task == 'rtasks':
		stasks,ptasks,rtasks = load_tasks()
		tasks = rtasks
		b_array = df.drop('Subject',axis=1)
		b_array = b_array.fillna(df.mean())
		for c in b_array.columns.values:
			if c not in tasks: 
				b_array = b_array.drop(c,axis=1)
	if task not in ['ptasks','stasks','rtasks','all','all_p']:
		b_array = df[task].values
		b_array[np.isnan(b_array)] = np.nanmean(b_array)
		tasks = [task]

	layers_array = np.array([2,3,4,5,6,7,8,9,10])
	neurons_array = np.array([10,15,25,50,75,100,150,200,300,400])

	loop_n = 400
	try:
		if int(components) == 7 or int(components) == 17: 
			loop_n = int(components)
	except: pass
	pc = np.load('/%s/results/pc.npy'%(homedir))


	pc_mod_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],))
	mod_acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
	pc_acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
	acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
	full_acc_array = np.zeros((layers_array.shape[0],neurons_array.shape[0],loop_n,b_array.shape[1]))
	mod_array = np.zeros((layers_array.shape[0],neurons_array.shape[0]))
	
	# pc_mod_array[:] = np.nan
	# mod_acc_array[:] = np.nan
	# pc_acc_array[:] = np.nan
	# med_array[:] = np.nan
	# acc_array[:] = np.nan
	# mod_array[:] = np.nan


	for lidx,layers in enumerate(layers_array):
		for nidx,neurons in enumerate(neurons_array):
			print layers,neurons
			if layers == 1: continue
			prediction_accs = np.zeros((loop_n,b_array.shape[1]))
			prediction = np.zeros((loop_n,b_array.shape[0],b_array.shape[1]))
			mods = np.zeros((loop_n,b_array.shape[1]))
			for task_idx,this_task in enumerate(tasks):
				prediction_accs[:,task_idx] = np.load('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,this_task,matrix,components))
				prediction[:,:,task_idx] = np.load('%s//results/%s_%s_%s_%s_%s_%s_prediction.npy'%(homedir,ctype,neurons,layers,this_task,matrix,components))
				mods[:,task_idx] = np.load('%s//results/%s_%s_%s_%s_%s_%s_network_structure.npy'%(homedir,ctype,neurons,layers,this_task,matrix,components))
			mean_prediction_acc = np.zeros(b_array.shape[1])
			for task_idx,this_task in enumerate(tasks):
				mean_prediction_acc[task_idx] = nan_pearsonr(np.nanmean(prediction[:,:,task_idx],axis=0),b_array[this_task])[0]
			#layer correlates PC with acc
			pc_acc_array[lidx,nidx] = nan_pearsonr(pc,np.nanmean(prediction_accs,axis=1))[0]
			# layer correlates Q with acc
			pc_mod_array[lidx,nidx] = pearsonr(pc,np.nanmean(mods,axis=1))[0]
			# layer correlates Q with acc
			mod_acc_array[lidx,nidx] = pearsonr(np.nanmean(prediction_accs,axis=1),np.nanmean(mods,axis=1))[0]
			acc_array[lidx,nidx] = mean_prediction_acc.mean()
			full_acc_array[lidx,nidx] = prediction_accs
			mod_array[lidx,nidx] = np.mean(mods)

	# pc_acc_array = []
	# for i in range(52):pc_acc_array.append(np.load('/%s/results/hub_corr.npy'%(homedir))[1:,:])
	# pc_acc_array = np.array(pc_acc_array).swapaxes(0,1).swapaxes(1,2)
	f, axes = plt.subplots(2, 4,figsize=(14.333,6),tight_layout=True)
	#architecture Q
	vals = mod_array/10000.
	ax = axes[0][0]
	sns.heatmap(vals,cmap='coolwarm',vmax=.25,vmin=0.05,ax=ax,cbar_kws={'label': 'DNN Q'})
	cbar = ax.collections[0].colorbar
	cbar.set_ticks([0.05,0.25])
	cbar.set_ticklabels([0.05,.25])
	cbar.ax.locator_params(nbins=2)
	cbar.set_label('architecture $\it{Q}$',rotation=270,labelpad=-10)
	ax.set_yticklabels(layers_array,fontsize=8)
	ax.set_xticklabels(neurons_array,fontsize=8)
	ax.set_ylabel('layers')
	ax.set_xlabel('neurons')
	#architecture accuracy
	vals = full_acc_array.mean(axis=-1).mean(axis=-1)
	ax = axes[0][1]
	sns.heatmap(vals,cmap='coolwarm',ax=ax,cbar_kws={'label': 'DNN Performance'})
	cbar = ax.collections[0].colorbar
	cmin,cmax = vals.min(),vals.max()
	cbar.set_ticks([cmin,cmax])
	cmin,cmax = np.around(vals.min(),3),np.around(vals.max(),3)
	cbar.set_ticklabels([cmin,cmax])
	cbar.set_label('architecture accuracy',rotation=270,labelpad=-10)
	ax.set_yticklabels(layers_array,fontsize=8)
	ax.set_xticklabels(neurons_array,fontsize=8)
	ax.set_ylabel('layers')
	ax.set_xlabel('neurons')
	#architecture,pc-accuracy
	ax = axes[0][2]
	vals = pc_acc_array
	sns.heatmap(vals,cmap='coolwarm',ax=ax)
	cbar = ax.collections[0].colorbar
	cmin,cmax = vals.min(),vals.max()
	cbar.set_ticks([cmin,cmax])
	cmin,cmax = np.around(vals.min(),2),np.around(vals.max(),2)
	cbar.set_ticklabels([cmin,cmax])
	cbar.set_label("Pearson's $\it{r}$ (PC,accuracy",rotation=270,labelpad=-8)
	ax.set_yticklabels(layers_array,fontsize=8)
	ax.set_xticklabels(neurons_array,fontsize=8)
	ax.set_ylabel('layers')
	ax.set_xlabel('neurons')
	#q,acc,pc polyfit
	ax = axes[0][3]
	plt.sca(ax)
	x,y=full_acc_array.mean(axis=-1).mean(axis=-1).flatten(),mod_array.flatten()/10000
	# for i in range(2,10):
	# 	polynomial_features= PolynomialFeatures(degree=i)
	# 	xp = polynomial_features.fit_transform(x.reshape(-1,1))
	# 	model = sm.OLS(y, xp).fit()
	# 	print i,model.rsquared,pearsonr(model.resid,pc_acc_array.mean(axis=2).flatten())
	polynomial_features= PolynomialFeatures(degree=4)
	xp = polynomial_features.fit_transform(x.reshape(-1,1))
	model = sm.OLS(y, xp).fit()
	# print i,model.rsquared,pearsonr(model.resid,pc_acc_array.mean(axis=2).flatten())
	ypred = model.predict(xp)
	scat = plt.scatter(x,y,c=pc_acc_array.flatten(),cmap='coolwarm',alpha=0.75)
	plt.colorbar(scat,ax=ax)
	# ax.set_xlim(0.0525,0.061)
	cbar = ax.collections[0].colorbar
	cmin,cmax=pc_acc_array.flatten().min(),pc_acc_array.flatten().max()
	cbar.set_ticks([cmin,cmax])
	cbar.set_ticklabels([np.around(cmin,2),np.around(cmax,2)])
	cbar.set_label("Pearson's $\it{r}$ (PC,accuracy)",rotation=270,labelpad=-10)
	ax.set_ylabel('architecture $\it{Q}$')
	ax.set_xlabel('architecture accuracy')
	sns.lineplot(x,ypred)
	ax.set_xlim(0.0525,0.061)
	#residuals from q,acc,pc polyfit
	ax = axes[1][0]
	x,y = model.resid,pc_acc_array.flatten()
	sns.regplot(x,y,ax=ax,color=sns.color_palette('coolwarm',20)[-2])
	r_val,p_val = nan_pearsonr(x,y)
	ax.text(.85,.1,'$\it{r}$=%s\n%s,df=98' %(np.around(r_val,2),log_p_value(p_val)),{'fontsize':6},horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	ax.set_xlim(x.min(),x.max())
	ax.set_ylim(y.min(),y.max())
	ax.set_xlabel("Residuals of $\\bf{d}$")
	ax.set_ylabel("architecture models hubs\nPearson's $\it{r}$ (PC,accuracy)")
	#if a task is captured by high q arch, high q arch captures hubs
	ax = axes[1][1]
	x,y= mod_acc_array.flatten(),pc_mod_array.flatten()
	sns.regplot(x,y,ax=ax,color=sns.color_palette('coolwarm',20)[-2])
	r_val,p_val = nan_pearsonr(x,y)
	ax.text(.85,.1,'$\it{r}$=%s\n%s,df=50' %(np.around(r_val,2),log_p_value(p_val)),{'fontsize':6},horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	ax.set_xlim(x.min(),x.max())
	ax.set_ylim(y.min(),y.max())
	ax.set_xlabel("High $\it{Q}$ architectures predict measure\nPearson's $\it{r}$ ($\it{Q}$,accuracy)")
	ax.set_ylabel("High $\it{Q}$ architectures predict hubs\nPearson's $\it{r}$ ($\it{Q}$,PC)")
	# what DNN do better when in isolation, in that there is a lot of information in each node?
	# We might think that these DNN do the best for connector hubs, which integrate information across the other nodes
	# so the information is already contained in single nodes, connector hubs
	# if so, we'd expect these DNN to do especially well at predicting connector hubs, we do! 
	vals = full_acc_array.max(axis=2).mean(axis=-1)-acc_array
	ax = axes[1][2]
	sns.heatmap(vals,cmap='coolwarm',ax=ax)
	cbar = ax.collections[0].colorbar
	cmin,cmax = vals.min(),vals.max()
	cbar.set_ticks([cmin,cmax])
	cmin,cmax = np.around(vals.min(),2),np.around(vals.max(),2)
	cbar.set_ticklabels([cmin,cmax])
	cbar.set_label('single node accuracy\n- combined accuracy',rotation=270,labelpad=-4)
	ax.set_yticklabels(layers_array,fontsize=8)
	ax.set_xticklabels(neurons_array,fontsize=8)
	ax.set_ylabel('layers')
	ax.set_xlabel('neurons')

	ax = axes[1][3]
	x,y =  (full_acc_array.max(axis=2).mean(axis=-1)-acc_array).flatten(),pc_acc_array.flatten()
	sns.regplot(x,y,color=sns.color_palette('coolwarm',20)[2],ax=ax)
	r_val,p_val = nan_pearsonr(x,y)
	ax.text(.85,.1,'$\it{r}$=%s\n%s,df=98' %(np.around(r_val,2),log_p_value(p_val)),{'fontsize':6},horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	ax.set_xlim(x.min(),x.max())
	ax.set_ylim(y.min(),y.max())
	ax.set_ylabel("Architecture models hubs\nPearson's $\it{r}$ (PC,accuracy)")
	ax.set_xlabel("Architecture's single node accuracy\n- combined accuracy")
	axes[0,0].set_title('a',loc='left',weight="bold")
	axes[0,1].set_title('b',loc='left',weight="bold")
	axes[0,2].set_title('c',loc='left',weight="bold")
	axes[0,3].set_title('d',loc='left',weight="bold")
	axes[1,0].set_title('e',loc='left',weight="bold")
	axes[1,1].set_title('f',loc='left',weight="bold")
	axes[1,2].set_title('g',loc='left',weight="bold")
	axes[1,3].set_title('h',loc='left',weight="bold")
	plt.tight_layout()
	# plt.show()
	plt.savefig('/%s/figures/archs.pdf'%(homedir))
	# plt.close()
	# plt.savefig('/%s/figures/archs.png'%(homedir),dpi=900)


	# t_full_acc_array = full_acc_array.reshape(90,52,400).mean(axis=0)
	# n_full_acc_array = np.zeros((52))
	# for t in range(52):
	# 	# n_full_acc_array[t] = scipy.stats.entropy(t_full_acc_array[t]+abs(np.min(t_full_acc_array[t])))
	# 	n_full_acc_array[t] = len(t_full_acc_array[t][t_full_acc_array[t]>0])
		
	# t_full_acc_array=scipy.stats.zscore(full_acc_array.reshape(90,52,400).mean(axis=0),axis=1)
	# sns.regplot(pc_acc_array.mean(axis=0).mean(axis=0),t_full_acc_array.max(axis=1))
	# plt.xlabel('predictive nodes are connector hubs')
	# plt.ylabel('predictive nodes are outliers')
	# plt.savefig('pc_outlier_%s.pdf'%(task))

	# sns.regplot(pc_acc_array.mean(axis=0).mean(axis=0),n_full_acc_array)
	# plt.xlabel('predictive nodes are connector hubs')
	# plt.ylabel('task is predicted well by many nodes')
	# plt.savefig('pc_lots_of_nodes_%s.pdf'%(task))


def figure4():
	task = 'all_p'
	df = pd.read_csv('/%s/results/%s_%s_%s_df.csv'%(homedir,task,matrix,'None'))
	df['dtype'] = 'node'
	tdf = pd.read_csv('/%s/results/%s_%s_%s_df.csv'%(homedir,task,matrix,'17'))
	tdf['dtype'] = 'system'
	df = df.append(tdf,ignore_index=True)

	cdf = pd.read_csv('/%s/results/%s_%s_%s_cdf.csv'%(homedir,task,matrix,'None'))
	cdf['dtype'] = 'node'
	ctdf = pd.read_csv('/%s/results/%s_%s_%s_cdf.csv'%(homedir,task,matrix,'17'))
	ctdf['dtype'] = 'system'
	cdf = cdf.append(ctdf,ignore_index=True)


	nn_color=sns.diverging_palette(220, 10, sep=80, n=7)[-2]
	nn_max=sns.diverging_palette(220, 10, sep=80, n=7)[6]
	nn_colors=sns.diverging_palette(220, 10, sep=80, n=200)[100:]
	linear_color=sns.diverging_palette(220, 10, sep=80, n=10)[0]
	linear_color2=sns.diverging_palette(220, 10, sep=80, n=10)[2]
	c_nn_17 = sns.color_palette('Reds')[-1]

	layers_array = np.array([1,2,3,4,5,6,7,8,9,10,])
	neurons_array = np.array([10,15,25,50,75,100,150,200,300,400,])

	top_tasks = df.groupby('task',sort=True)['prediction accuracy'].mean()
	
	task_labels = {'Strength_AgeAdj':'Strength', 'WM_Task_2bk_Acc':'Working Memory\n(2-back)', 'ReadEng_AgeAdj':'Reading','PicVocab_AgeAdj':'Vocabulary',\
	'PMAT24_A_CR':'Penn Matrix\nTest', 'VSPLOT_TC':'Penn Line\nOrientation', 'DDisc_AUC_40K':'Delayed\nDiscounting(40k)',\
	'Language_Task_Story_Avg_Difficulty_Level':'Reading\nComprehension', 'Endurance_AgeAdj':'Cardiovascular\nEndurance','NEOFAC_O':'Personality\nOpenness', \
	'Relational_Task_Rel_Acc':'Relational\nReasoning', 'DDisc_AUC_200':'Delayed\nDiscounting(200)','ListSort_AgeAdj':'Working Memory\n(List Sorting)', \
	'ProcSpeed_AgeAdj':'Processing\nSpeed', 'Flanker_AgeAdj':'Flanker\nTask','AngAggr_Unadj':'Agressive Anger', 'IWRD_TOT':'Penn Word Memory', 'CardSort_AgeAdj':'Card Sorting',\
	'NEOFAC_E':'Extraversion','SelfEff_Unadj':'Self-Efficacy', 'PicSeq_AgeAdj':'Picture Sequence Memory', 'Social_Task_TOM_Perc_TOM':'Theory of Mind (1)',\
	'NEOFAC_A':'Agreeableness', 'MeanPurp_Unadj':'Meaning and Purpose','Language_Task_Math_Avg_Difficulty_Level':'Math', 'InstruSupp_Unadj':'Social Support',
	'Friendship_Unadj':'Friendship', 'Sadness_Unadj':'Sadness', 'LifeSatisf_Unadj':'Life Satisfaction',
	'NEOFAC_C':'Conscientiousness', 'GaitSpeed_Comp':'Gait Speed', 'Dexterity_AgeAdj':'Dexterity', 'Noise_Comp':'Auditory Sensativity',
	'PSQI_Score':'Sleep Quality', 'Emotion_Task_Face_Acc':'Emotional Face Recognition', 'Mars_Final':'Mars Contrast Sensitivity',
	'PercStress_Unadj':'Perceived Stress', 'Loneliness_Unadj':'Loneliness', 'Taste_AgeAdj':'Taste Intensity',
	'PosAffect_Unadj':'Positive Affect', 'EmotSupp_Unadj':'Emotional Support', 'NEOFAC_N':'Neuroticism', 'AngAffect_Unadj':'Emotional Anger',
	'Social_Task_Random_Perc_Random':'Theory of Mind (2)', 'MMSE_Score':'Mini Mental Status Exam', 'PercHostil_Unadj':'Perceived Hostility',
	'ER40_CR':'Emotion Recognition', 'SCPT_TP':'Austained attention', 'Odor_AgeAdj':'Odor Identification', 'PercReject_Unadj':'Perceived Rejection',
	'AngHostil_Unadj':'Hostility', 'PainInterf_Tscore':'Problematic Pain'}

	csv_df = pd.DataFrame(columns=['HCP Name','Short Name'])
	csv_df['HCP Name'] = task_labels.keys()
	names = task_labels.values()
	for i,n in enumerate(names):
		names[i] = n.replace('\n',' ')

	csv_df['Short Name'] = names

	# text = []
	# freq = []

	# for t in top_tasks.keys():
	# 	tname = task_labels[t].replace('\n',' ')
	# 	text.append(tname)
	# 	freq.append(int(np.around(top_tasks[t],2)*100))

	# freq = np.array(freq) 
	# freq[freq<6] = 6



	# import wordcloud as wc
	# fpath = '/cbica/home/bertolem/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/Palatino.ttf'
	# wordcloud = wc.WordCloud(font_path=fpath,background_color="white",colormap=sns.cubehelix_palette(52,as_cmap=True),width=8500,height=5500).generate_from_frequencies(dict(zip(text,freq)))
	# plt.figure(figsize=(8.50,5.50),dpi=3000)
	# plt.imshow(wordcloud, interpolation="bilinear")
	# plt.axis("off")
	# plt.tight_layout()
	# # plt.show()
	# plt.savefig('/%s/figures/names.png'%(homedir),dpi=900)
	# plt.close()

	top_tasks = top_tasks.sort_values(ascending=False)
	top_tasks = np.array(top_tasks.keys()[:15]).astype(str)

	df[df['task'].isin(top_tasks)]

	top_df = df[df['task'].isin(top_tasks)]
	top_cdf = cdf[cdf['task'].isin(top_tasks)]


	sns.set(context='paper',font='Palatino',style='white',palette="pastel",color_codes=True)
	fig = plt.figure(figsize=(8,11))
	gs = GridSpec(13, 2, left=0.05, right=.95,top=.95,bottom=0.05,wspace=0.01,hspace=.2)	
		


	
	with sns.axes_style("whitegrid",{'font.family':'serif','context':"paper"}):

		i = df[(df.dtype=='system')&(df.model=='linear')]['prediction accuracy'].values.flatten()
		j = df[(df.dtype=='node')&(df.model=='linear')]['prediction accuracy'].values.flatten()
		x = df[(df.dtype=='system')&(df.model=='neural network')]['prediction accuracy'].values.flatten()
		y = df[(df.dtype=='node')&(df.model=='neural network')]['prediction accuracy'].values.flatten()

		rs =[i,j,x,y]
		ax1 = fig.add_subplot(gs[:4,0])
		ax1.set_title('a',loc='left',weight="bold")
		plt.sca(ax1)
		plt.ylim(-.2,.7)
		sns.boxenplot(data=[[i],[j],[x],[y]],palette=[linear_color2,linear_color,nn_color,nn_max],width=1)
		x1, x2 = 0, 1
		y1, h, col = .5, 0.02, 'k'
		plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=0.75, c=col)
		t,p = scipy.stats.ttest_ind(i,j)
		t,p = scipy.stats.ttest_ind(rs[0],rs[1])
		plt.text(.5, y1-h,'$\it{t}$=%s\n%s,df=830' %(np.around(t,3),log_p_value(p)),{'fontsize':6},ha='center', va='top')

		x1, x2 = 2, 3
		y1, h, col = .5, 0.02, 'k'
		plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=0.75, c=col)
		t,p = scipy.stats.ttest_ind(x,y)
		t,p = scipy.stats.ttest_ind(rs[2],rs[3])
		plt.text(2.5, y1-h,'$\it{t}$=%s\n%s,df=2079998' %(np.around(t,3),log_p_value(p)),{'fontsize':6},ha='center', va='top')

		x1, x2 = 0, 3
		y1, h, col = .57, 0.01, 'k'
		plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=0.75, c=col)
		t,p = scipy.stats.ttest_ind(i,y)
		t,p = scipy.stats.ttest_ind(rs[0],rs[3])
		plt.text(1.5, y1-h,'$\it{t}$=%s,%s,df=830' %(np.around(t,3),log_p_value(p)),{'fontsize':6},ha='center', va='top')

		x1, x2 = 0, 2
		y1, h, col = .3, 0.01, 'k'
		plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=0.75, c=col)
		t,p = scipy.stats.ttest_ind(i,x)
		t,p = scipy.stats.ttest_ind(rs[0],rs[2])
		plt.text(1, y1+h,'$\it{t}$=%s,%s,df=830' %(np.around(t,3),log_p_value(p)),{'fontsize':6},ha='center', va='bottom')

		x1, x2 = 1, 3
		y1, h, col = .35, 0.01, 'k'
		plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=0.75, c=col)
		t,p = scipy.stats.ttest_ind(j,y)
		t,p = scipy.stats.ttest_ind(rs[1],rs[3])
		plt.text(1, y1+h,'$\it{t}$=%s,-log10($\\it{p}$)>200,df=20798' %(np.around(t,3)),{'fontsize':6},ha='center', va='bottom')

		ax1.set_xticklabels(['linear (17)','linear (400)','neural (17)', 'neural (400)'])
		ax1.set_ylabel('nodal accuracy',labelpad=-36)

		i = cdf[(cdf.dtype=='system')&(cdf.model=='linear')].groupby('task').accuracy.mean().values
		j = cdf[(cdf.dtype=='node')&(cdf.model=='linear')].groupby('task').accuracy.mean().values
		x = cdf[(cdf.dtype=='system')&(cdf.model=='neural network')].groupby('task').accuracy.mean().values
		y = cdf[(cdf.dtype=='node')&(cdf.model=='neural network')].groupby('task').accuracy.mean().values

		ax1 = fig.add_subplot(gs[:4,1])
		ax1.set_title('b',loc='left',weight="bold")
		plt.sca(ax1)
		plt.ylim(-.2,.7)
		sns.boxenplot(data=[[i],[j],[x],[y]],palette=[linear_color2,linear_color,nn_color,nn_max],width=1)
		x1, x2 = 0, 3
		y1, h, col = .645, 0.02, 'k'
		plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=0.75, c=col)

		x1, x2 = 1, 3
		y1, h, col = .635, 0.02, 'k'
		plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=0.75, c=col)

		x1, x2 = 2, 3
		y1, h, col = .625, 0.02, 'k'
		plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=0.75, c=col)

		t,p = scipy.stats.ttest_rel(i,y)
		plt.text(.0, y1-h,'$\it{t}$>5,-log10($\\it{p}$)>5,df=50',{'fontsize':6},ha='center', va='top')
		ax1.set_xticklabels(['linear (17)','linear (400)','neural (17)', 'neural (400)'])
		ax1.set_yticklabels([])
		ax1.set_ylabel('combined accuracy',labelpad=-20)

	sns.set(style='white',font='Palatino')
	nn_400_accs = []
	nn_17_accs = []
	for l in layers_array:
		for n in neurons_array:
			nn_400_accs.append(df[(df.model=='neural network')&(df.layers==l)&(df.neurons==n)&(df.dtype=='node')]['prediction accuracy'])
			nn_17_accs.append(df[(df.model=='neural network')&(df.layers==l)&(df.neurons==n)&(df.dtype=='system')]['prediction accuracy'])

	nn_400_accs = np.array(nn_400_accs)
	nn_17_accs = np.array(nn_17_accs)

	ax = fig.add_subplot(gs[5,0])
	ax.set_title('c',loc='left',pad=42,weight='bold')
	sns.plt.sca(ax)
	max400 = np.nanmax(nn_400_accs,axis=0)
	max17 = np.nanmax(nn_17_accs,axis=0)
	kde1 = sns.kdeplot(max400,color=nn_max,label='neural max (400)',**{'linestyle':'--'})
	ax.tick_params('x',direction='in',pad=0.5,labelsize=8)
	ax.get_legend().remove()
	ax.set_yticks([])
	ax = ax.twinx()
	kde2 = sns.kdeplot(max17,color=nn_color,label='neural max (17)',**{'linestyle':'--'})
	ax.set_yticks([])
	ax.get_legend().remove()
	ax = ax.twinx()
	kde3 = sns.kdeplot(df[(df.model=='linear')&(df.dtype=='system')]['prediction accuracy'].values,color=linear_color,label='linear (17)',**{'linestyle':'--'})
	ax.set_yticks([])
	ax.get_legend().remove()
	ax = ax.twinx()
	kde4 = sns.kdeplot(cdf[(cdf.model=='neural network')&(cdf.dtype=='node')].accuracy.values, color=nn_max, lw=1.5,label='linear (17)')
	ax.set_yticks([])
	ax.get_legend().remove()
	ax = ax.twinx()
	kde5 = sns.kdeplot(cdf[(cdf.model=='linear')&(cdf.dtype=='node')].accuracy.values, color=linear_color, lw=1.5,label='combined linear (400)')
	ax.set_yticks([])
	ax.get_legend().remove()
	ax = ax.twinx()	
	kde6 = sns.kdeplot(cdf[(cdf.model=='linear')&(cdf.dtype=='system')].accuracy.values, color=linear_color2, lw=1.5,label='combined linear (17)')
	ax.set_yticks([])
	ax.get_legend().remove()
	ax = ax.twinx()
	kde7 = sns.kdeplot(np.nanmean(nn_17_accs,axis=0),color=nn_color,label='neural (400)')
	ax.set_yticks([])
	ax.get_legend().remove()
	ax = ax.twinx()
	kde8 = sns.kdeplot(cdf[(cdf.model=='neural network')&(cdf.dtype=='system')].accuracy.values, color=c_nn_17, lw=1.5,label='combined neural (17)',**{'linestyle':':'})
	ax.set_yticks([])
	ax.get_legend().remove()

	ax.text(0.01, .95, 'All Measures',{'fontsize':9}, color='black',ha="left", va="top", transform=ax.transAxes,)
	sns.despine()
	lns = ['neural max (400)','neural max (17)','neural (400)','linear (17)','combined neural (400)','combined neural (17)','combined linear (17)','combined linear (400)']
	ax.legend([kde1.lines[0],kde2.lines[0],kde7.lines[0],kde3.lines[0],kde4.lines[0],kde8.lines[0],kde5.lines[0],kde6.lines[0]],lns,ncol=4,loc=(0,1.25),labelspacing=.1,columnspacing=.1,**{'fontsize':10})
	
	plt_idx = 6
	plt_idx2 = 0
	for idx,task in enumerate(top_tasks):

		ax = fig.add_subplot(gs[plt_idx,plt_idx2])
		sns.plt.sca(ax)

		nn_400_accs = []
		nn_17_accs = []
		for l in layers_array:
			for n in neurons_array:
				nn_400_accs.append(df[(df.model=='neural network')&(df.layers==l)&(df.neurons==n)&(df.dtype=='node')&(df.task==task)]['prediction accuracy'])
				nn_17_accs.append(df[(df.model=='neural network')&(df.layers==l)&(df.neurons==n)&(df.dtype=='system')&(df.task==task)]['prediction accuracy'])

		nn_400_accs = np.array(nn_400_accs)
		nn_17_accs = np.array(nn_17_accs)

		colors = [linear_color2,linear_color,nn_color,nn_max]
		mean400 = np.nanmean(nn_400_accs,axis=0)
		mean17 = np.nanmean(nn_17_accs,axis=0)
		max400 = np.nanmax(nn_400_accs,axis=0)
		max17 = np.nanmax(nn_17_accs,axis=0)
		kde1 = sns.kdeplot(max400,color=nn_max,label='neural max (400)',**{'linestyle':'--'})
		ax.tick_params('x',direction='in',pad=0.5,labelsize=8)
		ax.get_legend().remove()
		ax.set_yticks([])
		ax = ax.twinx()
		kde2 = sns.kdeplot(max17,color=nn_color,label='neural max (17)',**{'linestyle':'--'})
		ax.set_yticks([])
		ax.get_legend().remove()
		ax = ax.twinx()
		kde3 = sns.kdeplot(df[(df.model=='linear')&(df.dtype=='system')&(df.task==task)]['prediction accuracy'].values,color=linear_color,label='linear (17)',**{'linestyle':'--'})
		ax.set_yticks([])
		ax.get_legend().remove()
		ax = ax.twinx()
		kde4 = sns.kdeplot(cdf[(cdf.model=='neural network')&(cdf.dtype=='node')&(cdf.task==task)].accuracy.values, color=nn_max, lw=1.5,label='linear (17)')
		ax.set_yticks([])
		ax.get_legend().remove()
		ax = ax.twinx()
		kde5 = sns.kdeplot(df[(df.model=='neural network')&(df.dtype=='node')&(df.task==task)]['prediction accuracy'].values,color=nn_color,label='neural (400)')
		ax.set_yticks([])
		ax.get_legend().remove()
		kde5 = sns.kdeplot(df[(df.model=='neural network')&(df.dtype=='system')&(df.task==task)]['prediction accuracy'].values,color=c_nn_17,label='neural (17)',**{'linestyle':':'})
		ax.set_yticks([])
		ax.get_legend().remove()
		
		line1 = ax.axvline(cdf[(cdf.model=='linear')&(cdf.dtype=='node')&(cdf.task==task)].accuracy.values, color=linear_color, lw=1.5,label='combined linear (400)')
		line2 = ax.axvline(cdf[(cdf.model=='linear')&(cdf.dtype=='system')&(cdf.task==task)].accuracy.values, color=linear_color2, lw=1.5,label='combined linear (17)')

		ax.text(0.01, .95, task_labels[task],{'fontsize':9}, color='black',ha="left", va="top", transform=ax.transAxes,)
		sns.despine()
		plt_idx = plt_idx + 1
		if plt_idx == 13: 
			plt_idx = 5
			plt_idx2 = 1
	plt.savefig('/%s/figures/figdist_pal.pdf'%(homedir))

# os.system('qsub -l h_vmem=1G,s_vmem=1G -pe threaded 30 -N s%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/ /%s/prediction.py -r structure -neurons %s -layers %s -task %s -matrix %s -components %s' %(5,50,'WM_Task_2bk_Acc',homedir,5,50,'WM_Task_2bk_Acc','all','None'))
# os.system('qsub -l h_vmem=1G,s_vmem=1G -pe threaded 30 -N s%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/ /%s/prediction.py -r structure -neurons %s -layers %s -task %s -matrix %s -components %s' %(10,25,'WM_Task_2bk_Acc',homedir,10,25,'WM_Task_2bk_Acc','all','None'))
# os.system('qsub -l h_vmem=4G,s_vmem=4G -pe threaded 12 -N s%s_%s%s -j y -b y -o /cbica/home/bertolem/sge/ -e /cbica/home/bertolem/sge/ /%s/prediction.py -r structure -neurons %s -layers %s -task %s -matrix %s -components %s' %(10,400,'WM_Task_2bk_Acc',homedir,10,400,'WM_Task_2bk_Acc','all','None'))

neurons = int(neurons)
layers = int(layers)
neurons_array = np.zeros((layers))
neurons_array[:] = neurons
neurons_array = tuple(neurons_array.astype(int))

if runtype == 'run':
	run(task=task,matrix=matrix,components=components)

if runtype == 'structure':
	while True:
		try: 
			from oct2py import octave
			break
		except: pass
	nn_structure(task=task,matrix=matrix,components=None,write_graph=True)

# break
# # m = sns.kdeplot(mean400,color=nn_max,label='neural (400)')
# # m = sns.kdeplot(max400,color=nn_max,label='neural max (400)',**{'linestyle':'--'})
# # m = sns.kdeplot(mean17,color=nn_color,label='neural (17)',**{'linestyle':(0,(1,1))})
# # m = sns.kdeplot(max17,color=nn_color,label='neural max (17)',**{'linestyle':'--'})
# # m = sns.kdeplot(df[(df.model=='linear')&(df.dtype=='system')&(df.task==task)]['prediction accuracy'].values,label='linear (17)',color=linear_color2,**{'linestyle':(0,(1,1))})
# # m = sns.kdeplot(df[(df.model=='linear')&(df.dtype=='node')&(df.task==task)]['prediction accuracy'].values,label='linear (400)',color=linear_color)
# # m.set_yticks([])
# # m = ax.axvline(cdf[(cdf.model=='neural network')&(cdf.dtype=='node')&(cdf.task==task)].accuracy.values[0], color='black', lw=1.5,label='combined neural (400)')
# # m = ax.axvline(cdf[(cdf.model=='linear')&(cdf.dtype=='node')&(cdf.task==task)].accuracy.values[0], color='grey', lw=1.5,label='combined linear (400)')
# # m = ax.axvline(cdf[(cdf.model=='neural network')&(cdf.dtype=='system')&(cdf.task==task)].accuracy.values[0], color='black', lw=1.5,label='combined neural (17)',**{'linestyle':(0,(1,1))})
# # m = ax.axvline(cdf[(cdf.model=='linear')&(cdf.dtype=='system')&(cdf.task==task)].accuracy.values[0], color='grey', lw=1.5,label='combined linear (17)',**{'linestyle':(0,(1,1))})
# ax.text(0, .9, task, {'fontsize':8},color='black',ha="left", va="center", transform=ax.transAxes)
# sns.despine()
# ax.get_legend().remove()
# # my_xticks = ax.get_xticks()
# # ax.set_xticks((my_xticks[1],my_xticks[-2]))
# # ax.tick_params(direction="in", pad=-20)
# graph_array =graph_array[loop_n:,loop_n:]
# graph_array = graph_array + graph_array.transpose()
# # final_array = np.zeros((graph_array.shape))
# # for i,j in combinations(range(graph_array.shape[0]),2):
# # 	if graph_array[i,j] != 0:
# # 		x,y = graph_array[i][graph_array[i]!=0],graph_array[j][graph_array[j]!=0]
# # 		final_array = pearsonr(x,y)[0]
# # graph_array[graph_array!=0] = scipy.stats.zscore(graph_array[graph_array!=0])

# pos_graph_array = graph_array.copy()
# # pos_graph_array[pos_graph_array!=0.0] = pos_graph_array[pos_graph_array!=0.0] + abs(np.nanmin(pos_graph_array))

# # neg_graph_array = graph_array.copy()
# # neg_graph_array = neg_graph_array * -1
# # neg_graph_array[neg_graph_array!=0.0] = neg_graph_array[neg_graph_array!=0.0] + abs(np.nanmin(neg_graph_array))

# pg = brain_graphs.matrix_to_igraph(pos_graph_array,cost=.035,mst=True)
# pos_c = pg.community_fastgreedy(weights='weight').as_clustering()

# # ng = brain_graphs.matrix_to_igraph(neg_graph_array,cost=.05,mst=True)
# # neg_c = ng.community_fastgreedy(weights='weight').as_clustering()
# posmod[node] = pos_c.modularity
# # negmod[node] = neg_c.modularity
# print nan_pearsonr(pc,posmod)
# # print nan_pearsonr(pc,negmod)



# full_model_arrays = np.array(full_model_arrays).astype('float16')
# com = np.array(com)
# np.save('%s//results/%s_%s_%s_%s_%s_%s_network_structure.npy'%(homedir,ctype,neurons,layers,task,matrix,components),mod)
# np.save('%s//results/%s_%s_%s_%s_%s_%s_network_membership.npy'%(homedir,ctype,neurons,layers,task,matrix,components),com)
# np.save('%s//results/%s_%s_%s_%s_%s_%s_network_weights.npy'%(homedir,ctype,neurons,layers,task,matrix,components),full_model_arrays)
# graph_array = graph_array + graph_array.transpose()
# graph_array =graph_array[neurons:,neurons:]
# pos_graph_array = graph_array.copy()
# pos_graph_array[pos_graph_array!=0.0] = pos_graph_array[pos_graph_array!=0.0] + abs(np.nanmin(pos_graph_array))

# neg_graph_array = graph_array.copy()
# neg_graph_array = neg_graph_array * -1
# neg_graph_array[neg_graph_array!=0.0] = neg_graph_array[neg_graph_array!=0.0] + abs(np.nanmin(neg_graph_array))

# pg = brain_graphs.matrix_to_igraph(pos_graph_array,cost=.01,mst=True)
# pos_c = pg.community_infomap(edge_weights='weight')
# pos_c = pg.community_fastgreedy(weights='weight').as_clustering()

# ng = brain_graphs.matrix_to_igraph(neg_graph_array,cost=.01,mst=True)
# c = g.community_infomap(edge_weights='weight')
# neg_c = ng.community_fastgreedy(weights='weight').as_clustering()
# mod[node] = np.sum([neg_c.modularity,pos_c.modularity])
# mod[node] = pos_c.modularity
# # com.append(c.membership)
# print nan_pearsonr(pc,mod)


# node_mod = [] 
# pos_graphs = []
# neg_graphs = []
# for l in range(1,layers):
# 	graph_array = model.coefs_[l].copy()
# 	graph_array = graph_array + graph_array.transpose()
# 	pos_graph_array = graph_array.copy()
# 	pos_graph_array[pos_graph_array!=0.0] = pos_graph_array[pos_graph_array!=0.0] + abs(np.nanmin(pos_graph_array))
# 	pg = brain_graphs.matrix_to_igraph(pos_graph_array,cost=.1,mst=True)
# 	pg.vs['slice'] = l
# 	pg.vs['id'] = range(neurons)
	
# 	neg_graph_array = graph_array.copy()
# 	neg_graph_array = neg_graph_array * -1
# 	neg_graph_array[neg_graph_array!=0.0] = neg_graph_array[neg_graph_array!=0.0] + abs(np.nanmin(neg_graph_array))
# 	ng = brain_graphs.matrix_to_igraph(neg_graph_array,cost=.1,mst=True)
# 	ng.vs['slice'] = l
# 	ng.vs['id'] = range(neurons)
# 	pos_graphs.append(pg)
# 	neg_graphs.append(ng)

# pos_membership, _ = louvain.find_partition_temporal(pos_graphs,louvain.CPMVertexPartition,interslice_weight=2,resolution_parameter=0.1)
# # 1/0
# neg_membership, _ = louvain.find_partition_temporal(neg_graphs,louvain.CPMVertexPartition,interslice_weight=.5,resolution_parameter=0.1)
# nodemod = []
# for l in range(layers-1):
# 	nodemod.append(VertexClustering(pos_graphs[l],membership=pos_membership[l],modularity_params={'weights':'weight'}).recalculate_modularity())
# 	# nodemod.append(VertexClustering(neg_graphs[l],membership=neg_membership[l],modularity_params={'weights':'weight'}).recalculate_modularity())

# mod[node] = np.nanmean(nodemod)
# print nan_pearsonr(mod,pc)
# continue

# pc_models = np.array(models)[np.argsort(pc)[-5:]]
# coefs = []
# for m in pc_models:
# 	coefs.append(m.coefs_)
# model_array = np.nanmean(coefs,axis=0)
# model_array = model_array[0:layers]

# node_mod = [] 
# pos_graphs = []
# neg_graphs = []
# for l in range(1,layers):
# 	graph_array = model.coefs_[l].copy()
# 	graph_array = graph_array + graph_array.transpose()
# 	pos_graph_array = graph_array.copy()
# 	pos_graph_array[pos_graph_array!=0.0] = pos_graph_array[pos_graph_array!=0.0] + abs(np.nanmin(pos_graph_array))
# 	pg = brain_graphs.matrix_to_igraph(pos_graph_array,cost=.1,mst=True)
# 	pg.vs['slice'] = l
# 	pg.vs['id'] = range(neurons)
	
# 	neg_graph_array = graph_array.copy()
# 	neg_graph_array = neg_graph_array * -1
# 	neg_graph_array[neg_graph_array!=0.0] = neg_graph_array[neg_graph_array!=0.0] + abs(np.nanmin(neg_graph_array))
# 	ng = brain_graphs.matrix_to_igraph(neg_graph_array,cost=.1,mst=True)
# 	ng.vs['slice'] = l
# 	ng.vs['id'] = range(neurons)
# 	pos_graphs.append(pg)
# 	neg_graphs.append(ng)

# pos_membership, _ = louvain.find_partition_temporal(pos_graphs,louvain.CPMVertexPartition,interslice_weight=np.mean(pos_graph_array),resolution_parameter=0.1)
# neg_membership, _ = louvain.find_partition_temporal(neg_graphs,louvain.CPMVertexPartition,interslice_weight=np.mean(neg_graph_array),resolution_parameter=0.1)
# nodemod = []
# for l in range(layers-1):
# 	nodemod.append(VertexClustering(pos_graphs[l],membership=pos_membership[l],modularity_params={'weights':'weight'}).recalculate_modularity())
# 	nodemod.append(VertexClustering(neg_graphs[l],membership=neg_membership[l],modularity_params={'weights':'weight'}).recalculate_modularity())
# mod[node] = np.nanmean(nodemod)
# print nan_pearsonr(mod,pc)
# continue

# flat_model_array = []

# global edges
# global b_array
# global matrices

# ptype = 'nn'
# pc = np.load('/%s/results/pc.npy'%(homedir))
# prediction_accs = np.load('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,task,matrix,components))
# df = behavior()
# subjects = df.Subject.values

# matrices = load_matrices(subjects,matrix,components)

# b_array = df[task].values
# b_array[np.isnan(b_array)] = np.nanmean(b_array)

# loop_n = 400
# try:
# 	if int(components) == 7 or int(components) == 17: 
# 		loop_n = int(components)
# 		if loop_n == 17: loop_n = 16
# except: pass 
# model_arrays = []
# if components != 'None':
# 	if components != None:
# 		loop_n = int(components) 

# full_model_arrays = []



# pool = Pool(40)
# models = pool.map(nn_multi,range(loop_n))
# pos_graphs = []
# neg_graphs = []
# for l in range(1,layers):
# 	graph_array = model.coefs_[l].copy()
# 	graph_array = graph_array + graph_array.transpose()
# 	pos_graph_array = graph_array.copy()
# 	pos_graph_array[pos_graph_array!=0.0] = pos_graph_array[pos_graph_array!=0.0] + abs(np.nanmin(pos_graph_array))
# 	pg = brain_graphs.matrix_to_igraph(pos_graph_array,cost=.1,mst=True)

# 	# pg = pg.community_fastgreedy(weights='weight').as_clustering()
# 	# node_mod.append(pg.modularity)
# 	# continue
	
# 	pg.vs['slice'] = l
# 	pg.vs['id'] = range(neurons)
# 	pos_graphs.append(pg)
# 	# neg_graphs.append(ng)

# # posmod[node] = np.nanmean(node_mod)
# # print nan_pearsonr(posmod,pc)
# # print nan_pearsonr(posmod,prediction_accs)

# nodemod = []
# for i in range(5):
# 	if layers == 2: pos_membership = [louvain.find_partition(pos_graphs[0],louvain.CPMVertexPartition,resolution_parameter=0.1).membership]
# 	else: pos_membership, _ = louvain.find_partition_temporal(pos_graphs,louvain.CPMVertexPartition,interslice_weight=pos_graph_array[pos_graph_array>0].mean(),resolution_parameter=0.1,weight_attr='weight')
	
# 	for l in range(layers-1):
# 		nodemod.append(VertexClustering(pos_graphs[l],membership=pos_membership[l],modularity_params={'weights':'weight'}).recalculate_modularity())

# posmod[node] = np.nanmean(nodemod)
# # com[node] = len(np.unique(pos_membership))
# print nan_pearsonr(posmod,pc)
# print nan_pearsonr(posmod,prediction_accs)
# # print nan_pearsonr(com,pc)


# os.system('qstat -s r > qstat.txt')
# jobs_names = pd.read_csv('qstat.txt',skiprows=[0,1],header=None,sep=' ')[3].values
# jobs = pd.read_csv('qstat.txt',skiprows=[0,1],header=None,sep=' ')[1].values
# for j,n in zip(jobs,jobs_names):
# 	print n, os.system('qstat -j %s | grep maxvmem'%(j))



	# df = df[df.system!='Limbic']
	# df['prediction accuracy'][df.model == 'linear'] = scipy.stats.zscore(df['prediction accuracy'][df.model == 'linear'])
	# df['prediction accuracy'][df.model == 'neural network'] = scipy.stats.zscore(df['prediction accuracy'][df.model == 'neural network'])
	# z_stat_d = {}
	# for system in df.system.unique():
	# 	x = df['prediction accuracy'][(df.model == 'neural network')&(df.system == system)]
	# 	y = df['prediction accuracy'][(df.model == 'linear')&(df.system == system)]
	# 	z_stat_d[system] = scipy.stats.ttest_ind(x,y)
	


	# f.text(0,1,'a',{'size':10},weight="bold",verticalalignment='top',horizontalalignment='left')
	# f.text(0.5,1,'b',{'size':10},weight="bold",verticalalignment='top')
	# f.text(0,0.48,'c',{'size':10},weight="bold",horizontalalignment='left')
	# f.text(0.5,0.48,'d',{'size':10},weight="bold")
	# sns.violinplot(data=df,y='prediction accuracy',x='system',hue='model')
	# sns.boxenplot(data=nn_df[nn_df.system!='Limbic'],x='system',y='prediction accuracy',palette=final_colors,ax=axes[0,0])
	# axes[0,1].remove()
	# axes[0,1]=plt.subplot(2,2,2,sharey=axes[0,0])
	# f.sca(axes[0,0])
	# plt.xlabel('')
	# plt.xticks(rotation=90)

	

	# names = ['Visual','Sensory\nMotor','Dorsal\nAttention','Ventral\nAttention','Control','Default']	
	# for idx,system in enumerate(names):
	# 	t_val,p_val = stat_d[system]
	# 	if t_val > 0:
	# 		plt.text(idx,label_min,'$\it{t}$=%s\n%s' %(np.around(t_val,1),log_p_value(p_val)),{'fontsize':6},horizontalalignment='center',verticalalignment='center')

	# sns.boxenplot(data=ols_df[ols_df.system!='Limbic'],x='system',y='prediction accuracy',palette=final_colors,ax=axes[0,1])
	# f.sca(axes[0,1])
	# # plt.xticks([])
	# plt.xlabel('')
	# plt.xticks(rotation=90)

	# sns.boxenplot(data=df[df.system!='Limbic'],x='system',y='prediction accuracy',hue='model',ax=axes[1,0],)
	# f.sca(axes[1,0])
	# plt.xticks([])
	# plt.yticks([])
	# plt.xlabel('')
	# axes[1,0].legend(loc=2,ncol=2,fontsize='small')


	# axes[1,1].remove()
	# axes[1,1]=plt.subplot(2,2,4,projection='3d')
	# ax = axes[1,1]
	# f.sca(axes[1,1])
	# ax = f.gca(projection='3d')

	# x,y	= np.meshgrid(plot_neurons_array,plot_layers_array)
	# ax.plot_wireframe(x, y, r_array, 1, color ='black')
	# ax.set_xticks(np.arange(len(neurons_array)))
	# ax.set_xticklabels(neurons_array,fontdict={'fontsize':'x-small'},rotation=-15)
	# ax.set_yticks(np.arange(len(layers_array)))
	# ax.set_yticklabels(layers_array,fontdict={'fontsize':'x-small'},rotation=15)
	# ax.set_ylabel('layers')
	# ax.set_xlabel('neurons')
	# ax.set_zlabel('accuracy')
	# plotp = np.array(p)
	# plotp[plotp<(plotp.mean()-plotp.std()*1)] = plotp.mean()-plotp.std()*1
	# plotp[plotp>(plotp.mean()+plotp.std()*1)] = plotp.mean()+plotp.std()*1
	# ax.view_init(40,-135)
	# scatter = ax.scatter(n,l, r, c=plotp,cmap='Reds',s=30,depthshade=False,edgecolors="black")
	# # cbar = f.colorbar(scatter, shrink=0.5, aspect=30,pad = 0.1)
	# # plt.tight_layout()
	# f.sca(axes[1,0])
	# names = ['Visual','Sensory\nMotor','Dorsal\nAttention','Ventral\nAttention','Control','Default']	
	# for idx,system in enumerate(names):
	# 	t_val,p_val = z_stat_d[system]
	# 	plt.text(idx,-3.3,'$\it{t}$=%s\n%s' %(np.around(t_val,1),log_p_value(p_val)),{'fontsize':6},horizontalalignment='center',verticalalignment='center')
	# f.subplots_adjust(bottom=.6)
	# plt.savefig('prediction_figure_%s_%s_%s.pdf'%(task,matrix,components),bbox_inches="tight")
	# plt.show()


	# m = sns.kdeplot(df[(df.model=='linear')&(df.dtype=='system')]['prediction accuracy'].values,label='linear (17)',color=linear_color2,**{'linestyle':(0,(1,1))})
	# m = sns.kdeplot(df[(df.model=='linear')&(df.dtype=='node')]['prediction accuracy'].values,label='linear (400)',color=linear_color)
	# m = ax.axvline(cdf[(cdf.model=='linear')&(cdf.dtype=='node')].accuracy.mean(), color='grey', lw=1.5,label='combined linear (400)')
	# m = ax.axvline(cdf[(cdf.model=='neural network')&(cdf.dtype=='system')].accuracy.mean(), color='black', lw=1.5,label='combined neural (17)',**{'linestyle':(0,(1,1))})
	# m = ax.axvline(cdf[(cdf.model=='linear')&(cdf.dtype=='system')].accuracy.mean(), color='grey', lw=1.5,label='combined linear (17)',**{'linestyle':(0,(1,1))})

	# m = ax.axvline(cdf[(cdf.model=='neural network')&(cdf.dtype=='node')].accuracy.mean(), color='black', lw=1.5,label='combined neural (400)')
	# m = ax.axvline(cdf[(cdf.model=='linear')&(cdf.dtype=='node')].accuracy.mean(), color='grey', lw=1.5,label='combined linear (400)')
	# m = ax.axvline(cdf[(cdf.model=='neural network')&(cdf.dtype=='system')].accuracy.mean(), color='black', lw=1.5,label='combined neural (17)',**{'linestyle':(0,(1,1))})
	# m = ax.axvline(cdf[(cdf.model=='linear')&(cdf.dtype=='system')].accuracy.mean(), color='grey', lw=1.5,label='combined linear (17)',**{'linestyle':(0,(1,1))})




# def figure5():
# 	# TODO, matrix, where regions are brains/tasks edges are pearson r between layers/neuron acc, does this recreate behavXbehave or FC?
# 	df = behavior()
# 	subjects = df.Subject.values
# 	if task == 'all_p':
# 		stasks,ptasks,rtasks = load_tasks()
# 		# tasks = ptasks
# 		tasks = np.zeros((len(stasks) + len(ptasks))).astype('S64')
# 		tasks[:len(stasks)] = stasks
# 		tasks[len(stasks):] = ptasks
# 		# tasks = tasks
# 		b_array = df.drop('Subject',axis=1)
# 		b_array = b_array.fillna(df.mean())
# 		for c in b_array.columns.values:
# 			if c not in tasks: 
# 				b_array = b_array.drop(c,axis=1)
# 	if task not in ['ptasks','stasks','rtasks','all','all_p']:
# 		b_array = df[task].values
# 		b_array[np.isnan(b_array)] = np.nanmean(b_array)
# 		tasks = [task]

# 	loop_n = 400
# 	try:
# 		if int(components) == 7 or int(components) == 17: 
# 			loop_n = int(components)
# 	except: pass

# 	print loop_n
# 	if loop_n == 400: pc = np.load('/%s/results/pc.npy'%(homedir))
# 	else:
# 		yeo_labels = yeo_partition(int(components))[1]
# 		if loop_n == 17: loop_n = 16
# 		pc = np.zeros(int(loop_n))
# 		pc_raw = np.load('/%s/results/pc.npy'%(homedir))
# 		for i in range(int(loop_n)):
# 			pc[i] = np.mean(pc_raw[np.where(yeo_labels==i)])


# 	layers_array = np.array([1,2,3,4,5,6,7,8,9,10,])
# 	neurons_array = np.array([10,15,25,50,75,100,150,200,300,400,])

# 	n_prediction_accs = np.zeros((len(layers_array),len(neurons_array),400))
# 	t_prediction_accs = np.zeros((len(layers_array),len(neurons_array),b_array.shape[1]))
# 	for lidx,layers in enumerate(layers_array):
# 		for nidx,neurons in enumerate(neurons_array):
# 			print layers,neurons
# 			prediction_accs = np.zeros((400,b_array.shape[1]))
# 			for task_idx,this_task in enumerate(tasks):
# 				prediction_accs[:,task_idx] = np.load('%s//results/%s_%s_%s_%s_%s_%s.npy'%(homedir,ctype,neurons,layers,this_task,matrix,components))
# 			n_prediction_accs[lidx,nidx,:] = np.nanmean(prediction_accs,axis=1)
# 			t_prediction_accs[lidx,nidx,:] = np.nanmean(prediction_accs,axis=0)

# 	n_nn_matrix = np.corrcoef(n_prediction_accs.reshape(100,400).swapaxes(0,1))
# 	pearsonr(n_nn_matrix[np.triu_indices(400,1)],mean_matrix[np.triu_indices(400,1)])

# 	t_nn_matrix = np.corrcoef(t_prediction_accs.reshape(100,52).swapaxes(0,1))
# 	pearsonr(t_nn_matrix[np.triu_indices(52,1)],np.array(b_array.corr())[np.triu_indices(52,1)])

# 	m = n_nn_matrix
# 	m = m + m.transpose()
# 	m = np.tril(m,-1)
# 	m = m + m.transpose()

# 	g = brain_graphs.matrix_to_igraph(m,cost=.15,mst=True)
# 	g = brain_graphs.brain_graph(VertexClustering(g,yeo_partition(17)[1],params={'weight':'weight'}))
# def multi_find1(i):
# 	global c				
# 	grouped = np.arange(c.shape[1])
# 	np.random.RandomState(i).shuffle(grouped)
# 	sorted_c = np.zeros((c.shape))
# 	for l in range(c.shape[0]): sorted_c[l,grouped.astype(int)] = c[l,:]
# 	sorted_c = sorted_c.astype(int)
# 	thisd = 0
# 	for l in range(c.shape[0]):
# 		bd = np.diff([sorted_c[l]]) == 0
# 		thisd = thisd +(bd[bd==True]).shape[0]
# 	# for n in range(c.shape[1]):
# 	# 	bd = np.diff(sorted_c[:,n]) == 0
# 	# 	thisd = thisd +(bd[bd==True]).shape[0]
# 	return thisd

# def multi_find2(d):
# 	global c
# 	global grouped
# 	temp,rseed = d
# 	gtemp = grouped.copy()
# 	# get random points
# 	swap = np.random.RandomState(rseed).choice(gtemp,temp,replace=False).astype(int)
# 	# select them in the full array
# 	sg = gtemp[swap]
# 	# shuffle them
# 	np.random.RandomState(rseed).shuffle(sg)
# 	# put them back
# 	gtemp[swap] = sg

# 	sorted_c = np.zeros((c.shape))
# 	for l in range(c.shape[0]): sorted_c[l,gtemp.astype(int)] = c[l,:]
# 	sorted_c = sorted_c.astype(int)
# 	mthisd = 0
# 	for l in range(c.shape[0]):
# 		bd = np.diff([sorted_c[l]]) == 0
# 		mthisd = mthisd +(bd[bd==True]).shape[0]
# 	for n in range(c.shape[1]):
# 		bd = np.diff(sorted_c[:,n]) == 0
# 		mthisd = mthisd +(bd[bd==True]).shape[0]
# 	return mthisd,gtemp




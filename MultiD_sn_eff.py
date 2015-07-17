import matplotlib
matplotlib.use('Agg')
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.optimize import fsolve

def Mid_Bins(arr):
	new_array=[]
	for i in range(len(arr)-1):
		new_array.append((arr[i]+arr[i+1])/2.0)

	return new_array
def query_db():
	conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
	cur = conn.cursor()
	cur.execute("SELECT peak_date, ra, dec, ab_magb, redshift, x1, color, int_dis, found from sn_mc where ra>310;")
	print 'Database Query Complete'
	m=cur.fetchall()
	cur.close()
	m=np.array(m, dtype=np.float)
	print len(m[:,0])
	return m



def Bin_Me(arr, final_bins):
	in_arrb, in_arr_bin=np.histogram(arr, bins=final_bins)
	in_arrb, in_arr_bin=np.histogram(arr, bins=in_arr_bin[in_arrb>0])

	return np.array(in_arr_bin)
'''
                              Table "public.sn_mc"
    
0 peak_date 
1 ra        
2 dec       
3 ab_magb   
4 redshift  
5 x1        
6 color     
7 int_dis   
8 found     
'''

'''Lets make a multid grid for peakdate, redshift, x1'''
def make_grid(m):
	first_grid=[m[:,0],m[:,4],m[:,5]]

	good_grid=[m[:,0][m[:,8]==True],m[:,4][m[:,8]==True],m[:,5][m[:,8]==True]]

	pdbins=np.linspace(min(m[:,0]), max(m[:,0]), 50)
	redbins=np.linspace(min(m[:,4]), max(m[:,4]), 50)
	x1bins=np.linspace(min(m[:,5]), max(m[:,5]), 50)

	H1,edges1 = np.histogramdd(first_grid, bins=(pdbins,redbins,x1bins,))
	#print edges1
	H2, edges2 = np.histogramdd(good_grid, bins=(pdbins, redbins, x1bins,))

	eff_grid=np.divide(H2.astype(float), H1.astype(float))
	#print eff_grid
	np.save('Supernova_Efficiency_Grid', eff_grid)
	np.save('Bin_Edges', edges2)
	print 'Bin Edges Saved'
	print 'Eff Grid Saved'
	

def Load_Grid(gridname, binname):
	g=np.load(gridname)
	ebin=np.load(binname)
	return g, ebin



def Reg_Grid(uneGrid, ebin, order):
	#uG=np.load(str(uneGrid)+'.npy')
	uG=np.nan_to_num(uneGrid)
	zoom_grid=ndimage.interpolation.zoom(uG, (float(max(uG.shape))/uG.shape[0], float(max(uG.shape))/uG.shape[1],float(max(uG.shape))/uG.shape[2]),order=order, mode='nearest')
	peak_Zoom=ndimage.interpolation.zoom(ebin[0], (float(max(uG.shape))/ebin[0].shape[0]),order=order, mode='nearest')
	red_Zoom=ndimage.interpolation.zoom(ebin[1], (float(max(uG.shape))/ebin[1].shape[0]),order=order, mode='nearest')
	x1_Zoom=ndimage.interpolation.zoom(ebin[2], (float(max(uG.shape))/ebin[2].shape[0]),order=order, mode='nearest')
	
	print zoom_grid.shape, peak_Zoom.shape, red_Zoom.shape, x1_Zoom.shape
	return zoom_grid, peak_Zoom, red_Zoom, x1_Zoom

m=query_db()
make_grid(m)
'''
eff_grid, ebin=Load_Grid('Supernova_Efficiency_Grid.npy', 'Bin_Edges.npy')
zoom_grid, peak_Zoom, red_Zoom, x1_Zoom=Reg_Grid(eff_grid, ebin, 1)
def Remap_Ranges(care, rang):
	return (care-min(rang))*(len(rang))/((max(rang)-min(rang)))

def Interp_On_Zoom(peak, red, x1, zoom_grid=zoom_grid, peak_Zoom=peak_Zoom, red_Zoom=red_Zoom, x1_Zoom=x1_Zoom):
	effs=ndimage.map_coordinates(zoom_grid,[[Remap_Ranges(peak, peak_Zoom)],[Remap_Ranges(red, red_Zoom)],[Remap_Ranges(x1, x1_Zoom)]],order=1)
	return effs
'''

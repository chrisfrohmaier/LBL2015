import matplotlib
matplotlib.use('Agg')
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def Load_Grid(gridname, binname):
	g=np.load(gridname)
	ebin=np.load(binname)
	return g, ebin



def Reg_Grid(uG, ebin, order):
	#uG=np.load(str(uneGrid)+'.npy')
	uG=np.nan_to_num(uG)
	print 'Zoom Grid Shape: ', uG.shape
	zoom_grid=ndimage.interpolation.zoom(uG.astype('float16'), (100,100,100,100,100),order=order, mode='nearest')
	peak_Zoom=ndimage.interpolation.zoom(ebin[0], (1000),order=order, mode='nearest')
	red_Zoom=ndimage.interpolation.zoom(ebin[1], (200),order=order, mode='nearest')
	x1_Zoom=ndimage.interpolation.zoom(ebin[2], (200),order=order, mode='nearest')
	ab_Zoom=ndimage.interpolation.zoom(ebin[3], (200), order=order, mode='nearest')
	c_Zoom=ndimage.interpolation.zoom(ebin[4], (100), order=order, mode='nearest')
	
	print zoom_grid.shape, peak_Zoom.shape, red_Zoom.shape, x1_Zoom.shape, ab_Zoom.shape, c_Zoom.shape
	return zoom_grid, peak_Zoom, red_Zoom, x1_Zoom, ab_Zoom, c_Zoom

#m=query_db()
#make_grid(m)
eff_grid, ebin=Load_Grid('Supernova_Efficiency_Grid.npy', 'Bin_Edges.npy')
print 'Grid Loaded'
zoom_grid, peak_Zoom, red_Zoom, x1_Zoom, ab_Zoom, c_Zoom=Reg_Grid(eff_grid, ebin, 1)

def Remap_Ranges(care, rang):
	return (care-min(rang))*(len(rang))/((max(rang)-min(rang)))

def eff(peakd, red, x1, ab, zoom_grid=zoom_grid, peak_Zoom=peak_Zoom, red_Zoom=red_Zoom, x1_Zoom=x1_Zoom, ab_Zoom=ab_Zoom):
	c_collapse=zoom_grid[Remap_Ranges(peak, peak_Zoom),Remap_Ranges(red, red_Zoom),Remap_Ranges(x1, x1_Zoom),Remap_Ranges(ab, ab_Zoom)]
	print 'Color Efficiency Array: ', c_collapse
	print 'Average Efficiency: ', c_collapse.mean


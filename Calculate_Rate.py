import matplotlib
matplotlib.use('Agg')
import psycopg2
import numpy as np
from astropy.convolution import convolve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage
from scipy.optimize import fsolve
from shapely.geometry import Polygon, Point
from shapely.geometry import MultiPolygon
from descartes.patch import PolygonPatch
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()

def Load_Grid(gridname, binname):
	g=np.load(gridname)
	ebin=np.load(binname)
	return g, ebin



def Reg_Grid(uneGrid, ebin, order):
	#uG=np.load(str(uneGrid)+'.npy')
	uG=np.nan_to_num(uneGrid)
	#uG=convolve(uneGrid, )
	#uG=uneGrid
	zoom_grid=ndimage.interpolation.zoom(uG, (float(max(uG.shape))/uG.shape[0], float(max(uG.shape))/uG.shape[1],float(max(uG.shape))/uG.shape[2]),order=order, mode='nearest')
	peak_Zoom=ndimage.interpolation.zoom(ebin[0], (float(max(uG.shape))/ebin[0].shape[0]),order=order, mode='nearest')
	red_Zoom=ndimage.interpolation.zoom(ebin[1], (float(max(uG.shape))/ebin[1].shape[0]),order=order, mode='nearest')
	x1_Zoom=ndimage.interpolation.zoom(ebin[2], (float(max(uG.shape))/ebin[2].shape[0]),order=order, mode='nearest')
	
	print zoom_grid.shape, peak_Zoom.shape, red_Zoom.shape, x1_Zoom.shape
	return zoom_grid, peak_Zoom, red_Zoom, x1_Zoom

#m=query_db()
#make_grid(m)
eff_grid, ebin=Load_Grid('Supernova_Efficiency_Grid.npy', 'Bin_Edges.npy')
zoom_grid, peak_Zoom, red_Zoom, x1_Zoom=Reg_Grid(eff_grid, ebin, 1)
def Remap_Ranges(care, rang):
	return (care-min(rang))*(len(rang))/((max(rang)-min(rang)))

def Interp_On_Zoom(peak, red, x1, zoom_grid=zoom_grid, peak_Zoom=peak_Zoom, red_Zoom=red_Zoom, x1_Zoom=x1_Zoom):
	effs=ndimage.map_coordinates(zoom_grid,[[Remap_Ranges(peak, peak_Zoom)],[Remap_Ranges(red, red_Zoom)],[Remap_Ranges(x1, x1_Zoom)]],order=1)
	return effs
z=0.12
print 'Redshift:', z
print 'Volume:', cosmo.comoving_volume(z)
def Get_Supernovae(z):
	cur.execute("SELECT distinct(sncosmo_fits.ptfname), sncosmo_fits.ra, sncosmo_fits.dec, sncosmo_fits.redshift, sncosmo_fits.t0, sncosmo_fits.x1 from sncosmo_fits join ptfname on sncosmo_fits.ptfname=ptfname.ptfname join subtraction on ptfname.sub_id=subtraction.id join ptffield on ptffield.id=subtraction.ptffield where sncosmo_fits.ra>310. and sncosmo_fits.ra<360. and sncosmo_fits.dec>-7. and sncosmo_fits.dec<20. and sncosmo_fits.redshift<%s and sncosmo_fits.c>-0.3 and sncosmo_fits.c<0.4 and sncosmo_fits.x1<3. and sncosmo_fits.x1>-3. and ptffield.color_excess<0.1 and sncosmo_fits.pass_cut=True;",(float(z),))
	#cur.execute("SELECT distinct(sncosmo_fits.ptfname), sncosmo_fits.ra, sncosmo_fits.dec, sncosmo_fits.redshift, sncosmo_fits.t0, sncosmo_fits.x1 from sncosmo_fits join ptfname on sncosmo_fits.ptfname=ptfname.ptfname join subtraction on ptfname.sub_id=subtraction.id join ptffield on ptffield.id=subtraction.ptffield where sncosmo_fits.ra>107. and sncosmo_fits.ra<270 and sncosmo_fits.dec>-2. and sncosmo_fits.dec<85. and sncosmo_fits.redshift<%s and ptffield.color_excess<0.1 ;",(float(z),))

	print 'Database Query Complete'
	m=cur.fetchall()
	cur.close()
	m=np.array(m)
	print 'Number of Object:', len(m[:,0])
	return m

m=Get_Supernovae(z)

'''
m[:,0] ptfname
m[:,1] ra
m[:,2] dec
m[:,3] redshift
m[:,4] t0
m[:,5] x1
'''
eff_array=[]
for i in range(0, len(m[:,0])):
	eff_array.append(Interp_On_Zoom(m[:,4][i].astype(float), m[:,3][i].astype(float), m[:,5][i].astype(float)))
e=np.array(eff_array)
#print e

table=np.column_stack((m,e))
print table
#for i in range(0,len(table[:,0])):
#	print table[:,0][i]
 
def Get_Fields_Max_Area_2010(ujd_start, ujd_stop, e):
	flat_cols=['#1abc9c','#2ecc71','#3498db','#9b59b6','#34495e','#f39c12','#d35400','#c0392b','#7f8c8d']
	conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier') #Connecting to the SN Group Server Database
	cur = conn.cursor()
	cur.execute("SELECT distinct (subtraction.ptffield), min(subtraction.ra_ul), max(subtraction.dec_ul), max(subtraction.ra_ur), max(subtraction.dec_ur), max(subtraction.ra_lr), min(subtraction.dec_lr), min(subtraction.ra_ll), min(subtraction.dec_ll) from subtraction JOIN ptffield ON subtraction.ptffield=ptffield.id  where subtraction.filter='R' and (ujd>%s and ujd<%s) and height<10 and width<10 and ptffield.color_excess<0.1 and ra_ll>310. and dec_ll>-7. and dec_lr>-7. and ra_ul>310. and dec_ur<20. and dec_ul<20. GROUP BY subtraction.ptffield, ptffield.color_excess;",(float(ujd_start),float(ujd_stop),)) #Get Everything from the Subtraction Table
	m=cur.fetchall()
	cur.close()
	print 'DB_Done'
	sub=np.array(m) #All the Subtraction Information
	'''
	0 ptffield, 
	1 ra_ul, 
	2 dec_ul, 
	3 ra_ur, 
	4 dec_ur, 
	5 ra_lr, 
	6 dec_lr, 
	7 ra_ll, 
	8 dec_ll, 
	'''
	# for i in range(0,len(sub)):
	# 	print (sub[:,1][i],sub[:,2][i]),(sub[:,3][i],sub[:,4][i]),(sub[:,5][i],sub[:,6][i]),(sub[:,7][i],sub[:,8][i])
	# print len(sub[:,0])
	#plt.subplot(111, projection="mollweide")
	for i in range(0,len(sub[:,0])):
		area=Polygon([(np.min((sub[:,1][i],sub[:,7][i])),np.max((sub[:,2][i],sub[:,4][i]))),(np.max((sub[:,3][i],sub[:,5][i])),np.max((sub[:,2][i],sub[:,4][i]))),(np.max((sub[:,5][i],sub[:,3][i])),np.min((sub[:,8][i],sub[:,6][i]))),(np.min((sub[:,7][i],sub[:,1][i])),np.min((sub[:,8][i],sub[:,6][i])))])
		patch = PolygonPatch(area, facecolor=np.random.choice(flat_cols,1)[0],zorder=2)
		plt.gca().add_patch(patch)
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(15,12.5)
	plt.xlim(305,360)
	plt.ylim(-23,32)

	

	sima=Polygon([(305,-23),(305,31.6),(360,31.6),(360,-23)])
	simp = PolygonPatch(sima, facecolor='k',zorder=0)
	plt.gca().add_patch(simp)

	plt.scatter(e[:,1].astype(float), e[:,2].astype(float), label='Spec Confirmed SNe Ia = '+str(len(e[:,0])), color='w', zorder=200)
	plt.legend()
	plt.title('2010')
	plt.xlabel('Ra')
	plt.ylabel('Dec')
	#plt.show()
	plt.savefig('2010_Eff_Points.png', dpi=300, bbox_inches='tight')
	plt.close()	

area2010=Get_Fields_Max_Area_2010(2455317.5,2455500.5, table)	

sum_array=[]
for i in range(0,len(table[:,0][table[:,6].astype(float)>0])):
	x=(1.+table[:,3][table[:,6].astype(float)>0][i].astype(float))/((table[:,6][table[:,6].astype(float)>0][i].astype(float))*0.50137)
	sum_array.append(x)

tot=np.sum(sum_array)
print tot	

v=((4.*np.pi*1328.95)/(3.*41253.))*(cosmo.comoving_volume(z)-cosmo.comoving_volume(0.02))

print 'Mean Redhsift:', np.mean(table[:,3].astype(float))
print 'Rate:', (1/v)*tot

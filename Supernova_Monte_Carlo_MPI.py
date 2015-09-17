from mpi4py import MPI
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from matplotlib.patches import Circle
from matplotlib.patches import Polygon as matPolygon
from shapely.geometry import Polygon, Point
from descartes.patch import PolygonPatch
from astropy.time import Time
import subprocess, math, sncosmo, psycopg2
import time as TI
from scipy import ndimage
import random
#The PTF48r Bandpass needs to be registered
bpass=np.loadtxt('PTF48R.dat')
wavelength=bpass[:,0]
transmission=bpass[:,1]
band=sncosmo.Bandpass(wavelength,transmission, name='ptf48r')
sncosmo.registry.register(band, force=True)

Mag_Zoom=np.load('Mag_Zoom.npy')
LMT_Zoom=np.load('LMT_Zoom.npy')
MED_Zoom=np.load('MED_Zoom.npy')
SEE_Zoom=np.load('SEE_Zoom.npy')
zoom_grid=np.load('zoom_grid.npy')

conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier') #Connecting to the SN Group Server Database
cur = conn.cursor()

conn2 = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier') #Connecting to the SN Group Server Database
cur2 = conn2.cursor()

def Get_Fields_Max_Area_2010(ujd_start, ujd_stop):
	flat_cols=['#1abc9c','#2ecc71','#3498db','#9b59b6','#34495e','#f39c12','#d35400','#c0392b','#7f8c8d']
	conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier') #Connecting to the SN Group Server Database
	cur = conn.cursor()
	cur.execute("SELECT distinct (subtraction.ptffield), min(subtraction.ra_ul), max(subtraction.dec_ul), max(subtraction.ra_ur), max(subtraction.dec_ur), max(subtraction.ra_lr), min(subtraction.dec_lr), min(subtraction.ra_ll), min(subtraction.dec_ll) from subtraction JOIN ptffield ON subtraction.ptffield=ptffield.id  where subtraction.filter='R' and (ujd>%s and ujd<%s) and height<10 and width<10 and ptffield.color_excess<0.1 and ra_ll>305 and dec_ll>-23 and dec_lr>-23 and ra_ul>305 and dec_ur<31.6 and dec_ul<31.6  GROUP BY subtraction.ptffield, ptffield.color_excess;",(float(ujd_start),float(ujd_stop),)) #Get Everything from the Subtraction Table
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
		patch = PolygonPatch(area, facecolor='white',zorder=2)
		plt.gca().add_patch(patch)
	#fig = matplotlib.pyplot.gcf()
	#fig.set_size_inches(20,10)
	#plt.xlim(0,360)
	#plt.ylim(-35,90)

	#cords=Convert_2010_Coords()

	sima=Polygon([(305,-23),(305,31.6),(360,31.6),(360,-23)])
	simp = PolygonPatch(sima, facecolor='k',zorder=0)
	plt.gca().add_patch(simp)
	'''
	plt.scatter(cords[:,0], cords[:,1], label='Spec Confirmed SNe Ia = '+str(len(cords[:,0])), color='w', zorder=200)
	plt.legend()
	plt.title('2010')
	plt.xlabel('Ra')
	plt.ylabel('Dec')
	plt.show()
	#plt.savefig('The_2010_Sky.png', dpi=300, bbox_inches='tight')
	plt.close()	
	'''

def ccd0_defect(com, alpha):
	bad_area=[]
	xt=com[0]
	yt=com[1]

	jonest=np.matrix([[np.cos(alpha), -np.sin(alpha), xt], [np.sin(alpha), np.cos(alpha), yt], [0,0,1.]])
	good_area=[]

	xl_scale=abs(com[0]-com[2])/((2048*1.01)/3600)
	xr_scale=abs(com[4]-com[6])/((2048*1.01)/3600)
	
	
	for i in [(147.*xl_scale,1501.), (177.*xl_scale,1501.),(177.*xr_scale,4097.),(147.*xr_scale,4097.)]:
		x=i[0]*(1.01/3600)
		y=i[1]*(1.01/3600)
		pixel_coords=np.matrix([[x],[y],[1.]])
		#print 'Pixel:', pixel_coords
		cords=np.dot(jonest,pixel_coords)
		#print 'New Coordinates:', cords
		#print cords.item(0)
		bad_area.append((cords.item(0),yt-(cords.item(1)-yt)))
	return Polygon(bad_area)

def all_CCD_check(com, ra, dec):	
	sima=Polygon([(com[0],com[1]),(com[2],com[3]),(com[4],com[5]),(com[6],com[7])])
	#simp = PolygonPatch(sima, facecolor='k',zorder=0)
	#plt.gca().add_patch(simp)
	alpha=np.arctan(np.divide((np.subtract(com[6],com[0])),np.subtract(com[7],com[1])))
	#print alpha

	xt=com[0]
	yt=com[1]

	jonest=np.matrix([[np.cos(alpha), -np.sin(alpha), xt], [np.sin(alpha), np.cos(alpha), yt], [0,0,1.]])
	#print jonest

	good_area=[]

	xl_scale=abs(com[0]-com[2])/((2048*1.01)/3600)
	xr_scale=abs(com[4]-com[6])/((2048*1.01)/3600)
	
	yu_scale=abs(com[3]-com[5])/((4096*1.01)/3600)
	
	yl_scale=abs(com[1]-com[7])/((4096*1.01)/3600)

	for i in [(75.*xl_scale,75.), (2048.*xl_scale,75.),(2048.*xr_scale,4000.),(75.*xr_scale,4000.)]:
		x=i[0]*(1.01/3600)
		y=i[1]*(1.01/3600)
		pixel_coords=np.matrix([[x],[y],[1.]])
		#print 'Pixel:', pixel_coords
		cords=np.dot(jonest,pixel_coords)
		#print 'New Coordinates:', cords
		#print cords.item(0)
		good_area.append((cords.item(0),yt-(cords.item(1)-yt)))
	
	garea=[good_area[0], (good_area[1][0],com[3]-((75.*1.01/3600)*yu_scale)), (good_area[2][0],com[5]+((96.*1.01/3600)*yu_scale)),good_area[3]]
	#print good_area
	#print [(com[0],com[1]),(com[2],com[3]),(com[4],com[5]),(com[6],com[7])]
	simg1=Polygon(garea)
	if com[8]==0:
		simb0=ccd0_defect(com, alpha)
		simg1=simg1.difference(simb0)
	return simg1.contains(Point(ra,dec))

def Get_Obs_Conditions(Ra, Dec, peak_date, cur=cur):
	
	cur.execute("SELECT geo_sub.ujd, geo_sub.ptffield, geo_sub.ccdid, geo_sub.lmt_mg_new, (geo_sub.seeing_new/geo_sub.seeing_ref), geo_sub.medsky_new, geo_sub.good_pix_area, ptffield.color_excess, geo_sub.ra_ll,geo_sub.dec_ll,geo_sub.ra_ul,geo_sub.dec_ul,geo_sub.ra_ur,geo_sub.dec_ur,geo_sub.ra_lr,geo_sub.dec_lr FROM geo_sub JOIN ptffield ON geo_sub.ptffield=ptffield.id WHERE ptffield.is_sdss=True and geo_sub.filter='R' and ujd>(%s-25.) and ujd<(%s+55.) and height<10 and width<10 and geo_sub.deep_ref_id<=17783 and st_contains(geo_sub.geo_ccd, ST_GeometryFromText('Point(%s %s)',4326))=True  ORDER BY geo_sub.ujd ASC;",(float(peak_date),float(peak_date),float(Ra),float(Dec),)) #Get Everything from the Subtraction Table
	m=cur.fetchall()
	
	m=np.array(m)
	'''Whats in m?
	m[:,0]	| Date
	m[:,1]	| PTFField
	m[:,2]	| ccdid
	m[:,3]	| lmt_mg_new
	m[:,4]	| Seeing_ratio
	m[:,5]	| medsky_new
	m[:,6]	| good_pix_area	
	m[:,7]	| color_excess
	m[:,8]	| ra_ll
	m[:,9]	| dec_ll
	m[:,10]	| ra_ul
	m[:,11]	| dec_ul
	m[:,12]	| ra_ur
	m[:,13]	| dec_ur
	m[:,14]	| ra_lr
	m[:,15]	| dec_lr
	'''
	#print 'mlen', len(m[:,0])
	#print m
	#print len(m[:])
	if len(m[:]) != 0:
		#com1=[m[:,8],m[:,9],m[:,10],m[:,11],m[:,12],m[:,13],m[:,14],m[:,15],m[:,16],m[:,17],m[:,2]]
		in_ccd=[]
		for i in range(0,len(m[:,0])):
			#print i
			com=[m[:,8][i],m[:,9][i],m[:,10][i],m[:,11][i],m[:,12][i],m[:,13][i],m[:,14][i],m[:,15][i],m[:,2][i]]
			in_ccd.append(all_CCD_check(com, Ra, Dec))
		m=np.column_stack((m,in_ccd)) #Now m[:,16] is a true/false column of whether object is in the good region
		#print m
		return m[m[:,16]==True]	
	else:
		return m
def Remap_Ranges(care, rang):
	return (care-min(rang))*(len(rang))/((max(rang)-min(rang)))

def Interp_On_Zoom(mag, lmt, med, see, zoom_grid=zoom_grid, Mag_Zoom=Mag_Zoom, LMT_Zoom=LMT_Zoom, MED_Zoom=MED_Zoom, SEE_Zoom=SEE_Zoom):
	effs=ndimage.map_coordinates(zoom_grid,[[Remap_Ranges(mag, Mag_Zoom)],[Remap_Ranges(lmt, LMT_Zoom)],[Remap_Ranges(med, MED_Zoom)],[Remap_Ranges(see, SEE_Zoom)]],order=1)
	return effs	

def Pass_Selection(sn_arr, find_bool, peak_date):
	'''
	'''

	low=False
	high=False
	#print 'sn_arr', sn_arr[:,0]
	#print 'All diff:', np.diff(sn_arr[:,0])
	check_low=sn_arr[:,0][(find_bool==True) & (sn_arr[:,0]<peak_date)]
	#print check_low
	#print 'Differences:', np.diff(check_low)
	if len(np.diff(check_low)[np.diff(check_low)>=0.5])>=1:
		low=True
	#	print 'Check', np.diff(check_low)[np.diff(check_low)>0.5]
	#	print 'Great Success on low'
	check_high=sn_arr[:,0][(find_bool==True) & (sn_arr[:,0]>peak_date)] 
	#print check_high
	#print 'Differences:', np.diff(check_high)
	if len(np.diff(check_high)[np.diff(check_high)>=0.5])>=1:
		high=True
	#	print 'Check', np.diff(check_high)[np.diff(check_high)>0.5]
	#	print 'Great Success on high'
	if low and high == True:
		#print 'YES!'
		return True
	else:
		#print 'NO!'
		return False

def Check_Detection(sn_par):
	'''This returns a truth array where each element in the array is a true false statement about whether that 
	observation would have been detected
	'''
	probs=(Interp_On_Zoom(sn_par[:,1], sn_par[:,4], sn_par[:,6], sn_par[:,5])).T

	tot_prob=probs[:,0]
	#print 'Prob',probs[:,0]
	#print 'Tot_Prob', tot_prob
	ran=np.random.uniform(0,1,len(tot_prob))
	#print ran
	x=(probs>ran)
	#print x
	#print 'Length Before: {}, Length after: {}'.format(len(sn_par[:,0]), len(sn_par[:,0][x==True]))
	#print sn_par[:,0][x==True]
	#print sn_par[x==True]
	#print sn_par[x==True][:,0][0]
	#print len(probs), len(snpar[:,1])

	return x
def Random_sneParameters(low_ujd, high_ujd, low_ra, high_ra, low_dec, high_dec):
	'''This needs to be changed to accept an area to draw coordinates from a range - Done
	'''
	
	umin=np.radians(low_ra)/(2.*np.pi)
	umax=np.radians(high_ra)/(2.*np.pi)

	vmin=(np.cos(np.radians(90.-low_dec))+1.0)/2.
	vmax=(np.cos(np.radians(90.-high_dec))+1.0)/2.

	pDec=random.uniform(vmin, vmax)
	pRa=random.uniform(umin, umax)
	#peak_date=np.random.uniform(min(sub[:,0]), max(sub[:,0]))
	
	#maxdate=high_ujd-low_ujd
	#pd=np.random.ran()
	explosion_date=np.random.uniform(low_ujd, high_ujd)
	#print peak_date
	xone=np.random.uniform(-3.0,3.0)
	color=np.random.uniform(-0.2,0.4)
	zedshift=np.random.uniform(0.0,0.12)
	int_dis=np.random.normal(0.,0.15)

	stretch=0.98+(0.091*xone)+(0.003*xone*xone)-(0.00075*xone*xone*xone)	

	peak_date=explosion_date+(18.98*(1.+zedshift)*stretch)
	Ra=np.degrees(pRa*2.*np.pi)
	Dec=90.-np.degrees(np.arccos((2.*pDec)-1.))
	#print Ra, Dec, peak_date
	return Ra, Dec, peak_date, xone, color, zedshift, int_dis 
	

def Gen_SN(peak_date, Ra, Dec, redshift, colour,x_1, int_dis, cur=cur):
	#Use below if on Iridis
	source = sncosmo.SALT2Source(modeldir="/scratch/cf5g09/Monte_Carlos/salt2-4")

	##Use below if not on iridis
	#source=sncosmo.get_source('salt2',version='2.4') 
	

	alpha=0.141
	beta=3.101
	m=Get_Obs_Conditions(Ra, Dec, peak_date, cur=cur)

	mabs= -19.05 - alpha*x_1 + beta*colour + int_dis #Setting the absolute Magnitude of the Supernova
	
	#print 'MW Extinction E(B-V): ', m[:,7][0]
	dust = sncosmo.CCM89Dust()
	model=sncosmo.Model(source=source, effects=[dust], effect_names=['mw'], effect_frames=['obs']) 
	model.set(z=redshift,t0=peak_date,x1=x_1, c=colour) #Setting redshift
	
	model.set_source_peakabsmag(mabs,'bessellb','ab', cosmo=FlatLambdaCDM(H0=70,Om0=0.3)) #Fixing my peak absolute magnitude
	#model.set(x1=x_1, c=colour)
	absmagb=model.source_peakabsmag('bessellb','ab', cosmo=FlatLambdaCDM(H0=70,Om0=0.3))
	absmag_r=model.source_peakabsmag('ptf48r','ab', cosmo=FlatLambdaCDM(H0=70,Om0=0.3))

	band=sncosmo.get_bandpass('ptf48r') #Retrieving the ptf48r bandpass 
	if m.size==0:
		return mabs, absmag_r, False, 9999.99, False
	model.set(mwebv=m[:,7][0])
	
	
	maglc=model.bandmag('ptf48r','ab',m[:,0]) #Creating a magnitude array of the lightcurve  
	fluxlc=model.bandflux('ptf48r',m[:,0]) #Creating a flux array of the lightcurve
	
	'''
	m[:,2]	| ccdid
	m[:,3]	| lmt_mg_new
	m[:,4]	| Seeing_ratio
	m[:,5]	| medsky_new
	m[:,6]	| good_pix_area	
	'''
	##Getting Rid of NANs in the Mag arrays
	time_array=m[:,0][(~np.isnan(maglc))]
	mag_lc=maglc[~np.isnan(maglc)]
	flux_lc=fluxlc[~np.isnan(maglc)]
	ccd_lc=m[:,2][~np.isnan(maglc)]
	lmt_lc=m[:,3][~np.isnan(maglc)]
	see_rat=m[:,4][~np.isnan(maglc)]
	med_lc=m[:,5][~np.isnan(maglc)]
	pix_lc=m[:,6][~np.isnan(maglc)]
	#print maglc
	#print mag_lc
	#print time_array[mag_lc<20], mag_lc[mag_lc<20]
	#print time_array[mag_lc<20], mag_lc[mag_lc<20]
	sn_par=np.array((time_array[mag_lc<20], mag_lc[mag_lc<20], flux_lc[mag_lc<20], ccd_lc[mag_lc<20], lmt_lc[mag_lc<20], see_rat[mag_lc<20], med_lc[mag_lc<20], pix_lc[mag_lc<20] )).T
	if sn_par.size == 0:
		#print '--------------------------------------SNPAR--------------------------------------'
		return mabs, absmag_r, False, 9999.99, False

	'''snapr
	snpar[:,0]	| time
	snpar[:,1]	| mag_lc
	snpar[:,2]	| flux_lc
	snpar[:,3]	| ccd_lc
	snpar[:,4]	| lmt_lc
	snpar[:,5]	| see_rat
	snpar[:,6]	| med_lc
	snpar[:,7]	| pix_lc
	'''
	#print sn_par

	return  mabs, absmag_r, sn_par, m[:,7][0], True #m[:,7][0] is color_excess

def update_sn_mc_table(peak_date, ra, dec, abmag_r, redshift, x1, color, int_dis, found, ebv, cur2=cur2):
	cur2.execute("INSERT INTO sn_mc (peak_date, ra, dec, abmag_r, redshift, x1, color, int_dis, found, ebv) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, %s)",(float(peak_date), ra, dec, ab_magb, redshift, x1, color, int_dis, found, ebv))
	conn2.commit()


N_MODELS_TOTAL = 50000000
ra_array=np.ones(N_MODELS_TOTAL)
dec_array=np.ones(N_MODELS_TOTAL)
found_array=np.ones(N_MODELS_TOTAL)
time_init = TI.time()
nproc = MPI.COMM_WORLD.Get_size()   	# number of processes
my_rank = MPI.COMM_WORLD.Get_rank()   	# The number/rank of this process

my_node = MPI.Get_processor_name()    	# Node where this MPI process runs

# total number of models to run	
n_models = N_MODELS_TOTAL / nproc		# number of models for each thread
remainder = N_MODELS_TOTAL - ( n_models * nproc )	# the remainder. e.g. your number of models may 
# little trick to spread remainder out among threads
# if say you had 19 total models, and 4 threads
# then n_models = 4, and you have 3 remainder
# this little loop would distribute these three 
if remainder < my_rank + 1:
	my_extra = 0
	extra_below = remainder
else:
	my_extra = 1
	extra_below = my_rank

# where to start and end your loops for each thread
my_nmin = (my_rank * n_models) + extra_below
my_nmax = my_nmin + n_models + my_extra

# total number you actually do
ndo = my_nmax - my_nmin
#print 'My_Rank:', my_rank
#print 'my_node:', my_node
#print 'Time', TI.time()*(my_rank+1*np.pi)

for i in range( my_nmin, my_nmax):
	Ra, Dec, peak_date, xone, color, zedshift, int_dis=Random_sneParameters(2455256.5,2455500.5,107.,270.,-2.,75.)

	absmagb, absmag_r, sn_par, ebv, good=Gen_SN(peak_date, Ra, Dec, zedshift, color,xone, int_dis, cur=cur)
	#print sn_par
	if good==False:
		Pass=False
	elif good==True:		
		find_bool=Check_Detection(sn_par)
		Pass=Pass_Selection(sn_par, find_bool[:,0], peak_date)
	#print Ra, Dec, absmagb, absmag_r, Pass
	update_sn_mc_table(peak_date, Ra, Dec, absmag_r, zedshift, xone, color, int_dis, Pass, ebv)

#uncomment this at the end of your script
cur.close()
time2 = TI.time()	
time_tot = time2 - time_init
# always call this when finishing up
MPI.Finalize()
'''
q=np.array([ra_array,dec_array,found_array]).T
#print q
#print q[:,0][q[:,2]==True]

plt.scatter(q[:,0][q[:,2]==True],q[:,1][q[:,2]==True], color='green', s=5., zorder=100)
plt.scatter(q[:,0][q[:,2]==False],q[:,1][q[:,2]==False], color='red', s=2., zorder=1)
Get_Fields_Max_Area_2010(2455317.5,2455500.5)
plt.show()
'''
print "Time to do x:", time_tot

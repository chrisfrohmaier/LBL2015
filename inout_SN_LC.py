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

def Get_Obs_Conditions(Ra, Dec, peak_date, cur=cur):
	
	cur.execute("SELECT geo_sub.ujd, geo_sub.ptffield, geo_sub.ccdid, geo_sub.lmt_mg_new, (geo_sub.seeing_new/geo_sub.seeing_ref), geo_sub.medsky_new, geo_sub.good_pix_area, ptffield.color_excess, geo_sub.sub_zp FROM geo_sub JOIN ptffield ON geo_sub.ptffield=ptffield.id WHERE geo_sub.filter='R' and ujd>(%s-20.) and ujd<(%s+50.) and height<10 and width<10 and st_contains(geo_sub.geo_ccd, ST_GeometryFromText('Point(%s %s)',4326))=True  ORDER BY geo_sub.ujd ASC;",(float(peak_date),float(peak_date),float(Ra),float(Dec),)) #Get Everything from the Subtraction Table
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
	m[:,8]	| sub_zp
	'''
	
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
	if len(np.diff(check_low)[np.diff(check_low)>0.5])>=2:
		low=True
	#	print 'Check', np.diff(check_low)[np.diff(check_low)>0.5]
	#	print 'Great Success on low'
	check_high=sn_arr[:,0][(find_bool==True) & (sn_arr[:,0]>peak_date)] #CHECK THIS!!!!
	#print check_high
	#print 'Differences:', np.diff(check_high)
	if len(np.diff(check_high)[np.diff(check_high)>0.5])>=2:
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
	'''Fill this in
	'''
	probs=(Interp_On_Zoom(sn_par[:,1], sn_par[:,4], sn_par[:,6], sn_par[:,5])).T
	tot_prob=(np.multiply(probs[:,0],np.divide(sn_par[:,7],0.6603)))
	#print 'Prob',probs[:,0]
	#print 'Tot_Prob', tot_prob
	ran=np.random.uniform(0,1,len(tot_prob))
	#print ran
	x=(tot_prob>ran)
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
	#source = sncosmo.SALT2Source(modeldir="/scratch/cf5g09/Monte_Carlos/salt2-4")

	##Use below if not on iridis
	source=sncosmo.get_source('salt2',version='2.4') 
	

	alpha=0.141
	beta=3.101
	m=Get_Obs_Conditions(Ra, Dec, peak_date, cur=cur)

	mabs= -19.05 - alpha*x_1 + beta*colour + int_dis #Setting the absolute Magnitude of the Supernova
	if m.size==0:
		return mabs, 9999.99, False, 9999.99
	#print 'MW Extinction E(B-V): ', m[:,7][0]
	dust = sncosmo.CCM89Dust()
	model=sncosmo.Model(source=source, effects=[dust], effect_names=['mw'], effect_frames=['obs']) 
	model.set(z=redshift,t0=peak_date,x1=x_1, c=colour, mwebv=m[:,7][0]) #Setting redshift
	
	model.set_source_peakabsmag(mabs,'bessellb','ab', cosmo=FlatLambdaCDM(H0=70,Om0=0.3)) #Fixing my peak absolute magnitude
	#model.set(x1=x_1, c=colour)

	band=sncosmo.get_bandpass('ptf48r') #Retrieving the ptf48r bandpass 
    
	
	
	maglc=model.bandmag('ptf48r','ab',m[:,0]) #Creating a magnitude array of the lightcurve  
	fluxlc=model.bandflux('ptf48r',m[:,0]) #Creating a flux array of the lightcurve
	absmagb=model.source_peakabsmag('bessellb','ab', cosmo=FlatLambdaCDM(H0=70,Om0=0.3))
	absmag_r=model.source_peakabsmag('ptf48r','ab', cosmo=FlatLambdaCDM(H0=70,Om0=0.3))
	'''
	m[:,2]	| ccdid
	m[:,3]	| lmt_mg_new
	m[:,4]	| Seeing_ratio
	m[:,5]	| medsky_new
	m[:,6]	| good_pix_area	
	m[:,7]	| color excess
	m[:,8]	| sub_zp
	'''
	##Getting Rid of NANs in the Mag arrays
	time_array=m[:,0][~np.isnan(maglc)]
	mag_lc=maglc[~np.isnan(maglc)]
	flux_lc=fluxlc[~np.isnan(maglc)]
	ccd_lc=m[:,2][~np.isnan(maglc)]
	lmt_lc=m[:,3][~np.isnan(maglc)]
	see_rat=m[:,4][~np.isnan(maglc)]
	med_lc=m[:,5][~np.isnan(maglc)]
	pix_lc=m[:,6][~np.isnan(maglc)]
	ebv_lc=m[:,7][~np.isnan(maglc)]
	zp_lc=m[:,8][~np.isnan(maglc)]
	#print maglc
	#print mag_lc
	sn_par=np.array((time_array, mag_lc, flux_lc, ccd_lc, lmt_lc, see_rat, med_lc, pix_lc, zp_lc )).T
	'''snapr
	snpar[:,0]	| time
	snpar[:,1]	| mag_lc
	snpar[:,2]	| flux_lc
	snpar[:,3]	| ccd_lc
	snpar[:,4]	| lmt_lc
	snpar[:,5]	| see_rat
	snpar[:,6]	| med_lc
	snpar[:,7]	| pix_lc
	snpar[:,8]	| sub_zp
	'''
	#print sn_par

	return  mabs, absmag_r, sn_par, m[:,7][0] #m[:,7][0] is color_excess

def update_io_sn_mc_table(peak_date, ra, dec, ab_magb, redshift, x1, color, int_dis, found, ebv, cur2=cur2):
	cur2.execute("INSERT INTO inout_sn_mc (peak_date, ra, dec, ab_magb, redshift, x1, color, int_dis, found, ebv) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, %s) RETURNING sn_id",(float(peak_date), ra, dec, ab_magb, redshift, x1, color, int_dis, found, ebv))
	snid=cur2.fetchone()
	conn2.commit()
	return snid
def update_lc_table(snid, date_array, mag_array, zp_array, cur2=cur2):
	for i in range(0,len(date_array)):
		cur2.execute("INSERT INTO inout_lc (snid, date, mag, zeropoint) VALUES (%s,%s,%s,%s);",(float(snid), date_array[i], mag_array[i], zp_array[i]))
	conn2.commit()


N_MODELS_TOTAL = 500000
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
	Ra, Dec, peak_date, xone, color, zedshift, int_dis=Random_sneParameters(2455317.5,2455500.5,310,360,-7.,20.)

	absmagb, absmag_r, sn_par, ebv=Gen_SN(peak_date, Ra, Dec, zedshift, color,xone, int_dis, cur=cur)
	if absmag_r==9999.99:
		Pass=False
	elif absmag_r<9999.99:
		find_bool=Check_Detection(sn_par)
		Pass=Pass_Selection(sn_par, find_bool, peak_date)
	#print Ra, Dec, Pass
	snid=update_io_sn_mc_table(peak_date, Ra, Dec, absmagb, zedshift, xone, color, int_dis, Pass, ebv)
	if sn_par==False:
		continue
	update_lc_table(snid[0], sn_par[:,0], sn_par[:,1], sn_par[:,8])

#uncomment this at the end of your script
cur.close()
time2 = TI.time()	
time_tot = time2 - time_init
# always call this when finishing up
MPI.Finalize()
print "Time to do x:", time_tot

import matplotlib, psycopg2
matplotlib.use('Agg')
import numpy as np
import astropy
import sncosmo
import matplotlib.pyplot as plt
from mpi4py import MPI
flat_cols=['#1abc9c','#2ecc71','#3498db','#9b59b6','#34495e','#f39c12','#d35400','#c0392b','#7f8c8d']
l=np.genfromtxt('PTFNAME_Coord_List_Ia.dat', usecols=(0,), delimiter=',', dtype=None)
source=sncosmo.get_source('salt2',version='2.4') 
model=sncosmo.Model(source=source) 
bpass=np.loadtxt('PTF48R.dat')
wavelength=bpass[:,0]
transmission=bpass[:,1]
band=sncosmo.Bandpass(wavelength,transmission, name='ptf48r')
sncosmo.registry.register(band, force=True)

def Check_Dates(dates, peakd):
	low=False
	high=False
	print dates
	dates_trim=dates[(dates>(peakd-20.)) & (dates<(peakd+50.))]
	
	dates_trim=np.sort(dates_trim)
	print dates_trim
	#print 'sn_arr', sn_arr[:,0]
	#print 'All diff:', np.diff(sn_arr[:,0])
	check_low=dates_trim[(dates_trim<peakd)]
	#print check_low
	#print 'Differences:', np.diff(check_low)
	if len(np.diff(check_low)[np.diff(check_low)>0.5])>=2:
		low=True
	#	print 'Check', np.diff(check_low)[np.diff(check_low)>0.5]
	#	print 'Great Success on low'
	check_high=dates_trim[(dates_trim>peakd)] #CHECK THIS!!!!
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

N_MODELS_TOTAL = len(l)  # total number of models to run
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
##Connecting to the Database
conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()

for i in range( my_nmin, my_nmax):

	try:
		hml=np.load('phot10/'+str(l[i])+'_LC.npy')

		
		
		ab=np.zeros(len(hml[:,1]), dtype='|S2')
		for i in range(len(ab)):
			ab[i]='ab'
		hml=np.column_stack((hml,ab))	
		band=np.zeros(len(hml[:,1]), dtype='|S6')
		for i in range(len(band)):
			band[i]='ptf48r'
		hml=np.column_stack((hml,band))
		hml_dat=astropy.table.Table(data=hml, names=('ptfname', 'time', 'magnitude', 'mag_err', 'flux', 'flux_err', 'zp_new', 'zp', 'ra', 'dec', 'zpsys', 'filter'), dtype=('str','float','float','float','float','float','float','float','float','float','str','str'))
		print 'Doing:', hml[:,0][0]

		cur.execute("SELECT redshift, obsdate, phase from specinfo where ptfname=%s;",(str(hml[:,0][0]),))
		zed=cur.fetchone()

		if len(zed)==0:
			print 'Bad Query'
			break
		#print zed[0]
		
		model.set(z=zed[0])
		#res, fitted_model=sncosmo.mcmc_lc(hml_dat, model, ['t0','x0','x1','c'], bounds={'x1':(-3.5,3.5), 'c':(-0.35,0.45)}, nburn=100, nsamples=5000)
		res, fitted_model=sncosmo.fit_lc(hml_dat, model, ['t0','x0','x1','c'], bounds={'x1':(-3.5,3.5), 'c':(-0.35,0.45)})
		pdate=res.parameters[1]
		pass_4cut=Check_Dates(hml[:,1].astype(float), pdate)
		print hml[:,0][0], pass_4cut
		

		
		fig=sncosmo.plot_lc(hml_dat, model=fitted_model, errors=res.errors, color=np.random.choice(flat_cols), figtext=str(hml[:,0][0])+'\n'+str(pass_4cut), xfigsize=10, pulls=False)
		plt.axvline(-20., color='black', linestyle='--')
		plt.axvline(+50., color='black', linestyle='--')
		plt.savefig('LC_Fixed/'+str(hml[:,0][0])+'.png', dpi=150, bbox_inches='tight')
		plt.close()
		print '### Parameters ###'
		print str(hml[:,0][0]), float(zed[0]), float(0), float(res.parameters[1]), float(res.errors['t0']),float(res.parameters[2]), float(res.errors['x0']),  float(res.parameters[3]), float(res.errors['x1']), float(res.parameters[4]), float(res.errors['c']), float(hml[:,8][0]), float(hml[:,9][0])
		print 'chi2', sncosmo.chisq(hml_dat, fitted_model)
		#cur.execute("INSERT INTO sncosmo_fits (ptfname, redshift, redshift_err, t0, t0_err, x0, x0_err, x1, x1_err, c, c_err, ra, dec, pass_cut) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);",(str(hml[:,0][0]), float(zed[0]), float(0), float(res.parameters[1]), float(res.errors['t0']),float(res.parameters[2]), float(res.errors['x0']),  float(res.parameters[3]), float(res.errors['x1']), float(res.parameters[4]), float(res.errors['c']), float(hml[:,8][0]), float(hml[:,9][0]), pass_4cut,))
		#conn.commit()
		print 'Done:', hml[:,0][0]
		
	except ValueError:
		print 'Value Error'	
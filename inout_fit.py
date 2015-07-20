import matplotlib, psycopg2
matplotlib.use('Agg')
import numpy as np
import astropy
import sncosmo
import matplotlib.pyplot as plt
from mpi4py import MPI
flat_cols=['#1abc9c','#2ecc71','#3498db','#9b59b6','#34495e','#f39c12','#d35400','#c0392b','#7f8c8d']
#l=np.genfromtxt('PTFNAME_Coord_List_Ia.dat', usecols=(0,), delimiter=',', dtype=None)
source=sncosmo.get_source('salt2',version='2.4') 
model=sncosmo.Model(source=source) 
bpass=np.loadtxt('PTF48R.dat')
wavelength=bpass[:,0]
transmission=bpass[:,1]
band=sncosmo.Bandpass(wavelength,transmission, name='ptf48r')
sncosmo.registry.register(band, force=True)

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

lcs=np.loadtxt('inout_snid_list.dat', usecols=(0,),dtype='int')

for i in range( my_nmin, my_nmax):

	try:
		cur.execute("SELECT date, 'ptf48r', 10^((mag-zeropoint)/(-2.5)),|/10^((mag-zeropoint)/(-2.5)), zeropoint, 'ab' FROM inout_lc WHERE snid=%s", (int(lcs[i])))
		m=cur.fetchall()
		m=np.array(m)
		
		hml_dat=astropy.table.Table(data=m, names=('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys'), dtype=('float','str','float','float','float','float'))
		print 'Doing:', hml[:,0][0]

		
		
		res, fitted_model=sncosmo.mcmc_lc(hml_dat, model, ['z','t0','x0','x1','c'], bounds={'z':(0,0.15),'x1':(-3.5,3.5), 'c':(-0.35,0.45)}, nburn=100, nsamples=5000)
		#res, fitted_model=sncosmo.fit_lc(hml_dat, model, ['t0','x0','x1','c'], bounds={'x1':(-3.5,3.5), 'c':(-0.35,0.45)}, verbose=True)
		#res, fitted_model=sncosmo.nest_lc(hml_dat, model, ['t0','x0','x1','c'], bounds={'x1':(-3.5,3.5), 'c':(-0.35,0.45)},)
				
		fig=sncosmo.plot_lc(hml_dat, model=fitted_model, errors=res.errors, color=np.random.choice(flat_cols), xfigsize=10)
		
		plt.savefig('Fitted_LCs/'+str(lcs[i])+'.png', dpi=150, bbox_inches='tight')
		plt.close()
		# print '### Parameters ###'
		# print str(hml[:,0][0]), float(zed[0]), float(0), float(res.parameters[1]), float(res.errors['t0']),float(res.parameters[2]), float(res.errors['x0']),  float(res.parameters[3]), float(res.errors['x1']), float(res.parameters[4]), float(res.errors['c']), float(hml[:,8][0]), float(hml[:,9][0])
		# print 'chi2', sncosmo.chisq(hml_dat, fitted_model)
		# print 'ndof', len(hml_dat)-4. #len(data)-len(vparam_names)
		# print 'red_chi2', sncosmo.chisq(hml_dat, fitted_model)/(len(hml_dat)-4.)
		# print 'absolute magnitue', fitted_model.source_peakabsmag('bessellb','ab')
		# print 'chi2', res.chisq
		# print 'res.ndof', res.ndof
		# print 'red_chisq', res.chisq/res.ndof
		#cur.execute("INSERT INTO sncosmo_fits (ptfname, redshift, redshift_err, t0, t0_err, x0, x0_err, x1, x1_err, c, c_err, ra, dec, pass_cut, redchi2, abs_mag) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);",(str(hml[:,0][0]), float(zed[0]), float(0), float(res.parameters[1]), float(res.errors['t0']),float(res.parameters[2]), float(res.errors['x0']),  float(res.parameters[3]), float(res.errors['x1']), float(res.parameters[4]), float(res.errors['c']), float(hml[:,8][0]), float(hml[:,9][0]), pass_4cut,float(sncosmo.chisq(hml_dat, fitted_model)/(len(hml_dat)-4.)), float(fitted_model.source_peakabsmag('bessellb','ab')),))
		#conn.commit()
		#print 'Done:', hml[:,0][0]
		
	except ValueError:
	 	print 'Value Error'
	
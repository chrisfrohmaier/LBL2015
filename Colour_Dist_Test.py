import matplotlib
#matplotlib.use('Agg')
import psycopg2
import numpy as np
from scipy.special import erf
from mpi4py import MPI

def Mid_Bins(arr):
	new_array=[]
	for i in range(len(arr)-1):
		new_array.append((arr[i]+arr[i+1])/2.0)

	return new_array



def SkewG(x,g,a,mu,s):
	efunc=erf((g*(x-mu))/(s*np.sqrt(2.)))
	return (a/(s*np.sqrt(2.*np.pi)))*(np.exp((-(x-mu)**2.)/(2.*s**2.)))*(1.+efunc)




#print 'Skewx', 'SkewFrac', 'Mean C', 'Count', 'Number Pass'
def Update_DB_from_Color_Data(lowc, highc, mskewa, cur2):
	#print lowc, highc
	x=np.mean((lowc,highc))
	
	cur2.execute("SELECT COUNT(*) from sn_mc where (color >=%s and color <%s);",((float(lowc),float(highc),)))
	print cur2.query
	print 'Count Done for Colour: ', x
	count=cur2.fetchone()[0]
	print 'Number in bin: ', count
	skewx=SkewG(x,1.8192627275,0.997793919871,-0.105487431764,0.117890808366)
	print skewx, skewx/mskewa, x, count, int(count*skewx/mskewa)
	setF=int(count*skewx/mskewa)
	print 'Updating Colour: ', x
	#cur2.execute("UPDATE sn_mc SET colour_pass = False WHERE sn_id IN (SELECT sn_id from sn_mc where (color >=%s and color <%s) limit %s);",((float(lowc),float(highc),int(setF),)) )
	cur2.execute("INSERT INTO colour_dis_sn_mc SELECT * from sn_mc where (color >=%s and color <%s) limit %s;",((float(lowc),float(highc),int(setF),)) )
	print cur2.query
	conn2.commit()
	
	print 'Done Colour :', x

def Fix_Broken_Bins(lowc, highc, mskewa):
	conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
	cur = conn.cursor()
	x=np.mean((lowc,highc))
	print 'Fixing Colour: ', x
	cur.execute("UPDATE sn_mc SET colour_pass = True WHERE (color >=%s and color <%s)",((float(lowc),float(highc),)) )
	
	conn.commit()
	print 'Fixed Colour: ', x
	print 'Redoing Colour: ', x
	Update_DB_from_Color_Data(lowc,highc, mskewa)
	conn.close()

#m=query_db()
bins=np.linspace(-0.2,0.4,500)
skewa=[SkewG(x,1.8192627275,0.997793919871,-0.105487431764,0.117890808366) for x in Mid_Bins(bins)]

N_MODELS_TOTAL = len(bins)-1
ra_array=np.ones(N_MODELS_TOTAL)
dec_array=np.ones(N_MODELS_TOTAL)
found_array=np.ones(N_MODELS_TOTAL)

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
	print 'Bin Number: ', i 
	print bins[i],bins[i+1]
	conn2 = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
	print 'Opened Connection'
	cur2 = conn2.cursor()
	Update_DB_from_Color_Data(bins[i],bins[i+1], max(skewa), cur2)
	cur2.close()
	conn2.close()
	print 'Closed Connection'
'''
Fix_Bins=[1,2,3,4,5,6,7,8,9,10]
for i in Fix_Bins:
	Fix_Broken_Bins(bins[i],bins[i+1], max(skewa))
'''
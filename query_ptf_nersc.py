#import matplotlib
#matplotlib.use('GTKCairo')
import psycopg2
import numpy as np
import matplotlib.pyplot as plt

l=np.genfromtxt('Ibc.dat', usecols=(0,), delimiter=',', dtype=None)
for i in l:
	
	con = psycopg2.connect(host='scidb2.nersc.gov', user='subptf', password='p33d$kyy', database='subptf')
	cur = con.cursor()
	print 'Nersc DB being Queried'
	cur.execute("SELECT ptfname.ptfname, subtraction.ujd, candidate.mag, candidate.mag_err, candidate.flux, candidate.flux_err, subtraction.ub1_zp_new, subtraction.sub_zp, candidate.ra, candidate.dec from ptfname join candidate on ptfname.candidate_id=candidate.id join subtraction on candidate.sub_id=subtraction.id where ptfname.ptfname=%s and subtraction.filter='R';", (str(i),))
	print cur.query

	print 'Query Done'
	m=cur.fetchall()
	cur.close()
	m=np.array(m)
	#print 'length of array:', len(m[:,1]), len(m[:,2])
	#print m[:,1].astype(float),m[:,2].astype(float)
	print m[:,0][0]
	np.save(str(m[:,0][0])+'_LC',m)

'''
plt.scatter(m[:,1].astype(float),m[:,2].astype(float))
plt.errorbar(m[:,1].astype(float),m[:,2].astype(float),yerr=m[:,3].astype(float), linestyle="None")
plt.gca().invert_yaxis()
#plt.savefig('11fe_LC.png')
plt.show()
'''
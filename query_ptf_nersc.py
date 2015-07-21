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
	cur.execute("SELECT subtraction.ujd, cand.flux, cand.flux_err, subtraction.sub_zp, 'ptf48r', 'ab' from subtraction LEFT OUTER JOIN (select candidate.sub_id, candidate.flux, candidate.flux_err from candidate where q3c_radial_query(candidate.ra, candidate.dec, 199.95694049, 41.983848597, 0.00083)) cand on subtraction.id=cand.sub_id where (subtraction.ujd > 2455317.5 and subtraction.ujd < 2455501.5 and subtraction.filter='R') and q3c_poly_query( 199.95694049, 41.983848597, ARRAY[subtraction.ra_ll, subtraction.dec_ll, subtraction.ra_ul, subtraction.dec_ul, subtraction.ra_ur, subtraction.dec_ur, subtraction.ra_lr, subtraction.dec_lr]) order by subtraction.ujd asc;", (str(i),))
	print cur.query

	print 'Query Done'
	m=cur.fetchall()
	cur.close()
	m=np.array(m)
	#print 'length of array:', len(m[:,1]), len(m[:,2])
	#print m[:,1].astype(float),m[:,2].astype(float)
	print l[i]
	np.save(str(l[i])+'_LC',m)

'''
plt.scatter(m[:,1].astype(float),m[:,2].astype(float))
plt.errorbar(m[:,1].astype(float),m[:,2].astype(float),yerr=m[:,3].astype(float), linestyle="None")
plt.gca().invert_yaxis()
#plt.savefig('11fe_LC.png')
plt.show()
'''
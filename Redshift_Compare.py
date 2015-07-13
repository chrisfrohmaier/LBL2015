import matplotlib, psycopg2
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

##Connecting to the Database
conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()

cur.execute("SELECT distinct(specinfo.ptfname), sncosmo_fits.redshift, sncosmo_fits.redshift_err, specinfo.redshift from sncosmo_fits join specinfo on sncosmo_fits.ptfname=specinfo.ptfname;")
print 'Database Query Complete'
m=cur.fetchall()
cur.close()
m=np.array(m)
print max(m[:,3])
print len(m[:,3].astype(float))
print len(m[:,1].astype(float))
print len(np.subtract(m[:,3].astype(float),m[:,1].astype(float)))
plt.scatter(m[:,3].astype(float), np.subtract(m[:,3].astype(float),m[:,1].astype(float)))
plt.errorbar(m[:,3].astype(float), np.subtract(m[:,3].astype(float),m[:,1].astype(float)), yerr=m[:,2].astype(float), linestyle='None')
# coefficients=np.polyfit(m[:,3].astype(float), m[:,1].astype(float), deg=1)
# poly=np.poly1d(coefficients)
# ys=poly(m[:,3].astype(float))
# plt.plot(m[:,3].astype(float), ys, label='Linear Fit')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Spectroscopic Redshift')
plt.ylabel(r'$\Delta$ z')
plt.title("Spectroscopically confirmed Ia between May and October \n with more than 20 epochs")

fig = matplotlib.pyplot.gcf()
#plt.axis('equal')
#plt.legend()
fig.set_size_inches(10,7)
#plt.xlim(0.,0.35)
#plt.ylim(0.,0.35)
plt.savefig('Redshift_Eff.png', dpi=150)
plt.savefig('All_Ia_Gr8_20.png', dpi=150, bboxinches='tight')
plt.show()
import matplotlib, psycopg2
matplotlib.use('Agg')
import numpy as np
import astropy
import sncosmo
import matplotlib.pyplot as plt
from mpi4py import MPI
flat_cols=['#1abc9c','#2ecc71','#3498db','#9b59b6','#34495e','#f39c12','#d35400','#c0392b','#7f8c8d']

conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()

cur.execute("SELECT inout_sn_mc.sn_id, inout_sn_mc.peak_date, inout_sn_mc.ab_magb, inout_sn_mc.redshift, inout_sn_mc.x1, inout_fit.redshift, inout_fit.t0, inout_fit.x1, inout_fit.abs_mag, inout_fit.redchi2 from inout_sn_mc JOIN inout_fit on inout_sn_mc.sn_id=inout_fit.snid;")
m=cur.fetchall()
m=np.array(m)

'''
m[:,0] snid
---Input---
m[:,1]	peakdate
m[:,2]	ab_magb
m[:,3]	redshift
m[:,4]	x1
---Output---
m[:,5]	redshift
m[:,6]	t0
m[:,7]	x1
m[:,8]	abs_mg
m[:,9]	chi2
'''

t0_diff=m[:,1]-m[:,6]
ab_mg_diff=m[:,2]-m[:,8]
z_diff=m[:,3]-m[:,5]
x1_diff=m[:,4]-m[:,7]

plt.suptitle("Differences between the input supernovae and the output supernovae\n Input-Output. Total number of Objects: "+str(len(m[:,0])))
plt.subplot(2,2,1)
plt.xlabel('Peak Date Difference (Days)')
plt.hist(t0_diff, bins=100, color=flat_cols[0])

plt.subplot(2,2,2)
plt.xlabel('Absolute Magnitude DIfference (B Band)')
plt.hist(ab_mg_diff, bins=100, color=flat_cols[2])

plt.subplot(2,2,3)
plt.xlabel('Redshift Difference')
plt.hist(z_diff, bins=100, color=flat_cols[4])

plt.subplot(2,2,4)
plt.xlabel('X1 Difference')
plt.hist(x1_diff, bins=100, color=flat_cols[6])

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15,12)

plt.savefig('../LC_Fitter/Sim_LC_Fit/Diff_Hist.png', dpi=300, bbox_inches='tight')
plt.close()
import numpy as np
import psycopg2
from astropy.time import Time

name, obsdate, telescope, instrument, class_type, redshift, phase=np.loadtxt('specinfo.txt', usecols=(0,1,2,3,4,5,6), comments='#', delimiter=',', dtype=('S6,S32,S32,S32,S8,<f8,<f8'), unpack=True)

phase_mask=np.isnan(phase)
phase[phase_mask]=99999

z_mask=np.isnan(redshift)
redshift[z_mask]=99999
t=Time(obsdate, format='iso', scale='utc')
obsd=t.jd
conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()

for i in range(0,len(name)):
	print name[i], obsd[i], telescope[i], instrument[i], class_type[i], redshift[i], phase[i]
	cur.execute("INSERT INTO specinfo (ptfname, obsdate, telescope, instrument, type, redshift, phase) VALUES (%s,%s,%s,%s,%s,%s,%s)", (name[i], obsd[i], telescope[i], instrument[i], class_type[i], redshift[i], phase[i]))

conn.commit()

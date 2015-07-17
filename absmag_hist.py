import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()

cur.execute("SELECT abs_mag from sncosmo_fits;")
n=cur.fetchall()
n=np.array(n)

cur.execute("SELECT abs_mag from sncosmo_fits where  pass_cut=True;")
m=cur.fetchall()
m=np.array(m)

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1) 
                                              

plt.hist(n[:,0].astype(float), bins=100, label='Total Sample', color='#9b59b6')
plt.hist(m[:,0].astype(float), bins=100, label='Passed Selection', color='#2ecc71')
plt.xlabel(r'Absolute Magnitude (B Band)')
plt.ylabel('Number')
plt.legend()
plt.title(r'Absolute Magnitude Distribution')
plt.savefig('../LC_Fitter/abs_mag_hist.png', dpi=150)
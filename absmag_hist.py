import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()

cur.execute("SELECT abs_mag from sncosmo_fits where abs_mag>-22;")
n=cur.fetchall()
n=np.array(n)

cur.execute("SELECT abs_mag from sncosmo_fits where  pass_cut=True and abs_mag>-22;")
m=cur.fetchall()
m=np.array(m)


                                              
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(7,5)
plt.hist(n[:,0].astype(float), bins=30, label='Total Sample', color='#3498db')
plt.hist(m[:,0].astype(float), bins=30, label='Passed Selection', color='#7f8c8d')
plt.xlabel(r'Absolute Magnitude (B Band)')
plt.ylabel('Number')
plt.legend()
plt.title(r'Absolute Magnitude Distribution')
plt.savefig('../LC_Fitter/abs_mag_hist.png', dpi=150)
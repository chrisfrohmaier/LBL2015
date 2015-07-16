import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()

cur.execute("SELECT redchi2 from sncosmo_fits where redchi2>0 and redchi2<50;")
n=cur.fetchall()
n=np.array(n)

cur.execute("SELECT redchi2 from sncosmo_fits where redchi2>0 and redchi2<50 and pass_cut=True;")
m=cur.fetchall()
m=np.array(m)

plt.hist(n[:,0].astype(float), bins=100, label='Total Sample')
plt.hist(m[:,0].astype(float), bins=100, label='Passed Selection')
plt.xlabel(r'Reduced $\chi^2$')
plt.ylabel('Number')
plt.title(r'Reduced $\chi^2$ Fits')
plt.savefig('../LC_Fitter/Chisq_hist.png', dpi=150)
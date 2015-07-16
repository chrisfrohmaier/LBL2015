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

fig = plt.figure()                                                               
ax = fig.add_subplot(1,1,1) 
major_ticks = np.arange(0, 51, 5)                                              
minor_ticks = np.arange(0, 51, 1)                                               

ax.set_xticks(major_ticks)                                                       
ax.set_xticks(minor_ticks, minor=True)                                           
ax.set_yticks(major_ticks)                                                       
ax.set_yticks(minor_ticks, minor=True) 
plt.hist(n[:,0].astype(float), bins=100, label='Total Sample', color='#9b59b6')
plt.hist(m[:,0].astype(float), bins=100, label='Passed Selection', color='#2ecc71')
plt.xlabel(r'Reduced $\chi^2$')
plt.xticks(range(0,len(x))rotation=45)
plt.ylabel('Number')
plt.legend()
plt.title(r'Reduced $\chi^2$ Fits')
plt.savefig('../LC_Fitter/Chisq_hist.png', dpi=150)
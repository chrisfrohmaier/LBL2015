import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()

cur.execute("SELECT redchi2 from sncosmo_fits where redchi2>0 and redchi2<Inifinty;")
m=cur.fetchall()
m=np.array(m)

plt.hist(m[:,0].astype(float), bins=10)
plt.show()
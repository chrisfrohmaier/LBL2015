import matplotlib
matplotlib.use('Agg')
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
def Mid_Bins(arr):
	new_array=[]
	for i in range(len(arr)-1):
		new_array.append((arr[i]+arr[i+1])/2.0)

	return new_array
def SkewG(x,g,a,mu,s):
	efunc=erf((g*(x-mu))/(s*np.sqrt(2.)))
	return (a/(s*np.sqrt(2.*np.pi)))*(np.exp((-(x-mu)**2.)/(2.*s**2.)))*(1.+efunc)


conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()
cur.execute("SELECT color, found from sn_mc where colour_pass=True and ra<310;")
print 'Database Query Complete'
m=cur.fetchall()
cur.close()
m=np.array(m, dtype=np.float)

print len(m[:,0])

def Bin_Me(arr, final_bins):
	in_arrb, in_arr_bin=np.histogram(arr, bins=final_bins)
	in_arrb, in_arr_bin=np.histogram(arr, bins=in_arr_bin[in_arrb>0])

	return np.array(in_arr_bin)
'''
                              Table "public.sn_mc"

0 color
1 found

'''
flat_cols=['#1abc9c','#2ecc71','#3498db','#9b59b6','#34495e','#f39c12','#d35400','#c0392b','#7f8c8d']

colour=m[:,0]
#mag_bin=np.linspace(min(mag),max(mag),20)
hist,bine=np.histogram(m[:,0], bins=np.linspace(-0.2,0.4,100))
center = (bine[:-1] + bine[1:]) / 2
#, color=flat_cols[1], label='Simulated Sample Distribution', normed=True
hist=hist/max(hist)
plt.bar(center, hist, align='center', color=flat_cols[1], label='Simulated Sample Distribution')

bins=np.linspace(-0.2,0.4,10000)
skewa=[SkewG(x,1.8192627275,0.997793919871,-0.105487431764,0.117890808366) for x in Mid_Bins(bins)]
#skewa=skewa/max(skewa)

plt.plot(Mid_Bins(bins),skewa, label='Betoule Distribution', color=flat_cols[7])


plt.ylabel('Number')
plt.xlabel('Colour')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15,10)
plt.savefig('Post_Bin_Colour_Dist.png', dpi=150, bbox_inches='tight')
#plt.show()
plt.close()
import matplotlib
#matplotlib.use('Agg')
import psycopg2
import numpy as np
from scipy.special import erf

def Mid_Bins(arr):
	new_array=[]
	for i in range(len(arr)-1):
		new_array.append((arr[i]+arr[i+1])/2.0)

	return new_array



def SkewG(x,g,a,mu,s):
	efunc=erf((g*(x-mu))/(s*np.sqrt(2.)))
	return (a/(s*np.sqrt(2.*np.pi)))*(np.exp((-(x-mu)**2.)/(2.*s**2.)))*(1.+efunc)




print 'Skewx', 'SkewFrac', 'Mean C', 'Count', 'Number Pass'
def Update_DB_from_Color_Data(lowc, highc, mskewa):
	#print lowc, highc
	conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
	cur = conn.cursor()
	cur.execute("SELECT COUNT(*) from sn_mc where (color >%s and color <%s);",((float(lowc),float(highc),)))
	count=cur.fetchone()[0]
	x=np.mean((lowc,highc))
	skewx=SkewG(x,1.8192627275,0.997793919871,-0.105487431764,0.117890808366)
	print skewx, skewx/mskewa, x, count, int(count*skewx/mskewa)
	setF=int(count)-int(count*skewx/mskewa)
	cur.execute("UPDATE sn_mc SET colour_pass = False WHERE sn_id IN (SELECT sn_id from sn_mc where (color >%s and color <%s) limit %s);",((float(lowc),float(highc),int(setF),)) )
	cur.commit()
	print 'Done Colour :', x
#m=query_db()
bins=np.linspace(-0.2,0.4,100)
skewa=[SkewG(x,1.8192627275,0.997793919871,-0.105487431764,0.117890808366) for x in Mid_Bins(bins)]
for i in range(len(bins)-1):
	Update_DB_from_Color_Data(bins[i],bins[i+1], max(skewa))

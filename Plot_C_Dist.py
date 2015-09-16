import matplotlib
matplotlib.use('Agg')
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
def Mid_Bins(arr):
	new_array=[]
	for i in range(len(arr)-1):
		new_array.append((arr[i]+arr[i+1])/2.0)

	return new_array
conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier')
cur = conn.cursor()
cur.execute("SELECT peak_date, ra, dec, ab_magb, redshift, x1, color, int_dis, found from sn_mc where ra<310;")
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

0 peak_date
1 ra
2 dec
3 ab_magb
4 redshift
5 x1
6 color
7 int_dis
8 found
'''
flat_cols=['#1abc9c','#2ecc71','#3498db','#9b59b6','#34495e','#f39c12','#d35400','#c0392b','#7f8c8d']

colour=m[:,6]
#mag_bin=np.linspace(min(mag),max(mag),20)
n_colour, bin_colour= np.histogram(colour, bins=100) #Binning ALL the data for a 'Total' array
n_colour2, bin_colour2 = np.histogram(colour[m[:,8]==True], bins=bin_colour) #Binning the succesfully recovered dat
a
n_colour_eff=np.divide(n_colour2.astype(float), n_colour.astype(float)) #Successful divided by total gives efficien
cy in each bin
tot_colour, totbm =np.histogram(colour, bins=bin_colour)
tot_err=np.divide(1., np.sqrt(tot_colour))

#plt.plot(Mid_Bins(colour_bin), n_colour_eff)
plt.errorbar(Mid_Bins(bin_colour), n_colour_eff, yerr=tot_err, color=np.random.choice(flat_cols))
#plt.xticks(np.arange(min(colour),max(colour)+0.01, 0.01))
plt.yticks(np.arange(0, 0.51, 0.1))
#for the minor ticks, use no labels; default NullFormatter
plt.ylim(0,0.5)
plt.xlim(min(colour),max(colour))
plt.axhline(0.5, color='black', linestyle='dashed', linewidth=0.5)
plt.title('colour fraction Recovery')
plt.ylabel('Fraction Recovered')
plt.xlabel('Colour')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15,10)
plt.savefig('Post_Bin_Colour_Dist.png', dpi=150, bbox_inches='tight')
#plt.show()
plt.close()
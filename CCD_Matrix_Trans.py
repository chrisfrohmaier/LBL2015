import matplotlib
#matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from descartes.patch import PolygonPatch
import psycopg2

conn = psycopg2.connect(host='srv01050.soton.ac.uk', user='frohmaier', password='rates', database='frohmaier') #Connecting to the SN Group Server Database
cur = conn.cursor()

cur.execute("SELECT ra_ll,dec_ll,ra_ul,dec_ul,ra_ur,dec_ur,ra_lr,dec_lr, ccdid from geo_sub where new_filename like 'PTF201104283317_2_o_45544%';") #Get Everything from the Subtraction Table
k=cur.fetchall()
cur.close()
k=np.array(k)
print len(k)
def ccd0_defect(m, alpha):
	bad_area=[]
	xt=m[0]
	yt=m[1]

	jonest=np.matrix([[np.cos(alpha), -np.sin(alpha), xt], [np.sin(alpha), np.cos(alpha), yt], [0,0,1.]])
	good_area=[]

	xl_scale=abs(m[0]-m[2])/((2048*1.01)/3600)
	xr_scale=abs(m[4]-m[6])/((2048*1.01)/3600)
	
	
	for i in [(147.*xl_scale,1501.), (177.*xl_scale,1501.),(177.*xr_scale,4097.),(147.*xr_scale,4097.)]:
		x=i[0]*(1.01/3600)
		y=i[1]*(1.01/3600)
		pixel_coords=np.matrix([[x],[y],[1.]])
		#print 'Pixel:', pixel_coords
		cords=np.dot(jonest,pixel_coords)
		#print 'New Coordinates:', cords
		#print cords.item(0)
		bad_area.append((cords.item(0),yt-(cords.item(1)-yt)))
	return Polygon(bad_area)
	

for m in k:
	sima=Polygon([(m[0],m[1]),(m[2],m[3]),(m[4],m[5]),(m[6],m[7])])
	simp = PolygonPatch(sima, facecolor='k',zorder=0)
	plt.gca().add_patch(simp)

	alpha=np.arctan(np.divide((np.subtract(m[6],m[0])),np.subtract(m[7],m[1])))

	
	
	#print alpha

	xt=m[0]
	yt=m[1]

	jonest=np.matrix([[np.cos(alpha), -np.sin(alpha), xt], [np.sin(alpha), np.cos(alpha), yt], [0,0,1.]])
	#print jonest

	good_area=[]

	xl_scale=abs(m[0]-m[2])/((2048*1.01)/3600)
	xr_scale=abs(m[4]-m[6])/((2048*1.01)/3600)
	
	yu_scale=abs(m[3]-m[5])/((4096*1.01)/3600)
	
	yl_scale=abs(m[1]-m[7])/((4096*1.01)/3600)

	for i in [(75.*xl_scale,75.), (2048.*xl_scale,75.),(2048.*xr_scale,4000.),(75.*xr_scale,4000.)]:
		x=i[0]*(1.01/3600)
		y=i[1]*(1.01/3600)
		pixel_coords=np.matrix([[x],[y],[1.]])
		#print 'Pixel:', pixel_coords
		cords=np.dot(jonest,pixel_coords)
		#print 'New Coordinates:', cords
		#print cords.item(0)
		good_area.append((cords.item(0),yt-(cords.item(1)-yt)))
	
	garea=[good_area[0], (good_area[1][0],m[3]-((75.*1.01/3600)*yu_scale)), (good_area[2][0],m[5]+((96.*1.01/3600)*yu_scale)),good_area[3]]
	#print good_area
	#print [(m[0],m[1]),(m[2],m[3]),(m[4],m[5]),(m[6],m[7])]
	simg1=Polygon(garea)
	if m[8]==0:
		simb0=ccd0_defect(m, alpha)
		simg1=simg1.difference(simb0)
	simg2 = PolygonPatch(simg1, facecolor='blue',zorder=5)
	plt.gca().add_patch(simg2)
plt.xlim(120,360)
plt.ylim(-20,85)
plt.xlabel('Ra')
plt.ylabel('Dec')
plt.show()
'''
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(30,15)
plt.savefig('CCD_Areas.png', bbox_inches='tight', dpi=300)
'''
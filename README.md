# LBL2015
Work at LBL in July 2015

##Tasks
1. Check sample of Ias is correct
	
	- Is there a better way of doing this?

<del>2. Get the photometric sample. i.e. Those non-spec confirmed, where is this table?</del>

	- <del>If there is a host assocoiated with these then look up in  NED/SDSS for the host redshift</del>

3. Find objects we have missed.

##Finding missing objects
For all candidates in PTF between 1st May and 31st October we scan through the database. We apply our selection cuts of ebv<0.1, filter='R', is_sdss=True.

There are about 33 million candidates remaining and we store an id and an average ra and dec for each.

Next we create a new table of all these objects observed more than 4 times.

##Reduced Chi Squared distribution for the SALT2.4 fits to real-time photometry

![ScreenShot](https://dl.dropboxusercontent.com/u/37570643/LBL_July2015/Chisq_hist.png)

##Selection Criteria
Changing the criteria to 5 points cuts Peter's list of missed candidate potentials in half.

If we enforce a new slection crieteria of at least 5 detections with 2 points before peak and 2 points after peak than we should be good.

##Abs Mag distribution 
Just looking at the spectroscopically confirmed SNe Ias I plot their absolute B band magnitude (Bessell B filter) as taken from sncosmo's best fit SALT2.4 lightcurve.

![ScreenShot](https://dl.dropboxusercontent.com/u/37570643/LBL_July2015/abs_mag_hist.png)

and those with a reduced $\chi^2$ < 5

![ScreenShot](https://dl.dropboxusercontent.com/u/37570643/LBL_July2015/abs_mag_hist_red5.png)

##PTFNAME but no Type
We want to answer the question about whether there is anything in with a ptfname but not a spectroscopic classification of a Ia. 

We queried the database for objects classified as a supernova but no further typing. We found a handful of objects and typed each one by hand. There were no Ias. 

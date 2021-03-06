# LBL2015
Work at LBL in July 2015

##Tasks
1. Check sample of Ias is correct ✔
	
	- Is there a better way of doing this? ✔
     - *Answer*: Yes, check the database for the new table `ptftrans` we created. This holds a more complete list of transients, even with sub-classes and sub-sub-classes.

2. <del>Get the photometric sample. i.e. Those non-spec confirmed, where is this table?</del> ✔

	-  <del>If there is a host assocoiated with these then look up in  NED/SDSS for the host redshift</del>

3. Find objects we have missed.

##<del>Finding missing objects</del>

**Note** We have chaged our method, further down the page is an updated method descibed in population bcand.

<del>For all candidates in PTF between 1st May and 31st October we scan through the database. We apply our selection cuts of ebv\<0.1, filter='R', is_sdss=True</del>

<del>There are about 33 million candidates remaining and we store an id and an average ra and dec for each.</del>

<del>Next we create a new table of all these objects observed more than 4 times.</del>

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

We queried the database for objects classified as a supernova but no further typing. We found a handful of objects and typed each one by hand. 

**There were no SNe Ia**

##Comparing the Simulated Ia input parameters to the sncosmo fits
I have created 3 new tables on the sngroup server called `inout_*`. These tables store simulated light curve data.
The `inout_sn_mc` table it a replica of `sn_mc` and holds the simulated parameters and the found True/False statements.
`inout_lc` is a brand new table. This stores the light curve data, e.g. ujd, mag etc. We don't store this information for our actual simualtions as this would be too data heavy.
`inout_fit` stores the fit parameters from sncosmo on the data in `inout_lc`

We compare the `inout_sn_mc` and `inout_lc` tables to see if we can reliably recreate the input supernova parameters after they have been thorough our PTF simulated pipeline. I have performed this test for ~50,000 supernovae.
###Difference histograms
![ScreenShot](https://dl.dropboxusercontent.com/u/37570643/LBL_July2015/Diff_Hist.png)
###Reduced Chi Squared of fit to recovered lightcurve
![ScreenShot](https://dl.dropboxusercontent.com/u/37570643/LBL_July2015/inout_redchi2.png)

##Query for PTFNAME objects that match itself
We want to find objects in PTFNAME that might actually be the same object. We query the database using the following query.
```sql
SELECT t1.qname, t1.qra, t1.qdec, t2.pname, t2.pra, t2.pdec from (select ptfname.ptfname AS qname, avg(candidate.ra) AS qra, avg(candidate.dec) AS qdec from ptfname, candidate where ptfname.candidate_id=candidate.id and type=3 group by qname) t1, (select ptfname.ptfname AS pname, avg(candidate.ra) AS pra, avg(candidate.dec) AS pdec from ptfname, candidate where ptfname.candidate_id=candidate.id and type=3 group by pname) t2 where t1.qname!=t2.pname and q3c_join(t1.qra, t1.qdec, t2.pra, t2.pdec, 0.00277) order by qra, qdec;
```
We get this result:

|qname |       qra        |       qdec       | pname |       pra        |       pdec
|------|------------------|------------------|-------|------------------|------------------
|09hdp | 3.84656933463333 |     30.722037382 | 09hdo | 3.84656933463333 |     30.722037382
|11pab |     44.626592771 |    6.30684528065 | 11pdt | 44.6274125285714 | 6.30744218361429
|10bwo | 116.418055017727 | 28.2948363358182 | 09hhv | 116.419909386667 | 28.2963363073333
|11fzz |    167.694550345 | 54.1051724694792 | 11giv | 167.694718892292 | 54.1036960426875
|11aib |    176.351850232 |    68.9532739803 | 09gpu | 176.352413199375 | 68.9519071746875
|11cmw | 179.742309353448 | 56.1886503645173 | 11czq |      179.7401251 | 56.1881194361111
|11byi | 185.438112007727 | 12.2570977032727 | 11bpw | 185.437523118788 | 12.2579958847273
|10myz | 224.557215567273 | 48.1912054614546 | 10oet | 224.559480336667 | 48.1911150336667
|11gjh | 248.057541854737 | 38.6555497056316 | 12eog | 248.058205301053 | 38.6564946971053
|11khk |    255.741854236 |    40.5185381536 | 11klx | 255.741927199231 | 40.5204022789231
|12lee |     354.92284852 |    25.1098173968 | 12lcr |   354.9245924475 | 25.1103071780833

Some of these objects are repeat observations, others are unique supernovae that happened in the same galaxy. Was hoping for a lensed supernova :-(

#The B-Cand Table
##Criteria for object to pass our selection cut

- Observed between April 1st and Oct 31st
- Use all possible subtractions  with RB score ≥0.07
- in SDSS
- Colour excess ≤0.1
- no reference after April 1st
- (will with and withou the x,y cuts)

###Bcand Negative Subs table

- Same as above except with negative subtractions and without the RB score.

======

(with 2+ negative subs)

1. get_neg_list.f
2. get_list.f

- 2+ detections
- t_{sep} > 0.5days
- mag < 20.0 (all observations brighter than 20.)
- no match in neg list ($3 \sigma \sqrt(positive^2 + negative^2)$)

-----> This goes into making the `bcand` table

Further filters on `bcand`:

- no star association
- np ptftrans association
- has 1+ non-detection before RB
- will also store the first jd of detection

(smear will also contain the x,y edges effect + CTE if we choose to include it)

#Nugent's BCand README
We have run the following query to get the 2010 list of candidates which we may have missed in PTF.

select candidate.id, subtraction.ujd, candidate.ra, candidate.dec from candidate, subtraction, ptffield, rb_classifier, deep_ref whe
re candidate.sub_id=subtraction.id and subtraction.deep_ref_id=deep_ref.id and rb_classifier.candidate_id=candidate.id and subtracti
on.ptffield=ptffield.id and is_sdss='t' and color_excess <= 0.1 and (candidate.id > 119808777 and candidate.id < 366784713)  and sub
traction.filter='R' and realbogus >= 0.07 and candidate.mag < 20.0 and candidate.pos_sub='t' and deep_ref.date < '2010-04-01'  order
 by ra;

candidate.id = 119808777 corresponds to the first candidate on JD 2455288.95065    2010-04-02T10:48:56.160
candidate.id = 366784713 corresponds to the last candidate on  JD 2455501.05003    2010-10-31T13:12:02.531

To get the list of negative subs:

select candidate.id, subtraction.ujd, candidate.ra, candidate.dec from candidate, subtraction, ptffield, deep_ref where candidate.su
b_id=subtraction.id and subtraction.deep_ref_id=deep_ref.id and subtraction.ptffield=ptffield.id and is_sdss='t' and color_excess <=
 0.1 and (candidate.id > 119808777 and candidate.id < 366784713)  and subtraction.filter='R' and candidate.pos_sub='f' and deep_ref.
date < '2010-04-01'  order by ra;

  32,390,708 bcand_neg.dat
   6,736,371 bcand_pos.dat

Then run get_list_neg.f on the result bcand_neg.dat to get a (id, count, ra, dec, sig) where I require count >= 2

Then run get_list.f on the result bcand_pos.dat to get a (id, jd, count, sepcnt, ra, dec, sig) where I require 2+ detections (count)
 with 1+ detections separated by 0.49 days

Objects are considered a match if they are within 3"

on sgn02 compiled with g95 --free-form -O3 get_list.f

to get fort.22 which I move to positive.dat
to get fort.23 which I move to negative.dat

443,631 cands in positive.dat
5,020,865 cand in negative.dat

Tables are created:

\i bcand_pos.schema
\i bcand_neg.schema

and inserted:

\copy bcand_pos (cand_id, jd, count, sepcnt, ra, dec, sig) from '/project/projectdirs/deepsky/rates/missed/code/positive.dat'  delim
iter ',';
\copy bcand_neg (cand_id, count, ra, dec, sig) from '/project/projectdirs/deepsky/rates/missed/code/negative.dat'  delimiter ',';

Now fix the sig's in case they are too low:

update bcand_pos set sig=0.5 where sig < 0.5;
update bcand_neg set sig=0.5 where sig < 0.5;

And now create sig3:

update bcand_pos set sig3 = sig/3600\*3;
update bcand_neg set sig3 = sig/3600\*3;

'''Code to take an input ra and dec and return an array which holds the light curve
Or instead of an array input the result into a new table
'''
import sys
import psycopg2

ra=sys.argv[1]
dec=sys.argv[2]

print 'Ra, Dec: ', ra, dec

con = psycopg2.connect(host='scidb2.nersc.gov', user='subptf', password='p33d$kyy', database='subptf')
cur = con.cursor()
print 'Nersc DB being Queried'

cur.execute("SELECT subtraction.ujd, subtraction.deep_ref_id, coalesce(cand.flux,-9999) AS flux, coalesce(cand.flux_err,-9999) AS fluxerr, coalesce(cand.rb,-9999) as RB, subtraction.sub_zp, subtraction.lmt_mg_new, subtraction.ub1_zp_new from subtraction LEFT OUTER JOIN (select candidate.sub_id, candidate.id as id, candidate.flux, candidate.flux_err, candidate.pos_sub AS pos_sub, rb_classifier.realbogus AS rb from candidate, rb_classifier where q3c_radial_query(candidate.ra, candidate.dec, %s, %s, 0.0002) and candidate.id=rb_classifier.candidate_id and rb_classifier.realbogus>0.07) cand on subtraction.id=cand.sub_id where (subtraction.ujd > 2455287.5 and subtraction.ujd < 2455501.5) and q3c_poly_query( %s, %s, ARRAY[subtraction.ra_ll, subtraction.dec_ll, subtraction.ra_ul, subtraction.dec_ul, subtraction.ra_ur, subtraction.dec_ur, subtraction.ra_lr, subtraction.dec_lr]) order by subtraction.ujd asc;",(float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[1]),float(sys.argv[2]),))
m=cur.fetchall()
print m
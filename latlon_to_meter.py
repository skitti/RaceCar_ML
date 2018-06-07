##############################################################################
from math import sin, cos, radians, asin, sqrt 

def distance_traveled(row):
    lon1 = row['MA_GPSLONG_SHIFT']
    lat1 = row['MA_GPSLAT_SHIFT']
    #print(row)
    lon2 = row['MA_GPSLONG']
    lat2 = row['MA_GPSLAT']
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    meter = 6367 * c *1000
    #print(km)
    return meter
    
def distance_traveled2(lon1,lat1,lon2,lat2):

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    meter = 6367 * c *1000
    #print(km)
    return meter
	
def distance_traveled_lol(row):
	x1 = row['MA_GPSLONG_SHIFT']
	y1 = row['MA_GPSLAT_SHIFT']
	
	x2 = row['MA_GPSLONG']
	y2 = row['MA_GPSLAT']
	d  = sqrt(x2**2+y2**2)-sqrt(x1**2+y1**2) 
	return d
	
def distance_traveled_lol2(x1,y1,x2,y2):
	d  = sqrt(x2**2+y2**2)-sqrt(x1**2+y1**2) 
	return d
	
#!/usr/bin/env python

"""
Experimenting with SV simulations

Approximations:
  * When an exposure is split (often), it just uses airmass and conditions
    of start of exposure, not for each individual exposure
  * Doesn't handle exposures that start under one obscondition (e.g. DARK)
    and then transition to another
  * Doesn't max out requested number of tiles for a program
"""

import os, sys, re
import datetime
import random

import numpy as np
import fitsio

from astropy.time import Time
import astropy.units as u
import astropy.coordinates
from astropy.table import Table, vstack

from desisurvey.ephem import Ephemerides, get_object_interpolator
from desisurvey.etc import exposure_time
from surveysim.weather import Weather
from desiutil.depend import add_dependencies

import argparse

parser = argparse.ArgumentParser(usage = "{prog} [options]")
# parser.add_argument("-i", "--input", type=str,  help="input data")
parser.add_argument("-o", "--outdir", type=str,  help="input data", required=True)
parser.add_argument("--firstnight", type=str,  help="first night YEARMMDD", default='20200101')
parser.add_argument("--numnights", type=int, help="first night YEARMMDD", default=31)
parser.add_argument("--randseed", type=int, help="random seed", default=1)
parser.add_argument("--debug", action="store_true", help="start ipython at end of run")

args = parser.parse_args()

#- Input sanity checks
if not re.match('20\d{6}', args.firstnight):
    print('--firstnight should be YEARMMDD')
    sys.exit(1)

#- Reproducibly random
random.seed(args.randseed)
np.random.seed(args.randseed)

#- Assume input tiles come with this code
indir = os.path.abspath(os.path.dirname(__file__))

LQ4_tiles = Table.read(indir+'/BRIGHT_LRG+QSO_SV_10_4x_superset20.fits')
LQ4_tiles.meta['MAXTILES'] = 10
LQ4_tiles.meta['DONETILE'] = 0
LQ4_tiles.meta['PROGRAM'] = 'SV_LQ4'
LQ4_tiles['PROGRAM'] = 'SV_LQ4'
LQ4_tiles['GOAL_DARK'] = 4
LQ4_tiles['GOAL_GRAY'] = 0
LQ4_tiles['GOAL_BRIGHT'] = 0
LQ4_tiles['NUMOBS_DARK'] = 0
LQ4_tiles['NUMOBS_GRAY'] = 0
LQ4_tiles['NUMOBS_BRIGHT'] = 0

LQ8_tiles = Table.read(indir+'/FAINT_LRG+QSO_SV_10_8x_superset20.fits')
LQ8_tiles.meta['MAXTILES'] = 10
LQ8_tiles.meta['DONETILE'] = 0
LQ8_tiles.meta['PROGRAM'] = 'SV_LQ8'
LQ8_tiles['PROGRAM'] = 'SV_LQ8'
LQ8_tiles['GOAL_DARK'] = 8
LQ8_tiles['GOAL_GRAY'] = 0
LQ8_tiles['GOAL_BRIGHT'] = 0
LQ8_tiles['NUMOBS_DARK'] = 0
LQ8_tiles['NUMOBS_GRAY'] = 0
LQ8_tiles['NUMOBS_BRIGHT'] = 0

QSO_tiles = Table.read(indir+'/QSO_SV_10_4x_superset19.fits')
QSO_tiles.meta['MAXTILES'] = 10
QSO_tiles.meta['DONETILE'] = 0
QSO_tiles.meta['PROGRAM'] = 'SV_QSO'
QSO_tiles['PROGRAMNAME'] = 'SV_QSO'
QSO_tiles['GOAL_DARK'] = 4
QSO_tiles['GOAL_GRAY'] = 0
QSO_tiles['GOAL_BRIGHT'] = 0
QSO_tiles['NUMOBS_DARK'] = 0
QSO_tiles['NUMOBS_GRAY'] = 0
QSO_tiles['NUMOBS_BRIGHT'] = 0

ELG_tiles = Table.read(indir+'/ELG_SV_25_4x_superset50.fits')
ELG_tiles.meta['MAXTILES'] = 25
ELG_tiles.meta['DONETILE'] = 0
ELG_tiles.meta['PROGRAM'] = 'SV_ELG'
ELG_tiles['PROGRAMNAME'] = 'SV_ELG'
ELG_tiles['GOAL_DARK'] = 1
ELG_tiles['GOAL_GRAY'] = 3
ELG_tiles['GOAL_BRIGHT'] = 0
ELG_tiles['NUMOBS_DARK'] = 0
ELG_tiles['NUMOBS_GRAY'] = 0
ELG_tiles['NUMOBS_BRIGHT'] = 0

BGS_tiles = Table.read(indir+'/BGS_SV_30_3x_superset58.fits')
BGS_tiles.meta['MAXTILES'] = 30
BGS_tiles.meta['DONETILE'] = 0
BGS_tiles.meta['PROGRAM'] = 'SV_BGS'
BGS_tiles['PROGRAMNAME'] = 'SV_BGS'
BGS_tiles['GOAL_DARK'] = 1
BGS_tiles['GOAL_GRAY'] = 0
BGS_tiles['GOAL_BRIGHT'] = 3
BGS_tiles['NUMOBS_DARK'] = 0
BGS_tiles['NUMOBS_GRAY'] = 0
BGS_tiles['NUMOBS_BRIGHT'] = 0

MWS_tiles = Table.read(indir+'/MWS_SV_25_3x_superset50.fits')
MWS_tiles.meta['MAXTILES'] = 25
MWS_tiles.meta['DONETILE'] = 0
MWS_tiles.meta['PROGRAM'] = 'SV_MWS'
MWS_tiles['PROGRAMNAME'] = 'SV_MWS'
MWS_tiles['GOAL_DARK'] = 1
MWS_tiles['GOAL_GRAY'] = 0
MWS_tiles['GOAL_BRIGHT'] = 3
MWS_tiles['NUMOBS_DARK'] = 0
MWS_tiles['NUMOBS_GRAY'] = 0
MWS_tiles['NUMOBS_BRIGHT'] = 0

#- Weight programs by number of times they need to be observed in dark time
program_tiles = [LQ8_tiles, LQ4_tiles, QSO_tiles, ELG_tiles, BGS_tiles, MWS_tiles]
program_weights = np.array([8.0, 4, 4, 1, 1, 1])
program_weights /= np.sum(program_weights)

#- Load pre-calculated ephemeris
os.environ['DESISURVEY_OUTPUT'] = '/project/projectdirs/desi/datachallenge/surveysim2018/shared/'
# os.environ['DESISURVEY_OUTPUT'] = '/data/desi/surveysim/shared/'
start_date = datetime.date(2019, 1, 1)
stop_date = datetime.date(2025, 12, 31)
ephem = Ephemerides(start_date, stop_date, \
    restore=os.getenv('DESISURVEY_OUTPUT')+'/ephem_2019-01-01_2025-12-31.fits')

#- Load weather model
weather = Weather(seed=args.randseed)

#- Where is KPNO?
kpno_lon, kpno_lat, kpno_height = -111.6*u.deg, 31.964*u.deg, 2120*u.m
kpno = astropy.coordinates.EarthLocation.from_geodetic(kpno_lon, kpno_lat, kpno_height)

def separation(ra1, dec1, ra2, dec2):
    '''Returns angular separation in degrees, for ra1,dec1 and ra2,dec2 in degrees'''
    #- Haversine formula
    phi1, theta1 = np.radians(dec1), np.radians(ra1)
    phi2, theta2 = np.radians(dec2), np.radians(ra2)
    r = 2*np.arcsin(np.sqrt(np.sin(0.5*(phi2-phi1))**2 + np.cos(phi1)*np.cos(phi2)*np.sin(0.5*(theta2-theta1))**2))
    return np.degrees(r)

def get_airmass(tiles, obstime):
    """
    Return airmass of tiles (Table) at given obstime (astropy Time)
    """
    lst = obstime.sidereal_time(kind='mean', longitude=kpno_lon).to('deg').value
    
    d = separation(lst, kpno_lat.value, tiles['RA'], tiles['DEC'])

    #- Approximate airmass
    airmass = 1/np.cos(np.radians(d))
    airmass[d>80] = 100

    return airmass.data

def get_moon(moon, now):
    '''Return moonra, moondec, moonalt, moonfrac, using moon interpolator'''
    moonra, moondec = moon(now.mjd)
    lst = now.sidereal_time(longitude=kpno_lon, kind='mean').to('deg').value
    moonalt = 90-separation(moonra, moondec, lst, kpno_lat.to('deg').value)
    moonfrac = ephem.get_moon_illuminated_fraction(now.mjd)
    return moonra, moondec, moonalt, moonfrac

#- start and stop of this svsim run
year = args.firstnight[0:4]
month = args.firstnight[4:6]
day = args.firstnight[6:8]
svsim_start = Time('{}-{}-{}T18:00:00'.format(year, month, day), format='isot') + 7*u.hour
svsim_end = svsim_start + (args.numnights + 0.5)*u.day

#- Start "tonight" and "now" from the previous day to trigger initial
#- "new night" calculation at start of observing loop
tonight = svsim_start - 1*u.day
dawn_mjd = ephem.get_night(tonight)['dawn']
now = Time(dawn_mjd + 1e-3, format='mjd')

expid = 1000
max_exptime = 25*60     #- maximum exposure time in seconds

#- We'll fill this list with the observations
obslist = list()

while True:
    #- Are we done?
    if now > svsim_end:
        break

    #- New night
    if now.mjd > dawn_mjd:
        tonight += 1*u.day
        ephem_tonight = ephem.get_night(tonight)
        dusk_mjd = ephem_tonight['dusk']
        dawn_mjd = ephem_tonight['dawn']
        darkgraybright, darkgraybright_mjds = ephem.get_night_program(tonight)
        now = Time(dusk_mjd, format='mjd') + 1*u.s
        moon = get_object_interpolator(ephem_tonight, 'moon')
        tiles_observed_tonight = list()
        night = int((now - 7*u.hour).datetime.strftime('%Y%m%d'))
        print('Moving on to new night {}'.format(night))
        
        #- Add calibration exposures, starting at 4pm KPNO time
        dt = (now-7*u.hour-12*u.hour).datetime
        calibtime = Time(datetime.datetime(dt.year, dt.month, dt.day, 16, 0, 0))+7*u.hour            
        for calibtype in ['arc', 'flat']:
            for i in range(3):
                obs = dict(NIGHT=night, EXPID=expid, TILEID=-1, PASS=-1, RA=0, DEC=32, EBMV=0,
                           PROGRAM='CALIB', FLAVOR=calibtype, MJD=calibtime.mjd,
                           SEEING=1.0, TRANSPARENCY=1.0, AIRMASS=1.0,
                           MOONALT=-90, MOONSEP=180, MOONFRAC=0,
                           EXPTIME=10.0, TOTEXPTIME=10.0, CONDITIONS='DAY',
                           LST=0.0,
                          )
                expid += 1
                calibtime += 2*u.min
                obslist.append(obs)

    w = weather.get(now)
    if w['open'] and w['transparency']>0.3 and w['seeing']<2.5:    
        i = np.searchsorted(darkgraybright_mjds, now.mjd) - 1
        assert i>=0
        conditions = darkgraybright[i]
        print('{} KPNO {}'.format((now-7*u.hour).datetime.strftime('%Y-%m-%d %H:%M'), conditions))
    
        for tiles in np.random.choice(program_tiles, size=len(program_tiles),
                                      p=program_weights, replace=False):
            if tiles.meta['MAXTILES'] == tiles.meta['DONETILE']:
                continue

            if not np.any(tiles['GOAL_'+conditions]):
                print('  {} no {} obs needed'.format(tiles.meta['PROGRAM'], conditions.lower()))
                continue
            
            #- for BGS/MWS tiles under DARK conditions, use BRIGHT exptime calc
            if conditions == 'DARK' and (tiles is BGS_tiles or tiles is MWS_tiles):
                exptime_conditions = 'BRIGHT'
            else:
                exptime_conditions = conditions

            #- Guess exptime for airmass=1.8 tile under these conditions
            #- to minimize chasing a setting tile across airmass>2
            moonra, moondec, moonalt, moonfrac = get_moon(moon, now)
            exptime = exposure_time(
                program=exptime_conditions,
                seeing=w['seeing'],
                transparency=w['transparency'],
                airmass=1.8,
                EBV=0.03,
                moon_alt=moonalt,
                moon_sep=70.0, #- placeholder
                moon_frac=moonfrac)

            #----- Criteria for whether to consider a tile -----
            #- Visible airmass<2 for at least typical exposure time
            airmass1 = get_airmass(tiles, now)
            airmass2 = get_airmass(tiles, now+exptime)
            isVisible = (airmass1 < 2.0) & (airmass2 < 2.0)
            
            #- Will still be visible in a month sometime during the night
            futureOK = tiles['LAST_OBS_DATE'] > now.mjd + 30
            
            #- Need more obs under these conditions
            moreNeeded = tiles['GOAL_'+conditions] > tiles['NUMOBS_'+conditions]
            
            #- Haven't already observed this tile tonight
            okTonight = ~np.in1d(tiles['TILEID'], tiles_observed_tonight)
    
            jj = np.where(isVisible & futureOK & moreNeeded & okTonight)[0]
            print('  {} {} tiles available'.format(tiles.meta['PROGRAM'], len(jj)))

            if len(jj) > 0:                
                #- Prioritize re-selecting tiles that are already observed
                p = (3*tiles['NUMOBS_'+conditions][jj] + 1).astype(float)
                p /= np.sum(p)
                
                j = np.random.choice(jj, p=p)
                tileid = tiles['TILEID'][j]
                tiles['NUMOBS_'+conditions][j] += 1
                tiles_observed_tonight.append(tileid)

                #- Will this obs finish this tile?
                #- Remember that some tiles span DARK/GRAY/BRIGHT
                if (tiles['NUMOBS_DARK'][j] == tiles['GOAL_DARK'][j]) and \
                   (tiles['NUMOBS_GRAY'][j] == tiles['GOAL_GRAY'][j]) and \
                   (tiles['NUMOBS_BRIGHT'][j] == tiles['GOAL_BRIGHT'][j]):
                    tiles.meta['DONETILE'] += 1

                obs = dict()
                for key in ['TILEID', 'PASS', 'RA', 'DEC']:
                    obs[key] = tiles[key][j]
        
                obs['PROGRAM'] = tiles.meta['PROGRAM']
                obs['CONDITIONS'] = conditions
                obs['EBMV'] = tiles['EBV_MED'][j]
                obs['FLAVOR'] = 'science'
                obs['NIGHT'] = night
                obs['MJD'] = now.mjd
        
                obs['SEEING'] = w['seeing']
                obs['TRANSPARENCY'] = w['transparency']
                
                #- airmass in 10 min, approximately mid exposure
                obs['AIRMASS'] = get_airmass(tiles[j:j+1], now+10*u.min)[0]

                moonra, moondec, moonalt, moonfrac = get_moon(moon, now)
                moonsep = separation(moonra, moondec, tiles['RA'][j], tiles['DEC'][j])
                obs['MOONALT'] = moonalt
                obs['MOONSEP'] = moonsep
                obs['MOONFRAC'] = moonfrac

                print('  selected tile {} at airmass {:.2f}'.format(
                    tileid, obs['AIRMASS']))
                
                #- Recalculate exptime for this tile; convert to seconds
                exptime = exposure_time(
                    program=exptime_conditions,
                    seeing=obs['SEEING'],
                    transparency=obs['TRANSPARENCY'],
                    airmass=obs['AIRMASS'],
                    EBV=obs['EBMV'],
                    moon_frac=obs['MOONFRAC'],
                    moon_sep=obs['MOONSEP'],
                    moon_alt=obs['MOONALT']).to('s').value

                #- Approximate varying conditions, but don't try to handle
                #- program changes or re-calculating total exptime
                nexp = int(exptime / max_exptime) + 1
                single_exptime = exptime / nexp
                print('    {} exposures -> {:.1f} min total'.format(nexp, exptime/60))
                for i in range(nexp):
                    if i > 0:
                        w = weather.get(now)
                        obs['SEEING'] = w['seeing']
                        obs['TRANSPARENCY'] = w['transparency']
                        obs['AIRMASS'] = get_airmass(tiles[j:j+1], now+single_exptime/2 * u.s)[0]
                    
                    obs['EXPID'] = expid
                    obs['EXPTIME'] = single_exptime
                    obs['TOTEXPTIME'] = exptime
                    lst = now.sidereal_time(kind='mean', longitude=kpno_lon).to('deg').value
                    obs['LST'] = lst
                    obslist.append(obs.copy())
                    expid += 1
                    now += (exptime / nexp) * u.s

                    #- conservative inter-exposure overhead for same tile
                    if i < nexp-1:
                        now += 2*u.min

                #- found a tile for this timeslot; don't check other programs
                break

    #- Advance by 10 min; approximates long slews and observing inefficiencies
    now += 10*u.min
 
#- TODO: write outputs (exposures list, subset of tiles observed)

x = exposures = Table(obslist)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir, exist_ok=True)

exposures.meta['EXTNAME'] = 'EXPOSURES'
exposures.meta['RANDSEED'] = args.randseed
add_dependencies(exposures.meta)
outfile = os.path.join(args.outdir, 'sv_exposures.fits')
exposures.write(outfile, overwrite=True)
print('Wrote {}'.format(outfile))

for tiles in program_tiles:
    filename = os.path.join(args.outdir, tiles.meta['PROGRAM'] + '-tiles.fits')
    ii = (tiles['NUMOBS_DARK'] > 0) | (tiles['NUMOBS_GRAY'] > 0) | (tiles['NUMOBS_BRIGHT'] > 0)
    tiles.meta['EXTNAME'] = 'TILES'
    tiles.meta['RANDSEED'] = args.randseed
    add_dependencies(tiles.meta)
    tiles[ii].write(filename, overwrite=True)
    print('Wrote {}'.format(filename))

#- Create concatentation of tiles
for tiles in program_tiles:
    del tiles.meta['PROGRAM']
    del tiles.meta['DONETILE']
    del tiles.meta['MAXTILES']

tiles = vstack(program_tiles)
ii = (tiles['NUMOBS_DARK'] > 0) | (tiles['NUMOBS_GRAY'] > 0) | (tiles['NUMOBS_BRIGHT'] > 0)
filename = os.path.join(args.outdir, 'SV-tiles.fits')
tiles[ii].write(filename, overwrite=True)
print('Wrote {}'.format(filename))

if args.debug:
    import IPython
    IPython.embed()

    
    

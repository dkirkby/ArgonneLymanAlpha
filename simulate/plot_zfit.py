#!/usr/bin/env python

import astropy.io.fits as pyfits
import numpy as np
import sys
import pylab # for debugging
import math

if len(sys.argv)<2 :
    print sys.argv[0],"results.fits"
    sys.exit(1)

h=pyfits.open(sys.argv[1])
print h[1].columns.names
t=h[1].data

ok=np.where(t["BEST_Z"]!=0)[0] # in case it's still running

a0=pylab.subplot(1,2,1)
tz=t["TRUE_Z"][ok]
dz=t["BEST_Z"][ok]-t["TRUE_Z"][ok]
ez=t["BEST_Z_ERR"][ok]
#a0.errorbar(tz,dz,ez,fmt="o",color="gray",alpha=0.4)
a0.plot(tz,dz,"o",color="gray",alpha=0.4)
a0.set_xlabel("redshift")
a0.set_ylabel(r"$\Delta z$")

zreq=np.linspace(0.5,4.,10)
a0.plot(zreq,0*zreq,'-.',c='k')
requirement=0.0025*(1+zreq)
a0.plot(zreq,requirement,'--',c='k')
a0.plot(zreq,-requirement,'--',c='k')
requirement=0.0004*(1+zreq)
a0.plot(zreq,requirement,'--',c='k')
a0.plot(zreq,-requirement,'--',c='k')

if 1 :
    bins=np.linspace(0.45,3.55,10)
    h1,junk=np.histogram(tz,bins=bins)
    if np.sum(h1>1)>0 :
        hz,junk=np.histogram(tz,bins=bins,weights=dz)
        hz2,junk=np.histogram(tz,bins=bins,weights=dz**2)
        mean=hz/(h1+(h1==0))
        rms=np.sqrt(hz2/(h1+(h1==0))-mean**2)
        err=rms/np.sqrt(h1-1+10*(h1<=1))
        nodes=bins[:-1]+(bins[1]-bins[0])/2.
        xerr=np.gradient(bins)[:-1]/2.
        a0.errorbar(nodes[h1>1],mean[h1>1],yerr=rms[h1>1],xerr=xerr[h1>1],fmt="o",color="red",lw=2)
        a0.errorbar(nodes[h1>1],mean[h1>1],yerr=err[h1>1],xerr=xerr[h1>1],fmt="o",color="k",lw=2)
a0.text(1.5,0.025,r"requirement:")
a0.text(1.5,0.022,r"$\sigma z < 0.0025 (1+z)$")
a0.text(1.5,0.019,r"$\bar{\Delta z} < 0.0004 (1+z)$")

gmag_range=[22-0.125,23+0.125]

a1=pylab.subplot(1,2,2)
ok=np.where(t["BEST_Z"]!=0)[0]
dz=t["BEST_Z"][ok]-t["TRUE_Z"][ok]
ez=t["BEST_Z_ERR"][ok]
gmag=t["GMAG"][ok]
#a1.errorbar(gmag,dz,ez,fmt="o",color="gray",alpha=0.4)
a1.plot(gmag,dz,"o",color="gray",alpha=0.4)
a1.plot(gmag_range,[0,0],'-.',c='k')
requirement=0.0025*(1+2.)
a1.plot(gmag_range,[requirement,requirement],'--',c='k')
a1.plot(gmag_range,[-requirement,-requirement],'--',c='k')
requirement=0.0004*(1+2.)
a1.plot(gmag_range,[requirement,requirement],'--',c='k')
a1.plot(gmag_range,[-requirement,-requirement],'--',c='k')



if 1 :
    bins=np.linspace(22-0.125,23+0.125,6)
    h1,junk=np.histogram(gmag,bins=bins)
    if np.sum(h1>1)>0 :
        hz,junk=np.histogram(gmag,bins=bins,weights=dz)
        hz2,junk=np.histogram(gmag,bins=bins,weights=dz**2)
        mean=hz/(h1+(h1==0))
        rms=np.sqrt(hz2/(h1+(h1==0))-mean**2)
        err=rms/np.sqrt(h1-1+10*(h1<=1))
        nodes=bins[:-1]+(bins[1]-bins[0])/2.
        xerr=np.gradient(bins)[:-1]/2.
        a1.errorbar(nodes[h1>1],mean[h1>1],yerr=rms[h1>1],xerr=xerr[h1>1],fmt="o",color="red",lw=2)
        a1.errorbar(nodes[h1>1],mean[h1>1],yerr=err[h1>1],xerr=xerr[h1>1],fmt="o",color="k",lw=2)



a1.set_xlabel("g mag (AB, SDSS filter)")
a1.set_ylabel(r"$\Delta z$")

if 0 :
    a2=pylab.subplot(2,2,3)
    a2.hist((t["BEST_Z"][ok]-t["TRUE_Z"][ok])/t["BEST_Z_ERR"][ok],bins=10)
    a2.set_xlabel(r"$\Delta z / \sigma z$")

pylab.show()


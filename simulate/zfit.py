#!/usr/bin/env python

from desispec.log import get_logger
from desispec.io.brick import Brick

import astropy.io.fits as pyfits
import argparse
import numpy as np
import sys
import pylab # for debugging
import math


def simple_zscan(wave,flux,ivar,template_wave,template_flux,zstep=0.001,zmin=0.1,zmax=3.,ntrack=3,recursive=True,zscan=None,chi2scan=None) :
    """
    args :
      wave : list of 1D array of wavelength in A
      flux : list of 1D array of flux in ergs/s/cm2/A
      ivar : list of 1D array of flux inverse variance
      template_wave : rest-frame wavelength of template
      template_flux : flux of template
      zstep  : step of redshift scan (linear)
      zmin   : min value of redshift scan
      zmax   : max value of redshift scan
      ntrack : number of local minima to track and record (default is 3)
      recursive : internal parameter, one must leave it to True
    
    returns :
       a dictionnary with results of the fit including redshift, uncertainty, chi2,
       for ntrack best fit local minima. 
        
    """
    if zscan is None :
        zscan=[] # used for final uncertainty
        chi2scan=[] # used for final uncertainty


    log=get_logger()
    
    nframes=len(wave)
    
    ndf=0
    for index in range(nframes) :
        ndf += np.sum(ivar[index]>0)
    
    ndf-=2 # FOR THE MOMENT WE FIT ONLY ONE AMPLITUDE PER Z
    if ndf<=0 :
        log.warning("ndf=%d skip this spectrum")
        return None,None
    
    log.debug("zstep=%f"%zstep)
    log.debug("nframes=%d"%nframes)

    
    nframes=len(wave)
            
    log.debug("z min=%f z max=%f"%(zmin,zmax))
    
    
    nbz=ntrack
    best_zs=-12.*np.ones((nbz))
    best_z_errors=np.zeros((nbz))
    
    very_large_number=1e12
    
    best_chi2s=very_large_number*np.ones((nbz))
    best_chi2pdfs=very_large_number*np.ones((nbz))
    
    chi2s_at_best_z_minus_zstep=np.zeros((nbz))
    chi2s_at_best_z_plus_zstep=np.zeros((nbz))
    #z_at_best_z_minus_zstep=np.zeros((nbz)) # debug
    #z_at_best_z_plus_zstep=np.zeros((nbz)) # debug
    
    previous_chi2=0
    previous_z=0 # debug
    previous_is_at_rank=-1
        
    best_z_amplitude=0
    best_z_amplitude_ivar=0
    chi2_has_increased=True
    
    
    for z in np.linspace(zmin,zmax,num=int((zmax-zmin)/zstep+1)) :

        sum_ivar_flux_prof = 0
        sum_ivar_prof2     = 0
        ndf                = 0
        chi2_0             = 0.
        for frame_index in range(nframes) :
            
            model  = np.interp(wave[frame_index],template_wave*(1+z),template_flux)
            
            sum_ivar_flux_prof += np.sum(ivar[frame_index]*flux[frame_index]*model)
            sum_ivar_prof2     += np.sum(ivar[frame_index]*model**2)
            ndf                += np.sum((ivar[frame_index]>0)*(model!=0))
            chi2_0             += np.sum(ivar[frame_index]*flux[frame_index]**2*(model!=0))

        amp = sum_ivar_flux_prof/sum_ivar_prof2
        amp_ivar = sum_ivar_prof2
        ndf -= 1

        

        chi2 = chi2_0 - 2*amp* sum_ivar_flux_prof + amp**2*sum_ivar_prof2
        chi2pdf = chi2/ndf
        
        zscan.append(z) # debugging
        chi2scan.append(chi2) # debugging

        if chi2<np.max(best_chi2s) :
            
            need_insert = chi2_has_increased # True
            chi2_has_increased = False
            
            # first find position depending on delta z and chi2
#            tmp=np.where(abs(z-best_zs)<0.1)[0]
#            if tmp.size>0 : # there is an existing close redshift we replace this entry
            if not need_insert :
                #i=tmp[0]                
                #need_insert = False # and we don't have to shift the others
                i=np.argmin(z-best_zs)
            else :
                # find position depending on value of chi2
                i=np.where(chi2<best_chi2s)[0][0] # take the first slot where chi2 smaller                            
                #print "DEBUG INSERT for chi2pdf=%f z=%f ndf=%d i=%d OTHERS z=%s chi2=%s"%(chi2pdf,z,ndf,i,str(best_zs[best_chi2pdfs<very_large_number]),str(best_chi2pdfs[best_chi2s<very_large_number]))
                best_zs[i+1:]=best_zs[i:-1]
                best_chi2pdfs[i+1:]=best_chi2pdfs[i:-1]
                best_chi2s[i+1:]=best_chi2s[i:-1]
                chi2s_at_best_z_minus_zstep[i+1:]=chi2s_at_best_z_minus_zstep[i:-1]
                chi2s_at_best_z_plus_zstep[i+1:]=chi2s_at_best_z_plus_zstep[i:-1]
                #z_at_best_z_minus_zstep[i+1:]=z_at_best_z_minus_zstep[i:-1]
                #z_at_best_z_plus_zstep[i+1:]=z_at_best_z_plus_zstep[i:-1]
                

        if chi2<best_chi2s[i] :
            
            best_chi2s[i]=chi2
            best_chi2pdfs[i]=chi2pdf
            best_zs[i]=z
            chi2s_at_best_z_minus_zstep[i]=previous_chi2
            #z_at_best_z_minus_zstep[i]=previous_z
            best_z_amplitude=amp
            best_z_amplitude_ivar=amp_ivar
            previous_is_at_rank=i

            if not need_insert :
                #print "DEBUG IMPROVE for chi2=%f z=%f i=%d BEST=%s"%(chi2,z,i,str(best_zs[best_chi2s<very_large_number]))
                # but this means we may have to change the ranks
                indices=np.argsort(best_chi2s)
                if np.sum(np.abs(indices-range(best_chi2s.size)))>0 : # need swap
                    best_chi2s=best_chi2s[indices]
                    best_chi2pdfs=best_chi2pdfs[indices]
                    best_zs=best_zs[indices]
                    chi2s_at_best_z_minus_zstep=chi2s_at_best_z_minus_zstep[indices]
                    chi2s_at_best_z_plus_zstep=chi2s_at_best_z_plus_zstep[indices]
                    #z_at_best_z_minus_zstep=z_at_best_z_minus_zstep[indices]
                    #z_at_best_z_plus_zstep=z_at_best_z_plus_zstep[indices]
                    
            

        else :
            # check change of chi2 is significant
            chi2_has_increased = (chi2>best_chi2s[i]+1.)
            if previous_is_at_rank>=0 :
                chi2s_at_best_z_plus_zstep[previous_is_at_rank]=chi2
                #z_at_best_z_plus_zstep[previous_is_at_rank]=z
                
            previous_is_at_rank=-1

        
        previous_chi2=chi2
        previous_z=z
    for rank in range(best_zs.size) :

        #print "DEBUG rank %d z %f<%f<%f chi2 %f,%f,%f"%(rank,z_at_best_z_minus_zstep[rank],best_zs[rank],z_at_best_z_plus_zstep[rank],
        #chi2s_at_best_z_minus_zstep[rank],
        #best_chi2s[rank],chi2s_at_best_z_plus_zstep[rank])                              
        

        # we can use the values about best_chi2 to guess the uncertainty on z with a polynomial fit
        coeffs=np.polyfit([best_zs[rank]-zstep,best_zs[rank],best_zs[rank]+zstep],[chi2s_at_best_z_minus_zstep[rank],best_chi2s[rank],chi2s_at_best_z_plus_zstep[rank]],2)
        a=coeffs[0]
        b=coeffs[1]
        c=coeffs[2]
    
        #if 0 and not recursive and rank==0 : # DEBUGGING
        #    bchi2=best_chi2s[rank]
        #    pylab.plot([best_zs[rank]-zstep,best_zs[rank],best_zs[rank]+zstep],[chi2s_at_best_z_minus_zstep[rank]-best_chi2s[rank],0,chi2s_at_best_z_plus_zstep[rank]-best_chi2s[rank]],"o")

        best_z_errors[rank] = zstep
        if a>0 :
            best_zs[rank]       = -b/(2*a)
            best_chi2s[rank]    = c-b**2/(4*a)
            best_z_errors[rank] = 1./math.sqrt(a)
    
        #if 0 and not recursive and rank==0 : # DEBUGGING
        #    bz=best_zs[rank]
        #    ez=best_z_errors[rank]
        #    zz=np.linspace(bz-3*zstep,bz+3*zstep,100)
        #    pol=np.poly1d(coeffs)(zz)
        #    pylab.plot(zz,pol-bchi2,c="r")
        #    pylab.plot([bz,bz],[0,1],c="k")
        #    pylab.plot([bz-ez,bz-ez],[0,1],c="k")
        #    pylab.plot([bz+ez,bz+ez],[0,1],c="k")
        
        #    print "DEBUG z=%f +- %f"%(best_zs[rank],best_z_errors[rank])
        #    pylab.show()
        
    
    if recursive :

        #print "DEBUG z=%s,dchi2=%s"%(str(best_zs),str(best_chi2s-best_chi2s[0]))
        #pylab.figure()
        #pylab.plot(zscan,chi2scan)
        
        best_results=None
        
        rank_labels = np.array(["BEST","SECOND","THIRD"]) # I know it's a bit ridiculous
        for i in range(rank_labels.size,ntrack) :
            rank_labels=np.append(rank_labels,np.array(["%dTH"%(i+1)])) # even more ridiculous
        # naming for output
        labels=np.array(["BEST","SECOND","THIRD"]) # I know it's a bit ridiculous
        for i in range(labels.size,ntrack) :
            labels=np.append(labels,np.array(["%dTH"%(i+1)])) # even more ridiculous

        #log.info("refining the tracked z values : %s"%str(best_zs))
        previous_zstep=zstep
        for rank in range(best_zs.size) :
            # second loop about minimum
            # where we save things to compute errors
            tmp_z       = best_zs[rank]
            #tmp_z_error = best_z_errors[rank]
            #if tmp_z_error>zstep :
            #    tmp_z_error = zstep
            #tmp_z_error=max(tmp_z_error,0.005)
            #tmp_z_error=max(tmp_z_error,0.0001)
            #z_nsig=2.
            zmin=tmp_z-0.01
            zmax=tmp_z+0.01
            zstep=(zmax-zmin)/100
            #log.info("refining z=%f with scan in %f<z<%f dz=%f"%(tmp_z,zmin,zmax,zstep))
            tmp_results = simple_zscan(wave,flux,ivar,template_wave,template_flux,zstep=zstep,zmin=zmin,zmax=zmax,ntrack=1,recursive=False,zscan=zscan,chi2scan=chi2scan)

            #log.info("refined : z=%f chi2=%f -> %f chi2=%f"%(tmp_z,best_chi2s[rank],tmp_results["BEST_Z"],tmp_results["BEST_CHI2"]))
            
            if rank == 0 :
                # this is the best
                best_results=tmp_results
            else :
                # here we replace the best values
                label=labels[rank]
                keys1=best_results.keys()
                for k1 in keys1 :
                    if k1.find("BEST")==0 :
                        k2=k1.replace("BEST",label)
                        best_results[k2] = tmp_results[k1]
        
        # swap results if it turns out that the ranking has been modified by the improved fit     
        chi2=np.zeros((ntrack))
        for i,l in zip(range(ntrack),rank_labels) :
            chi2[i]=best_results[l+"_CHI2PDF"]
        indices=np.argsort(chi2)
        if np.sum(np.abs(indices-range(ntrack)))>0 : # need swap
            swapped_best_results={}
            for i in range(ntrack) :
                new_label=rank_labels[i]
                old_label=rank_labels[indices[i]]
                for k in best_results :
                    if k.find(old_label)==0 :
                        swapped_best_results[k.replace(old_label,new_label)]=best_results[k]
            #print "DEBUG : has swapped results"
            #print best_results
            #print swapped_best_results
            best_results=swapped_best_results
            
        # check chi2 scan to increase error bar is other local minimum
        # with delta chi2<1
        zscan=np.array(zscan)
        chi2scan=np.array(chi2scan)
        dz=np.max(np.abs(zscan[chi2scan<(best_results["BEST_CHI2"]+1)]-best_results["BEST_Z"]))
        #print "DEBUG dz=",dz
        if dz>best_results["BEST_Z_ERR"] :
            log.warning("increase BEST_Z_ERR %f -> %f because of other local minimum"%(best_results["BEST_Z_ERR"],dz))
            best_results["BEST_Z_ERR"]=dz

        log.info("best z=%f+-%f chi2/ndf=%3.2f snr=%3.1f dchi2=%3.1f for dz=%f"%(best_results["BEST_Z"],best_results["BEST_Z_ERR"],best_results["BEST_CHI2PDF"],best_results["BEST_SNR"],best_results["SECOND_CHI2"]-best_results["BEST_CHI2"],abs(best_results["BEST_Z"]-best_results["SECOND_Z"])))
        return best_results


    
    
    res={}
    res["BEST_Z"]=best_zs[0]
    res["BEST_Z_ERR"]=best_z_errors[0]
    res["BEST_CHI2"]=best_chi2s[0]
    res["BEST_CHI2PDF"]=best_chi2pdfs[0]
    res["BEST_SNR"]=best_z_amplitude*math.sqrt(best_z_amplitude_ivar)
    res["BEST_AMP"]=best_z_amplitude
    
    return res
def luminosity_function(data, z_max=6.0, area_sq_deg=10000.):
    """Transform a data array from Nathalie into a tuple gbin, zbin, nqso.
    """
    ng, nz = data.shape
    # g-band magnitude bin centers are in the first column.
    gbin = data[:, 0]
    nz = nz - 1
    # Check that g-band bins are equally spaced.
    assert np.allclose(np.diff(gbin),  gbin[1] - gbin[0])
    # redshift bins are equally spaced from 0 up to z_max.
    zbin = z_max * (0.5 + np.arange(nz)) / nz
    # The remaining columns give predicted numbers of QSO in a 10,000 sq.deg. sample.
    # Normalize to densisities per sq.deg.
    nqso = data[:, 1:].reshape((ng, nz)) / area_sq_deg
    return gbin, zbin, nqso


def save(filename,gmag,z_true,best_z_array,best_z_error_array,best_chi2_array,best_chi2pdf_array,best_snr_array,delta_chi2_array) :
    
    cols=[]
    cols.append(pyfits.Column(name='BEST_Z', format='D', array=best_z_array))
    cols.append(pyfits.Column(name='BEST_Z_ERR', format='D', array=best_z_error_array))
    cols.append(pyfits.Column(name='BEST_CHI2', format='D', array=best_chi2_array))
    cols.append(pyfits.Column(name='BEST_CHI2PDF', format='D', array=best_chi2pdf_array))
    cols.append(pyfits.Column(name='DELTA_CHI2', format='D', array=delta_chi2_array))
    cols.append(pyfits.Column(name='BEST_SNR', format='D', array=best_snr_array))
    cols.append(pyfits.Column(name='TRUE_Z', format='D', array=z_true[:best_z_array.size]))
    cols.append(pyfits.Column(name='GMAG', format='D', array=gmag[:best_z_array.size]))
    
    cols = pyfits.ColDefs(cols)
    tbhdu = pyfits.BinTableHDU.from_columns(cols)
    output_hdulist = pyfits.HDUList([pyfits.PrimaryHDU(),tbhdu])
    output_hdulist.writeto(filename,clobber=True)
    print "wrote %s"%filename
    sys.stdout.flush()

def bin_index(bin_centers, low_edge):
    """Find the index of the bin with the specified low edge, where bins is an array of equally-spaced bin centers.
    """
    delta = bin_centers[1] - bin_centers[0]
    min_value = bin_centers[0] - 0.5 * delta
    index = int(round((low_edge - min_value) / delta))
    if abs((low_edge - min_value) / delta - index) > 1e-5:
        raise ValueError('low_edge = {} is not aligned with specified bins.'.format(low_edge))
    return index

def sample(data, num_samples, g_min=19, g_max=23, z_min=0, z_max=6, seed=None):
    """Generate random samples of (g,z) within the specified cuts.
    """
    gbin, zbin, nqso = luminosity_function(data)
    z_min_cut = bin_index(zbin, z_min)
    z_max_cut = bin_index(zbin, z_max)
    g_min_cut = bin_index(gbin, g_min)
    g_max_cut = bin_index(gbin, g_max)
    nqso_cut = nqso[g_min_cut:g_max_cut, z_min_cut:z_max_cut]
    # Calculate the flattened CDF
    cdf = np.cumsum(nqso_cut.ravel())
    cdf /= np.float(cdf[-1])
    # Pick random flattened indices.
    generator = np.random.RandomState(seed)
    r = generator.rand(num_samples)
    indices = np.searchsorted(cdf, r)
    # Unravel to get g,z indices.
    g_indices, z_indices = np.unravel_index(indices, (len(gbin), len(zbin)))
    # Spread points out uniformly in each 2D bin.
    dg, dz = gbin[1] - gbin[0], zbin[1] - zbin[0]
    g = gbin[g_min_cut + g_indices] + dg * (generator.rand(num_samples) - 0.5)
    z = zbin[z_min_cut + z_indices] + dz * (generator.rand(num_samples) - 0.5)
    return g, z

def main() :

    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--b', type = str, default = None, required=True,
                        help = 'path of DESI brick in b')
    parser.add_argument('--r', type = str, default = None, required=True,
                        help = 'path of DESI brick in r')
    parser.add_argument('--z', type = str, default = None, required=True,
                        help = 'path of DESI brick in z')
    parser.add_argument('--t', type = str, default = None, required=True,
                        help = 'path of QSO template (for the moment the desimodel one)')
    parser.add_argument('--outfile', type = str, default = None, required=True,
                        help = 'path of output file')
    parser.add_argument('--s', type = int, default = None, required=False,
                        help = 'only one spectrum (for debugging)')
    parser.add_argument('--num-samples', type=int, default=1000, required=False,
                        help = 'Number of samples to fit')
    parser.add_argument('--g-min', type=double, default=19, required=False,
                        help = 'Minimum g band magnitude for the QLF grid')
    parser.add_argument('--g-max', type=double, default=23, required=False,
                        help = 'Maximum g band magnitude for the QLF grid')
    parser.add_argument('--z-min', type=double, default=0, required=False,
                        help = 'Minimum redshift for the QLF grid')
    parser.add_argument('--z-max', type=double, default=6, required=False,
                        help = 'Maximum redshift for the QLF grid')
    
    args = parser.parse_args()
    log=get_logger()
    z_min = args.z-min
    z_max = args.z-max
    g_min = args.g-min
    g_max = args.g-max
    num_samples = args.num-samples
    log.info("starting")
    log.info("z_min = ", z_min)
    log.info("z_max = ", z_max)
    log.info("g_min = ", g_min)
    log.info("g_max = ", g_max)
    log.info("num_samples = ", num_samples)
    log.warning("HARDCODED PALANQUE2012 AND 2015 TABLES")
    table2012 = np.array([
        15.75, 50, 11, 7, 4, 4, 4, 4, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        16.25, 92, 34, 20, 14, 13, 13, 12, 12, 10, 8, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        16.75, 159, 96, 62, 43, 42, 41, 39, 37, 31, 25, 22, 21, 12, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        17.25, 249, 248, 182, 131, 130, 128, 120, 114, 96, 77, 65, 58, 34, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        17.75, 354, 558, 483, 381, 387, 384, 365, 347, 296, 238, 192, 158, 91, 44, 22, 11, 5, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        18.25, 461, 1076, 1125, 1009, 1066, 1074, 1050, 1008, 876, 713, 553, 431, 246, 119, 59, 29, 15, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        18.75, 548, 1790, 2224, 2318, 2565, 2671, 2715, 2642, 2374, 1982, 1528, 1126, 650, 318, 161, 80, 40, 20, 10, 5, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        19.25, 582, 2624, 3751, 4464, 5166, 5601, 5928, 5899, 5550, 4825, 3775, 2744, 1626, 832, 428, 216, 108, 56, 28, 14, 7, 4, 2, 1, 0, 0, 0, 0, 0, 0,
        19.75, 503, 3517, 5561, 7237, 8674, 9779, 10631, 10847, 10630, 9685, 7904, 5822, 3695, 2047, 1103, 569, 292, 151, 75, 39, 20, 10, 5, 3, 1, 1, 0, 0, 0, 0,
        20.25, 198, 4473, 7528, 10277, 12641, 14591, 16079, 16737, 16843, 15972, 13518, 10254, 7186, 4500, 2627, 1445, 760, 405, 203, 106, 54, 27, 14, 7, 4, 2, 1, 0, 0, 0,
        20.75, 0, 4976, 9650, 13461, 16826, 19623, 21758, 22913, 23360, 22704, 19534, 14971, 11576, 8356, 5491, 3330, 1886, 1035, 540, 286, 145, 75, 39, 19, 10, 5, 3, 1, 1, 0,
        21.25, 0, 4569, 12028, 16929, 21338, 24976, 27767, 29400, 30142, 29653, 25414, 19052, 15753, 12809, 9602, 6619, 4164, 2484, 1363, 744, 391, 201, 105, 53, 28, 14, 7, 4, 2, 1,
        21.75, 0, 2676, 14806, 20913, 26454, 31008, 34512, 36621, 37628, 37214, 31417, 22460, 19085, 16765, 13967, 10897, 7823, 5190, 3161, 1833, 999, 532, 285, 142, 75, 38, 19, 10, 5, 2,
        22.25, 0, 84, 15784, 25646, 32491, 38098, 42423, 45057, 46339, 45926, 38025, 25572, 21704, 19852, 17646, 15057, 12150, 9109, 6291, 4023, 2400, 1338, 736, 381, 203, 102, 52, 27, 13, 7,
        22.75, 0, 0, 9053, 31359, 39749, 46622, 51924, 55165, 56750, 56291, 45702, 28725, 23970, 22325, 20498, 18418, 16044, 13305, 10392, 7496, 5006, 3077, 1810, 975, 530, 276, 141, 73, 36, 19,
        23.25, 0, 0, 232, 29955, 48563, 56956, 63439, 67410, 69360, 68823, 54820, 32149, 26141, 24537, 22848, 21047, 19095, 16860, 14383, 11591, 8748, 6068, 3930, 2330, 1333, 714, 373, 198, 98, 51,
        23.75, 0, 0, 0, 6251, 54772, 69546, 77462, 82312, 84692, 84046, 65808, 35985, 28366, 26707, 25016, 23292, 21523, 19635, 17592, 15282, 12711, 9950, 7240, 4836, 3014, 1753, 959, 519, 260, 137,
        24.25, 0, 0, 0, 0, 15743, 79815, 94554, 100480, 103394, 102610, 79067, 40365, 30717, 28956, 27187, 25426, 23677, 21910, 20107, 18177, 16049, 13701, 11073, 8398, 5869, 3795, 2254, 1299, 682, 364,
        24.75, 0, 0, 0, 0, 0, 18500, 100062, 122644, 126193, 125238, 95158, 45393, 33235, 31346, 29458, 27598, 25781, 24001, 22249, 20487, 18656, 16705, 14505, 12119, 9500, 6953, 4650, 2915, 1656, 927,
        25.25, 0, 0, 0, 0, 0, 0, 8489, 90591, 153382, 152869, 114645, 51218, 35949, 33912, 31882, 29890, 27956, 26090, 24287, 22535, 20802, 19060, 17206, 15222, 12983, 10546, 7997, 5625, 3605, 2163,
        25.75, 0, 0, 0, 0, 0, 0, 0, 0, 38955, 134619, 138257, 57968, 38880, 36680, 34488, 32343, 30266, 28272, 26363, 24538, 22780, 21075, 19370, 17641, 15781, 13728, 11472, 9007, 6581, 4416,
        26.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28228, 37600, 40111, 39669, 37301, 34985, 32744, 30600, 28552, 26608, 24759, 22997, 21298, 19641, 17974, 16232, 14366, 12237, 9959, 7525
    ]).reshape(22, 31) 
    
    table2015a = np.array([
    15.75, 30, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    16.25, 60, 8, 4, 5, 5, 4, 4, 4, 4, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    16.75, 117, 29, 17, 19, 18, 17, 16, 16, 15, 12, 8, 6, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    17.25, 216, 101, 62, 70, 69, 64, 61, 62, 59, 45, 32, 22, 13, 7, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    17.75, 358, 312, 224, 255, 253, 235, 227, 231, 224, 171, 121, 82, 47, 25, 13, 7, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    18.25, 525, 788, 722, 855, 869, 819, 803, 824, 811, 630, 452, 309, 171, 88, 46, 22, 9, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    18.75, 703, 1563, 1890, 2393, 2544, 2493, 2507, 2612, 2622, 2112, 1572, 1096, 603, 309, 157, 76, 28, 8, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    19.25, 898, 2490, 3740, 5086, 5758, 5971, 6214, 6580, 6745, 5779, 4613, 3369, 1913, 1004, 516, 249, 93, 26, 10, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    19.75, 1125, 3445, 5827, 8319, 9913, 10805, 11590, 12422, 12937, 11839, 10261, 8011, 4902, 2771, 1499, 753, 289, 78, 31, 12, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    20.25, 1399, 4456, 7930, 11585, 14183, 15895, 17350, 18718, 19660, 18783, 17275, 14270, 9517, 5936, 3513, 1919, 804, 228, 91, 34, 10, 2, 1, 0, 0, 0, 0, 0, 0, 0,
    20.75, 1734, 5616, 10195, 15029, 18589, 21065, 23176, 25094, 26479, 25795, 24410, 20801, 14695, 9899, 6391, 3856, 1851, 599, 248, 94, 27, 6, 1, 1, 0, 0, 0, 0, 0, 0,
    21.25, 2141, 7016, 12842, 18997, 23563, 26793, 29584, 32124, 34027, 33395, 31948, 27600, 20026, 14047, 9572, 6217, 3399, 1325, 598, 241, 73, 17, 4, 1, 1, 1, 0, 0, 0, 0,
    21.75, 2631, 8738, 16067, 23807, 29528, 33591, 37170, 40481, 43047, 42378, 40701, 35383, 25928, 18498, 12931, 8731, 5198, 2395, 1211, 541, 182, 45, 10, 3, 2, 2, 1, 1, 0, 0,
    22.25, 3211, 10871, 20058, 29754, 36864, 41912, 46457, 50760, 54210, 53457, 51424, 44870, 32982, 23683, 16738, 11500, 7140, 3651, 2036, 1022, 394, 110, 25, 8, 6, 4, 2, 2, 1, 1,
    22.75, 3875, 13520, 25026, 37157, 45968, 52212, 57971, 63564, 68202, 67344, 64840, 56742, 41732, 30041, 21339, 14774, 9334, 5003, 2969, 1636, 730, 239, 60, 21, 13, 8, 5, 3, 2, 1,
    23.25, 4591, 16812, 31220, 46395, 57302, 65011, 72306, 79586, 85821, 84853, 81750, 71739, 52744, 38010, 27078, 18823, 11969, 6500, 3980, 2322, 1159, 450, 133, 48, 30, 19, 12, 7, 5, 3,
    23.75, 5270, 20905, 38950, 57934, 71426, 80937, 90180, 99667, 108052, 106983, 103130, 90753, 66677, 48080, 34331, 23929, 15247, 8253, 5120, 3076, 1645, 733, 256, 102, 63, 39, 24, 15, 9, 6,
    24.25, 5713, 25993, 48598, 72353, 89037, 100762, 112480, 124858, 136130, 134989, 130200, 114903, 84351, 60850, 43543, 30420, 19392, 10379, 6468, 3939, 2184, 1061, 432, 192, 119, 73, 45, 27, 17, 10,
    24.75, 5464, 32318, 60643, 90374, 110997, 125444, 140309, 156474, 171625, 170467, 164509, 145614, 106798, 77072, 55275, 38702, 24667, 13006, 8110, 4969, 2803, 1428, 648, 317, 199, 123, 76, 46, 28, 17
    ]).reshape(19, 31)

    table2015b = np.array([
    15.75, 23, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
    16.25, 49, 4, 2, 2, 2, 2, 2, 2, 2, 1, 1, 16, 10, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
    16.75, 99, 16, 8, 10, 10, 9, 9, 8, 8, 5, 3, 40, 25, 12, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
    17.25, 190, 69, 38, 44, 45, 41, 39, 39, 36, 24, 15, 104, 65, 32, 15, 7, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    17.75, 326, 248, 165, 196, 199, 185, 177, 176, 163, 113, 69, 268, 167, 82, 39, 17, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
    18.25, 488, 699, 628, 775, 805, 763, 744, 751, 709, 501, 314, 679, 422, 211, 102, 46, 16, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
    18.75, 664, 1453, 1808, 2389, 2615, 2602, 2624, 2702, 2629, 1968, 1308, 1650, 1027, 532, 262, 119, 42, 12, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
    19.25, 866, 2337, 3638, 5131, 5991, 6356, 6674, 7031, 7076, 5840, 4349, 3696, 2334, 1283, 657, 307, 111, 30, 11, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
    19.75, 1113, 3252, 5609, 8203, 9995, 11093, 11997, 12825, 13218, 11932, 10033, 7168, 4740, 2850, 1566, 769, 288, 77, 27, 9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
    20.25, 1423, 4274, 7646, 11336, 14053, 15897, 17425, 18758, 19553, 18488, 16711, 11689, 8346, 5597, 3399, 1817, 723, 195, 70, 23, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0 ,
    20.75, 1814, 5517, 9994, 14889, 18542, 21093, 23243, 25116, 26334, 25315, 23523, 16660, 12773, 9487, 6456, 3873, 1709, 482, 177, 58, 15, 3, 1, 0, 0, 0, 0, 0, 0, 0 ,
    21.25, 2303, 7084, 12906, 19262, 23994, 27320, 30181, 32721, 34466, 33311, 31222, 21941, 17567, 14054, 10566, 7166, 3647, 1143, 436, 148, 38, 7, 1, 0, 0, 0, 0, 0, 0, 0 ,
    21.75, 2911, 9082, 16610, 24818, 30879, 35139, 38892, 42310, 44774, 43370, 40764, 27761, 22625, 18918, 15230, 11423, 6772, 2503, 1034, 367, 96, 19, 3, 1, 0, 0, 0, 0, 0, 0 ,
    22.25, 3652, 11639, 21358, 31945, 39682, 45108, 50015, 54614, 58082, 56344, 53020, 34486, 28144, 24081, 20162, 16146, 10868, 4879, 2277, 876, 241, 49, 9, 2, 1, 1, 0, 0, 0, 0 ,
    22.75, 4527, 14913, 27461, 41112, 50977, 57875, 64289, 70486, 75363, 73207, 68942, 42521, 34433, 29793, 25456, 21136, 15480, 8290, 4489, 1955, 585, 122, 22, 5, 3, 1, 1, 0, 0, 0 ,
    23.25, 5504, 19106, 35311, 52915, 65485, 74247, 82634, 90995, 97849, 95188, 89701, 52304, 41818, 36375, 31400, 26555, 20387, 12378, 7728, 3931, 1345, 302, 55, 14, 7, 3, 2, 1, 0, 0,
    23.75, 6479, 24477, 45409, 68119, 84128, 95249, 106225, 117520, 127141, 123880, 116812, 64335, 50637, 44159, 38330, 32717, 25709, 16775, 11668, 6924, 2830, 719, 137, 35, 18, 9, 4, 2, 1, 0 ,
    24.25, 7195, 31358, 58404, 87705, 108088, 122196, 136567, 151843, 165336, 161372, 152261, 79215, 61261, 53494, 46590, 39969, 31740, 21404, 15927, 10682, 5278, 1609, 337, 89, 45, 22, 11, 5, 3, 1,
    24.75, 7043, 40171, 75127, 112945, 138885, 156770, 175600, 196278, 215178, 210413, 198658, 97685, 74113, 64767, 56549, 48670, 38821, 26430, 20396, 14815, 8615, 3269, 793, 220, 112, 56, 28, 13, 6, 3
    ]).reshape(19, 31)
    
    table2015 = 0.5 * (table2015a + table2015b) # Average of the two methods
    ##log.warning("HARDCODED TRUE Z AND GMAG")
    gmag, z_true = sample(table2015, num_samples, g_min, g_max, z_min, z_max, 123)
    ##g_grid = np.array([ 22.  ,  22.25,  22.5 ,  22.75,  23.  ])
    ##z_grid = np.linspace(0.5,3.5,31)
    ##gmag = np.tile(g_grid,(z_grid.size,1)).T.ravel()
    ##z_true = np.tile(z_grid,(g_grid.size,1)).ravel()
    
    b_brick=pyfits.open(args.b)
    r_brick=pyfits.open(args.r)
    z_brick=pyfits.open(ags.z)
    
    
    # need to know the wave and flux and z of the templates
    vals=np.loadtxt(args.t).T
    log.warning("HARDCODED REDSHIFT OF REFERENCE TEMPLATE Z=2.4!!")
    template_wave=vals[0]/(1.+2.4)
    template_flux=vals[1]
    
    #pylab.plot(template_wave,template_flux)
    #pylab.show()

    #qso_spectra=np.where(b_brick.hdu_list[4].data["OBJTYPE"]=="QSO")[0]
    #qso_spectra=qso_spectra[0:2]
    qso_spectra=np.arange(b_brick[0].data.shape[0])
    if args.s is not None :
        qso_spectra=np.array([args.s])
        
    nqso=qso_spectra.size
    log.info("number of QSO = %d"%nqso)
    
    best_z_array=np.zeros((nqso))
    best_z_error_array=np.zeros((nqso))
    best_chi2_array=np.zeros((nqso))
    best_chi2pdf_array=np.zeros((nqso))
    best_snr_array=np.zeros((nqso))
    delta_chi2_array=np.zeros((nqso))
    
    for spec,q in zip(qso_spectra,range(nqso)) :
        flux=[b_brick[0].data[spec],r_brick[0].data[spec],z_brick[0].data[spec]]
        ivar=[b_brick[1].data[spec],r_brick[1].data[spec],z_brick[1].data[spec]]
        wave=[b_brick[2].data[spec],r_brick[2].data[spec],z_brick[2].data[spec]]    
        result = simple_zscan(wave,flux,ivar,template_wave,template_flux,zstep=0.001,zmin=0.3,zmax=4.5,ntrack=3,recursive=True)
        
        
        best_z_array[q]     = result["BEST_Z"]
        best_z_error_array[q]     = result["BEST_Z_ERR"]
        delta_chi2_array[q] = -result["BEST_CHI2"]+result["SECOND_CHI2"]
        best_chi2_array[q]  = result["BEST_CHI2"]
        best_chi2pdf_array[q]  = result["BEST_CHI2PDF"]
        best_snr_array[q]  = result["BEST_SNR"]
        
        log.info("SPEC=%d/%d redshift=%f+-%f chi2pdf=%f dchi2=%f"%(q,nqso,best_z_array[q],best_z_error_array[q],best_chi2pdf_array[q],delta_chi2_array[q]))

        # debugging
        #pylab.figure()
        #pylab.errorbar(wave[ivar>0],flux[ivar>0],1./np.sqrt(ivar[ivar>0]),fmt="o",color="b")
        #pylab.plot(template_wave*(1+result["BEST_Z"]),template_flux*result["BEST_AMP"],color="r")
        #pylab.show()

        if q%2==1 : # save intermediate results 
            save(args.outfile,gmag,z_true,best_z_array,best_z_error_array,best_chi2_array,best_chi2pdf_array,best_snr_array,delta_chi2_array)

    save(args.outfile,gmag,z_true,best_z_array,best_z_error_array,best_chi2_array,best_chi2pdf_array,best_snr_array,delta_chi2_array)

if __name__ == '__main__':
    main()
    

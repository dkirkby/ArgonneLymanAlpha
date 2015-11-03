#!/usr/bin/env python

from desispec.log import get_logger
from desispec.io.brick import Brick

import astropy.io.fits as pyfits
import argparse
import numpy as np
import sys
import pylab # for debugging
import math


def simple_zscan(wave,flux,ivar,template_wave,template_flux,zstep=0.001,zmin=0.,zmax=100.,wave_nsig=2.,ntrack=3,recursive=True) :
    """
    args :
      wave : list of 1D array of wavelength in A
      flux : list of 1D array of flux in ergs/s/cm2/A
      ivar : list of 1D array of flux inverse variance
      template_wave : rest-frame wavelength of template
      template_flux : flux of template
      zstep  : step of redshift scan (linear)
      zmin   : force min value of redshift scan (default is automatic)
      zmax   : force max value of redshift scan (default is automatic)
      wave_nsig : defines the range of wavelength used to fit this line (modified in refinement)
      ntrack : number of local minima to track and record (default is 3)
      recursive : internal parameter, one must leave it to True
    
    returns :
       a dictionnary with results of the fit including redshift, uncertainty, chi2,
       for ntrack best fit local minima. 
        
    """
    chris_debug=False
   
    log=get_logger()
    
    nframes=len(wave)
    if (chris_debug): print "Nframes = %d"%nframes

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
    
    previous_chi2=0
    previous_is_at_rank=-1
    #model=[]
    #for frame_index in range(nframes) :
    #    model.append(np.zeros((wave[frame_index].size))) # was it a bug ? (CB)
    

        
    
    best_z_amplitude=0
    best_z_amplitude_ivar=0
    
    
    # for debugging only
    #zz=[]
    #zchi2=[]

    #template_min_wave=template_wave[template_flux!=0][0]+5. # add safety margin because of interpolation
    #template_max_wave=template_wave[template_flux!=0][-1]-5. # add safety margin because of interpolation
    #print "template valid wavelength range =",template_min_wave,template_max_wave
    
    for z in np.linspace(zmin,zmax,num=int((zmax-zmin)/zstep+1)) :
        
        

        # reset (wo memory allocation)
        sum_ivar_flux_prof = 0
        sum_ivar_prof2     = 0
        ndf                = 0
        chi2_0             = 0.
        for frame_index in range(nframes) :
            #model  = np.interp(wave[frame_index],template_wave*(1+z),template_flux)*(wave[frame_index]>template_min_wave*(1+z))*(wave[frame_index]<template_max_wave*(1+z))
            model  = np.interp(wave[frame_index],template_wave*(1+z),template_flux)
            
            sum_ivar_flux_prof += np.sum(ivar[frame_index]*flux[frame_index]*model)
            sum_ivar_prof2     += np.sum(ivar[frame_index]*model**2)
            ndf                += np.sum((ivar[frame_index]>0)*(model!=0))
            chi2_0             += np.sum(ivar[frame_index]*flux[frame_index]**2*(model!=0))

        amp = sum_ivar_flux_prof/sum_ivar_prof2
        amp_ivar = sum_ivar_prof2
        ndf -= 1

        if ndf<2000 : # just a safety cut when we don't have enough overlap between template and data 
            continue

        chi2 = chi2_0 - 2*amp* sum_ivar_flux_prof + amp**2*sum_ivar_prof2
        chi2pdf = chi2/ndf
        
        # my debug
        #zz.append(z)
        #zchi2.append(chi2)
        
        if chi2<np.max(best_chi2s) :
        #if chi2pdf<np.max(best_chi2pdfs) :
            
            need_insert = True
            
            # first find position depending on delta z and chi2
            tmp=np.where(abs(z-best_zs)<0.1)[0]
            if tmp.size>0 : # there is an existing close redshift we replace this entry
                i=tmp[0]                 
                need_insert = False # and we don't have to shift the others
            else :
                # find position depending on value of chi2
                i=np.where(chi2<best_chi2s)[0][0] # take the first slot where chi2 smaller                            
                #i=np.where(chi2pdf<best_chi2pdfs)[0][0] # take the first slot where chi2 smaller             
                
                #print "DEBUG INSERT for chi2pdf=%f z=%f ndf=%d i=%d OTHERS z=%s chi2=%s"%(chi2pdf,z,ndf,i,str(best_zs[best_chi2pdfs<very_large_number]),str(best_chi2pdfs[best_chi2s<very_large_number]))
                best_zs[i+1:]=best_zs[i:-1]
                best_chi2pdfs[i+1:]=best_chi2pdfs[i:-1]
                best_chi2s[i+1:]=best_chi2s[i:-1]
                chi2s_at_best_z_minus_zstep[i+1:]=chi2s_at_best_z_minus_zstep[i:-1]
                chi2s_at_best_z_plus_zstep[i+1:]=chi2s_at_best_z_plus_zstep[i:-1]

        if chi2<best_chi2s[i] :
        #if chi2pdf<best_chi2pdfs[i] :
            
            best_chi2s[i]=chi2
            best_chi2pdfs[i]=chi2pdf
            best_zs[i]=z
            chi2s_at_best_z_minus_zstep[i]=previous_chi2
            best_z_amplitude=amp
            best_z_amplitude_ivar=amp_ivar
            previous_is_at_rank=i

            if not need_insert :
                # print "DEBUG IMPROVE for chi2=%f z=%f i=%d BEST=%s"%(chi2,z,i,str(best_zs[best_chi2s<very_large_number]))
                # but this means we may have to change the ranks
                indices=np.argsort(best_chi2s)
                if np.sum(np.abs(indices-range(best_chi2s.size)))>0 : # need swap
                #indices=np.argsort(best_chi2pdfs)
                #if np.sum(np.abs(indices-range(best_chi2pdfs.size)))>0 : # need swap

                    best_chi2s=best_chi2s[indices]
                    best_chi2pdfs=best_chi2pdfs[indices]
                    best_zs=best_zs[indices]
                    chi2s_at_best_z_minus_zstep=chi2s_at_best_z_minus_zstep[indices]
                    chi2s_at_best_z_plus_zstep=chi2s_at_best_z_plus_zstep[indices]
        
            

        else :
            if previous_is_at_rank>=0 :
                chi2s_at_best_z_plus_zstep[previous_is_at_rank]=chi2
            previous_is_at_rank=-1

        
        previous_chi2=chi2
        
    
    
    
    #if recursive :
        #log.info("first pass best z =%f chi2/ndf=%f, second z=%f dchi2=%f, third z=%f dchi2=%f"%(best_zs[0],best_chi2pdfs[0],best_zs[1],best_chi2s[1]-best_chi2s[0],best_zs[2],best_chi2s[2]-best_chi2s[0]))
        #pylab.plot(zz,zchi2)
        #pylab.show()
        #sys.exit(12)
    
    
    for rank in range(best_zs.size) :
        # we can use the values about best_chi2 to guess the uncertainty on z with a polynomial fit
        coeffs=np.polyfit([best_zs[rank]-zstep,best_zs[rank],best_zs[rank]+zstep],[chi2s_at_best_z_minus_zstep[rank],best_chi2s[rank],chi2s_at_best_z_plus_zstep[rank]],2)
        a=coeffs[0]
        b=coeffs[1]
        c=coeffs[2]
    
        best_z_errors[rank] = zstep
        if a>0 :
            best_zs[rank]       = -b/(2*a)
            best_chi2s[rank]    = c-b**2/(4*a)
            best_z_errors[rank] = 1./math.sqrt(a)
    
    
    if recursive :
        best_results=None
        
        rank_labels = np.array(["BEST","SECOND","THIRD"]) # I know it's a bit ridiculous
        for i in range(rank_labels.size,ntrack) :
            rank_labels=np.append(rank_labels,np.array(["%dTH"%(i+1)])) # even more ridiculous
        
        for rank in range(best_zs.size) :
            # second loop about minimum
            # where we save things to compute errors
            tmp_z       = best_zs[rank]
            tmp_z_error = best_z_errors[rank]
            if tmp_z_error>zstep :
                tmp_z_error = zstep
            tmp_z_error=max(tmp_z_error,0.0001)
            z_nsig=2.
            zmin=tmp_z-z_nsig*tmp_z_error
            zmax=tmp_z+z_nsig*tmp_z_error
            zstep=(zmax-zmin)/10
            tmp_results = simple_zscan(wave,flux,ivar,template_wave,template_flux,zstep=zstep,zmin=zmin,zmax=zmax,wave_nsig=5.,recursive=False)

            if rank == 0 :
                # this is the best
                best_results=tmp_results
            else :
                # here we replace the best values
                labels=np.array(["BEST","SECOND","THIRD"]) # I know it's a bit ridiculous
                for i in range(labels.size,ntrack) :
                    labels=np.append(labels,np.array(["%dTH"%(i+1)])) # even more ridiculous
                
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
            best_results=swapped_best_results

        log.info("best z=%f+-%f chi2/ndf=%3.2f snr=%3.1f dchi2=%3.1f for dz=%f"%(best_results["BEST_Z"],best_results["BEST_Z_ERR"],best_results["BEST_CHI2PDF"],best_results["BEST_SNR"],best_results["SECOND_CHI2"]-best_results["BEST_CHI2"],abs(best_results["BEST_Z"]-best_results["SECOND_Z"])))
        return best_results
    
    res={}
    res["BEST_Z"]=best_zs[0]
    res["BEST_Z_ERR"]=best_z_errors[0]
    res["BEST_CHI2"]=best_chi2s[0]
    res["BEST_CHI2PDF"]=best_chi2pdfs[0]
    res["BEST_SNR"]=best_z_amplitude*math.sqrt(best_z_amplitude_ivar)
    return res



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
    
    args = parser.parse_args()
    log=get_logger()

    log.info("starting")
    
    log.warning("HARDCODED TRUE Z AND GMAG")
    g_grid = np.array([ 22.  ,  22.25,  22.5 ,  22.75,  23.  ])
    z_grid = np.array([ 1.  ,  1.25,  1.5 ,  1.75,  2.  ,  2.25,  2.5 ,  2.75,  3.  , 3.25,  3.5 ])
    gmag = np.tile(g_grid,(z_grid.size,1)).T.ravel()
    z_true = np.tile(z_grid,(g_grid.size,1)).ravel()
    
    b_brick=pyfits.open(args.b)
    r_brick=pyfits.open(args.r)
    z_brick=pyfits.open(args.z)
    
    
    # need to know the wave and flux and z of the templates
    vals=np.loadtxt(args.t).T
    template_wave=vals[0]/(1.+2.4)
    template_flux=vals[1]
    
    #pylab.plot(template_wave,template_flux)
    #pylab.show()

    #qso_spectra=np.where(b_brick.hdu_list[4].data["OBJTYPE"]=="QSO")[0]
    #qso_spectra=qso_spectra[0:2]
    qso_spectra=np.arange(b_brick[0].data.shape[0])
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
        result = simple_zscan(wave,flux,ivar,template_wave,template_flux,zstep=0.001,zmin=0.9,zmax=4.5,wave_nsig=2.,ntrack=3,recursive=True)
        
        
        best_z_array[q]     = result["BEST_Z"]
        best_z_error_array[q]     = result["BEST_Z_ERR"]
        delta_chi2_array[q] = -result["BEST_CHI2"]+result["SECOND_CHI2"]
        best_chi2_array[q]  = result["BEST_CHI2"]
        best_chi2pdf_array[q]  = result["BEST_CHI2PDF"]
        best_snr_array[q]  = result["BEST_SNR"]
        
        log.info("SPEC=%d/%d redshift=%f+-%f chi2pdf=%f dchi2=%f"%(q,nqso,best_z_array[q],best_z_error_array[q],best_chi2pdf_array[q],delta_chi2_array[q]))
    
        save(args.outfile,gmag,z_true,best_z_array,best_z_error_array,best_chi2_array,best_chi2pdf_array,best_snr_array,delta_chi2_array)
    save(args.outfile,gmag,z_true,best_z_array,best_z_error_array,best_chi2_array,best_chi2pdf_array,best_snr_array,delta_chi2_array)

if __name__ == '__main__':
    main()
    

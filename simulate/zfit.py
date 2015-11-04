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
    parser.add_argument('--s', type = int, default = None, required=False,
                        help = 'only one spectrum (for debugging)')
    
    args = parser.parse_args()
    log=get_logger()

    log.info("starting")
    
    log.warning("HARDCODED TRUE Z AND GMAG")
    g_grid = np.array([ 22.  ,  22.25,  22.5 ,  22.75,  23.  ])
    z_grid = np.linspace(0.5,3.5,31)
    gmag = np.tile(g_grid,(z_grid.size,1)).T.ravel()
    z_true = np.tile(z_grid,(g_grid.size,1)).ravel()
    
    b_brick=pyfits.open(args.b)
    r_brick=pyfits.open(args.r)
    z_brick=pyfits.open(args.z)
    
    
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
    

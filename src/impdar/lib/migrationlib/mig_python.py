#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""

Migration routines for ImpDAR

Much of this code is either directly referencing or written from older scripts in SeisUnix:
https://github.com/JohnWStockwellJr/SeisUnix/wiki
Options are:
    Kirchhoff (diffraction summation)
    Stolt (frequency wavenumber, constant velocity)
    Gazdag (phase shift, either constant or depth-varying velocity)
    SeisUnix (reference su routines directly)

Author:
Benjamin Hills
bhills@uw.edu
University of Washington
Earth and Space Sciences

Mar 12 2019

modified Johannes Aichele, 09 2025

"""

from __future__ import print_function
import sys


import numpy as np
import time
from scipy import sparse
from scipy.interpolate import griddata, interp1d, RectBivariateSpline
from scipy.integrate import cumulative_trapezoid


def migrationKirchhoffLoop(data, migdata, tnum, snum, dist, zs, zs2, tt_sec, vel, gradD, max_travel_time, nearfield):
    # Loop through every point in the image (trace and sample)
    # to look for a hyperbola propagating away from that point
    print('Migrating trace number:')
    for xi in range(tnum):
        print('{:d}, '.format(xi), end='')
        sys.stdout.flush()
        for ti in range(snum):
            # get the radial distances between points and the surface location for this trace
            rs = np.sqrt((dist - dist[xi])**2. + zs2[ti])
            # find the cosine of the angle of the tangent line, correct for obliquity factor
            with np.errstate(invalid='ignore'):
                costheta = zs[ti] / rs
            # get the exact indices from the array (closest to rs)
            Didx = np.argmin(np.abs(np.atleast_2d(tt_sec).transpose() - 2. * rs / vel), axis=0)
            # integrate the farfield term
            gradDhyp = gradD[Didx, np.arange(len(Didx))]
            gradDhyp[2. * rs / vel > max_travel_time] = 0.    # zero points that are outside of the domain
            integral = np.nansum(gradDhyp * costheta / vel)  # TODO: Yilmaz eqn 4.5 has an extra r in this weight factor???
            # integrate the nearfield term
            if nearfield:
                Dhyp = data[Didx, np.arange(len(Didx))]
                Dhyp[2. * rs / vel > max_travel_time] = 0.    # zero points that are outside of the domain
                integral += np.nansum(Dhyp * costheta / rs**2.)
            # sum the integrals and output
            migdata[ti, xi] = 1. / (2. * np.pi) * integral


def migrationKirchhoff(dat, vel=1.69e8, nearfield=False):
    """Kirchhoff Migration (Berkhout 1980; Schneider 1978; Berryhill 1979)

    This migration method uses an integral solution to the scalar wave equation Yilmaz (2001) eqn 4.5.
    The algorithm cycles through every sample in each trace, creating a hypothetical diffraciton
    hyperbola for that location,
        t(x)^2 = t(0)^2 + (2x/v)^2
    To migrate, we integrate the power along that hyperbola and assign the solution to the apex point.
    There are two terms in the integral solution, Yilmaz (2001) eqn 4.5, a far-field term and a
    near-field term. Most algorithms ignore the near-field term because it is small. Here there is an option,
    but default is to ignore.

    Parameters
    ---------
    dat: data as a class in the ImpDAR format
    vel: wave velocity, default is for ice
    nearfield: boolean to indicate whether or not to use the nearfield term in summation

    Output
    ---------
    dat: data as a class in the ImpDAR format (with dat.data now being migrated data)

    """

    print('Kirchhoff Migration (diffraction summation) of %.0fx%.0f matrix' % (dat.snum, dat.tnum))
    # check that the arrays are compatible
    _check_data_shape(dat)
    # start the timer
    start = time.time()
    # Calculate the time derivative of the input data
    gradD = np.gradient(dat.data, dat.travel_time / 1.0e6, axis=0)
    # Create an empty array to fill with migrated data
    migdata = np.zeros_like(dat.data, dtype=np.float64)

    # Try to cache some variables that we need lots
    tt_sec = dat.travel_time / 1.0e6
    max_travel_time = np.max(tt_sec)
    # Cache the depths
    zs = vel * tt_sec / 2.0
    zs2 = zs**2.

    migrationKirchhoffLoop(dat.data.astype(np.float64),
                          migdata,
                          dat.tnum,
                          dat.snum,
                          np.ascontiguousarray(dat.dist, dtype=np.float64) * 1.0e3,
                          np.ascontiguousarray(zs, dtype=np.float64),
                          np.ascontiguousarray(zs2, dtype=np.float64),
                          np.ascontiguousarray(tt_sec, dtype=np.float64),
                          vel,
                          np.ascontiguousarray(gradD, dtype=np.float64),
                          max_travel_time,
                          nearfield
                          )

    dat.data = migdata.copy()
    # print the total time
    print('')
    print('Kirchhoff Migration of %.0fx%.0f matrix complete in %.2f seconds'
          % (dat.snum, dat.tnum, time.time() - start))
    return dat


def migrationStolt(dat,vel=1.68e8,htaper=100,vtaper=1000):
    """Stolt Migration (Stolt, 1978, Geophysics)

    This is by far the fastest migration method. It is a simple transformation from
    frequency-wavenumber (FKx) to wavenumber-wavenumber (KzKx) space.

    Parameters
    ---------
    dat: data as a class in the ImpDAR format
    vel: wave velocity, default is for ice
    htaper: number of traces for the linear horizontal taper from the edges of the domain
    vtaper: number of samples for the vertical taper from the top and bottom.

    Output
    ---------
    dat: data as a class in the ImpDAR format (with dat.data now being migrated data)

    """

    print('Stolt Migration (f-k migration) of %.0fx%.0f matrix'%(dat.snum, dat.tnum))
    # check that the arrays are compatible
    _check_data_shape(dat)

    # save the start time
    start = time.time()
    # taper
    h = np.minimum(np.arange(dat.tnum),np.arange(dat.tnum)[::-1])/htaper
    v = np.minimum(np.arange(dat.snum),np.arange(dat.snum)[::-1])/vtaper
    h[h>1.] = 1.
    v[v>1.] = 1.
    H,V = np.meshgrid(h,v)
    dat.data = (dat.data*H*V).astype(dat.data.dtype)
    # 2D Forward Fourier Transform to get data in frequency-wavenumber space, FK = D(kx,z=0,ws)
    FK = np.fft.rfft2(dat.data, axes=(1, 0))
    # get the temporal frequencies
    ws = 2.*np.pi*np.fft.rfftfreq(dat.snum, d=dat.dt)
    # get the horizontal wavenumbers
    if np.mean(dat.trace_int) <= 0:
        Warning("The trace spacing, variable 'dat.trace_int', should be greater than 0. Using gradient(dat.dist) instead.")
        trace_int = np.gradient(dat.dist)
    else:
        trace_int = dat.trace_int
    kx = 2.*np.pi*np.fft.fftfreq(dat.tnum, d=np.mean(trace_int))
    # interpolate from frequency (ws) into wavenumber (kz)
    print(kx.shape, ws.shape, FK.real.shape)
    interp_real = RectBivariateSpline(np.fft.fftshift(kx), ws, np.fft.fftshift(FK.real, axes=[1]).T, kx=1, ky=1)
    interp_imag = RectBivariateSpline(np.fft.fftshift(kx), ws, np.fft.fftshift(FK.imag, axes=[1]).T, kx=1, ky=1)

    # interpolation will move from frequency-wavenumber to wavenumber-wavenumber, KK = D(kx,kz,t=0)
    KK = np.zeros_like(FK)
    print('Interpolating from temporal frequency (ws) to vertical wavenumber (kz):')
    print('Interpolating:')
    # for all temporal frequencies
    for zj in range(dat.snum // 2):
        kzj = ws[zj]*2./vel
        if zj%100 == 0:
            print(int(ws[zj]/1e6/2/np.pi), 'MHz, ', end='')
            sys.stdout.flush()
        # for all horizontal wavenumbers
        for xi in range(len(kx)):
            kxi = kx[xi]
            # migration conversion to wavenumber (Yilmaz equation C.53)
            wsj = vel/2.*np.sqrt(kzj**2.+kxi**2.)
            # get the interpolated FFT values, real and imaginary, S(kx,kz,t=0)
            KK[zj,xi] = interp_real(kxi,wsj)[0, 0] + 1j*interp_imag(kxi,wsj)[0, 0]
    # all vertical wavenumbers
    kz = ws*2./vel
    # grid wavenumbers for scaling calculation
    kX,kZ = np.meshgrid(kx,kz)
    # scaling for obliquity factor (Yilmaz equation C.56)
    with np.errstate(invalid='ignore'):
        scaling = kZ/np.sqrt(kX**2.+kZ**2.)
    KK *= scaling
    # the DC frequency should be 0.
    KK[0,0] = 0.+0j
    # 2D Inverse Fourier Transform to get back to distance spce, D(x,z,t=0)
    dat.data = np.fft.irfft2(KK, axes=(1, 0))

    # print the total time
    print('')
    print('Stolt Migration of %.0fx%.0f matrix complete in %.2f seconds'
          % (dat.snum, dat.tnum, time.time() - start))
    return dat


def migrationPhaseShift(dat,vel=1.69e8,vel_fn=None,htaper=100,vtaper=1000,nextPowerVal=1, **genfromtxt_kwargs):
    """

    Phase-Shift Migration
    case with constant velocity, v=constant (Gazdag 1978, Geophysics)
    case with layered velocities, v(z) (Gazdag 1978, Geophysics)
    case with vertical and lateral velocity variations, v(x,z) (Ristow and Ruhl 1994, Geophysics)

    Phase-shifting migration for constant, layered, and laterally varying velocity structures.
    This method works down from the surface, using the imaging principle
    (i.e. summing over all frequencies to get the solution at t=0)
    for the migrated section at each step.

    **
    The foundation of this script was taken from two places:
    Matlab code written by Andreas Tzanis, Dept. of Geophysics, University of Athens (2005)
    Seis Unix script sumigffd.c, Credits: CWP Baoniu Han, July 21th, 1997
    **

    Parameters
    ---------
    dat: data as a class in the ImpDAR format
    vel: v(x,z)
        Up to 2-D array with three columns for velocities (m/s), and z/x (m).
        Array structure is velocities in first column, z location in second, x location in third.
        If uniform velocity (i.e. vel=constant) input constant
        If layered velocity (i.e. vel=v(z)) input array with shape (#vel-points, 2) (i.e. no x-values)
    nextPowerVal: integer power of 2 for padding the number of traces (i.e. 1=2, 2=4, 3=8, etc)    
    vel_fn: filename for layered velocity input, .txt file with columns for v, x, z

    Output
    ---------
    dat: data as a class in the ImpDAR format (with dat.data now being migrated data)

    """

    print('Phase-Shift Migration of %.0fx%.0f matrix'%(dat.snum,dat.tnum))
    # check that the arrays are compatible
    _check_data_shape(dat)

    # save the start time
    start = time.time()
    # taper
    h = np.minimum(np.arange(dat.tnum),np.arange(dat.tnum)[::-1])/htaper
    v = np.minimum(np.arange(dat.snum),np.arange(dat.snum)[::-1])/vtaper
    h[h>1.] = 1.
    v[v>1.] = 1.
    H,V = np.meshgrid(h,v)
    dat.data *= H*V
    # pad the array with zeros up to the next power of 2 for discrete fft
    nt = 2**(nextPowerVal-1)*2**(np.ceil(np.log(dat.snum)/np.log(2))).astype(int)
    # get frequencies and wavenumbers
    if np.mean(dat.trace_int) <= 0:
        Warning("The trace spacing, variable 'dat.trace_int', should be greater than 0. Using gradient(dat.dist) instead.")
        trace_int = np.gradient(dat.dist)
    else:
        trace_int = dat.trace_int
    kx = 2.*np.pi*np.fft.fftfreq(dat.tnum,d=np.mean(trace_int))
    ws = 2.*np.pi*np.fft.fftfreq(nt,d=dat.dt)
    # 2D Forward Fourier Transform to get data in frequency-wavenumber space, FK = D(kx,z=0,ws)
    FK = np.fft.fft2(dat.data,(nt,dat.tnum))
    # Velocity structure from input
    if vel_fn is not None:
        try:
            vel = np.genfromtxt(vel_fn, **genfromtxt_kwargs)
            print('Velocities loaded from %s.'%vel_fn)
        except:
            raise TypeError('File %s was given for input velocity array, but cannot be loaded. Please reformat to txt file.'%vel_fn)
    if hasattr(vel, 'shape') and vel.shape == dat.data.shape:
        vmig = getVelocityMatrix(dat,vel)
        print('directly use velocity matrix')
    else:      
        vmig = getVelocityProfile(dat,vel)
    # Migration by phase shift, frequency-wavenumber (FKx) to time-wavenumber (TKx)
    TK = phaseShift(dat, vmig, vel, kx, ws, FK)
    # Transform from time-wavenumber (TKx) to time-space (TX) domain to get migrated section
    dat.data = np.fft.ifft(TK).real
    # print the total time
    print('')
    print('Phase-Shift Migration of %.0fx%.0f matrix complete in %.2f seconds'
          % (dat.snum, dat.tnum, time.time() - start))
    return dat


def migrationTimeWavenumber(dat,vel=1.69e8,vel_fn=None,htaper=100,vtaper=1000):
    """

    Time-Wavenumber Migration

    The migration is a reverse time migration in the (t,k) domain. In the
    first step, the data g(t,x) are Fourier transformed x->k into
    the time-wavenumber domain g(t,k).
    Then looping over wavenumbers, the data are then reverse-time
    finite-difference migrated, wavenumber by wavenumber.  The resulting
    migrated data m(tau,k), now in the tau (migrated time) and k domain,
    are inverse fourier transformed back into m(tau,xout) and written out.

    **
    The foundation of this script was taken from:
    Seis Unix script sumigtk.c, Credits: CWP Dave Hale, November 5th, 1990
    **

    Parameters
    ---------
    dat: data as a class in the ImpDAR format
    vel: v(x,z)
        Up to 2-D array with three columns for velocities (m/s), and z/x (m).
        Array structure is velocities in first column, z location in second, x location in third.
        If uniform velocity (i.e. vel=constant) input constant
        If layered velocity (i.e. vel=v(z)) input array with shape (#vel-points, 2) (i.e. no x-values)
    vel_fn: filename for layered velocity input, .txt file with columns for v, x, z

    Output
    ---------
    dat: data as a class in the ImpDAR format (with dat.data now being migrated data)

    """
    print('Time-Wavenumber Migration of %.0fx%.0f matrix'%(dat.snum, dat.tnum))
    # check that the arrays are compatible
    _check_data_shape(dat)

    # save the start time
    start = time.time()
    # taper
    h = np.minimum(np.arange(dat.tnum),np.arange(dat.tnum)[::-1])/htaper
    v = np.minimum(np.arange(dat.snum),np.arange(dat.snum)[::-1])/vtaper
    h[h>1.] = 1.
    v[v>1.] = 1.
    H,V = np.meshgrid(h,v)
    dat.data *= H*V
    # get wavenumbers
    if np.mean(dat.trace_int) <= 0:
        Warning("The trace spacing, variable 'dat.trace_int', should be greater than 0. Using gradient(dat.dist) instead.")
        trace_int = np.gradient(dat.dist)
    else:
        trace_int = dat.trace_int
    kx = 2.*np.pi*np.fft.fftfreq(dat.tnum,d=np.mean(trace_int))
    # 1D Forward Fourier Transform to get data in time-wavenumber space, TK = D(kx,z=0,ts)

    # Loop over wavenumbers for reverse time migration
    for k in kx:
        continue

    # 1D Inverse Fourier Transform to get data back into migrated time-distance space

    # print the total time
    print('')
    print('Time-Wavenumber Migration of %.0fx%.0f matrix complete in %.2f seconds'
          % (dat.snum, dat.tnum, time.time() - start))
    return dat

# -----------------------------------------------------------------------------
# Supporting functions
# -----------------------------------------------------------------------------

def phaseShift(dat, vmig, vels_in, kx, ws, FK):
    """

    Phase-Shift migration to get from frequency-wavenumber (FKx) space to time-wavenumber (TKx) space.
    This is for either constant or layered velocity v(z).

    **
    The foundation of this script was taken from Matlab code written by Andreas Tzanis,
    Dept. of Geophysics, University of Athens (2005)
    **

    Parameters
    ---------
    dat: data as a dictionary in the ImpDAR format
    vmig: migration velocity (m/s)
        can be constant or 1-D or 2-D array
    vels_in: v(x,z)
        Up to 2-D array with three columns for velocities (m/s), and z/x (m).
        Array structure is velocities in first column, z location in second, x location in third.
        If uniform velocity (i.e. vel=constant) input constant
        If layered velocity (i.e. vel=v(z)) input array with shape (#vel-points, 2) (i.e. no x-values)
    kx: horizontal wavenumbers
    ws: temporal frequencies
    FK: 2-D array of the data image in frequency-wavenumber space (FKx)

    Output
    ---------
    TK: 2-D array of the migrated data image in time-wavenumber space (TKx).

    """

    # initialize the time-wavenumber array to be filled with complex values
    TK = np.zeros((dat.snum,len(kx)))+0j

    # Uniform velocity case, vmig=constant
    if not hasattr(vmig,"__len__"):
        print('Constant velocity %s m/usec'%(vmig/1e6))
        # iterate through all frequencies
        print('Frequency: ',end='')
        sys.stdout.flush()

        print('Frequency: ')
        for iw in range(len(ws)):
            w = ws[iw]
            if w == 0.0:
                w = 1e-10/dat.dt
            if iw%100 == 0:
                print(int(w/1e6/(2.*np.pi)),'MHz',', ',end='')
                sys.stdout.flush()
            # remove frequencies outside of the domain
            vkx2 = (vmig*kx/2.)**2.
            ik = np.argwhere(vkx2 < w**2.)
            FFK = FK[iw,ik]
            # get the phase for shift
            phase = (-w*dat.dt*np.sqrt(1.0 - vkx2[ik]/w**2.)).real
            cp = np.conj(np.cos(phase)+1j*np.sin(phase))
            # Accumulate output image (time-wavenumber space) summed over all frequencies
            for itau in range(dat.snum):
                 FFK *= cp
                 TK[itau,ik] += FFK

    else:
        if not hasattr(vmig, 'shape'):
            raise ValueError('vmig needs to be an array or float')
        # Layered and/or lateral velocity case, vmig=v(x,z)
        if len(vmig) != dat.snum:
            raise ValueError('Interpolated velocity profile is not the length of the number of samples in a trace.')
        if hasattr(vmig[0], "__len__"):
            print('2-D velocity structure, Fourier Finite-Difference Migration')
            # Finite Difference Stencil
            stencil = Sp_Matr(dat.tnum,-2,1,1)
            FFX_last = 0.
        else:
            print('1-D velocity structure, Gazdag Migration')
            print('Velocities (m/s): %.2e',vels_in[:,0])
            print('Depths (m):',vels_in[:,1])
            print(r'Travel Times ($\mu$ sec):',dat.travel_time)
        # iterate through all output travel times
        for itau in range(dat.snum):
            tau = dat.travel_time[itau] / 1.0e6
            if itau%100 == 0:
                print('Time %.2e, ' %(tau), end='')
                sys.stdout.flush()
            # iterate through all frequencies
            for iw in range(len(ws)):
                w = ws[iw]
                if w == 0.0:
                    w = 1.0e-10 / dat.dt

                # Get foreground and background velocities
                if hasattr(vmig[itau], "__len__"):
                    vbg = np.min(vmig[itau])  # Stoffa et al. 1990's 1 / U_0 for the depth interval
                    vfg = vmig[itau]-vbg  # Stoffa et al. 1990's 1 / DeltaU
                    ufg  = 1. / vmig[itau] - 1. / vbg  # Stoffa's DeltaU
                else:
                    vbg = vmig[itau] # Stoffa et al. 1990's U_0 for the depth interval

                ### Retardation term
                # cosine squared
                coss = 1.0+0j - (0.5*vbg*kx/w)**2.
                # calculate phase for shift
                phase = (-w*dat.dt*np.sqrt(coss)).real
                cshift = np.conj(np.cos(phase)+1j*np.sin(phase))
                FK[iw] *= cshift

                if hasattr(vmig[itau],"__len__"):
                    # inverse fourier tranform to frequency-space domain
                    FFX = np.fft.ifft(FK[iw])

                    ### Thin-lens term (Stoffa et al. 1990)
                    phase2 = 2. * ufg * w * dat.dt + 1. * vbg * w * dat.dt
                    cshift2 = np.cos(phase2) + 1j*np.sin(phase2)
                    FFX *= cshift2

                    ### Diffraction term, Finite Difference operator
                    if itau > 0:
                        FFX = fourierFiniteDiff(dat,vfg,w,FFX,FFX_last,stencil)
                    FFX_last = FFX

                    # Fourier transform back to frequency-wavenumber domain
                    FK[iw] = np.fft.fft(FFX)

                # zero if outside domain
                idx = coss <= (tau/dat.travel_time[-1]/1e6)**2.
                FK[iw,idx] = 0.0 + 0j
                # sum over all frequencies
                TK[itau] += FK[iw]

    # Cut to original array size
    TK = TK[:,:dat.tnum]
    # Normalize for inverse FFT
    TK /= dat.snum
    return TK


def fourierFiniteDiff(dat, vs, w, FFX, FFX_last, stencil, alpha=0.5,beta=0.25):
    """

    Fourier Finite-Difference operator to correct for diffraction in the phase-shift method.
    This is for variable velocity v(x,z).

    Parameters
    ---------
    dat: data as a dictionary in the ImpDAR format
    vs: 1-D array of migration velocity (m/s)
    w: scalar temporal frequency
    FFX: 2-D array of the data image in frequency-space (FX)
    FFX_last: same as FFX but for the last iteration (i.e. tau-1)
    alpha: coefficient on second order term, default to 0.5 for 45-degree equation
    beta: coefficient on third order term, default to 0.25 for 45-degree equation

    Output
    ---------
    FFX: Updated input term, 2-D array of the data image in frequency-space (FX)

    """

    # Coefficients
    dx = np.mean(dat.trace_int)
    coeff1 = dat.dt*alpha*vs**2./(1j*4.*w*dx**2.)
    coeff2 = -beta*vs**2./(4.*w**2.*dx**2.)

    # Update equation, explicit backward Euler
    FFX = FFX_last + coeff1*(stencil*FFX) + coeff2*(stencil*FFX - stencil*FFX_last)
    return FFX


def Sp_Matr(N,diag,k1,k2,k3=0,k4=0,nx=0):
    A = sparse.lil_matrix((N, N))           #Function to create a sparse Matrix
    A.setdiag((diag)*np.ones(N))            #Set the diagonal
    A.setdiag((k1)*np.ones(N-1),k=1)        #Set the first upward off-diagonal.
    A.setdiag((k2)*np.ones(N-1),k=-1)       #Set the first downward off-diagonal
    A.setdiag((k3)*np.ones(N-nx),k=nx)      #Set for diffusion from above node
    A.setdiag((k4)*np.ones(N-nx),k=-nx)     #Set for diffusion from below node
    # Set Dirichlet boundary conditions
    A[0,0] = 1
    A[0,1:] = 0
    A[-1,-1] = 1
    A[-1,:-1] = 1
    return A


def getVelocityProfile(dat,vels_in):
    """

    Map the layered velocity structure into the shape of the data.

    Parameters
    ---------
    dat: data as a dictionary in the ImpDAR format
    vels_in: v(x,z)
        Up to 2-D array with three columns for velocities (m/s), and z/x (m).
        Array structure is velocities in first column, z location in second, x location in third.
        If uniform velocity (i.e. vel=constant) input constant
        If layered velocity (i.e. vel=v(z)) input array with shape (#vel-points, 2) (i.e. no x-values)

    Output
    ---------
    vmig: 2-D array of migration velocities (m/s), shape is (#traces, #samples).
        If constant input velocity, output is constant.
        If only z-component in input velocity array, output is v(z)

    """

    # return the input value if it is a constant
    if not hasattr(vels_in,"__len__"):
        return vels_in

    start = time.time()
    print('Interpolating the velocity profile.')

    if len(np.shape(vels_in)) != 2 or np.shape(vels_in)[1] == 1:
        raise ValueError('If non-constant vel, inputs needs to be 2d (v, z) or (v, z, x)')
    nlay, dimension = np.shape(vels_in)
    vel_v = vels_in[:,0]
    vel_z = vels_in[:,1]

    twtt = dat.travel_time.copy() / 1.0e6
    ### Layered Velocity
    if nlay == 1:
        raise ValueError('It does not make sense to only give one layer of velocity--if you want constant velocity just input v')
    elif dimension == 2:
        zs = np.max(vel_v)/2.*twtt      # depth array for maximum possible penetration
        zs[0] = twtt[0]*vel_v[0]/2.
        # If an input point is closest to a boundary push it to the boundary
        # This will suppress some desired errors though, so use this if to try to guard
        if (vel_z[0] > 1.1 * np.nanmin(zs) and vel_z[0] / np.nanmax(zs) > 1.0e-3) or vel_z[-1] * 1.1 < np.nanmax(zs):
            raise ValueError('Your velocity data doesnt come close to covering the depths in the data')
        if vel_z[0] > np.nanmin(zs):
            vel_v = np.insert(vel_v,0,vel_v[np.argmin(vel_z)])
            vel_z = np.insert(vel_z,0,np.nanmin(zs))
        if vel_z[-1] < np.nanmax(zs):
            vel_v = np.append(vel_v,vel_v[np.argmax(vel_z)])
            vel_z = np.append(vel_z,np.nanmax(zs))
        # Compute times from input velocity/location array (vels_in)
        vel_t = 2.*vel_z/vel_v
        # Interpolate to get t(z) for maximum penetration depth array
        tinterp = interp1d(vel_z,vel_t)
        tofz = tinterp(zs)
        # Compute z(t) from monotonically increasing t
        zinterp = interp1d(tofz,zs)
        zoft = zinterp(twtt)
        # Compute vmig(t) from z(t)
        vmig = 2.*np.gradient(zoft,twtt)

    ### Lateral Velocity Variations TODO: I need to check this more rigorously too.
    elif dimension == 3:
        vel_x = vels_in[:,2]    # Input velocities
        # Depth array for largest penetration range
        zs = np.linspace(np.min(vel_v)*twtt[0],
                 np.max(vel_v)*twtt[-1],
                 dat.snum)/2.
        # Use nearest neighbor interpolation to grid the input points onto a mesh
        if dat.dist is None or all(dat.dist == 0):
            raise ValueError('The distance vector was never set.')
        XS,ZS = np.meshgrid(dat.dist,zs)
        VS = griddata(np.transpose([vel_x,vel_z]),vel_v,np.transpose([XS.flatten(),ZS.flatten()]),method='nearest')
        VS = np.reshape(VS,np.shape(XS))

        # convert velocities into travel_time space for all traces
        vmig = np.zeros_like(VS)
        for i in range(dat.tnum):
            vel_z = ZS[:,i]
            vel_v = VS[:,i]
            # Compute times from input velocity/location array (vels_in)
            vel_t = 2*np.array([np.trapz(1./vel_v[:j],vel_z[:j]) for j in range(dat.snum)])
            # Interpolate to get t(z) for maximum penetration depth array
            tinterp = interp1d(ZS[:,i],vel_t)
            tofz = tinterp(zs)
            # Compute z(t) from monotonically increasing t
            zinterp = interp1d(tofz,zs)
            if twtt[-1] > tofz[-1]:
                raise ValueError('Two-way travel time array extends outside of interpolation range')
            zoft = zinterp(twtt)
            # Compute vmig(t) from z(t)
            vmig[:,i] = 2.*np.gradient(zoft,twtt)
    else:
        # We get here if the number of columns is bad
        raise ValueError('Input must be 2d with 2 or 3 columns')

    print('Velocity profile finished in %.2f seconds.'%(time.time()-start))

    return vmig


def _check_data_shape(dat):
    if np.size(dat.data, 1) != dat.tnum or np.size(dat.data, 0) != dat.snum:
        raise ValueError('The input array must be of size (snum, tnum)')


# Compute migration velocity from a physical velocity matrix (in (time, trace) space)
def getVelocityMatrix(dat, vels_in):
    """
    Given a physical velocity matrix (in (time, trace) space), compute the migration velocity profile (vmig)
    as required for migration: vmig = 2 * np.gradient(depth, time) for each trace.

    Parameters
    ----------
    dat: data as a class in the ImpDAR format
    vels_in: np.ndarray
        Physical velocity matrix of shape (dat.snum, dat.tnum), in m/s

    Returns
    -------
    vmig: np.ndarray
        Migration velocity matrix, shape (dat.snum, dat.tnum)
    """
    if not isinstance(vels_in, np.ndarray):
        raise TypeError("vels_in must be a numpy ndarray.")
    if vels_in.shape != dat.data.shape:
        raise ValueError(f"vels_in shape {vels_in.shape} does not match data shape {dat.data.shape}.")
    # Get time axis in seconds (assume dat.travel_time is in microseconds)
    if hasattr(dat, 'travel_time'):
        twtt = dat.travel_time.copy() / 1.0e6
    else:
        raise AttributeError("dat must have attribute 'travel_time' (in microseconds)")
    vmig = np.zeros_like(vels_in)
    for i in range(vels_in.shape[1]):
        # Integrate velocity to get depth as a function of time for this trace
        # depth(t) = \int_0^t v(t') * dt' / 2 (divide by 2 for two-way travel time)
        depth = cumulative_trapezoid(vels_in[:, i], twtt, initial=0) / 2
        # Compute migration velocity: 2 * dz/dt
        vmig[:, i] = 2 * np.gradient(depth, twtt)
    return vmig

def migrationKirchhoffPython_2(dat, vel=1.69e8, nearfield=False, progress=100, use_aperture=None,
                               aperture_angle_deg=None, fresnel=True, f0=None, fresnel_factor=1.0):
    """Optimized pure-Python Kirchhoff migration (same physics as migrationKirchhoff).

    Added options:
    aperture_angle_deg : float | None
        Half-cone angle (degrees) describing EM radiation / reception cone. Depth-dependent lateral
        limit = z * tan(theta). Covers unshielded (large angle ~80-90) to strongly shielded (small angle ~10-20).
    fresnel : bool
        If True and f0 provided, also compute first Fresnel radius R_F = sqrt( (vel/f0)*z / 2 ) and use
        effective lateral limit = min(angle_limit, fresnel_factor * R_F) (if angle given) or just fresnel limit.
    f0 : float | None
        Dominant frequency (Hz) for Fresnel calculation (required if fresnel True and fresnel constraint desired).
    fresnel_factor : float
        Multiplier on Fresnel radius (e.g. 1.0–2.0) to widen or tighten.

    Irregular spacing handling (NEW):
        Lateral quadrature weights w are applied so the lateral integral approximates ∫ A(x) dx for non-uniform x.
        w[0] = x1-x0, w[-1] = x_{N-1}-x_{N-2}, interior w_i = 0.5*(x_{i+1}-x_{i-1}).

    Units:
        Input dat.dist expected in kilometers (legacy ImpDAR convention). Internally converted to meters (dist_m = dat.dist * 1e3)
        for geometric calculations and weighting.

    Priority of lateral limiting per depth ti:
        1. If aperture_angle_deg or fresnel constraints given -> depth-dependent mask.
        2. Else if use_aperture (global meters) -> constant mask each trace.
        3. Else full aperture.
    """
    _check_data_shape(dat)
    print('Optimized Kirchhoff Migration (Python) of %.0fx%.0f matrix' % (dat.snum, dat.tnum))
    import numpy as _np, time as _time, math as _math
    start = _time.time()

    tt_sec = _np.ascontiguousarray(dat.travel_time.ravel(), dtype=_np.float64) / 1.0e6
    snum = dat.snum; tnum = dat.tnum
    if tt_sec.shape[0] != snum:
        raise ValueError('travel_time length mismatch')
    zs = vel * tt_sec / 2.0
    zs2 = zs * zs
    max_travel_time = tt_sec[-1]
    # Convert km -> m explicitly
    dist_m = _np.ascontiguousarray(dat.dist, dtype=_np.float64) * 1.0e3
    if dist_m.shape[0] != tnum:
        raise ValueError('dist length mismatch')
    gradD = _np.gradient(_np.ascontiguousarray(dat.data, dtype=_np.float64), tt_sec, axis=0)
    data_in = _np.ascontiguousarray(dat.data, dtype=_np.float64) if nearfield else None
    migdata = _np.zeros_like(gradD, dtype=_np.float64)
    two_over_vel = 2.0 / vel; inv_2pi = 1.0 / (2.0 * _np.pi)

    if aperture_angle_deg is not None:
        theta_rad = _math.radians(aperture_angle_deg)
        angle_limits = zs * _math.tan(theta_rad)
    else:
        angle_limits = None
    if fresnel and f0 is not None and f0 > 0:
        wavelength = vel / f0
        fresnel_limits = _np.sqrt((wavelength * zs) / 2.0) * fresnel_factor
    else:
        fresnel_limits = None

    def _depth_limit(idx_depth):
        al = angle_limits[idx_depth] if angle_limits is not None else None
        fl = fresnel_limits[idx_depth] if fresnel_limits is not None else None
        if al is not None and fl is not None:
            return min(al, fl)
        return al if al is not None else fl

    def _nearest_time_indices(times_target):
        idx = _np.searchsorted(tt_sec, times_target, side='left')
        over = idx >= snum; idx[over] = snum - 1
        mask = idx > 0
        left = idx - 1
        dt_r = _np.abs(tt_sec[idx] - times_target)
        dt_l = _np.empty_like(dt_r); dt_l[:] = _np.inf; dt_l[mask] = _np.abs(tt_sec[left[mask]] - times_target[mask])
        use_left = dt_l < dt_r; idx[use_left] = left[use_left]
        return idx

    use_ap = use_aperture if (use_aperture is not None and use_aperture > 0) else None

    for xi in range(tnum):
        if progress and (xi % progress == 0):
            elapsed = _time.time() - start
            eta = (elapsed * (tnum - xi) / xi) if xi > 0 else 0.0
            print(f'Trace {xi}/{tnum}  elapsed {elapsed:.1f}s ETA {eta:.1f}s')
        # Precompute full trace index arrays for no-aperture/global-aperture cases
        full_idx = _np.arange(tnum)
        if use_ap is not None and angle_limits is None and fresnel_limits is None:
            mask_global = _np.abs(dist_m - dist_m[xi]) <= use_ap
            dist_global = dist_m[mask_global]
            idx_global = full_idx[mask_global]
        for ti in range(snum):
            # Determine lateral subset for this depth
            if angle_limits is not None or fresnel_limits is not None:
                L = _depth_limit(ti)
                if L is None:  # fallback to global / full
                    if use_ap is not None:
                        mask_tr = _np.abs(dist_m - dist_m[xi]) <= use_ap
                        dist_sel = dist_m[mask_tr]; trace_idx = full_idx[mask_tr]
                    else:
                        dist_sel = dist_m; trace_idx = full_idx
                else:
                    mask_tr = _np.abs(dist_m - dist_m[xi]) <= L
                    if not mask_tr.any():
                        continue
                    dist_sel = dist_m[mask_tr]; trace_idx = full_idx[mask_tr]
            else:
                if use_ap is not None:
                    dist_sel = dist_global; trace_idx = idx_global
                else:
                    dist_sel = dist_m; trace_idx = full_idx

            dx = dist_sel - dist_m[xi]; dx2 = dx * dx
            rs = _np.sqrt(dx2 + zs2[ti])
            t_hyp = two_over_vel * rs
            outside = t_hyp > max_travel_time
            if outside.all():
                continue
            idx_time = _nearest_time_indices(t_hyp)
            if outside.any():
                idx_time[outside] = 0
            with _np.errstate(invalid='ignore', divide='ignore'):
                costheta = zs[ti] / rs
            if outside.any():
                costheta[outside] = 0.0
            grad_slice = gradD[idx_time, trace_idx]
            grad_slice[outside] = 0.0
            # Lateral quadrature weights for irregular spacing (meters)
            nloc = dist_sel.size
            if nloc == 1:
                w = _np.array([0.0], dtype=dist_sel.dtype)
            elif nloc == 2:
                w = _np.empty(2, dtype=dist_sel.dtype); w[:] = dist_sel[1] - dist_sel[0]
            else:
                w = _np.empty(nloc, dtype=dist_sel.dtype)
                w[0] = dist_sel[1] - dist_sel[0]
                w[-1] = dist_sel[-1] - dist_sel[-2]
                w[1:-1] = 0.5 * (dist_sel[2:] - dist_sel[:-2])
            if outside.any():
                w[outside] = 0.0
            integral = _np.nansum(grad_slice * costheta / vel * w)
            if nearfield:
                data_slice = data_in[idx_time, trace_idx]
                data_slice[outside] = 0.0
                with _np.errstate(divide='ignore', invalid='ignore'):
                    integral += _np.nansum(data_slice * costheta / (rs * rs) * w)
            migdata[ti, xi] = inv_2pi * integral

    dat.data = migdata
    print('\nOptimized Kirchhoff Migration complete in %.2f seconds' % (time.time() - start))
    return dat


try:
    import numba as _nb
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False


if _NUMBA_OK:
    @_nb.njit(parallel=True, fastmath=True)
    def _kirchhoff_numba_kernel(migdata,
                                gradD,
                                data_in,
                                dist_m,
                                zs,
                                zs2,
                                vel,
                                tt0,
                                dt,
                                max_travel_time,
                                nearfield,
                                depth_limits):
        # Numba kernel with lateral weighting and explicit km->m conversion already applied before call.
        inv_2pi = 1.0 / (2.0 * 3.141592653589793)
        two_over_vel = 2.0 / vel
        snum = zs.shape[0]
        tnum = dist_m.shape[0]
        for xi in _nb.prange(tnum):
            x0 = dist_m[xi]
            for ti in range(snum):
                z = zs[ti]; z2 = zs2[ti]
                lateral_limit = depth_limits[ti]
                integral = 0.0
                for direction in (0, 1):
                    if direction == 0:
                        k = xi; step = 1; stop_cond = tnum
                    else:
                        k = xi - 1; step = -1; stop_cond = -1
                    while k != stop_cond:
                        dx = dist_m[k] - x0
                        if lateral_limit >= 0.0 and (dx > lateral_limit or dx < -lateral_limit):
                            break
                        rs = (dx * dx + z2) ** 0.5
                        t_hyp = two_over_vel * rs
                        if t_hyp > max_travel_time:
                            break
                        it = int((t_hyp - tt0) / dt + 0.5)
                        if it < 0:
                            it = 0
                        elif it >= snum:
                            it = snum - 1
                        if rs > 0.0:
                            costheta = z / rs
                        else:
                            costheta = 0.0
                        # Lateral weight (irregular spacing) from full grid
                        if k == 0:
                            if tnum > 1:
                                w = dist_m[1] - dist_m[0]
                            else:
                                w = 0.0
                        elif k == tnum - 1:
                            w = dist_m[tnum - 1] - dist_m[tnum - 2]
                        else:
                            w = 0.5 * (dist_m[k + 1] - dist_m[k - 1])
                        integral += gradD[it, k] * costheta / vel * w
                        if nearfield:
                            integral += data_in[it, k] * costheta / (rs * rs) * w
                        k += step
                migdata[ti, xi] = inv_2pi * integral


def migrationKirchhoffPython_numba(dat,
                                   vel=1.69e8,
                                   nearfield=False,
                                   aperture_angle_deg=None,
                                   fresnel=True,
                                   f0=None,
                                   fresnel_factor=1.0,
                                   use_aperture=None):
    """Numba-accelerated Kirchhoff migration (with lateral weighting for irregular spacing).

    Physics: identical to migrationKirchhoffPython_2 (far-field + optional near-field) plus depth-dependent aperture.

    Units:
        Input dat.dist expected in kilometers. Internally converted to meters (dist_m = dat.dist * 1e3).
        All geometric / weighting calculations use meters.

    Weighting:
        Applies lateral quadrature weights w to approximate integral over x for irregular spacing.
    """
    if not _NUMBA_OK:
        raise RuntimeError("Numba not available. Install numba or use migrationKirchhoffPython_2.")
    _check_data_shape(dat)
    print(f'Numba Kirchhoff Migration of {dat.snum}x{dat.tnum} matrix')
    tt_sec = dat.travel_time.astype(np.float64) / 1.0e6
    dt = float(tt_sec[1] - tt_sec[0]); tt0 = float(tt_sec[0])
    snum = dat.snum; tnum = dat.tnum
    zs = vel * tt_sec / 2.0; zs2 = zs * zs
    max_travel_time = tt_sec[-1]
    # Explicit km -> m conversion
    dist_m = np.asarray(dat.dist, dtype=np.float64) * 1.0e3
    gradD = np.gradient(dat.data.astype(np.float64), tt_sec, axis=0)
    data_in = dat.data.astype(np.float64) if nearfield else np.zeros_like(gradD)
    depth_limits = np.full(snum, -1.0, dtype=np.float64)
    use_angle = aperture_angle_deg is not None
    use_fresnel = fresnel and (f0 is not None) and (f0 > 0.0)
    angle_limits = None; fresnel_limits = None
    if use_angle:
        theta = np.deg2rad(aperture_angle_deg); angle_limits = zs * np.tan(theta)
    if use_fresnel:
        wavelength = vel / f0; fresnel_limits = np.sqrt((wavelength * zs) / 2.0) * fresnel_factor
    if angle_limits is not None or fresnel_limits is not None:
        for i in range(snum):
            a = angle_limits[i] if angle_limits is not None else None
            fR = fresnel_limits[i] if fresnel_limits is not None else None
            if a is not None and fR is not None:
                depth_limits[i] = min(a, fR)
            elif a is not None:
                depth_limits[i] = a
            elif fR is not None:
                depth_limits[i] = fR
    elif use_aperture is not None and use_aperture > 0:
        depth_limits[:] = use_aperture
    migdata = np.zeros_like(gradD, dtype=np.float64)
    t0 = time.time()
    _kirchhoff_numba_kernel(migdata, gradD, data_in, dist_m, zs, zs2, vel, tt0, dt, max_travel_time, nearfield, depth_limits)
    dat.data = migdata
    print(f'Numba Kirchhoff complete in {time.time() - t0:.2f}s')
    return dat




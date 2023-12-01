#!/usr/bin/python3
"""Perform least squares fit of OSL-SAR dose,luminescence data to OTORX model.

If this code is run, as is, it will output a fit to sample data.
To run this on your own data, you need to modify three lines.  Toward the end
of the file, make these changes:
    1.  Set `TEST_DOSE` to the correct dose for your experiment
    2.  Set `DOSE` to the array of laboratory doses for your data
    3.  Set `LxTx` to the array of luminescence values for laboratory doses
        normalized by luminescence at the test dose

Reference:
Lawless & Timar-Gabor, "A new analytical model to fit both fine and
coarse grained quartz luminescence dose response curves," Radiation
Measurements

This code requires Python with the Numpy and Scipy packages installed.
"""

### Import required packages ###

import numpy as np
from scipy.special import lambertw as W
from scipy.optimize import curve_fit


### Define Functions ###

_ffmt = lambda vars: ' '.join(f'{x:.3g}' for x in vars)  # pretty-print a 1-D array

def nN2D(nN, Q, D63):
    """Return dose given n/N, Q, & D63.

    See Lawless & Timar-Gabor Eq 9

    >>> _ffmt(nN2D(nN=1-np.exp(-1), Q=np.array((0.5, 0.6, 0.7, 0.8)), D63=np.array((1, 10, 100, 1000))))
    '1 10 100 1e+03'
    >>> _ffmt(nN2D(nN=0.5, Q=np.array((0.5, 0.6, 0.7, 0.8)), D63=np.array((1, 10, 100, 1000))))
    '0.648 6.33 61.5 593'
    """
    return D63 * ((-np.log(1-nN) - Q*nN)/(1 - Q*(1-np.exp(-1))))

def D2nN(D, Q, D63):
    """Return n/N given dose D and parameters Q & D63.

    See Lawless & Timar-Gabor Equations 9 & 10

    Parameters
    ----------
    D : number or array of numbers
        Dose (same units as D63)
    Q : number or array of numbers
        Q = (1 - An/Am)(N/(N+ND))
    D63 : number or array of numbers
        Characteristic dose (in same units as D) when n/N=1- 1/e

    Returns
    -------
    r : three numbers
        Q, D63, c

    Examples
    --------
    >>> _ffmt(D2nN(1, Q=np.array((-10, -3, 0.1, 1)), D63=1))
    '0.632 0.632 0.632 0.632'
    """
    D = np.asarray(D)
    if np.all(abs(Q) < 1e-6):
        r = 1 - np.exp(-D/D63)
    elif np.any(abs(Q) < 1e-6):
        raise ValueError(f'Unsupported of zero and nonzero Q: Q={Q}')
    else:
        w = W(-Q * np.exp(-Q-(1.-Q*(1.-1./np.exp(1)))*D/D63))
        assert np.allclose(w.imag, 0)
        r = 1 + w.real/Q
    return r

# fn_otorx is a function which takes dose & log10 of the 3 parameters as arguments and returns n/N
fn_otorx = lambda D, logQ, logD63, logc: D2nN(D, 10**logQ, 10**logD63) * 10**logc / D2nN(TEST_DOSE, 10**logQ, 10**logD63)


def xy2otorxfit(dose, LxTx, pguess=None, sigma=None):
    """Return best fit parameters (Q, D63, c) given dose & intensity data.

    Parameters
    ----------
    dose : array of numbers
        Laboratory dose
    LxTx : array of numbers
        Luminescence at laboratory dose divided by luminescence for test dose
    pguess : None or three numbers
        Leave this as None and the code will select a reasonable first guess
        for the nonlinear fit
        If that is not working for you, supply a first guess for (Q, D63, c)
    sigma : None or array of numbers
        Weighting factors used by SciPy's curve_fit function.
        If you have error estimates for each data point in LxTx, you may enter that here.
        If sigma=1, then the absolute errors in luminescence will be minimized.
        If sigma is None or LxTx, then fractional errors will be minimized.

    Returns
    -------
    r : three numbers
        (Q, D63, c)
    """
    dose = np.asarray(dose)
    LxTx = np.asarray(LxTx)
    if sigma is None:
        sigma = LxTx  # minimize relative errors.
    if pguess is None:
        pguess = 0.9, np.max(dose), 1
    pguess = np.asarray(pguess)
    logpguess = np.log10(pguess)
    log_r, unused_cov = curve_fit(fn_otorx, dose, LxTx, p0=logpguess, sigma=sigma)
    r = 10**log_r
    return r  # r = (Q, D63, c)


### Define data, fit to OTORX model, and display results ###

# Replace the next three lines with your actual data:
TEST_DOSE = 17
DOSE = (17, 47, 94, 201, 402, 804, 1.07e+03, 1.68e+03, 3.35e+03, 5.36e+03, 6.7e+03, 8.04e+03,
        1e+04, 47, 94, 3.35e+03, 5.36e+03)
LxTx = (1.09, 2.3, 3.67, 5.63, 8.09, 10.3, 11.4, 13, 14.8, 15.8, 16.9, 17, 17.1, 2.26, 3.65,
        15.6, 15.9)

assert len(DOSE) == len(LxTx) # There must be one LxTx value for each dose

# Perform fit
Q, D63, c = xy2otorxfit(DOSE, LxTx)
print(f'Best fit: {Q=:.3g} {D63=:.3g} {c=:.3g}\n')

# Display results
fit = c * D2nN(DOSE, Q, D63) / D2nN(TEST_DOSE, Q, D63)
print('%10s %10s %10s' % ('Dose', 'Lx/Tx', 'Best Fit'))
for row in zip(DOSE, LxTx, fit):
    print('%10.0f %10.2f %10.2f' % tuple(row))

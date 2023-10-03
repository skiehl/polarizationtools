#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miscellaneous polarization tools.
"""

from math import sqrt
import numpy as np

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann", "Dmitry Blinov"]
__license__ = "BSD"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# FUNCTIONS
#==============================================================================

def debias_pol(pol, pol_err):
    """Returns the debiased polarization fraction.

    Parameters
    ----------
    pol : array-like
        Polarization fraction.
    pol_err : array-like
        Uncertainties of the polarization fraction.

    Returns
    -------
    pol_deb : numpy.ndarray
        Debiased polarization fraction.

    References
    ----------
    [1] Pavlidou et al, 2014 (eq. 1)
        https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.1693P/abstract
    """

    pol = np.asarray(pol)
    pol_err = np.asarray(pol_err)

    sel = pol / pol_err >= sqrt(2.)
    pol_deb = np.zeros(pol.size())
    pol_deb[sel] = np.sqrt(pol[sel]**2 - pol_err[sel]**2)

    return pol_deb

#==============================================================================

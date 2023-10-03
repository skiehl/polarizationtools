#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A tool to converte Stokes parameters to linear polarization.
"""

from math import sqrt, pi, exp, erf, cos
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm as dist_norm

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann", "Dmitry Blinov"]
__license__ = "BSD"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# CLASSES
#==============================================================================

class StokesConversion():
    """A tool to converte Stokes parameters to linear polarization.
    """

    #--------------------------------------------------------------------------
    def __init__(self):
        """Create instance of StokesConversion.

        Returns
        -------
        None.
        """

        self.stokes_i = None
        self.stokes_q = None
        self.stokes_u = None
        self.stokes_i_err = None
        self.stokes_q_err = None
        self.stokes_u_err = None
        self.pol = None
        self.pol_sq = None
        self.pol_mas = None
        self.pol_err = None
        self.unc = False

    #--------------------------------------------------------------------------
    def _set_stokes(self, stokes_q, stokes_u, stokes_i=None):
        """Set the Stokes data for conversion.

        Parameters
        ----------
        stokes_q : numpy.ndarray
            Stokes Q or q=Q/I.
        stokes_u : numpy.ndarray
            Stokes U or u=U/I.
        stokes_i : numpy.ndarray, optional
            Stokes I. If not provided, stokes_q and stokes_u inputs are
            interpreted as q=Q/I and u=U/I. The default is None.

        Returns
        -------
        None
        """

        self.single_val = False
        self.stokes_i = np.asarray(stokes_i)

        if stokes_i is None:
            self.stokes_q = np.asarray(stokes_q)
            self.stokes_u = np.asarray(stokes_u)
        else:
            self.stokes_q = np.asarray(stokes_q) / self.stokes_i
            self.stokes_u = np.asarray(stokes_u) / self.stokes_i

        # make single values arrays of dimension 1:
        if not self.stokes_q.shape:
            self.single_val = True
            self.stokes_q = np.expand_dims(self.stokes_q, 0)
            self.stokes_u = np.expand_dims(self.stokes_u, 0)
            if self.stokes_i is not None:
                self.stokes_i = np.expand_dims(self.stokes_i, 0)

    #--------------------------------------------------------------------------
    def _ratio_err(self, x, x_err, y, y_err):
        """Calculate the uncertainty of the ratio of two Gaussian distributed
        quantities.

        Parameters
        ----------
        x : numpy.ndarray
            Numerator quantity of the ratio.
        x_err : numpy.ndarray
            Uncertainty of the numerator.
        y : numpy.ndarray
            Denominator quantity of the ratio.
        y_err : numpy.ndarray
            Uncertainty of the denominator.

        Returns
        -------
        numpy.ndarray
            Uncertainty of the ratio.

        Notes
        -----
        The calculation is valid for an uncorrelated noncentral normal ratio.
        This approach currently does not allow zero values.

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Ratio_distribution#Normal_ratio_distributions
        """

        return (x / y)**2 * ((x_err / x)**2 + (y_err / y)**2)

    #--------------------------------------------------------------------------
    def _set_stokes_err(self, stokes_q_err, stokes_u_err, stokes_i_err=None):
        """Set the Stokes uncertainties.

        Parameters
        ----------
        stokes_q_err : numpy.ndarray
            Uncertainty of Stokes Q or q=Q/I.
        stokes_u_err : numpy.ndarray
            Uncertainty of Stokes U or u=U/I.
        stokes_i_err : numpy.ndarray, optional
            Uncertainty of Stokes I. If not provided, stokes_q_err and
            stokes_u_err inputs are interpreted as uncertainties of q=Q/I and
            u=U/I. The default is None.

        Returns
        -------
        None.
        """

        self.unc = True
        self.stokes_i_err = stokes_i_err

        if stokes_i_err is None:
            self.stokes_q_err = stokes_q_err
            self.stokes_u_err = stokes_u_err
        else:
            self.stokes_q_err = self._ratio_err(
                    self.stokes_q, stokes_q_err, self.stokes_i, stokes_i_err)
            self.stokes_u_err = self._ratio_err(
                    self.stokes_u, stokes_u_err, self.stokes_i, stokes_i_err)

    #--------------------------------------------------------------------------
    def _calc_pol_from_stokes(self):
        """Convertes Stokes q and u into linear fractional polarization and
        angle.

        Returns
        -----
        None.
        """

        self.pol_sq = self.stokes_q**2 + self.stokes_u**2
        self.pol = np.sqrt(self.pol_sq)
        self.evpa = 0.5 * np.arctan2(self.stokes_u, self.stokes_q)

    #--------------------------------------------------------------------------
    def _check_pol(self):
        """Check if fractional polarization values exceed 1.

        Raises
        ------
        Warning
            Raise if any values exceed 1.

        Returns
        -------
        None.
        """

        # check if any linear polarization values exceed 1 (by more than bit
        # precision):
        if np.any(np.logical_and(self.pol > 1., ~np.isclose(self.pol, 1.))):
            raise Warning(
                "Some fractional polarization values exceed 1.")

    #--------------------------------------------------------------------------
    def _calc_pol_err(self):
        """Calculate the  uncertainty of the fractional polarization.

        Returns
        -------
        None.

        Notes
        -----
        The estimate is defined in [1] eq. 31. Eq. 31 is valid for uncorrelated
        uncertainties in Stokes q and u. The correlated case is currently not
        implemented.

        References
        ----------
        [1] Plaszczynski et al, 2014
            https://ui.adsabs.harvard.edu/abs/2014MNRAS.439.4048P/abstract

        """

        norm = self.stokes_q**2 + self.stokes_u**2
        pol_var = (self.stokes_q**2 * self.stokes_q_err**2 +
                   self.stokes_u**2 * self.stokes_u_err**2) / norm
        self.pol_err = np.sqrt(pol_var)

    #--------------------------------------------------------------------------
    def _calc_pol_mas(self):
        """Calculate the  modified asymptotic (MAS) estimator of the fractional
        polarization.

        Returns
        -------
        None.

        Notes
        -----
        The estimate is defined in [1] eq. 37 and 30. Eq. 30 is valid for
        uncorrelated uncertainties in Stokes q and u. The correlated case is
        currently not implemented.

        References
        ----------
        [1] Plaszczynski et al, 2014
            https://ui.adsabs.harvard.edu/abs/2014MNRAS.439.4048P/abstract

        """

        norm = self.stokes_q**2 + self.stokes_u**2
        noise_bias_sq = (self.stokes_q**2 * self.stokes_u_err**2 +
                         self.stokes_u**2 * self.stokes_q_err**2) / norm

        self.pol_mas = self.pol - noise_bias_sq \
                * (1. - np.exp(-self.pol_sq / noise_bias_sq)) \
                / (2. * self.pol)

    #--------------------------------------------------------------------------
    def _evpa_pdf(self, theta, pol_snr):
        """Probability density function of EVPA measurements as defined by [1].

        Parameters
        ----------
        theta : float
            Angle in radians.
        pol_snr : float
            Fractional polarization signal-to-noise-ratio.

        Returns
        -------
        prob : float
            Probability density at angle theta.

        References
        -----
        [1] Naghizadeh-Khouei & Clarke, 1993
            https://ui.adsabs.harvard.edu/abs/1993A%26A...274..968N/abstract
        """

        const = 1. / sqrt(pi)
        eta = pol_snr * cos(2. * theta) / sqrt(2.)
        prob = const * (const + eta * exp(eta**2.) * (1. + erf(eta)))
        prob = prob * exp(-pol_snr**2. / 2.)

        return prob

    #--------------------------------------------------------------------------
    def _integrate_pdf(self, evpa, pol_snr):
        """Integration of EVPA PDF from -evpa to evpa.

        Parameters
        ----------
        evpa : float
            Angle in radians.
        pol_snr : float
            Fractional polarization signal-to-noise-ratio.

        Returns
        -------
        prob : float
            Integrated probability.
        """

        integ = quad(self._evpa_pdf, -evpa, evpa, args=(pol_snr))

        return integ[0]

    #--------------------------------------------------------------------------
    def _prob_diff(self, evpa, pol_snr, prob):
        """Calculate probability difference for minimization.

        Parameters
        ----------
        evpa : float
            Angle in radians.
        pol_snr : float
            Fractional polarization signal-to-noise-ratio.
        prob : float
            Target probability.

        Returns
        -------
        float
            Difference between estimated and target probability.
        """

        return abs(self._integrate_pdf(evpa, pol_snr) - prob)

    #--------------------------------------------------------------------------
    def _calc_evpa_err(self, sigma=1., approx_snr=20.):
        """Calculate the EVPA uncertainties.

        Parameters
        ----------
        sigma : float, optional
            Define the uncertainty in terms of sigma. The default is 1..
        approx_snr : float, optional
            For fractional polarization signal-to-noise-ratios larger than this
            value the EVPA uncertainty is approximated. The default is 20..

        Returns
        -------
        None.

        Notes
        -----
        This implements the method of [1].

        References
        -----
        [1] Naghizadeh-Khouei & Clarke, 1993
            https://ui.adsabs.harvard.edu/abs/1993A%26A...274..968N/abstract
        """

        prob = dist_norm.cdf(sigma) - dist_norm.cdf(-sigma)
        pol_snr = self.pol_mas / self.pol_err

        # for high SNR get approximation of EVPA uncertainty:
        hisnr = pol_snr >= approx_snr
        self.evpa_err = np.zeros(pol_snr.size, dtype=float)
        self.evpa_err[hisnr] = 0.5 / pol_snr[hisnr]

        # for low SNR calculate proper uncertainty:
        low_snr = np.nonzero(~hisnr)[0]
        for i in low_snr:
            # minimization:
            # start value pi/50. = 3.6 deg is just a reasonable guess
            res = minimize(
                    self._prob_diff, [pi/50.], args=(pol_snr[i], prob),
                    method='Nelder-Mead', tol=1e-5)

            # catch minimization issues:
            if not res.success:
                print('WARNING: Problem with data point {0:d}'.format(i))

            self.evpa_err[i] = res.x[0]

    #--------------------------------------------------------------------------
    def _return(self, sigma, unit):
        """Prepare the return tuple.

        Parameters
        ----------
        sigma : float
            Define the uncertainty in terms of sigma.
        unit : str
            If 'deg', the polarization angles and corresponding uncertainties
            are returned in degrees; otherwise in radians.

        Returns
        -------
        results : tuple
            Tuple with the linear polarization properties and (optionally) the
            corresponding uncertainties.
        """

        # include uncertainty estimates in return tuple:
        if self.unc:
            if unit=='deg':
                results = (
                        self.pol, self.pol_mas, self.pol_err*sigma,
                        np.degrees(self.evpa), np.degrees(self.evpa_err))
            else:
                results = (
                        self.pol, self.pol_mas, self.pol_err*sigma, self.evpa,
                        self.evpa_err)

        # no uncertainty estimates:
        else:
            if unit=='deg':
                results = (self.pol, np.degrees(self.evpa))
            else:
                results = (self.pol, self.evpa)

        # reduce to single values, if input was single values:
        if self.single_val:
            results = tuple([np.squeeze(res) for res in results])

        return results

    #--------------------------------------------------------------------------
    def convert(
            self, stokes_q, stokes_u, stokes_i=None, stokes_q_err=None,
            stokes_u_err=None, stokes_i_err=None, sigma=1., approx_snr=20.,
            unit='rad'):
        """Convert Stokes parameters to linear polarization properties.

        Parameters
        ----------
        stokes_q : numpy.ndarray
            Stokes Q or q=Q/I.
        stokes_u : numpy.ndarray
            Stokes U or u=U/I.
        stokes_i : numpy.ndarray, optional
            Stokes I. If not provided, stokes_q and stokes_u inputs are
            interpreted as q=Q/I and u=U/I. The default is None.
        stokes_q_err : numpy.ndarray, optional
            Uncertainty of Stokes Q or q=Q/I. If not provided, uncertainties
            are not calculated. The default is None.
        stokes_u_err : numpy.ndarray, optional
            Uncertainty of Stokes U or u=U/I. If not provided, uncertainties
            are not calculated. The default is None.
        stokes_i_err : numpy.ndarray, optional
            Uncertainty of Stokes I. Only required, if stokes_i, stokes_q_err,
            and stokes_u_err are given. If not provided, uncertainties
            are not calculated. The default is None.
        sigma : float, optional
            Define the uncertainty in terms of sigma. The default is 1..
        approx_snr : float, optional
            For fractional polarization signal-to-noise-ratios larger than this
            value the EVPA uncertainty is approximated. The default is 20..
        unit : str, optional
            If 'deg', the polarization angles and corresponding uncertainties
            are returned in degrees; otherwise in radians. The default is
            'rad'.

        Raises
        ------
        ValueError
            If 'sigma', 'approx_snr', or 'unit' have invalid inputs.

        Returns
        -------
        results : tuple
            If no uncertainties are given the tuple contains:
            fractional polarization and EVPA.
            If uncertainties are given, the tuple contains:
            fractional polarization, fractional polarization modified
            asymptotic estimator [1], fractional polarization uncertainty [1],
            EVPA, EVPA uncertainty [2]. EVPA and EVPA uncertainty are given in
            degrees or radians, as defined by the 'unit' input.

        References
        -----
        [1] Plaszczynski et al, 2014
            https://ui.adsabs.harvard.edu/abs/2014MNRAS.439.4048P/abstract
        [2] Naghizadeh-Khouei & Clarke, 1993
            https://ui.adsabs.harvard.edu/abs/1993A%26A...274..968N/abstract
        """

        # check and convert inputs:
        stokes_q = np.asarray(stokes_q, dtype=float)
        stokes_u = np.asarray(stokes_u, dtype=float)
        if stokes_i is not None:
            stokes_i = np.asarray(stokes_i, dtype=float)
        if stokes_q_err is not None:
            stokes_q_err = np.asarray(stokes_q_err, dtype=float)
        if stokes_u_err is not None:
            stokes_u_err = np.asarray(stokes_u_err, dtype=float)
        if stokes_i_err is not None:
            stokes_i_err = np.asarray(stokes_i_err, dtype=float)
        if type(sigma) not in (float, int) or sigma <= 0:
            raise ValueError(
                    "'sigma' needs to be int or float with value larger than "\
                    " zero.")
        if type(approx_snr) not in (float, int) or approx_snr <= 0:
            raise ValueError(
                    "'approx_snr' needs to be int or float with value larger "\
                    "than zero.")
        if not unit in ('rad', 'deg'):
            raise ValueError("'unit' needs to be 'rad' or 'deg'.")

        # store inputs:
        self._set_stokes(stokes_q, stokes_u, stokes_i=stokes_i)
        unc = False
        if stokes_q_err is not None and stokes_u_err is not None:
            unc = True
            self._set_stokes_err(
                    stokes_q_err, stokes_u_err, stokes_i_err=stokes_i_err)

        # calculate linear polarization properties:
        self._calc_pol_from_stokes()
        self._check_pol()

        # calculate uncertainties:
        if unc:
            self._calc_pol_err()
            self._calc_pol_mas()
            self._calc_evpa_err(sigma=sigma, approx_snr=approx_snr)

        # prepare return tuple:
        results = self._return(sigma, unit)

        return results

#==============================================================================

class StokesConversionSimple():
    """A tool to converte Stokes parameters to linear polarization and vice
    versa with simple Gaussian error propagation.
    """

    #--------------------------------------------------------------------------
    def __init__(self):
        """Create instance of StokesConversionSimple.

        Returns
        -------
        None
        """

        pass

    #--------------------------------------------------------------------------
    def _convert_angle(self, evpa, angle_unit_in, angle_unit_out):
        """Convert angle from radians to degree or vice versa, if needed.

        Parameters
        ----------
        evpa : numpy.ndarray
            Linear polarization angle in radians.
        angle_unit_in : str
            Unit of the input angle. Must be 'rad' or 'deg'.
        angle_unit_out : str
            Unit of the output angle. Must be 'rad' or 'deg'.

        Raises
        ------
        ValueError
            Raised, if angle_unit is not 'rad' or 'deg'.

        Returns
        -------
        evpa : numpy.ndarray
            The converted EVPA.
        """

        # check input:
        if angle_unit_in.lower() not in ['rad', 'deg']:
            raise ValueError(
                "`angle_unit` must be 'rad' or 'deg'.")
        if angle_unit_out.lower() not in ['rad', 'deg']:
            raise ValueError(
                "`angle_unit` must be 'rad' or 'deg'.")

        # convert from radians to degrees:
        if angle_unit_in == 'rad' and angle_unit_out == 'deg':
            evpa = np.degrees(evpa)

        # convert from radians to degrees:
        elif angle_unit_in == 'deg' and angle_unit_out == 'rad':
                evpa = np.radians(evpa)

        return evpa

    #--------------------------------------------------------------------------
    def _set_stokes(self, stokes_q, stokes_u, stokes_q_err, stokes_u_err):
        """Set the Stokes parameters for conversion.

        Parameters
        ----------
        stokes_q : array-like
            Stokes q=Q/I.
        stokes_u : array-like
            Stokes u=U/I.
        stokes_q_err : array-like
            Stokes q uncertainties.
        stokes_u_err : array-like
            Stokes u uncertainties.

        Raises
        ------
        ValueError
            Raised, if uncertainties are given for only one of the two Stokes
            parameters.

        Returns
        -------
        None
        """

        self.stokes_q = np.asarray(stokes_q)
        self.stokes_u = np.asarray(stokes_u)

        if stokes_q_err is None and stokes_q_err is None:
            self.propagate_err = False

        elif stokes_q_err is not None and stokes_q_err is not None:
            self.stokes_q_err = np.asarray(stokes_q_err)
            self.stokes_u_err = np.asarray(stokes_u_err)
            self.propagate_err = True

        else:
            raise ValueError(
                    "Either set uncertainties for both q and u or neither.")

    #--------------------------------------------------------------------------
    def _set_pol(self, pol, evpa, pol_err, evpa_err, angle_unit):
        """Set the Stokes parameters for conversion.

        Parameters
        ----------
        pol : array-like
            Linear fractional polarization.
        evpa : array-like
            Linear polarization angle in radians or degrees.
        pol_err : array-like
            UNcertainties of the linear fractional polarization.
        evpa_err : array-like
            Uncertainties of the linear polarization angle in radians or
            degrees.
        angle_unit : str
            Unit of the polarization angle. Must be 'rad' of 'deg'.

        Raises
        ------
        ValueError
            Raised, if uncertainties are given for only one of the two
            polarization parameters.

        Returns
        -------
        None
        """

        self.pol = np.asarray(pol)
        self.evpa = np.asarray(evpa)
        self.evpa = self._convert_angle(self.evpa, angle_unit, 'rad')
        self.evpa = np.mod(self.evpa, np.pi)

        # check polarzation input:
        if np.any(pol) > 1:
            raise ValueError(
                "Fractional linear polarization cannot be larger than 1.")

        if pol_err is None and evpa_err is None:
            self.propagate_err = False

        elif pol_err is not None and evpa_err is not None:
            self.pol_err = np.asarray(pol_err)
            self.evpa_err = np.asarray(evpa_err)
            self.propagate_err = True

        else:
            raise ValueError(
                    "Either set uncertainties for both pol and evpa or "
                    "neither.")

    #--------------------------------------------------------------------------
    def _stokes_to_pol(self):
        """Convertes Stokes q and u parameters into linear polarization
        fraction and angle.

        Raises
        ------
        ValueError
            Raised, when the input data results in fractional polarization
            larger than 1.

        Returns
        -------
        None
        """

        self.pol = np.sqrt(self.stokes_q**2 + self.stokes_u**2)

        # check that the fractional polarization does not exceed 1:
        if np.any(self.pol > 1):
            raise ValueError(
                    "Some Stokes q and u values result in a fractional "
                    "polarization larger than 1.")

        self.evpa = 0.5 * np.arctan2(self.stokes_u, self.stokes_q)

    #--------------------------------------------------------------------------
    def _stokes_to_pol_err(self):
        """Propagates Stokes uncertainties into linear polarization
        uncertainties.

        Returns
        -------
        None
        """

        self.pol_err = np.sqrt(
                (self.stokes_q**2 * self.stokes_q_err**2 \
                 + self.stokes_u**2 * self.stokes_u_err**2)) \
                / self.pol
        self.evpa_err = np.sqrt(
                self.stokes_q**2 * self.stokes_u_err**2 \
                + self.stokes_u**2 * self.stokes_q_err**2) / self.pol**2 / 2

    #--------------------------------------------------------------------------
    def _pol_to_stokes(self):
        """Convertes the linear polarization fraction and angle into Stokes q
        and u parameters.

        Returns
        -------
        None
        """

        sel = np.logical_and(
                self.evpa >= np.pi / 4, self.evpa < np.pi * 3 / 4)
        sign_q = np.where(sel, -1., 1.)
        sel = self.evpa >= np.pi / 2
        sign_u = np.where(sel, -1., 1.)

        tanevpa = np.tan(2 * self.evpa)
        tanevpasq = tanevpa**2
        self.stokes_q = sign_q * self.pol / np.sqrt(1 + tanevpasq)
        self.stokes_u = self.pol / np.sqrt(1 / tanevpasq + 1)
        self.stokes_u = np.copysign(self.stokes_u, sign_u)

    #--------------------------------------------------------------------------
    def _pol_to_stokes_err(self):
        """Propagates linear polarization uncertainties into Stokes
        uncertainties.

        Returns
        -------
        None
        """

        qsqusq = self.stokes_q**2 + self.stokes_u**2
        qqduqd = self.stokes_q**4 - self.stokes_u**4
        self.stokes_q_err = np.sqrt(
                qsqusq / qqduqd \
                * (self.stokes_q**2 * self.pol_err**2 \
                   - 4 * self.stokes_u**2 * qsqusq * self.evpa_err**2))
        self.stokes_u_err = np.sqrt(
                qsqusq /qqduqd \
                * (4 * self.stokes_q**2 * qsqusq * self.evpa_err**2 \
                   - self.stokes_u**2 * self.pol_err**2))

    #--------------------------------------------------------------------------
    def convert_stokes_to_pol(
            self, stokes_q, stokes_u, stokes_q_err=None, stokes_u_err=None,
            angle_unit='rad'):
        """Convert Stokes parameters to linear polarization properties.

        Parameters
        ----------
        stokes_q : array-like
            Stokes q=Q/I.
        stokes_u : array-like
            Stokes u=U/I.
        stokes_q_err : array-like, optional
            Uncertainty of Stokes q. If not provided, uncertainties are not
            propagated. The default is None.
        stokes_u_err : array-like, optional
            Uncertainty of Stokes u. If not provided, uncertainties are not
            propagated. The default is None.
        angle_unit : str, optional
            If 'deg', the polarization angles and corresponding uncertainties
            are returned in degrees; otherwise in radians. The default is
            'rad'.

        Returns
        -------
        pol : np.ndarray
            Fractional linear polarization in the range [0, 1].
        pol_err : np.ndarray
            Uncertainties of the fractional linear polarization. Only returned,
            if Stokes uncertainties are provided.
        evpa : np.ndarray
            Linear polarization angle in radian or degrees.
        pol_err : np.ndarray
            Uncertainties of the linear polarization angle in radians or
            degrees. Only returned, if Stokes uncertainties are provided.

        Notes
        -----
        This method does simple Gaussian error propagation. For a formally
        correct calculation of the uncertainties it is recommended to use
        the StokesConversion class instead.
        """

        self._set_stokes(stokes_q, stokes_u, stokes_q_err, stokes_u_err)

        # convert Stokes to polarization parameters:
        self._stokes_to_pol()

        # error propagation:
        if self.propagate_err:
            self._stokes_to_pol_err()

        # convert angle if needed:
        self.evpa = self._convert_angle(self.evpa, 'rad', angle_unit)

        # return results:
        if self.propagate_err:
            return self.pol, self.pol_err, self.evpa, self.evpa_err
        else:
            return self.pol, self.evpa

    #--------------------------------------------------------------------------
    def convert_pol_to_stokes(
            self, pol, evpa, pol_err=None, evpa_err=None, angle_unit='rad'):
        """Convert Stokes parameters to linear polarization properties.

        Parameters
        ----------
        pol : np.ndarray
            Fractional linear polarization in the range [0, 1].
        pol_err : np.ndarray
            Uncertainties of the fractional linear polarization. If not
            provided, uncertainties are not propagated. The default is None.
        evpa : np.ndarray
            Linear polarization angle in radian or degrees.
        evpa_err : np.ndarray
            Uncertainties of the linear polarization angle in radians or
            degrees. If not provided, uncertainties are not propagated. The
            default is None.
        angle_unit : str, optional
            Specify if the polarization angle and corresponding uncertainties
            are given in 'rad' or 'deg'. The default is 'rad'.

        Returns
        -------
        stokes_q : array-like
            Stokes q=Q/I.
        stokes_u : array-like
            Stokes u=U/I.
        stokes_q_err : array-like, optional
            Uncertainty of Stokes q. Only returned, if Stokes uncertainties are
            provided.
        stokes_u_err : array-like, optional
            Uncertainty of Stokes u. Only returned, if Stokes uncertainties are
            provided.

        Notes
        -----
        This method does simple Gaussian error propagation. The results may not
        reflect the true uncertainties.
        """

        self._set_pol(pol, evpa, pol_err, evpa_err, angle_unit)

        # convert Stokes to polarization parameters:
        self._pol_to_stokes()

        # error propagation:
        if self.propagate_err:
            self._pol_to_stokes_err()

        # return results:
        if self.propagate_err:
            return (self.stokes_q, self.stokes_q_err, self.stokes_u,
                    self.stokes_u_err)
        else:
            return self.stokes_q, self.stokes_u

#==============================================================================
# MAIN
#==============================================================================

if __name__ == '__main__':
    converter = StokesConversion()
    print(converter.convert(0.5, 0.1))
    print(converter.convert(5, 1, 7))
    #print(converter.convert(5, 1, 5))
    print(converter.convert(
            0.5, 0.1, stokes_q_err=0.2, stokes_u_err=0.1))
    print(converter.convert(
            [0.5], [0.1], stokes_i=[2], stokes_q_err=[0.5], stokes_u_err=[0.5],
            stokes_i_err=[0.5]))
    # TODO: the latter two cases crash with single values

    converter = StokesConversionSimple()
    print(converter.convert_stokes_to_pol(-1, 0, angle_unit='deg'))
    print(converter.convert_stokes_to_pol(-1, 0, 0.1, 0.5, angle_unit='deg'))
    print(converter.convert_pol_to_stokes(1, 90, angle_unit='deg'))
    print(converter.convert_pol_to_stokes(1, 90, 0.1, 0.25, angle_unit='deg'))

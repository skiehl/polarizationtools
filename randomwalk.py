#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulations of a random walk in the Stokes Q-U plane.
"""

import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann"]
__license__ = "BSD"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# CLASSES
#==============================================================================

class RWsim():
    """A tool to run random walk simulations in the Stokes Q-U plane.
    """

    #--------------------------------------------------------------------------
    def __init__(self):
        """Create instance of RWsim.
        """

        pass

    #--------------------------------------------------------------------------
    def _draw_from_powerlaw(self, index, minval, maxval, size=1):
        """Draws random numbers from a truncated power-law distribution.

        Parameters
        ----------
        index : float
            Power-law index.
        minval : float
            Lower limit of the distribution.
        maxval : float
            Upper limit of the distribution.
        size : int, optional
            Number of random data points to return. The default is 1.

        Returns
        -------
        random_numbers : numpy.ndarray
            Random numbers.
        """

        if index == -1:
            random_numbers = np.exp(
                np.random.uniform(size=size) * np.log(maxval / minval) \
                + np.log(minval))

        else:

            index += 1
            random_numbers = np.power(
                (maxval**index - minval**index) \
                * np.random.uniform(size=size) + minval**index, 1 / index)

        return random_numbers

    #--------------------------------------------------------------------------
    def _draw_from_ecdf(self, data, size=1):
        """Draws random number from an empirical cumulative distribution
        ferrorstion (ECDF) defined by given data.

        Parameters
        ----------
        data : list-like
            The data which defines the ECDF.
        size : int, optional
            Number of random data points to return. The default is 1.

        Returns
        -----
        random_numbers : np.ndarray
            Random numbers.
        """

        data = np.asarray(data)
        ecdf = ECDF(data)
        draw = np.random.uniform(low=ecdf.y[1], size=size)
        random_numbers = np.interp(draw, ecdf.y, ecdf.x)

        return random_numbers

    #--------------------------------------------------------------------------
    def _create_random_time(
            self, total, dist, param, recursion=0):
        """Creates random time data points with time steps following a given
        distribution.

        Parameters
        ----------
        total : float
            The total time to cover by the time data points
        dist : str
            Defines the distribution the time steps are drawn from. Choose from
            truncated 'powerlaw', 'lognormal' and 'ecdf'. The distribution
            parameters need to be set accordingly in the 'param'.
        param : list
            A list of distribution parameters depending on the chosen
            distribution:
            For 'powerlaw' give (1) the power-law index, (2) the lower, and
            (3) the upper limit of the truncated distribution.
            For 'lognormal' give the distribution (1) mu and (2) sigma.
            For 'ecdf' give an array of time steps (differences between time
            data points not the time data points!).
        recursion : int, optional
            Do not manually set a value. This parameter is needed internally
            when the drawn time steps are not enough to cover the targeted
            total time and a recursion call of the ferrorstion is necessery. The
            default is 0.

        Raises
        ------
        ValueError
            Raised if `dist` is neither 'lognormal', 'powerlaw', nor  'ecdf'.

        Returns
        -------
        time : numpy.ndarray
            Random time data points.
        """

        # determine number of time steps to create:
        if recursion:
            size = recursion

        elif dist=='powerlaw' and param[0] < -2.:
            mean_sampling = \
                (param[2]**(param[0] + 2) - param[1]**(param[0] + 2)) \
                / (param[2]**(param[0] + 1) - param[1]**(param[0] + 1)) \
                * (param[0] + 1) /(param[0] + 2)
            size = int(1.2 * total / mean_sampling)

        elif dist=='lognormal':
            mean_sampling = np.exp((param[0] + param[1]**2) / 2)
            size = int(1.2 * total / mean_sampling)

        elif dist=='ecdf':
            mean_sampling = np.mean(param)
            size = int(1.2 * total / mean_sampling)

        else:
            size = 100

        if size < 10:
            size = 10

        # create random time steps:
        if dist=='powerlaw':
            steps = self._draw_from_powerlaw(
                param[0], param[1], param[2], size=size)

        elif dist=='lognormal':
            steps = np.random.lognormal(
                mean=param[0], sigma=param[1], size=size)

        elif dist=='ecdf':
            steps = self._draw_from_ecdf(param, size=size)

        else:
            raise ValueError(
                f"Distribution type '{dist}' is not supported. Either set "
                "to 'powerlaw', 'lognormal', or 'ecdf'.")

        time = np.cumsum(steps)

        # recursion, if time steps do not cover total time:
        if time[-1] < total:
           size = int(np.ceil(2 * (1 - time[-1] / total) * size))
           more = self._create_random_time(
               total-time[-1], dist=dist, param=param, recursion=size)
           time = np.concatenate((time, more + time[-1]))

        if not recursion:
            time = time[time<=total]
            time = np.r_[0, time]

        return time

    #--------------------------------------------------------------------------
    def _create_time(self, total, dist='const', param=1):
        """

        Parameters
        ----------
        total : float
            The total time to cover by the time data points
        dist : str, optional
            Defines the distribution the time steps are drawn from. Choose from
            truncated 'powerlaw', 'lognormal' and 'ecdf'. The distribution
            parameters need to be set accordingly in the 'param'. The default
            is 'const'.
        param : float or list, optional
            A list of distribution parameters depending on the chosen
            distribution:
            For 'const' give a float that defines the fixed time interval.
            For 'powerlaw' give (1) the power-law index, (2) the lower, and
            (3) the upper limit of the truncated distribution.
            For 'lognormal' give the distribution (1) mu and (2) sigma.
            For 'ecdf' give an array of time steps (differences between time
            data points not the time data points!). The default is 1.

        Returns
        -------
        time : numpy.ndarray
            Time data points.
        """

        if dist == 'const':
            time = np.arange(0, total, param)

        else:
            time = self._create_random_time(total, dist, param)

        return time

    #--------------------------------------------------------------------------
    def _rw_simple(
            self, time, cells, variation, cell_pol=0.72, return_cells=False):
        """Polarization random walk based on the 'Simple Q,U random walk
        process' [1].

        Parameters
        ----------
        time : np.ndarray
            Time.
        cells : int
            Number of cells.
        variation : float
            Cell variation rate, the number of cells that change every unit
            time step.
        cell_pol : float, optional
            The fractional polarization of each cell. The default is 0.72.
        return_cells : bool, optional
            If True, all cell q and u parameters are returned. The default is
            False.

        Returns
        -------
        stokes_q, stokes_u : np.ndarrays
            The simulated, integrated, normalized Stokes parameters q=Q/I,
            u=U/I, if `return_cells=False`.
        stokes_q, stokes_u, cells_q, cells_u :
                np.ndarray, np.ndarray, np.ndarray, np.ndarray
            The simulated, integrated, normalized Stokes parameters q=Q/I,
            u=U/I and all cell q and u parameters, if `return_cells=True`.

        References
        ----------
        [1] Kiehlmann et al, 2016
            https://ui.adsabs.harvard.edu/abs/2016A%2526A...590A..10K/
        """

        time = np.asarray(time)

        # initialize randomized cells:
        # Stokes I, Q, U for each cell and each time step;
        # I constant (normed that total flux equals 1),
        # Q, U random samples from Gaussian distribution:
        cells_i = 1. /cells
        cells_q = np.random.normal(0, 1, (time.shape[0], cells))
        cells_u = np.random.normal(0, 1, (time.shape[0], cells))

        # set number of changing cells:
        # cumulative number of varying cells:
        nvar = np.rint(time * variation).astype(int)
        # current number of varying cells:
        nvar = np.diff(nvar)
        nvar = np.where(nvar > cells, cells, nvar)

        # iterate through random changes:
        # initially all cells are randomized; pick random cells that stay
        # constant and copy those:

        for i, n in enumerate(nvar, 1):
            # no cells change: copy latest results:
            if n == 0:
                cells_q[i] = cells_q[i-1]
                cells_u[i] = cells_u[i-1]

            # cells change:
            else:
                # select random cells that change:
                mask = np.unique(np.random.randint(0, cells, n))

                # copy constant cells:
                sel = np.ones(cells, dtype=bool)
                sel[mask] = False
                cells_q[i][sel] = cells_q[i-1][sel]
                cells_u[i][sel] = cells_u[i-1][sel]

        # normalize Q, U:
        norm = np.sqrt(cells_q**2. + cells_u**2.) / cell_pol
        cells_q = cells_q / norm * cells_i
        cells_u = cells_u / norm * cells_i

        # integrated Stokes:
        stokes_q = np.sum(cells_q, axis=1)
        stokes_u = np.sum(cells_u, axis=1)

        if return_cells:
            return stokes_q, stokes_u, cells_q, cells_u

        return stokes_q, stokes_u

    #--------------------------------------------------------------------------
    def _rw_ordered(
            self, time, cells, variation, cell_pol=0.72, flux='const',
            return_cells=False):
        """Polarization random walk based on the 'Ordered Q, U random walk
        process with constant I' or 'Ordered Q, U random walk process with
        decreasing I' [1].

        Parameters
        ----------
        time : np.ndarray
            Time.
        cells : int
            Number of cells.
        variation : float
            Cell variation rate, the number of cells that change every unit
            time step.
        cell_pol : float, optional
            The fractional polarization of each cell. The default is 0.72.
        flux : str, optional
            Set to 'const' for constant or 'decr' for decreasing cell flux
            density. The default is 'const'.
        return_cells : bool, optional
            If True, all cell q and u parameters are returned. The default is
            False.

        Returns
        -------
        stokes_q, stokes_u : np.ndarray, np.ndarray
            The simulated, integrated, normalized Stokes parameters q=Q/I,
            u=U/I, if `return_cells=False`.
        stokes_q, stokes_u, cells_q, cells_u :
                np.ndarray, np.ndarray, np.ndarray, np.ndarray
            The simulated, integrated, normalized Stokes parameters q=Q/I,
            u=U/I and all cell q and u parameters, if `return_cells=True`.

        Raises
        ------
        ValueError
            Raised if `flux` is neither 'const' nor 'decr'.

        References
        ----------
        [1] Kiehlmann et al, 2016
            https://ui.adsabs.harvard.edu/abs/2016A%2526A...590A..10K/
        """

        # initialize randomized cells:
        # Stokes I, Q, U for each cell and each time step;
        # I constant or decreasing down the shock front
        # (normed that total flux equals 1),
        # Q, U random samples from Gaussian distribution:
        if flux == 'const':
            cells_i = 1 / cells

        elif flux == 'decr':
            cells_i = np.linspace(0, 1, num=cells+1)
            cells_i = cells_i[1:] / np.sum(cells_i)
            cells_i = np.repeat(
                cells_i, time.shape[0], axis=0).reshape(-1, time.shape[0]).T

        else:
            raise ValueError("Parameter 'flux' has to be 'const' or 'decr'.")

        cells_q = np.random.normal(0., 1., (time.shape[0], cells))
        cells_u = np.random.normal(0., 1., (time.shape[0], cells))

        # set number of changing cells:
        # cumulative number of varying cells:
        nvar = np.rint(time * variation).astype(int)
        # current number of varying cells:
        nvar = np.diff(nvar)
        # max number of varying cells:
        nvar = np.where(nvar > cells, cells, nvar)

        # iterate through random changes:
        # initially all cells are randomized; keep the first nvar random values
        # and copy the other cells from the previous time steps cells 0 to
        # cells-nvar

        for i, n in enumerate(nvar, 1):
            # shift and copy constant cells:
            cells_q[i][n:] = cells_q[i-1][0:cells-n]
            cells_u[i][n:] = cells_u[i-1][0:cells-n]

        # normalize q, u:
        norm = np.sqrt(cells_q**2. + cells_u**2.) / cell_pol
        cells_q = cells_q / norm * cells_i
        cells_u = cells_u / norm * cells_i

        # integrated Stokes:
        stokes_q = np.sum(cells_q, axis=1)
        stokes_u = np.sum(cells_u, axis=1)

        if return_cells:
            return stokes_q, stokes_u, cells_q, cells_u

        return stokes_q, stokes_u

    #--------------------------------------------------------------------------
    def _add_noise(self, stokes_q, stokes_u, dist, param):
        """Create Gaussian "observational" noise.

        Parameters
        ----------
        stokes_q : np.ndarray
            Stokes q values.
        stokes_u : np.ndarray
            Stokes u values.
        dist : str
            Defines the distribution the noise levels are drawn from. Choose
            from 'const', truncated 'powerlaw', 'lognormal' and 'ecdf'. The
            distribution parameters need to be set accordingly in the 'param'.
        param : float or list, optional
            A list of distribution parameters depending on the chosen
            distribution:
            For 'const' give a float that defines the fixed time interval.
            For 'powerlaw' give (1) the power-law index, (2) the lower, and
            (3) the upper limit of the truncated distribution.
            For 'lognormal' give the distribution (1) mu and (2) sigma.
            For 'ecdf' give an array of time steps (differences between time
            data points not the time data points!). The default is 1.

        Raises
        ------
        ValueError
            Raised if `dist` is neither 'const', 'lognormal', 'powerlaw', nor
            'ecdf'.

        Returns
        -------
        stokes_q : numpy.ndarray
            Stokes q with added Gaussian noise.
        stokes_u : numpy.ndarray
            Stokes u with added Gaussian noise.
        stokes_err : numpy.ndarray
            The corresponding uncertainties.
        """

        size = stokes_q.shape[0]

        if dist == 'const':
            stokes_err = np.ones(size) * param

        elif dist=='lognormal':
            stokes_err = np.random.lognormal(
                mean=param[0], sigma=param[1], size=size)

        elif dist=='powerlaw':
            stokes_err = self._draw_from_powerlaw(
                param[0], param[1], param[2], size=size)

        elif dist=='ecdf':
            stokes_err = self._draw_from_ecdf(param, size=size)

        else:
            raise ValueError(
                f"Distribution type '{dist}' is not supported. Either set "
                "to 'lognormal', 'powerlaw', or 'ecdf'.")

        # add noise:
        stokes_q += np.random.normal(loc=0, scale=stokes_err)
        stokes_u += np.random.normal(loc=0, scale=stokes_err)

        # ensure that the polarization does not exceed one, when data points
        # are pushed over the unit circle by the noise:
        pol = np.sqrt(stokes_q**2 + stokes_u**2)
        stokes_q = np.where(pol>1, stokes_q/pol, stokes_q)
        stokes_u = np.where(pol>1, stokes_u/pol, stokes_u)

        return stokes_q, stokes_u, stokes_err

    #--------------------------------------------------------------------------
    def sim(self, process='simple', cells=10, variation=1., cell_pol=0.72,
            flux='const', time_total=100, time_dist='const', time_param=1,
            error_dist=None, error_param=1):
        """Run a random walk simulation in the Stokes Q-U plane.

        Parameters
        ----------
        process : TYPE, optional
            Random walk process type, a defined in [1]. Chose 'simple' or
            'ordered. The default is 'simple'.
        cells : int
            Number of cells.
        variation : float
            Cell variation rate, the number of cells that change every unit
            time step.
        cell_pol : float, optional
            The fractional polarization of each cell. The default is 0.72.
        flux : str, optional
            Set to 'const' for constant or 'decr' for decreasing cell flux
            density. Only releveant if `process='ordered'`. The default is
            'const'.
        time_total : float
            The total time to cover by the time data points
        time_dist : str, optional
            Defines the distribution the time steps are drawn from. Choose from
            truncated 'powerlaw', 'lognormal' and 'ecdf'. The distribution
            parameters need to be set accordingly in the 'param'. The default
            is 'const'.
        time_param : float or list, optional
            A list of distribution parameters depending on the chosen
            distribution:
            For 'const' give a float that defines the fixed time interval.
            For 'powerlaw' give (1) the power-law index, (2) the lower, and
            (3) the upper limit of the truncated distribution.
            For 'lognormal' give the distribution (1) mu and (2) sigma.
            For 'ecdf' give an array of time steps (differences between time
            data points not the time data points!). The default is 1.
        error_dist : str
            Defines the distribution the noise levels are drawn from. Choose
            from 'const', truncated 'powerlaw', 'lognormal' and 'ecdf'. The
            distribution parameters need to be set accordingly in the 'param'.
        error_param : float or list, optional
            A list of distribution parameters depending on the chosen
            distribution:
            For 'const' give a float that defines the fixed time interval.
            For 'powerlaw' give (1) the power-law index, (2) the lower, and
            (3) the upper limit of the truncated distribution.
            For 'lognormal' give the distribution (1) mu and (2) sigma.
            For 'ecdf' give an array of time steps (differences between time
            data points not the time data points!). The default is 1.

        Raises
        ------
        ValueError
            Raised if `process` is neither 'simple' nor 'ordered'.

        Returns
        -------
        time : np.ndarray
            Time of each data point.
        stokes_q : np.ndarray
            Normalized Stokes values, q=Q/I.
        stokes_u : np.ndarray
            Normalized Stokes values, u=U/I.
        stokes_err : np.ndarray
            Uncertainties corresponding to each pair of Stokes values. The same
            noise level is assumed for and added to both Stokes parameters.

        References
        ----------
        [1] Kiehlmann et al, 2016
            https://ui.adsabs.harvard.edu/abs/2016A%2526A...590A..10K/
        """

        # create time:
        time = self._create_time(time_total, time_dist, time_param)

        # Stokes random walk:
        if process=='simple':
            stokes_q, stokes_u = self._rw_simple(
                    time, cells, variation, cell_pol=cell_pol)

        elif process=='ordered':
            stokes_q, stokes_u = self._rw_ordered(
                    time, cells, variation, cell_pol=cell_pol, flux=flux)

        else:
            raise ValueError("`process` must be either 'simple' or 'ordered'.")

        # add noise:
        if error_dist:
            stokes_q, stokes_u, stokes_err = self._add_noise(
                stokes_q, stokes_u, error_dist, error_param)

        else:
            stokes_err = np.zeros(time.shape[0])

        return time, stokes_q, stokes_u, stokes_err

#==============================================================================

if __name__ == '__main__':
    sim = RWsim()
    time, stokes_q, stokes_u, stokes_err = sim.sim(
        time_dist='lognormal', time_param=(1, 1), error_dist='const', error_param=0.1)

    print(time)
    print(stokes_q)
    print(stokes_u)
    print(stokes_err)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis tools for electric vector position angle (EVPA) time-series data.
"""

import numpy as np

__author__ = "Sebastian Kiehlmann"
__credits__ = ["Sebastian Kiehlmann", "Dmitry Blinov"]
__license__ = "BSD"
__maintainer__ = "Sebastian Kiehlmann"
__email__ = "skiehlmann@mail.de"
__status__ = "Production"

#==============================================================================
# CLASSES
#==============================================================================

class EVPAanalyzer():
    """Analyze EVPA time-series data."""

    #--------------------------------------------------------------------------
    def __init__(self, time=None, evpa=None, evpa_err=None, unit='rad'):
        """Create instance of EVPAanalyzer.

        Parameters
        ----------
        time : array-like
            Time.
        evpa : array-like
            EVPA in radians or degrees.
        evpa_err : array-like, optional
            Uncertainties of the EVPA in radians or degrees. The default is
            None.
        unit : str, optional
            Unit of the EVPA and uncertainties. Must be 'rad' or 'deg'. The
            default is 'rad'.

        Returns
        -------
        None
        """

        if time is None:
            self.time = None
            self.evpa = None
            self.evpa_err = None
        else:
            self.set_data(evpa, evpa_err=evpa_err, unit=unit)

        self.evpa_adj = None

    #--------------------------------------------------------------------------
    def _data_exists(self, raise_err=False):
        """Check if data is stored.

        Parameters
        ----------
        raise_err : bool
            If True, an error is raised if not data is stored. Otherwise,
            no error is raised. The default is False.

        Raises
        ------
        ValueError
            Raised, if `raise_err` is True and no data is stored.

        Returns
        -------
        exists : bool
            True, if data is stored. False, otherwise.
        """

        exists = False

        if self.time is None:
            if raise_err:
                raise ValueError(
                    "No data. First use `set_data()` to provide time and EVPA."
                    )
        else:
            exists = True

        return exists

    #--------------------------------------------------------------------------
    def _adjusted_data_exists(self, raise_err=False):
        """Check if adjusted EVPA data is stored.

        Parameters
        ----------
        raise_err : bool
            If True, an error is raised if not adjusted data is stored.
            Otherwise, no error is raised. The default is False.

        Raises
        ------
        ValueError
            Raised, if `raise_err` is True and no adjusted data is stored.

        Returns
        -------
        exists : bool
            True, if adjusted data is stored. False, otherwise.
        """

        exists = False

        if self.evpa_adj is None:
            if raise_err:
                raise ValueError(
                    "Adjusted EVPA data required. First use `adjust()`.")
        else:
            exists = True

        return exists

    #--------------------------------------------------------------------------
    def _adjust_mcoa(self):
        """Adjusts an EVPA curve based on the assumption of minimal change
        of the amplitude between adjacent data points.

        Returns
        -------
        evpa_adj : np.ndarray
            The adjusted EVPA data.
        """

        # calculate difference:
        evpa_diff = np.r_[0, np.diff(self.evpa)]

        # determine offsets:
        sel = np.absolute(evpa_diff) > np.pi / 2
        offset = np.where(sel, evpa_diff / np.pi, 0)
        offset = np.round(offset, 0) * np.pi
        offset = np.cumsum(offset)

        # apply offset:
        evpa_adj = self.evpa - offset

        return evpa_adj

    #--------------------------------------------------------------------------
    def _adjust_mcor(self):
        """Adjusts an EVPA curve based on the assumption of minimal change
        of the rate between adjacent data points.

        Returns
        -------
        evpa_adj : np.ndarray
            The adjusted EVPA data.
        """

        # TODO: Get this from my RoboPol 2021 paper code.
        raise NotImplementedError()

    #--------------------------------------------------------------------------
    def _adjust_mcoa_multi(self, n):
        """Adjusts an EVPA curve based on the assumption of minimal change of
        the amplitude between each data point, concidering a given number of
        preceeding data points.

        Parameters
        ----------
        n : int
            Number of reference points considered in the data shifting
            decision. If only one reference point is chosen this method
            calls _adjust_mcoa().

        Returns
        -------
        evpa_adj : np.ndarray
            The adjusted EVPA data.
        """

        # check input:
        if not isinstance(n, int) or n < 1:
            raise ValueError("`n` must be integer equal to or larger than 1.")

        if n == 1:
            return self._adjust_mcoa()

        evpa_adj = np.array(self.evpa)

        # iterate through data points:
        for j in range(1, self.evpa.size):
            # get reference points:
            i = j - n
            i = i if i > 0 else 0
            evpa_ref = evpa_adj[i:j]

            # calculate difference and apply offset:
            evpa_diff = evpa_adj[j] - np.median(evpa_ref)
            offset = np.round(evpa_diff / np.pi, 0) * np.pi
            evpa_adj[j] -= offset

        return evpa_adj

    #--------------------------------------------------------------------------
    def set_data(self, time, evpa, evpa_err=None, unit='rad'):
        """Create instance of EVPAanalyzer.

        Parameters
        ----------
        time : array-like
            Time.
        evpa : array-like
            EVPA in radians or degrees.
        evpa_err : array-like, optional
            Uncertainties of the EVPA in radians or degrees. The default is
            None.
        unit : str, optional
            Unit of the EVPA and uncertainties. Must be 'rad' or 'deg'. The
            default is 'rad'.

        Raises
        ------
        ValueError
            Raised, if `unit` is not 'rad' or 'deg'.

        Returns
        -------
        None
        """

        # convert inputs to arrays, if needed:
        time = np.asarray(time)
        evpa = np.asarray(evpa)

        if evpa_err is not None:
            evpa_err = np.asarray(evpa_err)

        # convert to radians if necessary:
        if unit == 'rad':
            self.unit = 'rad'

        elif unit == 'deg':
            self.unit = 'deg'
            evpa = np.radians(evpa)

            if evpa_err is not None:
                evpa_err = np.radians(evpa_err)

        else:
            raise ValueError("`unit` must be 'rad' or 'deg'.")

        # save as attributes:
        self.time = time
        self.evpa = evpa
        self.evpa_err = evpa_err

        # normalize EVPA:
        self.normalize()

        # sort by time:
        i = np.argsort(time)
        self.time = self.time[i]
        self.evpa = self.evpa[i]

        if self.evpa_err is not None:
            self.evpa_err = self.evpa_err[i]

    #--------------------------------------------------------------------------
    def normalize(self, lower_limit=0.):
        """Shifts all EVPA data points into a 180 degrees interval.

        Parameters
        ----------
        lower_limit : float, default=0
            Sets the lower limit of the EVPA interval.

        Returns
        -----
        None
        """

        self._data_exists(raise_err=True)

        self.evpa -= np.floor(self.evpa / np.pi) * np.pi

        if lower_limit:
            sel = self.evpa >= lower_limit % np.pi
            self.evpa = np.where(sel, self.evpa - np.pi, self.evpa)
            self.evpa += (np.floor(lower_limit / np.pi) + 1) * np.pi

    #--------------------------------------------------------------------------
    def adjust(self, n=1, method='mcoa'):
        """Adjusts an EVPA curve based on the assumption of minimal change of
        the amplitude or rate between each data point, concidering a given
        number of preceeding data points.

        Parameters
        ----------
        n : int, optional
            The number of reference points for the adjustment. The default is
            1.
        method : str, optional
            Selects the method of adjustment. Must me 'mcoa' or 'mcor'. The
            default is 'mcoa'.

        Raises
        ------
        ValueError
            Raise, if `method` is not 'mcoa' or 'mcor'.

        Returns
        -------
        None
        """

        self._data_exists(raise_err=True)

        if method.lower() ==  'mcoa':
            self.adjustment = f'MCoA-{n}'
            self.evpa_adj = self._adjust_mcoa_multi(n)

        elif method.lower() == 'mcor':
            if n > 1:
                print('WARNING: Multiple reference points are not implemented'\
                      ' for the MCoR method. Proceeding with one reference '\
                      'point.')

            self.adjustment = 'MCoR'
            self.evpa_adj = self._adjust_mcor()

        else:
            raise ValueError("`method` must be 'mcoa' or 'mcor'.")

    #--------------------------------------------------------------------------
    def consistency(self, n_max=10, method='mcoa'):
        """Checks the consistency between adjusted EVPA curves using an
        increasing number of reference points.

        Parameters
        ----------
        n_max : int, optional
            Maximum number of reference points up to which the consistency is
            checked. The default is 10.
        method : str, optional
            Selects the method of adjustment. Must me 'mcoa' or 'mcor'. The
            default is 'mcoa'.

        Raises
        ------
        ValueError
            Raise, if `method` is not 'mcoa' or 'mcor'.

        Returns
        -------
        n_consistent : int
            Highest number of reference points which gives adjustment results
            consistent with fewer reference points.
        """

        self._data_exists(raise_err=True)

        # check for infinite consistency: if all data points are within a pi/2
        # interval, no data points will be shifted, thus the shifting
        # consistency is infinate:
        if np.max(self.evpa) - np.min(self.evpa) <= np.pi / 2:
            return np.inf

        # set largest number of reference points:
        n_max = self.evpa.size - 1 if self.evpa.size <= n_max else n_max

        # adjust EVPA and check consistency:
        n_consistent = 1
        evpa_ref = self._adjust_mcoa()

        for n in range(2, n_max+1):
            if method.lower() == 'mcoa':
                evpa_adj = self._adjust_mcoa_multi(n=n)
            elif method.lower() == 'mcor':
                raise NotImplementedError()
            else:
                raise ValueError("`method` must be 'mcoa' or 'mcor'.")

            if np.allclose(evpa_ref, evpa_adj):
                n_consistent = n
            else:
                break

        return n_consistent

    #--------------------------------------------------------------------------
    def get_data(self, unit='default'):
        """Returns the saved data.

        Parameters
        ----------
        unit : str, optional
            The unit of the EVPA. If 'default', the EVPAs are returned in the
            same unit as they were put in. Otherwise, the unit must be 'rad' or
            'deg'. The default is 'default'.

        Raises
        ------
        ValueError
            Raised, if `unit` is not 'default', 'rad', or 'deg'.

        Returns
        -------
        data : dict
            Contains the time and the normalized EVPA. The adjusted EVPA and
            uncertainties are included if they exist.
        """

        unit = self.unit if unit == 'default' else unit
        data = {'time': self.time}

        if unit == 'rad':
            data['evpa'] = self.evpa

            if self._adjusted_data_exists():
                data['evpa_adj'] = self.evpa_adj

            if self.evpa_err is not None:
                data['evpa_err'] = self.evpa_err

        elif unit == 'deg':
            data['evpa'] = np.degrees(self.evpa)

            if self._adjusted_data_exists():
                data['evpa_adj'] = np.degrees(self.evpa_adj)

            if self.evpa_err is not None:
                data['evpa_err'] = np.degrees(self.evpa_err)

        else:
            raise ValueError("`unit` must be 'default', 'rad', or 'deg'.")

        return data

#==============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis tools for electric vector position angle (EVPA) time-series data.
"""

from itertools import groupby
from operator import itemgetter
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

    # -------------------------------------------------------------------------
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
            self.set_data(time, evpa, evpa_err=evpa_err, unit=unit)

        self.evpa_adj = None
        self.rotation_indices = None

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def _adjust_mcoa(self):
        """Adjusts an EVPA curve based on the assumption of minimal change
        of the amplitude between adjacent data points as defined e.g. in [1].

        Returns
        -------
        evpa_adj : np.ndarray
            The adjusted EVPA data.

        References
        ----------
        [1] Kiehlmann et al, 2021
            https://ui.adsabs.harvard.edu/abs/2021MNRAS.507..225K/abstract
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

    # -------------------------------------------------------------------------
    def _adjust_mcor(self):
        """Adjusts an EVPA curve based on the assumption of minimal change
        of the rate between adjacent data points as defined in [1].

        Returns
        -------
        evpa_adj : np.ndarray
            The adjusted EVPA data.

        References
        ----------
        [1] Kiehlmann et al, 2021
            https://ui.adsabs.harvard.edu/abs/2021MNRAS.507..225K/abstract
        """

        # TODO: Get this from my RoboPol 2021 paper code.
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    def _adjust_mcoa_multi(self, n):
        """Adjusts an EVPA curve based on the assumption of minimal change of
        the amplitude between each data point, concidering a given number of
        preceeding data points, as defined in [1] and [2].

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

        References
        ----------
        [1] Kiehlmann, 2015
            https://ui.adsabs.harvard.edu/abs/2015PhDT.......630K/abstract
        [2] Kiehlmann et al, 2016
            https://ui.adsabs.harvard.edu/abs/2016A%26A...590A..10K/abstract
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

    # -------------------------------------------------------------------------
    def _identify_rot_smooth(
            self, sel=None, threshold_abs=None, threshold_factor=None,
            error_propagation=False):
        """Finds continuous parts of smooth variation in the EVPA time series,
        based on [1].

        Parameters
        ----------
        sel : np.ndarray, optional
            Indices to a subsection of data. If given, the identification runs
            only over this subsection of the data. The default is None.
        threshold_abs : float, optional
            Threshold of the absolute change in the rotation rate given in the
            unit of the EVPA input over the unit of the time input. For details
            see notes below. This argument overwrites `threshold_factor`. The
            default is None.
        threshold_factor : string, optional
            Threshold of the relative change in the rotation rate given as a
            unitless factor. For details see notes below. This argument is
            overwritten by `threshold_factor`. The default is None.
        error_propagation: bool, optional
            If True, the difference of the derivatives is reduced by the
            propagated errors. Note: EVPA errors are only considered, when
            `threshold_abs` is set. For `threshold_factor` this functionality
            is not implemented yet. The default is False.

        Raises
        ------
        ValueError
            Raised if `threshold_abs` is not larger than 1.
        ValueError
            Raised if `threshold_factor` is not larger than 0.
        ValueError
            Raised if neither `threshold_abs` nor `threshold_factor` is given.

        Returns
        -------
        indices : list of np.ndarrays
            Each element provides the indices to the data points that are part
            of an identified rotation.

        Notes
        -----
        Let the threshold be f and the previous derivative of 'evpa' is X1 and
        the current derivative is X2. The transition from X1 to X2 is
        considered smooth if X2 is in the following range:
        If `threshold_abs` is given: X2 in [X1-f, X2+f].
        If `threshold_factor` is given: X2 in [1/f*X1, f*X1].

        If `threshold_factor` is used this method implements the definition of
        an EVPA rotation as introduced by [1].

        References
        ----------
        [1] Blinov et al, 2015
            https://ui.adsabs.harvard.edu/abs/2015MNRAS.453.1669B/abstract
        """

        # check inputs:
        if threshold_abs is not None:
            if threshold_abs <= 0:
                raise ValueError(
                        "`threshold_abs` must be float larger than 0.")

            if self.unit == 'deg':
                threshold_abs = np.radians(threshold_abs)

        elif threshold_factor is not None:
            if threshold_factor <= 0:
                raise ValueError(
                        "`threshold_factor` must be float larger than 0.")

        else:
            raise ValueError(
                    "Either `threshold_abs` or `threshold_factor` must be set."
                    )

        if error_propagation:
            if self.evpa_err is None:
                print("WARNING: No EVPA uncertainties stored. Errors are not "\
                      "propagated.")
                error_propagation = False

            elif threshold_abs is None and threshold_factor is not None:
                raise NotImplementedError(
                        "Error propagation in combination with "\
                        "`threshold_factor` is not implemented yet.")
                # TODO: implement

        # get data:
        sel = np.arange(self.time.size) if sel is None else sel
        time = self.time[sel]
        evpa = self.evpa_adj[sel]

        if self.evpa_err is not None:
            evpa_err = self.evpa_err[sel]

        # identify rotations:
        dtime = np.diff(time)
        rate = np.diff(evpa) / dtime

        if threshold_abs:
            print('WARNING: `threshold_abs` seems to be flawed.')
            # TODO: bug fix
            drate = np.diff(rate)

            if error_propagation:
                rate_err = np.sqrt(
                        evpa_err[:-1]**2 + evpa_err[1:]**2) / dtime
                drate_err = np.sqrt(rate_err[:-1]**2 + rate_err[1:]**2)
                drate_red = np.abs(drate) - drate_err
                smooth = drate_red <= threshold_abs

            else:
                smooth = np.abs(drate) <= threshold_abs

        elif threshold_factor:
            smooth1 = np.logical_and(rate[1:] >= rate[:-1] / threshold_factor,
                                     rate[1:] <= rate[:-1] * threshold_factor)
            smooth2 = np.logical_and(rate[1:] <= rate[:-1] / threshold_factor,
                                     rate[1:] >= rate[:-1] * threshold_factor)
            smooth = np.where(rate[1:] > 0, smooth1, smooth2)

        # find ranges of continuous rotations:
        indices = []

        for k, g in groupby(
                    enumerate(np.nonzero(smooth)[0]), lambda x:x[0]-x[1]):
            group = list(map(itemgetter(1), g))
            indices.append(np.arange(group[0], group[-1]+3))

        return indices

    # -------------------------------------------------------------------------
    def _identify_rot_cont(self, sel=None, significance_level=1, verbose=0):
        """Identifies periods of unidirectional variability in the EVPA time
        series, as defined in [1] and [2].

        Parameters
        ----------
        sel : np.ndarray, optional
            Indices to a subsection of data. If given, the identification runs
            only over this subsection of the data. The default is None.
        significance_level : float, optional
            Factor by which the uncertainties are increased when calculating
            if a value change is significant. The default is 1.
        verbose : int, optional
            If 2 step-wise information is printed.

        Raises
        ------
        ValueError
            Raised if `significance_level` is smaller than 1.

        Returns
        -------
        indices : list of np.ndarrays
            Each element provides the indices to the data points that are part
            of an identified rotation.

        References
        ----------
        [1] Kiehlmann, 2015
            https://ui.adsabs.harvard.edu/abs/2015PhDT.......630K/abstract
        [2] Kiehlmann et al, 2016
            https://ui.adsabs.harvard.edu/abs/2016A%26A...590A..10K/abstract
        """

        # check input:
        if significance_level < 0:
            raise ValueError(
                    "`significance_level` must be equal to or larger than 0.")

        # get data:
        sel = np.arange(self.time.size) if sel is None else sel
        evpa = self.evpa_adj[sel]

        if self.evpa_err is None:
            evpa_err = np.zeros(evpa.size)
        else:
            evpa_err = self.evpa_err[sel]

        # create storage for rotation indices (start, stop) and amplitudes:
        indices = [[0,2]]
        amplitudes = []

        # first rotation amplitude:
        current_amplitude = evpa[1] - evpa[0]

        # first rotation significance, True if rotation amplitude larger than
        # root summed squared errors time significance level:
        current_sign = abs(current_amplitude) > significance_level \
                * np.sqrt(evpa_err[0]**2 + evpa_err[1]**2)

        if verbose > 1:
            print('\nPoint-wise results of the rotation identification.\n'\
                  'Note: Data point numbering starts with 0.\n')
            if current_sign:
                print('Data point     1        : First rotation significant.')
            else:
                print('Data point     1        : First rotation insignificant.'
                      )

        # iterate through data:
        for i in range(2, evpa.size):

            # get amplitude and significance of next rotation:
            new_amplitude = evpa[i] - evpa[i-1]
            new_sign = abs(new_amplitude) > significance_level \
                    * np.sqrt(evpa_err[i-1]**2 + evpa_err[i]**2)

            # State 1: YY+
            # current: significant, new: significant, rotation: continued
            if current_sign and new_sign \
                    and current_amplitude * new_amplitude >= 0:
                indices[-1][1] = i + 1
                current_amplitude = evpa[i] - evpa[indices[-1][0]]

                if verbose > 1:
                    print(f'Data point {i:5d} YY+    : '\
                          'Continued significant rotation.')

            # State 2: YY-
            # current: significant, new: significant, rotation: changed
            elif current_sign and new_sign:
                indices.append([i-1, i+1])
                amplitudes.append(current_amplitude)
                current_amplitude = new_amplitude

                if verbose > 1:
                    print(f'Data point {i:5d} YY-    : '\
                          'Changed significant rotation.')

            # State 3: YN+
            # current: significant, new: insignificant, rotation: continued
            elif current_sign and not new_sign and \
                    current_amplitude * new_amplitude >= 0:
                indices[-1][1] = i + 1
                current_amplitude = evpa[i] - evpa[indices[-1][0]]

                if verbose > 1:
                    print(f'Data point {i:5d} YN+    : '\
                          'Significant rotation continued insignificantly.')

            # State 4: YN-
            # current: significant, new: insignificant, rotation: changed
            elif current_sign and not new_sign:
                indices.append([i-1, i+1])
                amplitudes.append(current_amplitude)
                current_amplitude = new_amplitude
                current_sign = False

                if verbose > 1:
                    print(f'Data point {i:5d} YN-    : '\
                          'Significant rotation followed by insignificant '\
                          'rotation.')

            # State 5,6,7,8 NN
            # current: insignificant, new: insignificant
            elif not current_sign and not new_sign:

                indices[-1][1] = i+1
                temp = evpa[indices[-1][0]:indices[-1][1]]
                temp_err = evpa_err[indices[-1][0]:indices[-1][1]]

                # check all combinations of temporary data points with current
                # one for significant rotations - the largest difference might
                # not be significant if one data point has a large error:
                for j in range(temp.size-1):
                    current_amplitude = evpa[i] - temp[j]
                    if abs(current_amplitude) > significance_level \
                            * np.sqrt(evpa_err[i]**2 + temp_err[j]**2):
                        current_sign = True
                        break

                # Note: if a significant rotation is found current_sign is
                # True, current_amplitude reflects the sign of the rotation

                # State 5: NNN
                # insignificant rotation staying insignificant
                if not current_sign:
                    # there is nothing more happening here and that is correct.

                    if verbose > 1:
                        print(f'Data point {i:5d} NNN    : '\
                              'Insignificant rotations stays insignificant.')

                # State 6: NNYn/a first significant rotation
                # insignificant rotation becoming significant for the first
                # time
                elif len(amplitudes) == 0:

                    if current_amplitude >= 0:
                        j = np.argmin(temp)
                    else:
                        j = np.argmax(temp)

                    indices[-1][0] = j

                    if verbose > 1:
                        print(f'Data point {i:5d} NNYn/a : '\
                              'Insignificant rotations becoming significant '\
                              'for the first time.')

                # State 7: NNY+
                # insignificant rotation becoming significant, continuing
                elif current_amplitude *amplitudes[-1] >= 0:
                    del amplitudes[-1], indices[-1]
                    indices[-1][1] = i + 1
                    current_amplitude = evpa[i] - evpa[indices[-1][0]]

                    if verbose > 1:
                        print(f'Data point {i:5d} NNY+   : '\
                              'Insignificant rotations becoming significant, '\
                              'continuing former rotation.')

                # State 8: NNY-
                # insignificant rotation becoming significant, changing
                else:
                    if current_amplitude > 0:
                        indices[-1][0] += np.argmin(temp)
                        indices[-2][1] = indices[-1][0] + 1
                    else:
                        indices[-1][0] += np.argmax(temp)
                        indices[-2][1] = indices[-1][0] + 1
                    indices[-1][1] = i+1
                    amplitudes[-1] = \
                        evpa[indices[-2][0]] - evpa[indices[-2][1]-1]
                    current_sign = True

                    if verbose > 1:
                        print(f'Data point {i:5d} NNY-   : '\
                              'Insignificant rotations becoming significant, '\
                              'opposite to former rotation.')

            # State 9,10,11 NY
            # current: insignificant, new: significant
            else:
                # State 9 NYn/a first time
                # first time significant
                if len(amplitudes) == 0:
                    if new_amplitude > 0:
                        indices[-1] = [np.argmin(evpa[:i]), i+1]
                    else:
                        indices[-1] = [np.argmax(evpa[:i]), i+1]

                    current_sign = True
                    current_amplitude = \
                            evpa[indices[-1][-1]-1] - evpa[indices[-1][0]]

                    if verbose > 1:
                        print(f'Data point {i:5d} NYn/a  : '\
                              'First significant rotation.')

                # State 10 NY++/--
                # current: insignificant, new: significant, continues former
                elif (new_amplitude * amplitudes[-1] > 0):
                    del indices[-1], amplitudes[-1]
                    indices[-1][1] = i+1
                    current_amplitude = evpa[i] - evpa[indices[-1][0]]
                    current_sign = True

                    if verbose > 1:
                        print('Data point {i:5d} NY++/--: '\
                              'Insignificant rotation followed by '\
                              'significant rotation, continuing former one.')

                # State 11 NY+-/-+
                # current: insignificant, new: significant, opposite to former
                else:
                    temp = evpa[indices[-1][0]:i+1]
                    if new_amplitude > 0:
                        indices[-2][1] += np.argmin(temp)
                    else:
                        indices[-2][1] += np.argmax(temp)
                    indices[-1] = [indices[-2][1]-1, i+1]
                    amplitudes[-1] = \
                            evpa[indices[-2][1]-1] - evpa[indices[-2][0]]
                    current_amplitude = evpa[i] - evpa[indices[-1][0]]
                    current_sign = True

                    if verbose > 1:
                        print(f'Data point {i:5d} NY+-/-+: '\
                              'Insignificant rotation followed by '\
                              'significant rotation, opposite to former one.')

        # assess last data point/interval
        else:
            if current_sign:
                amplitudes.append(current_amplitude)
            elif len(amplitudes):
                del indices[-1]
                temp = evpa[indices[-1][1]-1:]
                if amplitudes[-1] > 0:
                    indices[-1][1] += np.argmax(temp)
                    amplitudes[-1] = \
                            evpa[indices[-1][1]-1] - evpa[indices[-1][0]]
                else:
                    indices[-1][1] += np.argmin(temp)
                    amplitudes[-1] = \
                            evpa[indices[-1][1]-1] - evpa[indices[-1][0]]
            else:
                del indices[-1]

        for i, (start, stop) in enumerate(indices):
            indices[i] = np.arange(start, stop)

        return indices

    #--------------------------------------------------------------------------
    def _split_data(self, gap):
        """Split the data at large time gaps.

        Parameters
        ----------
        gap : float
            Gap length threshold in the same unit as the input time data. The
            data is split when the time interval between data points exceeds
            this value.

        Returns
        -----
        out : list of np.ndarray
            Each element provides the indices to a split data set.
        """

        if len(self.time) < 2:
            return []

        # identify large gaps:
        split = np.r_[
                0, np.nonzero(np.diff(self.time)>gap)[0] + 1, self.time.size]
        indices = []

        # prepare list of indices to the split data sets:
        for start, stop in zip(split[:-1], split[1:]):
            indices.append(np.arange(start, stop))

        return indices

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def adjust(self, n=1, method='mcoa'):
        """Adjusts an EVPA curve based on the assumption of minimal change of
        the amplitude or rate between each data point, concidering a given
        number of preceeding data points, as defined e.g. in [1-3].

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

        References
        ----------
        [1] Kiehlmann, 2015
            https://ui.adsabs.harvard.edu/abs/2015PhDT.......630K/abstract
        [2] Kiehlmann et al, 2016
            https://ui.adsabs.harvard.edu/abs/2016A%26A...590A..10K/abstract
        [3] Kiehlmann et al, 2021
            https://ui.adsabs.harvard.edu/abs/2021MNRAS.507..225K/abstract
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

    # -------------------------------------------------------------------------
    def consistency(self, n_max=10, method='mcoa'):
        """Checks the consistency between adjusted EVPA curves using an
        increasing number of reference points, as defined in [1] and [2].

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

        References
        ----------
        [1] Kiehlmann, 2015
            https://ui.adsabs.harvard.edu/abs/2015PhDT.......630K/abstract
        [2] Kiehlmann et al, 2016
            https://ui.adsabs.harvard.edu/abs/2016A%26A...590A..10K/abstract
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
    def variation_estimator(self, sel=None, return_rates=False):
        """Calculates the variation estimator of an EVPA curve.
        The variation estimator is the average absolute offset of the
        point-wise derivative from the average derivative of the curve, as
        defined in [1] and [2].

        Parameters
        ----------
        sel : np.ndarray, optional
            Indices to a subsection of data. If given, the estimation runs
            only over this subsection of the data. The default is None.
        return_rates : bool, optional
            If True, the pairwise rotation rates are returned. Otherwise, not.
            The default is False.

        Returns
        -----
        variation : float, float
            Variation estimator.
        tendency : float
            Estimate of the secular trend of the EVPA curve.
        rates : np.ndarray, optional
            The pairwise derivatives of the EVPA curve. Only returned, if
            `return_rates=True`.

        References
        ----------
        [1] Kiehlmann, 2015
            https://ui.adsabs.harvard.edu/abs/2015PhDT.......630K/abstract
        [2] Kiehlmann et al, 2016
            https://ui.adsabs.harvard.edu/abs/2016A%26A...590A..10K/abstract
        """

        # get data:
        sel = np.arange(self.time.size) if sel is None else sel
        evpa = self.evpa_adj[sel]
        time = self.time[sel]

        # differences:
        devpa = np.diff(evpa)
        dtime = np.diff(time)

        # relative point-to-point variation:
        rates = devpa / dtime

        # tendency and variation estimator:
        tendency = np.mean(rates)
        variation = np.mean(np.absolute(rates - tendency))

        if return_rates:
            rates = np.r_[0, rates]
            return variation, tendency, rates
        else:
            return variation, tendency

    # -------------------------------------------------------------------------
    def identify_rotations(self, method, time_gap=None, verbose=0, **kwargs):
        """Identify rotations in the stored data.
        One may chose from two different definitions of a rotation: 'smooth' as
        introduced by [1] or 'continuous' by [2] and [3].

        Parameters
        ----------
        method : str
            Select either 'smooth' or 'continuous'.
        time_gap : float or None, optional
            If set, the data is split at time gaps larger than this threshold.
            Rotations will not include these large gaps. The default is None.
        verbose : int, optional
            If 1, returns some basic information about the identification. If
            2, provides details in case that `method='continuous'`. Set to 0 to
            turn off any notifications. The default is 0.
        **kwargs :
            Arguments forwarded to the smooth or continuous rotation
            identification methods. See details below.

        Keyword arguments for the smooth rotation identification
        --------------------------------------------------------
        threshold_abs : float, optional
            Threshold of the absolute change in the rotation rate given in the
            unit of the EVPA input over the unit of the time input. For details
            see notes below. This argument overwrites `threshold_factor`. The
            default is None.
        threshold_factor : string, optional
            Threshold of the relative change in the rotation rate given as a
            unitless factor. For details see notes below. This argument is
            overwritten by `threshold_factor`. The default is None.
        error_propagation: bool, optional
            If True, the difference of the derivatives is reduced by the
            propagated errors. Note: EVPA errors are only considered, when
            `threshold_abs` is set. For `threshold_factor` this functionality
            is not implemented yet. The default is False.

        Keyword arguments for the continuous rotation identification
        ------------------------------------------------------------
        significance_level : float, optional
            Factor by which the uncertainties are increased when calculating
            if a value change is significant. The default is 1.

        Raises
        ------
        ValueError
            Raised, if `time_gap` is neither float nor int or is equal to or
            smaller than 0.
        ValueError
            Raised, if `method` is neither 'smooth' nor 'continuous'.

        Returns
        -------
        None

        Notes
        -----
        If `method='smooth'`:
            Either `threshold_abs` or `threshold_factor` must be set.
            Let the threshold be f and the previous derivative of 'evpa' is X1
            and the current derivative is X2. The transition from X1 to X2 is
            considered smooth if X2 is in the following range:
            If `threshold_abs` is given: X2 in [X1-f, X2+f].
            If `threshold_factor` is given: X2 in [1/f*X1, f*X1].

            If `threshold_factor` is used this method implements the definition
            of an EVPA rotation as introduced by [1].
        If `method='continuous'`:
            This method implements the algorithm introduced by [2] and [3].

        References
        ----------
        [1] Blinov et al, 2015
            https://ui.adsabs.harvard.edu/abs/2015MNRAS.453.1669B/abstract
        [2] Kiehlmann, 2015
            https://ui.adsabs.harvard.edu/abs/2015PhDT.......630K/abstract
        [3] Kiehlmann et al, 2016
            https://ui.adsabs.harvard.edu/abs/2016A%26A...590A..10K/abstract
        """

        if verbose > 0:
            print(f'Identify {method} rotations..')

        # check that data exists:
        self._adjusted_data_exists(raise_err=True)

        # split data, if needed:
        if time_gap is None:
            split_indices = [np.arange(0, self.evpa.size)]

        else:
            if type(time_gap) not in [float, int] or time_gap <= 0:
                raise ValueError("`time_gap` must be larger than 0.")

            split_indices = self._split_data(time_gap)
            n_split = len(split_indices)

            if verbose > 0 and n_split > 1:
                print(f'Iterating through {n_split} split data sub-sets..')

        rotation_indices = []

        # identify smooth rotations:
        if method.lower() == 'smooth':
            self.rot_method = 'smooth'

            # iterate through split data sets:
            for sel in split_indices:
                if sel.size < 2:
                    continue

                temp_indices = self._identify_rot_smooth(sel=sel, **kwargs)

                for ind in temp_indices:
                    rotation_indices.append(ind + sel[0])

        # or identify continuous rotations:
        elif method.lower() == 'continuous':
            self.rot_method = 'continuous'

            # iterate through split data sets:
            for sel in split_indices:
                if sel.size < 2:
                    continue

                temp_indices = self._identify_rot_cont(
                        sel=sel, verbose=verbose, **kwargs)

                for ind in temp_indices:
                    rotation_indices.append(ind + sel[0])

        # otherwise, raise error:
        else:
            raise ValueError("`method` must be 'smooth' or 'continuous'.")

        self.rotation_indices = rotation_indices
        n_rot = len(rotation_indices)

        if verbose > 0:
            print(f'{n_rot} rotations identified.')

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    def get_rotations(
            self, stats=True, indices=False, data=False, amplitude_cut=0,
            number_cut=2, significance=None, significance_level=1,
            unit='default', verbose=1):
        """Get the identified rotations: statistics, indices, and/or data.

        Parameters
        ----------
        stats : bool, optional
            If True, return rotation statistics: the amplitudes, durations,
            rates, and the variation estimator are returned. The default is
            True.
        indices : bool optional
            If True, return the indices to the rotations in the data. The
            default is False.
        data : bool, optional
            If True, return the data for all rotations. The default is False.
        amplitude_cut : float, optional
            Only rotations with amplitudes larger than this threshold are
            returned. The cut must be given in the same unit as the input data,
            if `unit='default'` or the unit must be specified via setting
            `unit` to 'rad' or 'deg'. The default is 0.
        number_cut : int, optional
            Only rotations consisting of at least this amount of data points
            are returned. The default is 2.
        significance : str, optional
            Only rotations that meet the significance criterion are returned.
            If 'total', the total rotation amplitude must be larger than the
            corresponding uncertainty times `significance_level`.
            It 'pairwise', the EVPA change between each adjacent data pair
            must be larger than the corresponding uncertainty times
            `significance_level`. These criteria require that uncertainties
            have been provided. The default is None.
        significance_level : float, optional
            Factor considered for the significance criteria (see above for
            details). The default is 1.
        unit : str, optional
            If 'default', the EVPA-based rotation parameters are returned in
            the same unit as the original input data. Otherwise, specify the
            output unit by setting 'rad' or 'deg'. This input also specifies
            the unit of the `amplitude_cut`. The default is 'default'.
        verbose : int, optional
            If 1, prints some information about the count of simulations
            identified, rejected by the selection criteria, and returned.
            Set to 0 to turn off any information. The default is 1.

        Raises
        ------
        ValueError
            Raised if no ratations have been identified yet.
        ValueError
            Raised if `unit` is neither 'default', 'rad', or 'deg'.
        ValueError
            Raised if `amplitude_cut` is not float or int or smaller than 0.
        ValueError
            Raised if `number_cut` is not int or smaller than 2.
        ValueError
            Raised if `significance` is neither None, 'total', not 'pairwise'.
        ValueError
            Raised if `significance_level` is neither float nor int or smaller
            than or equal to 0.

        Returns
        -------
        results : dict
            Dictionary containing the various items requested. If `stats=True`
            the rotation properties are saved under keys 'ampl', 'dur',
            'rates', and 'var'. If `indices =True` a list is saved under the
            key 'indices', where each item is a numpy.ndarray of indices
            corresponding to the data that make up each rotation. If
            `data=True` each rotation's data is returned as a dict in a list
            under the key 'data'.`
        """

        # check if rotations exist:
        if self.rotation_indices is None:
            raise ValueError(
                    "No rotations stored. Run `identify_rotations` first.")

        # check input:
        if unit.lower() not in ['default', 'rad', 'deg']:
            raise ValueError("`unit` must be 'default', 'rad', or 'deg'.")

        if type(amplitude_cut) not in [int, float] or amplitude_cut < 0:
            raise ValueError(
                "`amplitude_cut` must be float equal to or larger than 0.")

        if not isinstance(number_cut, int) or number_cut < 2:
            raise ValueError(
                "`number_cut` must be integer equal to or larger than 2.")

        if significance not in [None, 'total', 'pairwise']:
            raise ValueError(
                    "`significance` must be 'total', 'pairwise', or None.")

        if significance and self.evpa_err is None:
            raise ValueError(
                    'No uncertainties stored. Cannot apply significance '\
                    'criterion.')

        if type(significance_level) not in [int, float] \
                or significance_level <= 0:
            raise ValueError(
                    "`significance_level` must be float larger than 0.")

        # convert amplitude cut:
        if (unit == 'default' and self.unit == 'deg') or unit == 'deg':
            amplitude_cut = np.radians(amplitude_cut)

        # storage for results:
        results = {}
        rotation_indices = []
        rotations_data = []
        amplitudes = []
        durations = []
        varest = []
        n_rot_total = len(self.rotation_indices)
        n_rot_skipped = 0

        # iterate through rotations:
        for i, sel in enumerate(self.rotation_indices):
            dur = self.time[sel[-1]] - self.time[sel[0]]
            ampl = self.evpa_adj[sel[-1]] - self.evpa_adj[sel[0]]
            var, __ = self.variation_estimator(sel=sel)

            # apply number cut:
            if sel.size < number_cut:
                n_rot_skipped += 1
                continue

            # apply amplitude cut:
            if np.abs(ampl) < amplitude_cut:
                n_rot_skipped += 1
                continue

            # apply total significance criterion:
            if significance == 'total':
                ampl_err = np.sqrt(
                        self.evpa_err[sel[-1]]**2 + self.evpa_err[sel[0]]**2)

                if np.absolute(ampl) <= ampl_err * significance_level:
                    n_rot_skipped += 1
                    continue

            # apply pairwise significance criterion:
            if significance == 'pairwise':
                ampls = np.absolute(
                        self.evpa_adj[sel[:-1]] - self.evpa_adj[sel[1:]])
                ampls_err = np.sqrt(
                        self.evpa_err[sel[:-1]]**2 + self.evpa_err[sel[1:]]**2)

                if np.any(ampls <= ampls_err * significance_level):
                    n_rot_skipped += 1
                    continue

            # store indices and statistics:
            rotation_indices.append(sel)
            durations.append(dur)
            amplitudes.append(ampl)
            varest.append(var)

            if data:
                rot = {'time': self.time[sel], 'evpa': self.evpa[sel],
                       'evpa_adj': self.evpa_adj[sel]}

                if self.evpa_err is not None:
                    rot['evpa_err'] = self.evpa_err[sel]

                rotations_data.append(rot)

        # print info:
        if verbose >= 1:
            print(f'{n_rot_total} rotations were identified.')

            if n_rot_skipped:
                print(f'{n_rot_skipped} rotations do not meet the selection '\
                      'criteria.')

            print(f'{n_rot_total-n_rot_skipped} rotations are returned.')


        # add statistics to results:
        if stats:
            amplitudes = np.array(amplitudes)
            durations = np.array(durations)
            rates = amplitudes / durations

            # change EVPA stats to degree, if needed:
            unit = self.unit if unit == 'default' else unit

            if unit.lower() == 'deg':
                amplitudes = np.degrees(amplitudes)
                rates = np.degrees(rates)
                varest = np.degrees(varest)

            results['ampl'] = amplitudes
            results['dur'] = durations
            results['rate'] = rates
            results['var'] = varest

        # add indices to results:
        if indices:
            results['indices'] = rotation_indices

        # add rotation data to results:
        if data:
            results['data'] = rotations_data

        return results

#==============================================================================

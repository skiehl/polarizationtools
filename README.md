# polarizationtools
Python3 tools to deal with astrophysical polarization data.

## Modules

* **evpatools.py** Tools to analyze EVPA time series data.
* **misc.py** Miscellaneous tools.
* **stokesconversion.py** Convert Stokes parameters into linear polarization parameters and vice versa.

### Module details

* **evpatools.py**
    * **EVPAanalyzer** class: Shift EVPA data points to account for the 180 degrees ambiguity, following [1].
* **misc.py**
    * **debias_pol** function: Debias the fractional linear polarization, following [2].
* **stokesconversion.py**
    * **StokesConversion** class: Convert Stokes parameters into linear polarization parameters with proper treatment of the uncertainties as explained in [3].
    * **StokesConversionSimple** class: Convert Stokes parameters into linear polarization parameters and vice versa with simple Gaussian error propagation.

## License

polarizationtools is licensed under the BSD 3-Clause License - see the
[LICENSE](https://github.com/skiehl/polarizationtools/blob/main/LICENSE) file.

## References

[1] https://ui.adsabs.harvard.edu/abs/2016A%26A...590A..10K/abstract
[2] https://ui.adsabs.harvard.edu
[3] https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.3715B/abstract

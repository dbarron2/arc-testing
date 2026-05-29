"""Planck spectral radiance using Astropy's BlackBody model."""

import astropy.units as u
from astropy.modeling.models import BlackBody

# Wavelength-based spectral radiance (B_lambda)
_SLAM = u.erg / (u.cm**2 * u.s * u.AA * u.sr)


def spectral_radiance(wavelength, temperature):
    """
    Spectral radiance B_lambda(T, lambda) from Planck's law.

    Parameters
    ----------
    wavelength : `~astropy.units.Quantity`
        Wavelength (e.g. ``500 * u.nm``, ``1 * u.um``).
    temperature : `~astropy.units.Quantity`
        Temperature (e.g. ``5778 * u.K``).

    Returns
    -------
    `~astropy.units.Quantity`
        Spectral radiance per unit wavelength (B_lambda).
    """
    bb = BlackBody(temperature=temperature, scale=1.0 * _SLAM)
    return bb(wavelength)


def spectral_radiance_nu(frequency, temperature):
    """
    Spectral radiance B_nu(T, nu) from Planck's law.

    Parameters
    ----------
    frequency : `~astropy.units.Quantity`
        Frequency (e.g. ``5e14 * u.Hz``).
    temperature : `~astropy.units.Quantity`
        Temperature (e.g. ``5778 * u.K``).

    Returns
    -------
    `~astropy.units.Quantity`
        Spectral radiance per unit frequency (B_nu) in erg / (cm² s Hz sr).
    """
    bb = BlackBody(temperature=temperature, scale=1.0)
    return bb(frequency)

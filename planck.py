"""
Planck spectral radiance.

Two implementations are provided:

- ``planck_lambda`` / ``planck_nu``: from first principles, no imports, plain floats.
- ``spectral_radiance`` / ``spectral_radiance_nu``: Astropy ``BlackBody``, unit-aware quantities.
"""

import astropy.units as u
from astropy.modeling.models import BlackBody

# ---------------------------------------------------------------------------
# From scratch (SI constants, no units)
# ---------------------------------------------------------------------------

# CODATA 2019
H = 6.62607015e-34  # Planck constant [J·s]
C = 299792458.0  # Speed of light [m/s]
K_B = 1.380649e-23  # Boltzmann constant [J/K]
_E = 2.718281828459045  # Euler's number


def _expm1(x):
    """exp(x) - 1 without importing math."""
    return _E**x - 1.0


def planck_lambda(wavelength, temperature):
    """
    Spectral radiance B_lambda(T, lambda) from Planck's law.

    B_lambda = (2 h c^2 / lambda^5) / (exp(h c / (lambda k T)) - 1)

    Parameters
    ----------
    wavelength : float
        Wavelength in meters.
    temperature : float
        Temperature in kelvin.

    Returns
    -------
    float
        Spectral radiance in W / (m^3 * sr).
    """
    exponent = H * C / (wavelength * K_B * temperature)
    return (2.0 * H * C**2 / wavelength**5) / _expm1(exponent)


def planck_nu(frequency, temperature):
    """
    Spectral radiance B_nu(T, nu) from Planck's law.

    B_nu = (2 h nu^3 / c^2) / (exp(h nu / (k T)) - 1)

    Parameters
    ----------
    frequency : float
        Frequency in hertz.
    temperature : float
        Temperature in kelvin.

    Returns
    -------
    float
        Spectral radiance in W / (m^2 * Hz * sr).
    """
    exponent = H * frequency / (K_B * temperature)
    return (2.0 * H * frequency**3 / C**2) / _expm1(exponent)


# ---------------------------------------------------------------------------
# Astropy BlackBody (unit-aware)
# ---------------------------------------------------------------------------

_SLAM = u.erg / (u.cm**2 * u.s * u.AA * u.sr)


def astropy_spectral_radiance(wavelength, temperature):
    """
    Spectral radiance B_lambda(T, lambda) via Astropy's BlackBody model.

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


def astropy_spectral_radiance_nu(frequency, temperature):
    """
    Spectral radiance B_nu(T, nu) via Astropy's BlackBody model.

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

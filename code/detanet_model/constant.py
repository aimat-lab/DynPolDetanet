"""Physical constants and reference data shared across the DetaNet model.

Everything here is a plain module-level tensor/float, imported either
explicitly (`from .constant import atom_masses`) or via the wildcard
`from .constant import *` used in `detanet_model/__init__.py` and
`spectra_simulator.py`.
"""
from torch import tensor,float32

# Atomic mass (amu) indexed by atomic number Z, i.e. atom_masses[Z].
# Index 0 is a dummy padding entry (there is no element with Z=0) so that
# atom_masses[z] can be indexed directly with a tensor of atomic numbers
# without an off-by-one correction. Covers Z=1 (H) through Z=109 (Mt).
atom_masses=tensor([0,1.008,  4.002602,  6.94,  9.0121831,  10.81,
                    12.011,  14.007,  15.999,  18.998403163,  20.1797,
                    22.98976928,  24.305,  26.9815385,  28.085,  30.973761998,
                    32.06,  35.45,  39.948,  39.0983,  40.078,
                    44.955908,  47.867,  50.9415,  51.9961,  54.938044,
                    55.845,  58.933194,  58.6934,  63.546,  65.38,
                    69.723,  72.63,  74.921595,  78.971,  79.904,
                    83.798,  85.4678,  87.62,  88.90584,  91.224,
                    92.90637,  95.95,  97.90721,  101.07,  102.9055,
                    106.42,  107.8682,  112.414,  114.818,  118.71,
                    121.76,  127.6,  126.90447,  131.293,  132.90545196,
                    137.327,  138.90547,  140.116,  140.90766,  144.242,
                    144.91276,  150.36,  151.964,  157.25,  158.92535,
                    162.5,  164.93033,  167.259,  168.93422,  173.054,
                    174.9668,  178.49,  180.94788,  183.84,  186.207,
                    190.23,  192.217,  195.084,  196.966569,  200.592,
                    204.38,  207.2,  208.9804,  208.98243,  209.98715,
                    222.01758,  223.01974,  226.02541,  227.02775,  232.0377,
                    231.03588,  238.02891,  237.04817,  244.06421,  243.06138,
                    247.07035,  247.07031,  251.07959,  252.083,  257.09511,
                    258.09843,  259.101,  262.11,  267.122,  268.126,
                    271.134,  270.133,  269.1338,  278.156,  281.165,],
                   dtype=float32)

# --------------------------------------------------------------------------
# Per-atom isolated-atom reference energies, indexed by atomic number Z
# (same padding-at-index-0 convention as atom_masses above). Used as the
# `atom_ref` argument to DetaNet when predicting total/atomization energies:
# the model predicts atomic contributions, and atom_ref[z] is added back per
# atom so that atomization energy = predicted energy - sum(atom_ref[z_i]).
# Zero entries are elements not present in the corresponding dataset's
# chemical space (e.g. QM9-derived sets only cover H, B, C, N, O, F).
# Values are given in Hartree in the source data and converted to eV here
# via the *27.21138 factor (matching `ha_ev` below), except qm7x_e_ref,
# which is already tabulated directly in eV.
# --------------------------------------------------------------------------

# QM9S reference energies, B3LYP/def2-TZVP level of theory.
ref_qm9s_energy=tensor([0,-0.5021542,0,
                                  0, 0, -24.6636882, -37.8592244, -54.6039774, -75.0952568, -99.7690884, 0,
                                  0, 0, -242.3844686, -289.39065, -341.2794682, -398.1299236, -460.1644113, 0,
                                  ],dtype=float32)*27.21138

# QM9 reference energies, B3LYP/6-31G(2df) level of theory.
qm9_e_ref=tensor([0,-0.500273,0,0,0,0,-37.846772,-54.583861,-75.064579,-99.717314,0],dtype=float32)*27.21138

# QM7-X reference energies, PBE0+MBD level of theory (already in eV).
qm7x_e_ref=tensor([0,0,0,0,0,0,-1027.592489146,-1484.274819088,-2039.734879322,
            0,0,0,0,0,0,0,-10828.707468187,-12516.444619523],dtype=float32)

# --------------------------------------------------------------------------
# Fundamental physical constants (SI units unless noted) and unit-conversion
# factors, used mainly by spectra_simulator.py to turn predicted quantum-
# chemical quantities (energies, dipole/transition moments) into simulated
# IR/Raman/UV-vis spectra with physically meaningful intensities.
# --------------------------------------------------------------------------

# Planck constant, J*s.
hp=6.62606896*1e-34

# pi.
pi=3.1415926

# Boltzmann constant, J/K.
Kb=1.380649 * 1e-23

# Avogadro's constant, mol^-1.
NA=6.02*1e23

# Vacuum permittivity, F/m.
vac=8.854187817*1e-12

# Speed of light in vacuum, m/s.
c=299792458

# eV -> Hartree conversion factor.
ev_ha=1/27.21138

# Angstrom -> Bohr conversion factor.
A_bohr=1/0.529177249

# Hartree -> eV conversion factor.
ha_ev=27.21138

# eV -> Joule conversion factor.
ev_joule=1.6021766208e-19

# Atomic mass unit -> kilogram conversion factor.
amu_kg = 1.660539040e-27

# Angstrom -> meter conversion factor.
A_m=1e-10

# Debye^2/Angstrom^2 -> nm*mol^-1 style scaling constant used in IR
# intensity conversion (paired with ir_coff below).
N_mD = 1e8

# Frequency unit conversion: wavenumber (cm^-1) -> Hz.
cm_hz=2.99792458e10

# IR intensity unit-conversion coefficient: (Debye^2/Angstrom^2/amu) -> km/mol.
ir_coff=42.2561

# Raman activity unit-conversion coefficient: (Bohr^6/Angstrom^2/amu) -> Angstrom^4/amu.
raman_coff=0.021958718449

# Reference wavelength-energy product (nm*eV) used to convert between
# excitation wavelength and excitation energy: wavelength[nm] = wlmax / E[eV].
wlmax=1239.85

# Atomic-unit dipole moment -> Debye conversion factor.
dipole_au_D=2.541746

# Hessian (eV/Angstrom^2/amu) -> SI (J/m^2/kg) conversion factor, i.e. to
# units directly usable for computing vibrational frequencies (rad/s or Hz)
# from the mass-weighted Hessian eigenvalues.
hess_t=ev_joule/((A_m**2)*amu_kg)

# Hessian (eV/Angstrom^2) -> force constant (mDyne/Angstrom) conversion factor.
hessian_to_fc=ev_joule*N_mD/A_m



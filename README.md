# FLFM_PSF_simulation_python-
Python version for PSF simulation for Fourier Light Field Microscopy

This repository contains a Python script for simulating the Point Spread Function (PSF) of a Single-Molecule Light-Field Microscope (SMLFM).

This script, adapted from an original MATLAB implementation, uses Fourier optics to model the microscope's optical path. It generates a 3D PSF stack by simulating the electric field from an isotropic emitter, applying a phase mask for the microlens array (MLA), and propagating the field to the image plane.

Key features include:

Configuration of system parameters (e.g., NA, focal lengths, MLA pitch) through a dedicated system class.
Simulation of the electric field at the back focal plane (BFP).
Generation of hexagonal microlens array center coordinates.
Fresnel propagation from the BFP to the image plane.
Visualization of the resulting PSF hyperstack.

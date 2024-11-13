# ECTOPLASM

This repository provides digital tools for Euler Characteristic Transform (ECT) based shape analysis in 3 dimensions for piecewise linear shapes, based on the preprint TBA.

A particularity of this algorithm is that it can compute the transforms and their distances in **closed form**, providing accurate inference and bypassing the need for selecting the discretization parameters.

The main functionalities of the package require just numpy and scipy. For aspects requiring auto-differentiability, torch and torchquad are also required. 

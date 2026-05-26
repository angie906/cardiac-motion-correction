# Motion Correction for Panoramic Cardiac Optical Mapping 

## Overview

This project implements a motion correction pipeline for optical mapping offline-images of beating mouse hearts.

The algorithm was developed for panoramic cardiac optical mapping systems and aims to compensate for tissue motion while preserving electrical propagation dynamics.

The pipeline uses dense optical flow estimation with adaptive reference registration to stabilize image sequences acquired from high-speed cardiac imaging experiments.

## Features

- Dense optical flow-based motion estimation
- Adaptive reference frame updating
- Frame warping and stabilization
- Binary heart segmentation (Watershed Segmentation)
- Morphological mask refinement
- Hole filling and contour filtering
- TIFF stack processing
- Optical mapping preprocessing workflow

## Processing Pipeline

The processing workflow consists of:

1. TIFF stack loading
2. Heart segmentation
3. Binary mask refinement
4. Dense optical flow estimation
5. Motion vector accumulation
6. Image warping
7. Adaptive reference update
8. Motion Correction Validation (Comparison Plotting)

# Change Log

All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).
 
## [Unreleased] - 2022-06-18
  
The following files were added

- plotvoxcloud : light source on , intended to render point cloud geometry , light source is always 30 degrees (counterclockwise around the z-axis) from the camera position, camera always at approximately twice the farthest distance from the origin. 

- plotvoxcloud2 : light source off , intended to render point clouds with attributes , tries to place the camera at a distance such that the point cloud occupies the whole field of view.

Original code in which these scripts were based : https://www.kaggle.com/datasets/daavoo/3d-mnist

AxisHelper was renamed to AxesHelper https://github.com/mrdoob/three.js/commit/845999e06fb8c64c30d413bf668aa9da68357e01

Suggestions:
- save plotVC.html in /tmp/ instead of in the current folder
- join the two scripts in one, giving the option to turn the light source on or off
- improve the way that the initial camera position is defined, none of the current methods work 100%

## [Unreleased] - 2022-06-23

I joined the two scripts plotvoxcloud and plotvoxcloud2 in a single script, by adding an input to control the lights in the plotvoxcloud script.
Improved the initial camera position, I think I figured out what was wrong.
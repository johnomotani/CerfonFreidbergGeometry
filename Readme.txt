Help on module CerfonFreidberg:

NAME
    CerfonFreidberg

FILE
    /home/jomotani/Pedestal/CerfonFreidbergGeometry/CerfonFreidberg.py

DESCRIPTION
    # Compute poloidal flux function Psi from analytic Grad-Shafranov solutions in
    # Cerfon & Freidberg, Physics of Plasmas 17, 032502 (2010); doi: 10.1063/1.3328818
    # Matlab implementation of solutions for c1..c12 by James Cook <j.w.s.cook@warwick.ac.uk> (2013)
    # Python implementation by John Omotani <omotani@chalmers.se> (2015)

CLASSES
    CerfonFreidberg
    
    class CerfonFreidberg
     |  x and y are normalised coordinates, x=R/R0 and y=Z/R0
     |  epsi is the inverse aspect ratio
     |  kapp is the elongation
     |  delt is the triangularity
     |  xsep, ysep are the coordinates of the X-point (ysep<0 by assumption)
     |  qsta is 'q_*' as defined in the Cerfon & Freidberg paper; it is not actually used here
     |  A is a parameter whose value determines the toroidal beta of the equilibrium (see C&F)
     |  R0 is the nominal major radius of the plasma
     |  B0 is the vacuum magnetic field at R0
     |  
     |  Methods defined here:
     |  
     |  BR(self)
     |      Returns a function that evaluates the R component the magnetic field at (R,Z)
     |  
     |  BZ(self)
     |      Returns a function that evaluates the Z component the magnetic field at (R,Z)
     |  
     |  Bp(self)
     |      Returns a function that evaluates the poloidal magnetic field at (R,Z)
     |  
     |  Bp_symbolic(self, R, Z)
     |      Returns a sympy expression for the poloidal magnetic field in terms of x and y
     |  
     |  Bt(self)
     |      Returns a function that evaluates the toroidal magnetic field at (R,Z)
     |  
     |  Bt_symbolic(self, R, Z)
     |      Returns a sympy expression for the toroidal magnetic field in terms of x and y
     |  
     |  Psi(self)
     |      Returns a function that evaluates the poloidal flux Psi at (R,Z)
     |  
     |  betat(self)
     |      Returns the toroidal beta
     |  
     |  calculateAAndPsi0FromBetatAndCurrent(self, betat, I)
     |      Iteratively calculate the integration constant A and the value of Psi0 from toroidal beta, betat, and plasma current, I.
     |  
     |  calculateAAndPsi0FromBetatAndq(self, betat, q, psiVal)
     |      Iteratively calculate the integration constant A and the value of Psi0 from toroidal beta, betat, and the safety factor, q on a specified flux surface with psi of psiVal.
     |  
     |  calculateCp(self)
     |      Calculates the normalised plasma circumference Cp
     |  
     |  calculatePsi0FromBetat(self, betat)
     |      Calculates Psi0 given an input toroidal beta
     |  
     |  calculatePsi0FromCurrent(self, I)
     |      Calculates Psi0 given a value for the plasma current
     |  
     |  current(self)
     |      Calculate the plasma current from the equilibrium.
     |  
     |  dBpdR(self)
     |      Returns a function that evaluates the R derivative of the poloidal magnetic field at (R,Z)
     |  
     |  dBpdZ(self)
     |      Returns a function that evaluates the Z derivative of the poloidal magnetic field at (R,Z)
     |  
     |  dBtdR(self)
     |      Returns a function that evaluates the R derivative of the toroidal magnetic field at (R,Z)
     |  
     |  dBtdZ(self)
     |      Returns a function that evaluates the Z derivative of the toroidal magnetic field at (R,Z)
     |  
     |  dPsidR(self)
     |      Returns a function that evaluates the R-derivative of the poloidal flux dPsi/dR at (R,Z)
     |  
     |  dPsidZ(self)
     |      Returns a function that evaluates the Z-derivative of the poloidal flux dPsi/dZ at (R,Z)
     |  
     |  getAxis(self)
     |      Finds the maximum of Psi, which is the magnetic axis. Sets the values of Raxis, Zaxis and Psiaxis, and also returns them.
     |  
     |  getFluxSurfaceGrid(self, psiNGrid, thetaGrid)
     |      Take 1d arrays of normalised flux, psiNGrid, and poloidal angle (centred on the magnetic axis), thetaGrid,
     |      and return a 2d arrays giving major radius, R[ipsi,itheta], and height, Z[ipsi,itheta], the Cartesian
     |      coordinates in the poloidal plane, and also the minor radius, r[ipsi,itheta], of the logically rectangular
     |      psiNGrid*thetaGrid grid.
     |  
     |  getMinorRadiusGrid(self, theta, psiNGrid, rguess=None)
     |      Find the minor radius for an array of psiN at some angle theta
     |  
     |  init(self, epsi, kapp, delt, xsep, ysep, A, R0, B0, Psi0=None)
     |      Initialize by giving desired parameter values. Psi0 can be calculated from other quantities, e.g. toroidal beta, if not already known
     |  
     |  initByName(self, machineName)
     |      Initialize using parameters representative of machines:
     |      "ITER": See R Aymar, Barabaschi and Shimomura 2002 Plasma Phys. Control. Fusion 44 519
     |      "NSTX": ??? Ono, Masayuki, S. M. Kaye, Y-KM Peng, G. Barnes, W. Blanchard, M. D. Carter, J. Chrzanowski et al. "Exploration of spherical torus physics in the NSTX device." Nuclear Fusion 40, no. 3Y (2000): 557.
     |  
     |  initPlotting(self, xmin, xmax, ymin, ymax)
     |      Set limits for plotting of flux surfaces, in x,y coordinates.
     |  
     |  p(self)
     |      Returns a function that evaluates the pressure at (R,Z)
     |  
     |  plotFluxSurfaces(self)
     |      Plot the flux surfaces from computed solutions.
     |  
     |  q(self, psiVal)
     |      Calculate safety factor, q for some psi-surface
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  integrationGridSize = 1000



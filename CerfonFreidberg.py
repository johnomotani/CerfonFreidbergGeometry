# Compute poloidal flux function Psi from analytic Grad-Shafranov solutions in
# Cerfon & Freidberg, Physics of Plasmas 17, 032502 (2010); doi: 10.1063/1.3328818
# Matlab implementation of solutions for c1..c12 by James Cook <j.w.s.cook@warwick.ac.uk> (2013)
# Python implementation by John Omotani <omotani@chalmers.se> (2015)

import numpy
import scipy.optimize
import scipy.integrate
import sympy
from matplotlib import pyplot
import math

class CerfonFreidberg:
    """
    x and y are normalised coordinates, x=R/R0 and y=Z/R0
    epsi is the inverse aspect ratio
    kapp is the elongation
    delt is the triangularity
    xsep, ysep are the coordinates of the X-point (ysep<0 by assumption)
    qsta is 'q_*' as defined in the Cerfon & Freidberg paper; it is not actually used here
    A is a parameter whose value determines the toroidal beta of the equilibrium (see C&F)
    R0 is the nominal major radius of the plasma
    B0 is the vacuum magnetic field at R0
    """

    integrationGridSize = 1000

    def initByName(self,machineName):
        """
        Initialize using parameters representative of machines:
        "ITER": See R Aymar, Barabaschi and Shimomura 2002 Plasma Phys. Control. Fusion 44 519
        "NSTX": ??? Ono, Masayuki, S. M. Kaye, Y-KM Peng, G. Barnes, W. Blanchard, M. D. Carter, J. Chrzanowski et al. "Exploration of spherical torus physics in the NSTX device." Nuclear Fusion 40, no. 3Y (2000): 557.
        """

        if machineName == "ITER":
            epsi = 0.32
            kapp = 1.7
            delt = 0.33
            xsep = 0.88
            ysep = -0.6
            qsta = 1.57
            #A = -0.155 # C&F's value
            A = -0.0825542587266
            R0 = 6.2
            B0 = 5.3
            #Psi0 = 228.820949438 # from betat using C&F's A
            Psi0 = 236.265277402
            # plotting stuff
            xmin = 0.5
            xmax = 1.5
            ymin = -0.8
            ymax = 0.6
        elif machineName == "NSTX":
            epsi = 0.78
            kapp = 2
            delt = 0.35
            xsep = 0.7
            ysep = -1.71
            qsta = 2
            A = -0.05
            R0 = 0.85
            B0 = 0.3
            Psi0 = None
            # plotting stuff
            xmin = 0
            xmax = 2
            ymin = -2
            ymax = 1.6
        else:
            print("Unknown machineName:",machineName)
            raise Exception

        self.init(epsi,kapp,delt,xsep,ysep,A,R0,B0,Psi0)

    def init(self,epsi,kapp,delt,xsep,ysep,A,R0,B0,Psi0=None):
        """
        Initialize by giving desired parameter values. Psi0 can be calculated from other quantities, e.g. toroidal beta, if not already known
        """

        self.epsi = epsi
        self.kapp = kapp
        self.delt = delt
        self.xsep = xsep
        self.ysep = ysep
        self.A = A
        self.R0 = R0
        self.B0 = B0
        self.Psi0 = Psi0 # call calculatePsi0FromBetat or calculatePsi0FromCurrent to get this value, if needed

        self.mu0 = 4.e-7*numpy.pi
        self.Cp = None # call calculateCp to get this value, if needed

        self.xmin = .8*(1.-self.epsi)
        self.xmax = 1.+self.epsi+.2*(1.-self.epsi)
        self.ymin = 1.2*self.ysep
        self.ymax = 1.2*self.epsi*self.kapp

        if A is not None:
            # Can only actually solve for coefficients if A is given, otherwise calculateAAndPsi0FromBetatAndCurrent() must be called to calculate A
            
            alph = numpy.arcsin(delt)
            N1 = -(1.+alph)**2/epsi/kapp**2
            N2 = (1.-alph)**2/epsi/kapp**2
            N3 = -kapp/epsi/numpy.cos(alph)**2

            x,y = sympy.symbols("x y")
            c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 = sympy.symbols("c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12")

            psi = CerfonFreidberg._getpsi(x,y,A,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12)
            psi_x = sympy.diff(psi,x)
            psi_xx = sympy.diff(psi_x,x)
            psi_y = sympy.diff(psi,y)
            psi_yy = sympy.diff(psi_y,y)

            # solve for coefficients
            s = sympy.solvers.solve( [psi.subs([(x,1.+epsi),(y,0.)]),
                                      psi.subs([(x,1.-epsi),(y,0.)]),
                                      psi.subs([(x,1.-delt*epsi),(y,kapp*epsi)]),
                                      psi.subs([(x,xsep),(y,ysep)]),
                                      psi_y.subs([(x,1.+epsi),(y,0.)]),
                                      psi_y.subs([(x,1.-epsi),(y,0.)]),
                                      psi_x.subs([(x,1.-delt*epsi),(y,kapp*epsi)]),
                                      psi_x.subs([(x,xsep),(y,ysep)]),
                                      psi_y.subs([(x,xsep),(y,ysep)]),
                                      (psi_yy+N1*psi_x).subs([(x,1.+epsi),(y,0.)]),
                                      (psi_yy+N2*psi_x).subs([(x,1.-epsi),(y,0.)]),
                                      (psi_xx+N3*psi_y).subs([(x,1.-delt*epsi),(y,kapp*epsi)])
                                     ]
                                     ,[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12]
                                   )
            # evaluate coefficients.
            self.c1 = s[c1]
            self.c2 = s[c2]
            self.c3 = s[c3]
            self.c4 = s[c4]
            self.c5 = s[c5]
            self.c6 = s[c6]
            self.c7 = s[c7]
            self.c8 = s[c8]
            self.c9 = s[c9]
            self.c10 = s[c10]
            self.c11 = s[c11]
            self.c12 = s[c12]

            self.Raxis = None
            self.Zaxis = None
            self.Psiaxis = None
            if Psi0:
                # Psiaxis only makes sense if the value of Psi0 is known
                self.getAxis() # Set the values of Raxis, Zaxis, Psiaxis

            #print("c1",c1)
            #print("c2",c2)
            #print("c3",c3)
            #print("c4",c4)
            #print("c5",c5)
            #print("c6",c6)
            #print("c7",c7)
            #print("c8",c8)
            #print("c9",c9)
            #print("c10",c10)
            #print("c11",c11)
            #print("c12",c12)

            #print("test1",CerfonFreidberg._getpsi(1.+epsi,0.,A,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12))
            #print("test2",CerfonFreidberg._getpsi(1.-epsi,0.,A,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12))
            #print("test3",CerfonFreidberg._getpsi(1.-delt*epsi,kapp*epsi,A,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12))
            #print("test4",CerfonFreidberg._getpsi(xsep,ysep,A,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12))
            #print("test5",psi_y.subs([
            #                            (x,1.+epsi),
            #                            (y,0.),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7),
            #                            (sympy.Symbol('c8'),c8),
            #                            (sympy.Symbol('c9'),c9),
            #                            (sympy.Symbol('c10'),c10),
            #                            (sympy.Symbol('c11'),c11),
            #                            (sympy.Symbol('c12'),c12)]))

            #print("test6",psi_y.subs([
            #                            (x,1.-epsi),
            #                            (y,0.),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7),
            #                            (sympy.Symbol('c8'),c8),
            #                            (sympy.Symbol('c9'),c9),
            #                            (sympy.Symbol('c10'),c10),
            #                            (sympy.Symbol('c11'),c11),
            #                            (sympy.Symbol('c12'),c12)]))

            #print("test7",psi_x.subs([
            #                            (x,1.-delt*epsi),
            #                            (y,kapp*epsi),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7),
            #                            (sympy.Symbol('c8'),c8),
            #                            (sympy.Symbol('c9'),c9),
            #                            (sympy.Symbol('c10'),c10),
            #                            (sympy.Symbol('c11'),c11),
            #                            (sympy.Symbol('c12'),c12)]))

            #print("test8",psi_x.subs([
            #                            (x,xsep),
            #                            (y,ysep),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7),
            #                            (sympy.Symbol('c8'),c8),
            #                            (sympy.Symbol('c9'),c9),
            #                            (sympy.Symbol('c10'),c10),
            #                            (sympy.Symbol('c11'),c11),
            #                            (sympy.Symbol('c12'),c12)]))

            #print("test9",psi_y.subs([
            #                            (x,xsep),
            #                            (y,ysep),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7),
            #                            (sympy.Symbol('c8'),c8),
            #                            (sympy.Symbol('c9'),c9),
            #                            (sympy.Symbol('c10'),c10),
            #                            (sympy.Symbol('c11'),c11),
            #                            (sympy.Symbol('c12'),c12)]))

            #print("test10",(psi_yy+N1*psi_x).subs([
            #                            (x,1.+epsi),
            #                            (y,0.),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7),
            #                            (sympy.Symbol('c8'),c8),
            #                            (sympy.Symbol('c9'),c9),
            #                            (sympy.Symbol('c10'),c10),
            #                            (sympy.Symbol('c11'),c11),
            #                            (sympy.Symbol('c12'),c12)]))

            #print("test11",(psi_yy+N2*psi_x).subs([
            #                            (x,1.-epsi),
            #                            (y,0.),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7),
            #                            (sympy.Symbol('c8'),c8),
            #                            (sympy.Symbol('c9'),c9),
            #                            (sympy.Symbol('c10'),c10),
            #                            (sympy.Symbol('c11'),c11),
            #                            (sympy.Symbol('c12'),c12)]))

            #print("test12",(psi_xx+N3*psi_y).subs([
            #                            (x,1.-delt*epsi),
            #                            (y,kapp*epsi),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7),
            #                            (sympy.Symbol('c8'),c8),
            #                            (sympy.Symbol('c9'),c9),
            #                            (sympy.Symbol('c10'),c10),
            #                            (sympy.Symbol('c11'),c11),
            #                            (sympy.Symbol('c12'),c12)]))

    def setPlotLimits(self,xmin,xmax,ymin,ymax):
        """
        Set limits for plotting of flux surfaces, in x,y coordinates.
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def plotFluxSurfaces(self):
        """
        Plot the flux surfaces from computed solutions.
        """
        x1 = numpy.linspace(self.xmin,self.xmax,100)
        y1 = numpy.linspace(self.ymin,self.ymax,100)

        psiArray = numpy.zeros((100,100))
        psi = self._psi_xy()
        for i in range(100):
            psiArray[:,i] = psi(x1[i],y1)
        if self.Psiaxis is None:
            psiAxis = self._getpsiAxisForPlotting()
        else:
            psiAxis = self.Psiaxis/self.Psi0
        pyplot.contour(x1,y1,psiArray,numpy.linspace(-.1*psiAxis,psiAxis,15))
        pyplot.show()

    def Psi(self):
        """
        Returns a function that evaluates the poloidal flux Psi at (R,Z)
        """
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z), self.Psi0*self._getpsi(R/self.R0,Z/self.R0,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12),"numpy")

    def dPsidR(self):
        """
        Returns a function that evaluates the R-derivative of the poloidal flux dPsi/dR at (R,Z)
        """
        return lambda R,Z: self.Psi0*self._dpsidx_xy()(R/self.R0,Z/self.R0)/self.R0

    def dPsidZ(self):
        """
        Returns a function that evaluates the Z-derivative of the poloidal flux dPsi/dZ at (R,Z)
        """
        return lambda R,Z: self.Psi0*self._dpsidy_xy()(R/self.R0,Z/self.R0)/self.R0

    def getAxis(self):
        """
        Finds the maximum of Psi, which is the magnetic axis. Sets the values of Raxis, Zaxis and Psiaxis, and also returns them.
        """
        #if (not self.Raxis) or (not self.Zaxis) or (not self.Psiaxis):
        result = scipy.optimize.minimize(lambda posArray: self._psi_xy()(posArray[0],posArray[1]),[1.,0.])
        x = result.x[0]
        y = result.x[1]
        psi = self._psi_xy()(x,y)
        self.Raxis = x*self.R0
        self.Zaxis = y*self.R0
        self.Psiaxis = psi*self.Psi0
        return self.Raxis, self.Zaxis, self.Psiaxis

    def _getpsiAxisForPlotting(self):
        """
        Finds the maximum of Psi, which is the magnetic axis. Sets the values of Raxis, Zaxis and Psiaxis, and also returns them.
        """
        #if (not self.Raxis) or (not self.Zaxis) or (not self.Psiaxis):
        result = scipy.optimize.minimize(lambda posArray: self._psi_xy()(posArray[0],posArray[1]),[1.,0.])
        x = result.x[0]
        y = result.x[1]
        psi = self._psi_xy()(x,y)
        return psi

    def _psi_xy(self):
        # normalised form
        x,y = sympy.symbols("x y")
        return sympy.lambdify((x,y), self._getpsi(x,y,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12),"numpy")

    def _dpsidx_xy(self):
        x,y = sympy.symbols("x y")
        psi = self._getpsi(x,y,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12)
        return sympy.lambdify((x,y),sympy.diff(psi,x),"numpy")

    def _dpsidy_xy(self):
        x,y = sympy.symbols("x y")
        psi = self._getpsi(x,y,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12)
        return sympy.lambdify((x,y),sympy.diff(psi,y),"numpy")

    def _psi_xy_symbolic(self,x,y):
        return self._getpsi(x,y,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9,self.c10,self.c11,self.c12)

    def _dpsidx_xy_symbolic(self,x,y):
        return sympy.diff(self._psi_xy_symbolic(x,y),x)

    def _dpsidy_xy_symbolic(self,x,y):
        return sympy.diff(self._psi_xy_symbolic(x,y),y)

    def _pOverPsi02_xy(self):
        return lambda x,y: -1./self.mu0/self.R0**4*(1.-self.A)*self._psi_xy()(x,y)

    def _BpOverPsi0_xy(self):
        return lambda x,y: 1./self.R0*numpy.sqrt(self._dpsidx_xy()(x,y)**2+self._dpsidy_xy()(x,y)**2) / x / self.R0

    def _findSeparatrix(self,theta):
        # Find minor radius of psi=0 in normalised (x-y) units
        psi = self._psi_xy()
        psiAxis = psi(1.,0.) # Not really the axis, but should be close enough
        sign = psiAxis/numpy.abs(psiAxis)
        psiAtTheta = lambda a: sign*psi(1.+a*numpy.cos(theta),a*numpy.sin(theta)) # sign so that this function always decreases away from the axis
        # Need to find an endpoint with positive psi
        rguess = ( 2.*self.epsi*numpy.cos(theta)**2 # catch inner and outer equatorial points which are at 1+epsi for theta=0 and 1-epsi for theta=2pi
                   + 2.*self.kapp*self.epsi*numpy.sin(theta)**2 # catch high point and low point which are at kapp*epsi and -kapp*epsi
                 )
        if rguess*numpy.cos(theta)<-1.:
            # Avoid having x=0.
            rguess = -.999/numpy.cos(theta)
        if psi(1.+rguess*numpy.cos(theta),rguess*numpy.sin(theta))*psiAxis<0.: # psi changes sign
            endpoint = rguess
        else:
            result = scipy.optimize.minimize_scalar(psiAtTheta, bounds=(0.,rguess),method='bounded')
            a_max = result.x
            psi_max = psi(1.+a_max*numpy.cos(theta),a_max*numpy.sin(theta))
            if psi_max*psiAxis<0.:
                endpoint = a_max
            else:
                # Very close to the X-point, just use a_max as the minor radius of psi=0 surface
                return a_max
        return scipy.optimize.brentq(psiAtTheta,0.,endpoint)

    def _betatOverPsi02(self):
        pOverPsi02 = self._pOverPsi02_xy()
        pOverPsi02_total = 0.
        volume = 0.
        Ntheta = self.integrationGridSize
        dtheta = 2.*numpy.pi/float(Ntheta)
        for theta in numpy.linspace(0.,2.*numpy.pi,Ntheta,endpoint=False):
            a = self._findSeparatrix(theta)
            integral,error = scipy.integrate.quad(
                    lambda r: pOverPsi02(1.+r*numpy.cos(theta),r*numpy.sin(theta)) * r * (1.+r*numpy.cos(theta))*2.*numpy.pi
                    ,0.,a)
            pOverPsi02_total += integral*dtheta
            #integral,error = scipy.integrate.quad(lambda r: r * (1.+r*numpy.cos(theta))*2.*numpy.pi,0.,a)
            integral = (a**2/2.+a**3/3.*numpy.cos(theta))*2.*numpy.pi
            volume += integral*dtheta
        pOverPsi02_average = pOverPsi02_total/volume
        return 2.*self.mu0*pOverPsi02_average/self.B0**2

    def betat(self):
        """
        Returns the toroidal beta
        """
        return self.Psi0**2*self._betatOverPsi02()

    def calculatePsi0FromBetat(self,betat):
        """
        Calculates Psi0 given an input toroidal beta
        """
        self.Psi0 = numpy.sqrt(betat/self._betatOverPsi02())
        self.getAxis() # Now that Psi0 is known, can compute Psiaxis, so call getAxis now 
        return self.Psi0

    def calculateCp(self):
        """
        Calculates the normalised plasma circumference Cp
        """
        Ntheta = self.integrationGridSize
        dtheta = 2.*numpy.pi/float(Ntheta)
        self.Cp = 0.
        aplus = self._findSeparatrix(-dtheta/2.)
        xplus = 1.+aplus*numpy.cos(-dtheta/2.)
        yplus = aplus*numpy.sin(-dtheta/2.)
        for theta in numpy.linspace(0.,2.*numpy.pi,Ntheta,endpoint=False):
            aminus = aplus
            xminus = xplus
            yminus = yplus
            aplus = self._findSeparatrix(theta+dtheta/2.)
            xplus = 1.+aplus*numpy.cos(theta+dtheta/2.)
            yplus = aplus*numpy.sin(theta+dtheta/2.)
            self.Cp += numpy.sqrt((xplus-xminus)**2+(yplus-yminus)**2)
            
        return self.Cp

    def _integralBpOverPsi0AroundSeparatrix(self):
        # loop integral of Bp over plasma surface
        # Integral with unnormalised lengths (eventually): i.e. includes factor of R0
        psi = self._psi_xy()
        psi_x = self._dpsidx_xy()
        psi_y = self._dpsidy_xy()
        Ntheta = self.integrationGridSize
        dtheta = 2.*numpy.pi/float(Ntheta)
        integralBpOverPsi0 = 0.
        aplus = self._findSeparatrix(-dtheta/2.)
        xplus = 1.+aplus*numpy.cos(-dtheta/2.)
        yplus = aplus*numpy.sin(-dtheta/2.)
        for theta in numpy.linspace(0.,2.*numpy.pi,Ntheta,endpoint=False):
            aminus = aplus
            xminus = xplus
            yminus = yplus
            aplus = self._findSeparatrix(theta+dtheta/2.)
            xplus = 1.+aplus*numpy.cos(theta+dtheta/2.)
            yplus = aplus*numpy.sin(theta+dtheta/2.)
            a = self._findSeparatrix(theta)
            x = 1.+a*numpy.cos(theta)
            y = a*numpy.sin(theta)
            this_psi_x = psi_x(x,y)
            this_psi_y = psi_y(x,y)
            dlOverR0 = numpy.sqrt((xplus-xminus)**2+(yplus-yminus)**2)
            integralBpOverPsi0 += numpy.sqrt(this_psi_x**2+this_psi_y**2)/x*dlOverR0 # sqrt(psi_x**2+psi_y**2)=R0**2*x*(Bp/Psi0)
            
        # Divide out left-over factor of R0
        integralBpOverPsi0 /= self.R0

        return integralBpOverPsi0

    def calculatePsi0FromCurrent(self,I):
        """
        Calculates Psi0 given a value for the plasma current
        """
        # Bpbar = mu0*I/R0/Cp => integral(Bp) = mu0*I
        self.Psi0 = self.mu0*I/self._integralBpOverPsi0AroundSeparatrix()
        self.getAxis() # Now that Psi0 is known, can compute Psiaxis, so call getAxis now 
        return self.Psi0

    def _testfuncBetatAndCurrent(self,A,betat,I):
        print("trying ",A)
        # Re-initialise with this value of A
        self.init(self.epsi,self.kapp,self.delt,self.xsep,self.ysep,A,self.R0,self.B0,self.Psi0)
        # Using the current as the test variable (solving for Psi0 with betat) seems to converge better??
        print("Psi0",self.calculatePsi0FromCurrent(I))
        currentbetat = self.betat()
        print("betat",currentbetat)
        print("testval",currentbetat-betat)
        print("")
        #return self.betat()-betat
        return currentbetat - betat
        #print("Psi0",self.calculatePsi0FromBetat(betat))
        #thisCurrent = self.current()
        #print("current",thisCurrent)
        #print("testval",thisCurrent - I)
        #return thisCurrent - I

    def _calculateAAndPsi0(self,testfunc,arg1,arg2):
        # Use given value of A as an initial guess, and find a root that gives the desired optimization quantities
        # Start with low resolution to get close
        origN = self.integrationGridSize
        self.integrationGridSize = 50
        Amax = .9999
        searching = True
        testlower = -1.
        testupper = Amax # Don't need to do this if betat is zero, since then A is 1. exactly, but checking A=1. when calculating Psi0 from betat gives a divide-by-zero error, so start just below 1.
        while searching:
            try:
                self.A = scipy.optimize.brentq(testfunc,testupper,testlower,args=(arg1,arg2))
                searching = False
            except ValueError as e:
                # Presumably upper and lower limits have the same sign of the test value, so use a new range
                testupper = testlower
                testlower *= 2.
                if testlower>=testupper:
                    raise ValueError("Failed to find good limits")
                print("changing range to search to ("+str(testlower)+","+str(testupper)+")")
        # Now refine with high resolution, assuming that we are within testInterval of the accurate solution already
        testInterval = 0.04
        self.integrationGridSize = origN
        firstA = self.A
        while True:
            try:
                testlower = firstA-testInterval
                testupper = firstA+testInterval
                if testupper>Amax:
                    testupper = Amax
                self.A = scipy.optimize.brentq(testfunc,testlower,testupper,args=(arg1,arg2))
                break
            except ValueError as error:
                if error.message == "f(a) and f(b) must have different signs":
                    # Presumably interval was too small, so try increasing it
                    if testInterval > 2.:
                        # Interval is bigger than the original one, something must have gone wrong
                        error.message = "Unexpected failure to find interval around zero of A."
                        raise error
                    else:
                        testInterval *= 2.
                        print("testInterval",testInterval)
        print("found A =",self.A,"and Psi0 =",self.Psi0,"for arg1 =",arg1,"and arg2 =",arg2)

    def calculateAAndPsi0FromBetatAndCurrent(self,betat,I):
        """
        Iteratively calculate the integration constant A and the value of Psi0 from toroidal beta, betat, and plasma current, I.
        """
        self._calculateAAndPsi0(self._testfuncBetatAndCurrent,betat,I)

    def q(self,psiVal):
        """
        Calculate safety factor, q for some psi-surface
        """
        # Integral with unnormalised lengths (eventually): i.e. includes factor of R0
        Ntheta = self.integrationGridSize
        dtheta = 2.*numpy.pi/float(Ntheta)
        thetaGrid = numpy.linspace(0.,2.*numpy.pi,Ntheta,endpoint=False)
        R,Z,r = self.getFluxSurfaceGrid([psiVal],thetaGrid[:])
        Bt = self.Bt()(R,Z)
        dthetadR = -numpy.sin(thetaGrid)/r
        dthetadZ = numpy.cos(thetaGrid)/r
        JTimesR = (self.dPsidR()(R,Z)*dthetadZ-self.dPsidZ()(R,Z)*dthetadR) # J=B.Grad(theta)=Bp.Grad(theta)
        q_integral = (Bt/JTimesR).sum()*dtheta
            
        # Divide by integral(dtheta)
        q_integral /= 2.*numpy.pi

        return q_integral

    def _testfuncBetatAndq(self,A,betat,q,psiVal):
        print("trying ",A)
        # Re-initialise with this value of A
        self.init(self.epsi,self.kapp,self.delt,self.xsep,self.ysep,A,self.R0,self.B0,self.Psi0)
        # Using the current as the test variable (solving for Psi0 with betat) seems to converge better??
        print("Psi0",self.calculatePsi0FromBetat(betat))
        currentq = self.q(psiVal)
        print("q",currentq)
        print("testval",currentq-q)
        print("")
        return currentq-q

    def calculateAAndPsi0FromBetatAndq(self,betat,q,psiVal):
        """
        Iteratively calculate the integration constant A and the value of Psi0 from toroidal beta, betat, and the safety factor, q on a specified flux surface with psi of psiVal.
        """
        self._calculateAAndPsi0(lambda A,betat,q: self._testfuncBetatAndq(A,betat,q,psiVal),betat,q)

    def current(self):
        """
        Calculate the plasma current from the equilibrium.
        """
        return self.Psi0*self._integralBpOverPsi0AroundSeparatrix()/self.mu0

    def p(self):
        """
        Returns a function that evaluates the pressure at (R,Z)
        """
        return lambda R,Z: -self.Psi0**2/self.mu0/self.R0**4*(1-self.A)*self._psi_xy()(R/self.R0,Z/self.R0)

    def pressurePsiN(self, psinorm):
        """
        Returns the pressure at normalised psi (0 = magnetic axis, 1 = plasma edge)
        """
        psi = 1 - psinorm # psi in Cerfon-Freidberg 2010 goes from 1 on axis to 0 at edge
        
        return self.Psi0**2 * (1. - self.A)*psi / (self.mu0*self.R0**4)

    def fpolPsiN(self, psinorm):
        """
        Returns the poloidal current function f = R*Bt at given normalised psi
        (0 = magnetic axis, 1 = plasma edge)
        """
        psi = 1 - psinorm # psi in Cerfon-Freidberg 2010 goes from 1 on axis to 0 at edge
        return self.R0*numpy.sqrt(self.B0**2 - 2.*self.Psi0**2*self.A*psi/self.R0**4)

    def Bt(self):
        """
        Returns a function that evaluates the toroidal magnetic field at (R,Z)
        """
        return lambda R,Z: numpy.sqrt(self.R0**2/R**2*(self.B0**2-2.*self.Psi0**2/self.R0**4*self.A*self._psi_xy()(R/self.R0,Z/self.R0)))

    def Bt_symbolic(self,R,Z):
        """
        Returns a sympy expression for the toroidal magnetic field in terms of x and y
        """
        x,y = sympy.symbols("x y")
        return sympy.sqrt(self.R0**2/R**2*(self.B0**2-2.*self.Psi0**2/self.R0**4*self.A*self._psi_xy_symbolic(x,y).subs([(x,R/self.R0),(y,Z/self.R0)])))

    def dBtdR(self):
        """
        Returns a function that evaluates the R derivative of the toroidal magnetic field at (R,Z)
        """
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z),sympy.diff(self.Bt_symbolic(R,Z),R),"numpy")

    def dBtdZ(self):
        """
        Returns a function that evaluates the Z derivative of the toroidal magnetic field at (R,Z)
        """
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z),sympy.diff(self.Bt_symbolic(R,Z),Z),"numpy")

    def Bp(self):
        """
        Returns a function that evaluates the poloidal magnetic field at (R,Z)
        """
        return lambda R,Z: self.Psi0/self.R0*numpy.sqrt(self._dpsidx_xy()(R/self.R0,Z/self.R0)**2+self._dpsidy_xy()(R/self.R0,Z/self.R0)**2) / R

    def Bp_symbolic(self,R,Z):
        """
        Returns a sympy expression for the poloidal magnetic field in terms of x and y
        """
        x,y = sympy.symbols("x y")
        return self.Psi0/self.R0*sympy.sqrt(self._dpsidx_xy_symbolic(x,y).subs([(x,R/self.R0),(y,Z/self.R0)])**2+self._dpsidy_xy_symbolic(x,y).subs([(x,R/self.R0),(y,Z/self.R0)])**2) / R

    def dBpdR(self):
        """
        Returns a function that evaluates the R derivative of the poloidal magnetic field at (R,Z)
        """
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z),sympy.diff(self.Bp_symbolic(R,Z),R),"numpy")

    def dBpdZ(self):
        """
        Returns a function that evaluates the Z derivative of the poloidal magnetic field at (R,Z)
        """
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z),sympy.diff(self.Bp_symbolic(R,Z),Z),"numpy")

    def BR(self):
        """
        Returns a function that evaluates the R component the magnetic field at (R,Z)
        """
        return lambda R,Z: self.Psi0/self.R0*self._dpsidx_xy()(R/self.R0,Z/self.R0) / R

    def BZ(self):
        """
        Returns a function that evaluates the Z component the magnetic field at (R,Z)
        """
        return lambda R,Z: self.Psi0/self.R0*self._dpsidy_xy()(R/self.R0,Z/self.R0) / R

    def getMinorRadiusGrid(self,theta,psiNGrid,rguess=None):
        """
        Find the minor radius for an array of psiN at some angle theta
        """

        if psiNGrid.ndim > 1:
            raise ValueError("psiNGrid has more than one dimension.")
        if psiNGrid.max() >= 1.:
            raise ValueError("psiNGrid contains value(s) greater than or equal to 1")
        if psiNGrid.min() <= 0.:
            raise ValueError("psiNGrid contains value(s) less than or equal to 0")
        if psiNGrid.ndim == 0:
            # enumerate cannot handle 0d array
            psiNGrid = psiNGrid[numpy.newaxis]

        PsiGrid = (1.-psiNGrid)*self.Psiaxis

        # Function to evalute Psi for a given minor radius
        Psi = self.Psi()
        sign = self.Psiaxis/numpy.abs(self.Psiaxis)
        PsiAtTheta = lambda r: Psi(self.Raxis+r*numpy.cos(theta),self.Zaxis+r*numpy.sin(theta)) # sign so that this function always decreases away from the axis
        # Find an endpoint where Psi has the opposite sign to Psiaxis
        ##if PsiAtTheta(self.epsi*self.R0)*self.Psiaxis<0.:
        ##    endpoint = self.epsi*self.R0
        ##else:
        ##    result = scipy.optimize.minimize_scalar(lambda r: sign*PsiAtTheta(r), bounds=(0.,2.*self.R0),method='bounded')
        ##    endpoint = result.x
        ###endpoint = self.R0*self._findSeparatrix(theta) # Assuming we are only interested in closed field lines, this avoids duplicating heuristics in _findSeparatrix

        # Need to find an endpoint with positive psi
        if not rguess:
            rguess = ( 2.*(self.R0*self.epsi-(self.Raxis-self.R0)*numpy.cos(theta))*numpy.cos(theta)**2 # catch inner and outer equatorial points which are at 1+epsi for theta=0 and 1-epsi for theta=2pi
                       + 2.*(self.R0*self.kapp*self.epsi-self.Zaxis*numpy.sin(theta))*numpy.sin(theta)**2 # catch high point and low point which are at kapp*epsi and -kapp*epsi
                     )
        if rguess*numpy.cos(theta)<-self.Raxis:
            #print("option A")
            # Avoid having R=0.
            rguess = -.999*self.Raxis/numpy.cos(theta)
        if PsiAtTheta(rguess)*self.Psiaxis<0.: # psi changes sign
            #print("option B")
            endpoint = rguess
        else:
            #print("option C, rguess:",rguess)
            result = scipy.optimize.minimize_scalar(lambda r: PsiAtTheta(r)/self.Psiaxis, bounds=(0.,rguess),method='bounded')
            r_max = result.x
            Psi_max = PsiAtTheta(r_max)
            endpoint = r_max

        #print("")
        #aval = self._findSeparatrix(theta)
        #psia = self._psi_xy()(1.+aval*numpy.cos(theta),aval*numpy.sin(theta))
        #print("aval",aval)
        #print("psia",psia)
        #print("Psia",Psi(self.R0*(1.+aval*numpy.cos(theta)),self.R0*aval*numpy.sin(theta)))
        #print("")
        #print("endpoint",endpoint)
        #print("theta",theta)
        #print("Psi at 0:",PsiAtTheta(0.),PsiAtTheta(0.)-PsiGrid[0])
        #print("Psi at endpoint:",PsiAtTheta(endpoint),PsiAtTheta(endpoint)-PsiGrid[0])
        ##print("Psi near endpoint",PsiAtTheta(endpoint-.1),PsiAtTheta(endpoint+.1))
        #print("PsiGrid",PsiGrid)
        #print("PsiNGrid",psiNGrid)
        #print("Psiaxis",self.Psiaxis)
        #print("RAxis",self.Raxis)
        #print("ZAxis",self.Zaxis)

        rGrid = numpy.zeros(PsiGrid.size)
        for i,PsiVal in enumerate(PsiGrid):
            rGrid[i] = scipy.optimize.brentq(lambda r: PsiAtTheta(r)-PsiVal,0.,endpoint)
        return rGrid

    def getFluxSurfaceGrid(self,psiNGrid,thetaGrid):
        """
        Take 1d arrays of normalised flux, psiNGrid, and poloidal angle (centred on the magnetic axis), thetaGrid,
        and return a 2d arrays giving major radius, R[ipsi,itheta], and height, Z[ipsi,itheta], the Cartesian
        coordinates in the poloidal plane, and also the minor radius, r[ipsi,itheta], of the logically rectangular
        psiNGrid*thetaGrid grid.
        """
        psiNGrid = numpy.array(psiNGrid)
        thetaGrid = numpy.array(thetaGrid)
        if psiNGrid.ndim > 1:
            raise ValueError("psiN has more than one dimension.")
        if thetaGrid.ndim > 1:
            raise ValueError("theta has more than one dimension.")
        if psiNGrid.max() >= 1.:
            raise ValueError("psiNGrid contains value(s) greater than or equal to 1")
        if psiNGrid.min() <= 0.:
            raise ValueError("psiNGrid contains value(s) less than or equal to 0")
        if thetaGrid.ndim == 0:
            # enumerate cannot handle 0d array
            thetaGrid = thetaGrid[numpy.newaxis]

        Npsi = psiNGrid.size
        Ntheta = thetaGrid.size

        R = numpy.zeros([Npsi,Ntheta])
        Z = numpy.zeros([Npsi,Ntheta])
        r = numpy.zeros([Npsi,Ntheta])

        rguess = None
        for itheta,t in enumerate(thetaGrid):
            r[:,itheta] = self.getMinorRadiusGrid(t,psiNGrid,rguess)
            rguess = ( 1.1 + 2.*(thetaGrid[(itheta+1)%Ntheta]-t) )*r[-1,itheta] # minor radius should not change too much between adjacent theta-grid points.
        R = self.Raxis + r*numpy.cos(thetaGrid[numpy.newaxis,:])
        Z = self.Zaxis + r*numpy.sin(thetaGrid[numpy.newaxis,:])

        return R,Z,r

    @staticmethod
    def _getpsi(x,y,A,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12): # get parts of psi
        p = ( 1./8.*x**4 + A*(x**2.*sympy.log(x)/2.-1./8.*x**4) + c1*1. + c2*x**2 + c3*(y**2 - x**2*sympy.log(x))
            + c4*(x**4 - 4.*x**2*y**2) + c5*(2.*y**4 - 9.*y**2*x**2
            + 3.*x**4*sympy.log(x) - 12.*x**2*y**2*sympy.log(x)) + c6*(x**6
            - 12.*x**4*y**2 + 8.*x**2*y**4) + c7*(8.*y**6 - 140.*y**4*x**2
            + 75.*y**2*x**4 - 15*x**6*sympy.log(x) + 180.*x**4*y**2*sympy.log(x)
            - 120.*x**2*y**4*sympy.log(x)) + c8*(y) + c9*(y*x**2)
            + c10*(y**3 - 3.*y*x**2*sympy.log(x)) + c11*(3.*y*x**4 - 4.*y**3*x**2)
            + c12*(8.*y**5 - 45.*y*x**4 - 80.*y**3*x**2*sympy.log(x)
            + 60.*y*x**4*sympy.log(x))
            )
        return p


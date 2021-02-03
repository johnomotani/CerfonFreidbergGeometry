# Compute poloidal flux function Psi from analytic Grad-Shafranov solutions in
# Cerfon & Freidberg, Physics of Plasmas 17, 032502 (2010); doi: 10.1063/1.3328818
# Matlab implementation of solutions for c1..c12 by James Cook <j.w.s.cook@warwick.ac.uk> (2013)
# Python implementation by John Omotani <omotani@chalmers.se> (2015)

import numpy
import scipy.optimize
import scipy.integrate
import sympy
from matplotlib import pyplot

class CerfonFreidbergSymmetric:
    """
    x and y are normalised coordinates, x=R/R0 and y=Z/R0
    epsi is the inverse aspect ratio
    kapp is the elongation
    delt is the triangularity
    qsta is 'q_*' as defined in the Cerfon & Freidberg paper; it is not actually used here
    A is a parameter whose value determines the toroidal beta of the equilibrium (see C&F)
    R0 is the nominal major radius of the plasma
    B0 is the vacuum magnetic field at R0
    """

    integrationGridSize = 1000

    def initByName(self,machineName):

        if machineName == "ITER":
            # See R Aymar, Barabaschi and Shimomura 2002 Plasma Phys. Control. Fusion 44 519
            epsi = 0.32
            kapp = 1.7
            delt = 0.33
            qsta = 1.57
            #A = -0.155 # C&F quoted value
            A = -0.118071320696
            R0 = 6.2
            B0 = 5.3
            #Psi0 = 223.933073188 # from betat using C&F's A
            Psi0 = 227.517571766
            # plotting stuff
            xmin = 0.5
            xmax = 1.5
            ymin = -0.8
            ymax = 0.6
        elif machineName == "NSTX":
            # ??? Ono, Masayuki, S. M. Kaye, Y-KM Peng, G. Barnes, W. Blanchard, M. D. Carter, J. Chrzanowski et al. "Exploration of spherical torus physics in the NSTX device." Nuclear Fusion 40, no. 3Y (2000): 557.
            epsi = 0.78
            kapp = 2
            delt = 0.35
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

        self.init(epsi,kapp,delt,A,R0,B0,Psi0)
        self.initPlotting(xmin,xmax,ymin,ymax)

    def init(self,epsi,kapp,delt,A,R0,B0,Psi0=None, xpoint=True):
        # In principle Psi0 can be calculated from other quantities, e.g. toroidal beta, if not already known
        # xpoint=True creates double-null equilibria, xpoint=False creates equilibria  without X-points

        self.epsi = epsi
        self.kapp = kapp
        self.delt = delt
        self.A = A
        self.R0 = R0
        self.B0 = B0
        self.Psi0 = Psi0 # call calculatePsi0FromBetat or calculatePsi0FromCurrent to get this value, if needed

        self.mu0 = 4.e-7*numpy.pi
        self.Cp = None # call calculateCp to get this value, if needed

        if A:
            # Can only actually solve for coefficients if A is given, otherwise calculateAAndPsi0FromBetatAndCurrent() must be called to calculate A
            
            alph = numpy.arcsin(delt)
            N1 = -(1.+alph)**2/epsi/kapp**2
            N2 = (1.-alph)**2/epsi/kapp**2
            N3 = -kapp/epsi/numpy.cos(alph)**2

            x,y = sympy.symbols("x y")
            c1,c2,c3,c4,c5,c6,c7 = sympy.symbols("c1 c2 c3 c4 c5 c6 c7")

            psi = CerfonFreidbergSymmetric._getpsi(x,y,A,c1,c2,c3,c4,c5,c6,c7)
            psi_x = sympy.diff(psi,x)
            psi_xx = sympy.diff(psi_x,x)
            psi_y = sympy.diff(psi,y)
            psi_yy = sympy.diff(psi_y,y)

            # solve for coefficients
            if xpoint:
                s = sympy.solvers.solve( [psi.subs([(x,1.+epsi),(y,0.)]),
                                          psi.subs([(x,1.-epsi),(y,0.)]),
                                          psi.subs([(x,1.-1.1*delt*epsi),(y,1.1*kapp*epsi)]),
                                          psi_x.subs([(x,1.-1.1*delt*epsi),(y,1.1*kapp*epsi)]),
                                          psi_y.subs([(x,1.-1.1*delt*epsi),(y,1.1*kapp*epsi)]),
                                          (psi_yy+N1*psi_x).subs([(x,1.+epsi),(y,0.)]),
                                          (psi_yy+N2*psi_x).subs([(x,1.-epsi),(y,0.)]),
                                         ]
                                         ,[c1,c2,c3,c4,c5,c6,c7]
                                       )
            else:
                s = sympy.solvers.solve( [psi.subs([(x,1.+epsi),(y,0.)]),
                                          psi.subs([(x,1.-epsi),(y,0.)]),
                                          psi.subs([(x,1.-delt*epsi),(y,kapp*epsi)]),
                                          psi_x.subs([(x,1.-delt*epsi),(y,kapp*epsi)]),
                                          (psi_yy+N1*psi_x).subs([(x,1.+epsi),(y,0.)]),
                                          (psi_yy+N2*psi_x).subs([(x,1.-epsi),(y,0.)]),
                                          (psi_xx+N3*psi_y).subs([(x,1.-delt*epsi),(y,kapp*epsi)])
                                         ]
                                         ,[c1,c2,c3,c4,c5,c6,c7]
                                       )
            # evaluate coefficients.
            self.c1 = s[c1]
            self.c2 = s[c2]
            self.c3 = s[c3]
            self.c4 = s[c4]
            self.c5 = s[c5]
            self.c6 = s[c6]
            self.c7 = s[c7]

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

            #print("test1",CerfonFreidbergSymmetric._getpsi(1.+epsi,0.,A,c1,c2,c3,c4,c5,c6,c7))
            #print("test2",CerfonFreidbergSymmetric._getpsi(1.-epsi,0.,A,c1,c2,c3,c4,c5,c6,c7))
            #print("test3",CerfonFreidbergSymmetric._getpsi(1.-delt*epsi,kapp*epsi,A,c1,c2,c3,c4,c5,c6,c7))
            #print("test4",psi_x.subs([
            #                            (x,1.-delt*epsi),
            #                            (y,kapp*epsi),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7)]))

            #print("test5",(psi_yy+N1*psi_x).subs([
            #                            (x,1.+epsi),
            #                            (y,0.),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7)]))

            #print("test6",(psi_yy+N2*psi_x).subs([
            #                            (x,1.-epsi),
            #                            (y,0.),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7)]))

            #print("test12",(psi_xx+N3*psi_y).subs([
            #                            (x,1.-delt*epsi),
            #                            (y,kapp*epsi),
            #                            (sympy.Symbol('c1'),c1),
            #                            (sympy.Symbol('c2'),c2),
            #                            (sympy.Symbol('c3'),c3),
            #                            (sympy.Symbol('c4'),c4),
            #                            (sympy.Symbol('c5'),c5),
            #                            (sympy.Symbol('c6'),c6),
            #                            (sympy.Symbol('c7'),c7)]))

    def initPlotting(self,xmin,xmax,ymin,ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def plotFluxSurfaces(self):
        x1 = numpy.linspace(self.xmin,self.xmax,100)
        y1 = numpy.linspace(self.ymin,self.ymax,100)

        psiArray = numpy.zeros((100,100))
        psi = self.psi_xy()
        for i in range(100):
            psiArray[:,i] = psi(x1[i],y1)
        psiAxis = psiArray.min()
        pyplot.contour(x1,y1,psiArray,numpy.linspace(-.1*psiAxis,psiAxis,15))
        pyplot.show()

    def Psi(self):
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z), self.Psi0*self._getpsi(R/self.R0,Z/self.R0,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7),"numpy")

    def dPsidR(self):
        return lambda R,Z: self.Psi0*self.dpsidx_xy()(R/self.R0,Z/self.R0)/self.R0

    def dPsidZ(self):
        return lambda R,Z: self.Psi0*self.dpsidy_xy()(R/self.R0,Z/self.R0)/self.R0

    def getAxis(self):
        if not self.Raxis or not self.Zaxis or not self.PsiAxis:
            result = scipy.optimize.minimize(lambda posArray: self.psi_xy()(posArray[0],posArray[1]),[1.,0.])
            x = result.x[0]
            y = result.x[1]
            psi = self.psi_xy()(x,y)
            self.Raxis = x*self.R0
            self.Zaxis = y*self.R0
            self.PsiAxis = psi*self.Psi0
        return self.Raxis, self.Zaxis, self.PsiAxis

    def psi_xy(self):
        # normalised form
        x,y = sympy.symbols("x y")
        return sympy.lambdify((x,y), self._getpsi(x,y,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7),"numpy")

    def dpsidx_xy(self):
        x,y = sympy.symbols("x y")
        psi = self._getpsi(x,y,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7)
        return sympy.lambdify((x,y),sympy.diff(psi,x),"numpy")

    def dpsidy_xy(self):
        x,y = sympy.symbols("x y")
        psi = self._getpsi(x,y,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7)
        return sympy.lambdify((x,y),sympy.diff(psi,y),"numpy")

    def psi_xy_symbolic(self,x,y):
        return self._getpsi(x,y,self.A,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7)

    def dpsidx_xy_symbolic(self,x,y):
        return sympy.diff(self.psi_xy_symbolic(x,y),x)

    def dpsidy_xy_symbolic(self,x,y):
        return sympy.diff(self.psi_xy_symbolic(x,y),y)

    def _pOverPsi02_xy(self):
        return lambda x,y: -1./self.mu0/self.R0**4*(1.-self.A)*self.psi_xy()(x,y)

    def _BpOverPsi0_xy(self):
        return lambda x,y: 1./self.R0*numpy.sqrt(self.dpsidx_xy()(x,y)**2+self.dpsidy_xy()(x,y)**2) / x / self.R0

    def _findSeparatrix(self,theta):
        # Find minor radius of psi=0
        psi = self.psi_xy()
        psiAxis = psi(1.,0.) # Not really the axis, but should be close enough
        sign = psiAxis/numpy.abs(psiAxis)
        psiAtTheta = lambda a: sign*psi(1.+a*numpy.cos(theta),a*numpy.sin(theta)) # sign so that this function always decreases away from the axis
        # Need to find an endpoint with positive psi
        rguess = ( 2.*self.epsi*numpy.cos(theta)**2 # catch inner and outer equatorial points which are at 1+epsi for theta=0 and 1-epsi for theta=2pi
                   + 2.*self.kapp*self.epsi*numpy.sin(theta)**2 # catch high point and low point which are at kapp*epsi and -kapp*epsi
                 )
        if rguess*numpy.cos(theta)<-1.:
            # Avoid having x=0.
            rguess = .999
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
        return self.Psi0**2*self._betatOverPsi02()

    def calculatePsi0FromBetat(self,betat):
        self.Psi0 = numpy.sqrt(betat/self._betatOverPsi02())
        self.getAxis() # Now that Psi0 is known, can compute Psiaxis, so call getAxis now 
        return self.Psi0

    def calculateCp(self):
        # normalised plasma circumference
        psi = self.psi_xy()
        psi_x = self.dpsidx_xy()
        psi_y = self.dpsidy_xy()
        Ntheta = self.integrationGridSize
        dtheta = 2.*numpy.pi/float(Ntheta)
        self.Cp = 0.
        for theta in numpy.linspace(0.,2.*numpy.pi,Ntheta,endpoint=False):
            a = self._findSeparatrix(theta)
            x = 1.+a*numpy.cos(theta)
            y = a*numpy.sin(theta)
            this_psi_x = psi_x(x,y)
            this_psi_y = psi_y(x,y)
            dsdtheta = a*numpy.sqrt(this_psi_x**2+this_psi_y**2)/(this_psi_x*numpy.cos(theta)+this_psi_y*numpy.sin(theta))
            self.Cp += dsdtheta*dtheta

    def _integralBpOverPsi0AroundSeparatrix(self):
        # loop integral of Bp over plasma surface
        # Integral with unnormalised lengths (eventually): i.e. includes factor of R0
        psi = self.psi_xy()
        psi_x = self.dpsidx_xy()
        psi_y = self.dpsidy_xy()
        Ntheta = self.integrationGridSize
        dtheta = 2.*numpy.pi/float(Ntheta)
        integralBpOverPsi0 = 0.
        for theta in numpy.linspace(0.,2.*numpy.pi,Ntheta,endpoint=False):
            a = self._findSeparatrix(theta)
            x = 1.+a*numpy.cos(theta)
            y = a*numpy.sin(theta)
            this_psi_x = psi_x(x,y)
            this_psi_y = psi_y(x,y)
            integralBpOverPsi0 += a*dtheta*(this_psi_x**2+this_psi_y**2)/(this_psi_y*numpy.sin(theta)+this_psi_x*numpy.cos(theta))
            
        integralBpOverPsi0 /= self.R0

        return integralBpOverPsi0

    def calculatePsi0FromCurrent(self,I):
        # Bpbar = mu0*I/R0/Cp => integral(Bp) = mu0*I
        self.Psi0 = self.mu0*I/self._integralBpOverPsi0AroundSeparatrix()
        self.getAxis() # Now that Psi0 is known, can compute Psiaxis, so call getAxis now 
        return self.Psi0

    def _testfunc(self,A,betat,I):
        print("trying ",A)
        # Re-initialise with this value of A
        self.init(self.epsi,self.kapp,self.delt,A,self.R0,self.B0,self.Psi0)
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

    def calculateAAndPsi0FromBetatAndCurrent(self,betat,I):
        # Use given value of A as an initial guess, and find a root that gives the desired beta and current
        #self.A = scipy.optimize.newton(self._testfunc,self.A,args=(betat,I))
        # Start with low resolution to get close
        origN = self.integrationGridSize
        self.integrationGridSize = 50
        self.A = scipy.optimize.brentq(self._testfunc,1.,-1.,args=(betat,I))
        # Now refine with high resolution, assuming that we are within testInterval of the accurate solution already
        testInterval = 0.01
        self.integrationGridSize = origN
        firstA = self.A
        while True:
            try:
                self.A = scipy.optimize.brentq(self._testfunc,firstA-testInterval,firstA+testInterval,args=(betat,I))
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
        print("found A =",self.A,"and Psi0 =",self.Psi0,"for betat =",betat,"and current =",I)

    def current(self):
        return self.Psi0*self._integralBpOverPsi0AroundSeparatrix()/self.mu0

    def p(self):
        return lambda R,Z: -self.Psi0**2/self.mu0/self.R0**4*(1-self.A)*self.psi_xy()(R/self.R0,Z/self.R0)

    def Bt(self):
        return lambda R,Z: numpy.sqrt(self.R0**2/R**2*(self.B0**2-2.*self.Psi0**2/self.R0**4*self.A*self.psi_xy()(R/self.R0,Z/self.R0)))

    def Bt_symbolic(self,R,Z):
        x,y = sympy.symbols("x y")
        return sympy.sqrt(self.R0**2/R**2*(self.B0**2-2.*self.Psi0**2/self.R0**4*self.A*self.psi_xy_symbolic(x,y).subs([(x,R/self.R0),(y,Z/self.R0)])))

    def dBtdR(self):
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z),sympy.diff(self.Bt_symbolic(R,Z),R),"numpy")

    def dBtdZ(self):
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z),sympy.diff(self.Bt_symbolic(R,Z),Z),"numpy")

    def Bp(self):
        return lambda R,Z: self.Psi0/self.R0*numpy.sqrt(self.dpsidx_xy()(R/self.R0,Z/self.R0)**2+self.dpsidy_xy()(R/self.R0,Z/self.R0)**2) / R

    def Bp_symbolic(self,R,Z):
        x,y = sympy.symbols("x y")
        return self.Psi0/self.R0*sympy.sqrt(self.dpsidx_xy_symbolic(x,y).subs([(x,R/self.R0),(y,Z/self.R0)])**2+self.dpsidx_xy_symbolic(x,y).subs([(x,R/self.R0),(y,Z/self.R0)])**2) / R

    def dBpdR(self):
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z),sympy.diff(self.Bp_symbolic(R,Z),R),"numpy")

    def dBpdZ(self):
        R,Z = sympy.symbols("R Z")
        return sympy.lambdify((R,Z),sympy.diff(self.Bp_symbolic(R,Z),Z),"numpy")

    def BR(self):
        return lambda R,Z: self.Psi0/self.R0*self.dpsidx_xy()(R/self.R0,Z/self.R0) / R

    def BZ(self):
        return lambda R,Z: self.Psi0/self.R0*self.dpsidy_xy()(R/self.R0,Z/self.R0) / R

    def getMinorRadiusGrid(self,theta,psiNGrid):
        # Find minor radius of an array of psiN at some angle theta

        if psiNGrid.ndim > 1:
            raise ValueError("psiNGrid has more than one dimension.")
        if psiNGrid.max() >= 1.:
            raise ValueError("psiNGrid contains value(s) greater than or equal to 1")
        if psiNGrid.min() <= 0.:
            raise ValueError("psiNGrid contains value(s) less than or equal to 0")
        if psiNGrid.ndim == 0:
            # enumerate cannot handle 0d array
            psiNGrid = psiNGrid[numpy.newaxis]

        PsiGrid = (1.-psiNGrid)*self.PsiAxis

        # Function to evalute Psi for a given minor radius
        Psi = self.Psi()
        sign = self.PsiAxis/numpy.abs(self.PsiAxis)
        PsiAtTheta = lambda r: sign*Psi(self.Raxis+r*numpy.cos(theta),self.Zaxis+r*numpy.sin(theta)) # sign so that this function always decreases away from the axis
        # Find an endpoint where Psi has the opposite sign to PsiAxis
        if PsiAtTheta(self.R0)*self.PsiAxis<0.:
            endpoint = self.R0
        else:
            result = scipy.optimize.minimize_scalar(lambda r: PsiAtTheta(r), bounds=(0.,self.R0),method='bounded')
            endpoint = result.x
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

        for itheta,t in enumerate(thetaGrid):
            r[:,itheta] = self.getMinorRadiusGrid(t,psiNGrid)
        R = self.Raxis + r*numpy.cos(thetaGrid[numpy.newaxis,:])
        Z = self.Zaxis + r*numpy.sin(thetaGrid[numpy.newaxis,:])

        return R,Z,r

    @staticmethod
    def _getpsi(x,y,A,c1,c2,c3,c4,c5,c6,c7): # get parts of psi
        p = ( 1./8.*x**4 + A*(x**2.*sympy.log(x)/2.-1./8.*x**4) + c1*1. + c2*x**2 + c3*(y**2 - x**2*sympy.log(x))
            + c4*(x**4 - 4.*x**2*y**2) + c5*(2.*y**4 - 9.*y**2*x**2
            + 3.*x**4*sympy.log(x) - 12.*x**2*y**2*sympy.log(x)) + c6*(x**6
            - 12.*x**4*y**2 + 8.*x**2*y**4) + c7*(8.*y**6 - 140.*y**4*x**2
            + 75.*y**2*x**4 - 15*x**6*sympy.log(x) + 180.*x**4*y**2*sympy.log(x)
            - 120.*x**2*y**4*sympy.log(x))
            )
        return p


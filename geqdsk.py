
from datetime import date
from numpy import zeros, linspace

def f2s(f):
    """
    Format a string containing a float
    """
    s = ""
    if f >= 0.0:
        s += " "
    return s + "%1.9E" % f

class ChunkOutput:
    def __init__(self, filehandle, chunksize=5):
        self.fh = filehandle
        self.counter = 0
        self.chunk = chunksize

    def write(self, value):
        self.fh.write(f2s(value))
        self.counter += 1
        if self.counter == self.chunk:
            self.fh.write("\n")
            self.counter = 0
        
def write_1d(val, out):
    for i in range(len(val)):
        out.write(val[i])

def write_2d(val, out):
    nx,ny = val.shape
    for x in range(nx):
        for y in range(ny):
            out.write(val[x,y])
    
def write(eq, fh, nx=65, ny=65):
    """
    Write a GEQDSK equilibrium file, given a CerfonFreidberg equilibrium object
    
    eq - CerfonFreidberg object
    fh - file handle
    
    """
    
    # First line: Identification string, followed by resolution
    fh.write("  CERFONFREIDBERG %s 3  %d  %d\n" % (date.today().strftime("%d/%m/%Y"), nx, ny))
    
    rmin = eq.xmin * eq.R0
    rmax = eq.xmax * eq.R0
    zmin = eq.ymin * eq.R0
    zmax = eq.ymax * eq.R0

    
    # Second line
    rdim = rmax - rmin # Horizontal dimension in meter of computational box
    zdim = zmax - zmin # Vertical dimension in meter of computational box
    rcentr = eq.R0   # R in meter of vacuum toroidal magnetic field BCENTR
    rleft = rmin  # Minimum R in meter of rectangular computational box
    zmid = 0.5*(zmin + zmax) # Z of center of computational box in meter

    fh.write(f2s(rdim)+f2s(zdim)+f2s(rcentr)+f2s(rleft)+f2s(zmid)+"\n")
    
    # Third line
    
    rmaxis, zmaxis, simag = eq.getAxis() # Find magnetic axis
    sibdry = 0.0  # Psi at boundary
    bcentr = eq.B0 # Vacuum magnetic field at rcentr

    fh.write(f2s(rmaxis) + f2s(zmaxis) + f2s(simag) + f2s(sibdry) + f2s(bcentr) + "\n")
    
    # 4th line
    
    cpasma = eq.current()  # Plasma current
    fh.write(f2s(cpasma) + f2s(simag) + f2s(0.0) + f2s(rmaxis) + f2s(0.0) + "\n")
    
    # 5th line
    
    fh.write(f2s(zmaxis) + f2s(0.0) + f2s(sibdry) + f2s(0.0) + f2s(0.0) + "\n")

    # fill arrays
    workk = zeros([nx])
    
    psi = zeros([ny,nx])
    
    psifunc = eq.Psi() # Returns a function for evaluating psi
    rarr = linspace(rmin, rmax, nx)
    zarr = linspace(zmin, zmax, ny)
    for y in range(ny):
        for x in range(nx):
            psi[y,x] = psifunc(rarr[x], zarr[y])
    
    psinorm = linspace(0.0, 1.0, nx, endpoint=False) # Does not include separatrix
    fpol = eq.fpolPsiN(psinorm)
    pres = eq.pressurePsiN(psinorm)

    qpsi = zeros([nx])
    for i in range(1,nx): # Exclude axis
        qpsi[i] = eq.q(psinorm[i])
    qpsi[0] = qpsi[1]
    
    # Write arrays
    co = ChunkOutput(fh)
    write_1d(fpol, co)
    write_1d(pres, co)
    write_1d(workk, co)
    write_1d(workk, co)
    write_2d(psi, co)
    write_1d(qpsi, co)

    # Boundary / limiters
    
    if co.counter != 0:
        fh.write("\n")
    fh.write("   0   0\n")

if __name__ == "__main__":
    # Test case
    print("Running geqdsk test case")
    
    from CerfonFreidberg import CerfonFreidberg
    
    eq = CerfonFreidberg()
    eq.initByName("ITER")
    
    with open("iter.geqdsk", "wt") as fh:
        write(eq, fh)
        
    print("Finished")

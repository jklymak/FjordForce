import numpy as np

def o2sat(theta, salt):

    sox1 = 1929.7
    sox2 = -117.46
    sox3 = 3.116
    sox4 =   -0.0306
    oA0=  2.00907
    oA1=  3.22014
    oA2=  4.05010
    oA3=  4.94457
    oA4= -2.56847e-1
    oA5=  3.88767
    oB0= -6.24523e-3
    oB1= -7.37614e-3
    oB2= -1.03410e-2
    oB3= -8.17083e-3
    oC0= -4.88682e-7
    SchmidtNoO2 = sox1 + sox2 * theta + sox3 * theta**2 + sox4 * theta**3

    aTT  = 298.15 -theta
    aTK  = 273.15 +theta
    aTS  = np.log(aTT/aTK)
    aTS2 = aTS*aTS
    aTS3 = aTS2*aTS
    aTS4 = aTS3*aTS
    aTS5 = aTS4*aTS

    oCnew  = (oA0 + oA1*aTS + oA2*aTS2 + oA3*aTS3 + oA4*aTS4 + oA5*aTS5
              +salt*(oB0 + oB1*aTS + oB2*aTS2 + oB3*aTS3) + oC0*salt**2)
    O2sat = np.exp(oCnew)

    # Convert from ml/l to umol/kg  (7 mL/L = 312 umol/kg)
    O2sat = O2sat/22391.6 * 1.e6

    return O2sat
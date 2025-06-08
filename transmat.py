import argparse
import numpy as np
import numpy.linalg as la

def main():
    parser = argparse.ArgumentParser(description='Python script for extracting key matrices from NBO FILE47 and extracting LMO energies')
    parser.add_argument('-f', '--file', help='Path to the FILE47 input file', required=True)
    parser.add_argument('-d', '--dmat', help='Path to LMO coefficients file, e.g. dmat for NLMOs', required=False)
    parser.add_argument('-a', '--print_all', help='Print full list of MO energies. By default, print -50eV < E < 50ev', action='store_true', required=False)
    args = parser.parse_args()
    
    #read lines of FILE47
    with open(args.file, 'r') as f:
        lines = f.readlines()

    #Parse number of basis functions from first line
    keywds = lines[0].split()
    for i,word in enumerate(keywds):
        if word == 'NBAS=':
            n = int(keywds[i+1])

    #Parse FOCK, OVERLAP, DENSITY, LCAOMO
    fock_found = False
    overlap_found = False
    LCAOMO_found = False
    density_found = False
    lf = 0
    F = np.empty(n**2)
    ls = 0
    S = np.empty(n**2)
    lc = 0
    C = np.empty(n**2)
    lp = 0
    P = np.empty(n**2)
    for i,line in enumerate(lines):
        if fock_found & (lf < n**2):
            lll = line.strip().split()
            if len(lll) == 3:
                F[lf] = lll[0]
                F[lf+1] = lll[1]
                F[lf+2] = lll[2]
                lf += 3
            elif len(lll) == 2:
                F[lf] = lll[0]
                F[lf+1] = lll[1]
                lf += 2
            elif len(lll) == 1:
                F[lf] = lll[0]
                lf += 1     
        if overlap_found & (ls < n**2):
            lll = line.strip().split()
            if len(lll) == 3:
                S[ls] = lll[0]
                S[ls+1] = lll[1]
                S[ls+2] = lll[2]
                ls += 3
            elif len(lll) == 2:
                S[ls] = lll[0]
                S[ls+1] = lll[1]
                ls += 2
            elif len(lll) == 1:
                S[ls] = lll[0]
                ls += 1
        if LCAOMO_found & (lc < n**2):
            lll = line.strip().split()
            if len(lll) == 3:
                C[lc] = lll[0]
                C[lc+1] = lll[1]
                C[lc+2] = lll[2]
                lc += 3
            elif len(lll) == 2:
                C[lc] = lll[0]
                C[lc+1] = lll[1]
                lc += 2  
            elif len(lll) == 1:
                C[lc] = lll[0]
                lc += 1
        if density_found & (lp < n**2):
            lll = line.strip().split()
            if len(lll) == 3:
                P[lp] = lll[0]
                P[lp+1] = lll[1]
                P[lp+2] = lll[2]
                lp += 3
            elif len(lll) == 2:
                P[lp] = lll[0]
                P[lp+1] = lll[1]
                lp += 2  
            elif len(lll) == 1:
                P[lp] = lll[0]
                lp += 1
        if '$OVERLAP' in line:
            overlap_found = True
        if '$FOCK' in line:
            fock_found = True
        if '$LCAOMO' in line:
            LCAOMO_found = True
        if '$DENSITY' in line:
            density_found = True
    
    #Reshape 1D arrays to matrices
    F_mat = F.reshape(n,n)
    C_mat = C.reshape(n,n).T
    S_mat = S.reshape(n,n).T
    P_mat = P.reshape(n,n).T

    #Tests
    Trace_PS = np.trace(np.matmul(P_mat,S_mat))

    I = np.matmul(np.matmul(C_mat.T,S_mat),C_mat)
    is_unit = np.allclose(I, np.identity(n))

    with open("transmat.out", 'w') as f:
        f.write("TESTS:\n"
                "Trace(DENSITY*OVERLAP) = " + str(Trace_PS)+"\n"
                "The above should be equal to the number of electrons to machine precision\n\n")
    if is_unit:
        with open("transmat.out", 'a') as f:
            f.write("LCAOMO.T*OVERLAP*LCAOMO is confirmed to be a unit matrix! :)\n\n")

    else:
        with open("transmat.out",'a') as f:
            f.write("WARNING: LCAOMO.T*OVERLAP*LCAOMO is NOT a unit matrix! :(\n\n")


    #Calculate energies
    E_mat_canonical = np.matmul(np.matmul(C_mat.T,F_mat),C_mat)
    E_canonical = E_mat_canonical.diagonal()*27.211407953 #convert to eV

    with open("transmat.out", 'a') as f:
        f.write("Canonical Orbital energies (eV):\n")
        for i,E in enumerate(E_canonical):
            if args.print_all:
                f.write(str(i+1)+": "+str(E)+"\n")
            elif (-50 < E) & (E < 50):
                f.write(str(i+1)+": "+str(E)+"\n")

   
    if args.dmat:
    #read lines of dmat
        with open(args.dmat, 'r') as f:
            lines = f.readlines()
        l = 0
        D = np.empty(n**2)
        for i,line in enumerate(lines):
            if (i>=3) & (l < n**2):
                #print(l)
                lll = line.strip().split()
                if len(lll) == 3:
                    D[l] = lll[0]
                    D[l+1] = lll[1]
                    D[l+2] = lll[2]
                    l+=3
                elif len(lll) == 2:
                    D[l] = lll[0]
                    D[l+1] = lll[1]
                    l += 2
                elif len(lll) == 1:
                    D[l] = lll[0]
                    l += 1
        
        D_mat = D.reshape(n,n).T
        E_mat_LMO = np.matmul(np.matmul(D_mat.T,F_mat),D_mat)
        E_LMO = E_mat_LMO.diagonal()*27.211407953 #convert to eV

        with open("transmat.out", 'a') as f:
            f.write("Requested LMO energies (eV):\n")
            for i,E in enumerate(E_LMO):
                if args.print_all:
                    f.write(str(i+1)+": "+str(E)+"\n")
                elif (-50 < E) & (E < 50):
                    f.write(str(i+1)+": "+str(E)+"\n")
if __name__ == '__main__':
    main()

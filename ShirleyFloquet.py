##############################################################
# This code contains different methods to treat generalized
# (multi-drive) Floquet Hamiltonians. A good description can
# be found in Phys. Rev. A 101, 032116 and in arXiv preprint
# arXiv:2302.12816 (2023).
##############################################################
import numpy as np
import scipy as sp
from scipy.linalg import *
import math as math
import cmath as cmath
from itertools import product

################################################################
##################### Perturbative treatement####################
################################################################
# The method "Perturbative_corrections(K,V,order)" computes the
# Perturbative block diagonalization of the matrix H=K+V, where
# K is the diagonal matrix in the operator basis and V is the
# perturbation. The correction is computed up to order "order"
# and the function then returns the two arrays "G,Hcorr",
# which contain the transformation and the  corrections to the
# Hamiltonian as described in the paper 	arXiv:2302.12816
##############################################################


def commutator(A, B):
    """
    Computes the commutator between A and B
    """
    return np.matmul(A, B)-np.matmul(B, A)

##############################################################


def nested_commutator(A, B):
    """
    A=array of matrices, B matrix computes the nested commutator
    [A1,[A2,...[An,B]]], where A1 is the first matrix in the array A
    """
    tmp = np.array(B,dtype=complex)
    for Ai in reversed(A):
        tmp = commutator(Ai, tmp)
    return tmp

##############################################################


def partition(number):
    '''
    Partition of integers, order matters i.e. (1,2) != (2,1)
    '''
    answer = set()
    answer.add((number, ))
    for x in range(1, number):
        for y in partition(number - x):
            answer.add((x, ) + y)
    return answer

##############################################################


def Dk(kappa, Projectors, X, nontrivial, nindices):
    '''
    Superoperator as defined in IBM paper
    '''
    length = len(kappa)
    result = np.zeros_like(X)
    for n, m in product(range(length), range(length)):
        # Avoids degenerate case
        if kappa[n] != kappa[m]:
            # Saves some matrix multiplications
            if (not n in nontrivial) and (not m in nontrivial):
                nind = nindices[n]  # Get non zero element of projector
                mind = nindices[m]  # Get non zero element of projector
                result[nind, mind] += X[nind, mind]/(kappa[n]-kappa[m])
            else:
                psin = Projectors[n]
                psim = Projectors[m]
                result += np.matmul(np.conj(psin).T, psim)*np.matmul(psin,
                                                            np.matmul(X, np.conj(psim.T)))/(kappa[n]-kappa[m])
    return result

##############################################################


def get_Projectors(X):
    '''
    Gets eigenvalues (kappa), eigenvectors (projectors)
    as well as nontrivial (the indices of eigenvectors 
    which are not canonical basis vectors) and nindices
    (the label of the canonical vector)
    '''
    kappa, eigvec = np.linalg.eigh(X)  # Spectral decomposition of X
    Projectors = np.zeros((len(kappa), 1, X.shape[0]), dtype=complex)
    nontrivial = []
    nindices = np.zeros(len(kappa), dtype=int)
    for j in range(len(kappa)):
        v = eigvec[:, j].reshape(1, -1)
        Projectors[j, :, :] = v
        nindices[j] = np.where(np.abs(v) > 1e-3)[1][0]
        if (np.abs(v)>1e-3).sum()>1: #If the eigenspace is degenerate
            nontrivial.append(j)
    return kappa, Projectors, nontrivial, nindices

##############################################################


def Pk(Projectors, X, nontrivial, nindices):
    '''
    Superoperator as defined in IBM paper
    '''
    length = len(Projectors)
    result = np.array(X,dtype=complex)
    for n in range(length):
        if (n not in nontrivial):
            nind = nindices[n]
            result[nind, nind] = 0
        if (n in nontrivial):
            Psin = np.matmul(np.conj(Projectors[n].T), Projectors[n])
            result = result-np.matmul(np.matmul(Psin, X), Psin)
    return result

##############################################################


def Gi(Projectors, kappa, Htmp, nontrivial, nindices):
    """
    Get transformation at the current order
    """
    return Dk(kappa, Projectors, Pk(Projectors, Htmp, nontrivial, nindices), nontrivial, nindices)

##############################################################


def Hi(Projectors, kappa, K, V, nontrivial, nindices, order):
    G = []  # Transformations
    Hcorrections = []  # Perturbative corrections

    ############
    # zeroth order
    H0 = np.array(K,dtype=complex)
    Hcorrections.append(H0)

    ############
    # First order
    G1 = Dk(kappa, Projectors, Pk(Projectors, V,
            nontrivial, nindices), nontrivial, nindices)
    G.append(G1)
    H1 = commutator(G1, K)+V
    Hcorrections.append(H1)

    ############
    # Higher orders
    for i in range(2, order+1):
        Htemp = np.zeros(K.shape, dtype=complex)  # ith correction
        # Get the possible commutators
        permutedindices = [list(np.array(x)-1) for x in partition(i)]
        for ind in permutedindices:
            if ind != [i-1]:  # Exclude the last one
                # G's we want in the nested commutator
                Gcommutator = (np.array(G))[np.array(ind)]
                Htemp += nested_commutator(Gcommutator, K) / \
                    math.factorial(len(ind))

        # Same thing for the potential part
        permutedindices = [list(np.array(x)-1) for x in partition(i-1)]
        for ind in permutedindices:
            Gcommutator = (np.array(G))[ind]
            Htemp += nested_commutator(Gcommutator, V)/math.factorial(len(ind))

        # Get the next order of the generator
        Gicorr = Gi(Projectors, kappa, Htemp, nontrivial, nindices)
        G.append(Gicorr)
        Hi = Htemp+commutator(Gicorr, K)  # Correction at the next order
        Hcorrections.append(Hi)

    return G, Hcorrections

##############################################################


def Perturbative_corrections(K, V, order):
    """
    Computes the perturbative corrections using
    Schrieffer-Wolff perturbation theory up to order "order"
    on the Hamiltonian H=K+V, where K is the unperturbed Hamiltonian
    in the operator basis and V is the perturbation. IMPORTANT: If there
    is a degeneracy in the diagonal elements of K, the corresponding 
    subspace must be included in K, otherwise the perturbation theory
    breaks down!
    """
    kappa, Projectors, nontrivial, nindices = get_Projectors(K)
    return Hi(Projectors, kappa, K, V, nontrivial, nindices, order)

################################################################
######################## Relevant subspace#######################
################################################################
# The method get_n_NN_Submatrix(M,ind,n) allows to have all relevant
# elements in the Hamiltonian to treat a interaction between the diagonal
# elements with indices ind up to n photon interactions
# (i.e. we get the effective subspace Hamiltonian will all the matrix
# elements that are connected to the elements M[ind,ind] with paths up to order n)
# For a parameter sweepe its more efficient to extract the relevant indices once
# with get_n_NN_Submatrix_index, and then use H[np.ix_(Indices, Indices)]
# Like this one only computes the indices once.
##############################################################
def get_NN(M):
    """
    Getting the direct neighbours for every diagonal element in M
    (i.e. getting the elements which are connected by non zero off diagonal 
    element)
    """
    distoneneighbours = dict([])
    for n in range(len(M)):
        neighboursn = []
        for m in range(len(M)):
            if (M[n, m] != 0) and (n != m):
                neighboursn.append(m)
        distoneneighbours[n] = neighboursn
    return distoneneighbours

##############################################################


def get_nth_NN(firstNN, nminusoneNN):
    """
    Gives nth nearest neighbours if we already have the (n-1) NN
    and a dictonary of first NN.
    """
    nNeighbours = set([])
    # Get the nearest neighbours of the (n-1)th nearest neighbours
    for element in nminusoneNN:
        # Nearest neighbours of the element
        nNeighbours.update(firstNN[element])
    Indices = set(nminusoneNN)
    # Add the new neighbours
    Indices.update(nNeighbours)
    Indices = list(Indices)
    return Indices

##############################################################


def get_n_NN_Submatrix_index(M, ind, n):
    """
    Gives the submatrix indices at order n
    """
    # Getting the direct neighbours for every diagonal element
    if n == 0:
        return M[np.ix_(ind, ind)]

    distoneneighbours = get_NN(M)
    # First order neighbours submatrix
    Indices = []
    for index in ind:
        Indices += distoneneighbours[index]
        Indices += [index]
    Indices = list(set(Indices))
    # Next orders
    for k in range(1, n):
        Indices = get_nth_NN(distoneneighbours, Indices)
    Indices = np.sort(Indices)
    return Indices

##############################################################


def get_n_NN_Submatrix(M, ind, n):
    """Gives the submatrix with all elements connected with paths
    of at most length n starting from the elements M[ind,ind]"""
    Indices = get_n_NN_Submatrix_index(M, ind, n)
    return M[np.ix_(Indices, Indices)]



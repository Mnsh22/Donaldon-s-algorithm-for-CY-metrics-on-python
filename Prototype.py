# Plan for general code:
# Step 1: Generate random points on the quintic,
# Step 2: Construct T-map
import numpy as np
import sympy as sp
import math
from itertools import combinations_with_replacement
from time import perf_counter

# STEP 1: Generate random points on the quintic,


# random points on unit 9-sphere corresponds to sample points in CP^4.
# Hence we generate random points on S9 via following function.
t1 = perf_counter()
def sample_point_C5_on_unit_sphere():

    v = np.random.randn(5) + 1j*np.random.randn(5)
    vec = v / np.linalg.norm(v)
    return vec

v = sample_point_C5_on_unit_sphere()

#print(np.linalg.norm(v)) # just checking if answer indeed gives me 1.

# Now need to project this point into CP^4 by removing the phase, ie: find a way such that both v and e^itheta
# end up mapping to the same coordinate. And one can do it by just multiplying v by the negative of the argument of the
# complex number of any one component. Ie:

def projected_S9_point_onto_coord_of_CP4(v):
    index = np.argmax(abs(v))
    w = np.exp(- 1j * np.angle(v[index])) * v
    return w

# Now we use the two random points on S9 to define a line in CP^4 intersecting X, Ie: following the polynomial equation.
def find_quintic_roots():
    v1 = sample_point_C5_on_unit_sphere()
    v2 = sample_point_C5_on_unit_sphere()
    p = projected_S9_point_onto_coord_of_CP4(v1)
    q = projected_S9_point_onto_coord_of_CP4(v2)

    polynomial = np.zeros(6, dtype=complex)  # Vector each containing a term in the expansion of the polynomial
    # cause np.roots works with vectors only.

    for k in range(6):
        polynomial[k] = sum([math.comb(5, k) * (p[i] ** (5 - k)) * (q[i] ** k) for i in range(5)])

        # Now that coeff is fully built, solve for roots
    roots = np.roots(polynomial[::-1])  # np.roots wants highest degree first

    return roots, p, q

# Now we need to repeat the process n-times n=p_M times.


def generate_quintic_points(p_M_points):
    points = []
    while len(points) < p_M_points:
        roots, p, q = find_quintic_roots()

        for t in roots:  # we gonna loop over the t in the roots function
            z = p + t * q
            Qw = np.sum(z ** 5)
            if np.abs(Qw) > 1e-15:  # threshold
               continue

            points.append(z)

    return np.array(points)  # shape: (n_points, 5)

sample = generate_quintic_points(p_M_points=62250) ### PUT DESIRED VALUE FOR N_p !!!!!!!!!!!!!

Error_sample = generate_quintic_points(p_M_points=40000)






















# STEP 2: Constructing T-map


# Now we first create a fn able to find the |z_J| term of the T-map

# First like suggested by the ML 4 CY paper we need to make sure we pick the right z_J (coming into the T-map eqn)

def coordinates_picking(sample):
    results = []
    for x in sample:  # x is a length-5 complex vector
        i_max = int(np.argmax(np.abs(x)))  # index of largest |x[i]|
        results.append(i_max)
    return results

fixed = coordinates_picking(sample)

#print(fixed)
#print(coordinates_picking(sample))


# Following function will set one of the coordinates of the array to 1 according to the function we defined above.
def coordinates_fixing(sample, fix):

    for i in range(len(sample)):
        b=fix[i] # b is the associated element for the i'th component of the list.
        sample[i] = sample[i]/sample[i][b]
    return sample

coord_fix_fn = coordinates_fixing(sample, fixed) # Need to do it if we wanna define a fn in terms of it.

#print(coordinates_fixing(sample)[1]) # just checking if it's right
#print("Count:", len(coordinates_fixing(sample)), "\n") # again just a check


def extra_trial_picking(coord_fix_fn):
    results = []
    for y in coord_fix_fn:  # y is a length-5 complex vector with one entry == 1
        scores = 5 * (np.abs(y) ** 4)  # |dQ/dz_i| ~ 5*|z_i|^4
        # knock out any entry equal to 1 (your fixed chart coord)
        mask = np.isclose(y, 1)  # handles 1+0j safely
        scores[mask] = -500

        #for i in range(5):
            #if np.abs(y[i]) == 1:
                #scores[i] = -500

        i_max = int(np.argmax(scores))
        results.append(i_max)
    return results

extras = extra_trial_picking(coord_fix_fn)
#print(extras[1])


# That was way more painful that it looked. Anyways now we can build the coordinates fixer.
def extra_coordinates_fixing(coord_fix_fn, extras):

    for i in range(len(extras)):

        b = extras[i]
        not_b_indices = [k for k in range(5) if k != b] # just like before but != means not equal so make a list of
        # numbers that are not the b index
        s = sum([ coord_fix_fn[i][k]**5 for k in not_b_indices ]) # NB: Need to equal this to a number or Int problem
        coord_fix_fn[i][b] = (-s)**(1/5)

    return coord_fix_fn

coordinates_for_every_p_M = extra_coordinates_fixing(coord_fix_fn, extras)

# Seems to work, by checking print(coord_fixing(sample)) and print(extras) the only no that should change from the first
# of the two print should be the component given by print(extras), and it matches, so should be right.
# Now this list contains all the vectors we need and nice cause it contains every coordinate for every point p_M
# according to the correct patches.
# Also nice cause we now can make a list also of all z_J easily just in case for the T-Map,
# just store coord_fix_fn[i][b] into a list and gg.

def z_J_container(extras, coordinates_for_every_p_M):

    list_of_z_J = []

    for i in range(len(extras)):
        b = extras[i]
        z_J_for_T_map = (coordinates_for_every_p_M[i][b]) #gonna list straigtaway the norm to the 8 since is what we need
        # for the T_map

        #print(z_J_for_T_map) #Need to check if it's right and so imma just print each no as it comes out and then
        # compare with the final list

        list_of_z_J.append(z_J_for_T_map)

    return list_of_z_J

container = z_J_container(extras, coordinates_for_every_p_M)



# Now we need to create the det function and we will have all the ingredients to make the T-map. To build such function
# we will need to first create the Jacobian matrix. Then the Kahler form over the CP^4 (FS). And finally the determinant
# of the pullback given by the determinant of two Jacobian matrices acting over the FS kahler form.

def Jacobian_matrix(cfepm, ext, fix, cont):

    Jacobians = []
    derivatives_Jacobians = []
    for y in range(len(sample)):

        g = cfepm[y]
        c = ext[y] # i'th element of extras, ie: index we fix to -(sum...)^1/5
        d = fix[y]  # i'th element of fixed, ie: index we fix to 1
        J = np.zeros(( 3, 5), dtype=complex)  # the main idea is take a zero x zero matrix and based
        dJ = np.zeros((5,3,5), dtype=complex)
        # of the conditions put the component there or not. Note ofc since we starting with a np.zero already
        # no need to put conditions that trivially satisfy a specific component being zero. I made a list
        # for all the no not referenced from the extra and fixed i'th component.
        b = [x for x in range(5) if x != c and x != d]

        for i in range(5): #The way the loop works is that for i between 0 to 5, the
        # computer chooses 1 to start off with. Why? Cause easier to deal w/ conditions w/ fixed column index.
        # Since that's mainly where the conditions lies, depending whether i has some value or not.
        # Now under such for i loop i put a for q loop saying that for each i the computer runs through a loop
        # of all 3 rows
            for q in range(3):
                h = b[q] # I define h to be the first or second or third element of b respectively
                m = [x for x in b if x != h]
                if i == c: # infamous conditions
                    J[q, i] = (-(g[h] ** 4) / (cont[y]) ** 4)
                    for r in range(5):
                        if r == c:
                            dJ[r, q, i] = (4 * ((g[h] ** 4) / ((cont[y]) ** 5)))
                        elif r == h:
                            dJ[r, q, i] = (-4 * (g[h] ** 3) / ((cont[y]) ** 4)) - (4 * (g[h] ** 8) / ((cont[y]) ** 9))
                        elif r in m:
                            dJ[r, q, i] = ((-4 * ((g[h] ** 4) * (g[r] ** 4))) / ((cont[y]) ** 9))

                elif i == h:
                    J[q, i] = 1 # infamous conditions

        Jacobians.append(J)
        derivatives_Jacobians.append(dJ)
    return Jacobians, derivatives_Jacobians

Jack, deriv_Jack = Jacobian_matrix(coordinates_for_every_p_M, extras, fixed, container)


# Now onto defining the metric, for the Kahler form.
def P4_FS_metric_builder(cfepm):
    container_of_metrics = []
    for x in range(len(fixed)):  # remember that g[i][j] means i'th row j'th column element.
        g = np.zeros((5, 5), dtype=complex)
        z = cfepm[x]
        norm = np.vdot(z, z).real
        for i in range(5):
            for j in range(5):
                if i == j:
                    g[i][j] = (1/np.pi) * ( (1/norm) - ((z[i]*np.conj(z[j])) /((norm)**2) ) )

                else:
                    g[i][j] = (-(1/np.pi)) * ( (z[j]*np.conj(z[i])) / ((norm)**2) )

        container_of_metrics.append(g)

    return container_of_metrics

metrics_at_each_p_M = P4_FS_metric_builder(coordinates_for_every_p_M)

#print(metrics_at_each_p_M[1]-metrics_at_each_p_M[998])

def determinant_builder_P4(g, J, extras):

    determinant_pullback_at_each_p_M = []

    for i in range(len(extras)):

        pullback = J[i] @ g[i] @ np.conj(J[i]).T # we note that the metric on FS hermitian and so is it's pullback by the jacobians.

        det_Kahler = np.linalg.det(pullback)

        determinant_pullback_at_each_p_M.append( det_Kahler )

    return determinant_pullback_at_each_p_M

det_Kahler_list = determinant_builder_P4(metrics_at_each_p_M, Jack, extras)



# We first define the monomials of the map.

n = 5  # number of coordinates we are considering
K = 3  # order polynomial we are considering

#def N_k_builder():
N_k = math.comb(n + K - 1, K) #we looking at k less than 5 anyways, (remember that for k>5 need to remove dof)

#print(N_k)

#creating a function that generates the list of monomials combination for a given k (user's choice)



# Now we need to add a code that creates the T-map. We divide the labour in two factors. First one Being N_k/Vol_CY and
# the second factor containing the sum over the matrix constructed by the monomials times the weight
# First factor given by:

def weight_list(cont, detlist, sample):


    weight_list = []

    for i in range(len(sample)):
        w = 1 / ( 25 * ((abs(cont[i])) ** 8) * (detlist[i]) )
        weight_list.append(w)
    return weight_list

w_M_list = weight_list(container, det_Kahler_list, sample)


def first_factor(N_k, wml, sample):

    Vol_CY = ((1/len(sample)) * sum(wml))
    first_fact = (N_k)/(Vol_CY)

    return first_fact

ff = first_factor(N_k, w_M_list, sample)



# We first define the monomials of the map.
def Monomial_list_coord_value(k, cfepm, sample):

    Monomial_list = []

    for i in range(len(sample)):
        z = cfepm[i]
        variables = [ z[0], z[1], z[2], z[3], z[4] ]
        combo = combinations_with_replacement(variables, k)
        gh = list(combo)
        Monomial_list.append(gh) #append each list of N_k combinations

    return Monomial_list

every_single_monomial_combination_tuple = Monomial_list_coord_value(3, coordinates_for_every_p_M, sample) #set the K you want here too
#print(every_single_monomial_combination_tuple[2])



# For the second factor we need to create first a list containing matrix that should contain s_alpha s_betabar at
# each p_M. Then construct the sum function timing by the weight given by 1/(25...determinant_list[i]).
# Starting by the numerator

def section_vector_list(esmct, sample):
    section_vec_list = []
    for i in range(len(sample)):
        x = esmct[i] # list of N_k tuples
        s = np.zeros(N_k, dtype=complex)
        for j in range(N_k):
            s[j] = np.prod(x[j]) # product of tuple values
        section_vec_list.append(s)
    return section_vec_list

svl = section_vector_list(every_single_monomial_combination_tuple, sample)


def T_map_iteration(h, ff, s, wml, n_iter, sample):
    for _ in range(n_iter):
        T_h = ff * (1/len(sample)) * sum([ ( ((np.einsum('m,n->mn', s[i], np.conj(s[i]))) * wml[i]) / (np.einsum('ab,a,b', h, s[i], np.conj(s[i]))) ) for i in range(len(sample)) ])

        #convergence = (1/len(sample)) * sum((np.einsum('ab,a,b', np.linalg.inv(T_h).T, s[i], np.conj(s[i]))/np.einsum('ab,a,b', h, s[i], np.conj(s[i]))) for i in range(len(sample)))
        #convergence1 = np.einsum('ab,ba', T_h.T,h)
        #print(convergence)
        #print('start other convergence', convergence1)
        h = np.linalg.inv(T_h).T #indices

        # Normalise to prevent scaling drift
        #h_inv /= (np.linalg.det(h_inv) ** (1/h_inv.shape[0]))

    return h

h0 = np.eye(N_k, dtype=complex)
#h_trial = np.array([[1,0.0005j,0,1000000j,0],[-0.0005j,1,0,0,0],[0,0,1,0,0],[-1000000j,0,0,1,0],[0,0,0,0,1]], dtype=complex)
h_new = T_map_iteration(h0, ff, svl, w_M_list, 10, sample)
#print(h_new)





























# STEP 3: Calculate the sigma error in the code. It's nice to have a nice recap of the variables we have.
# Np = 1000
# k = 1
# Iteration times = 20

N_t = 60000

def error_vol_CY(N_t, w_M_list):
    # Just like above here pick the desired N_k value over which the T-map should operate.
    Evcy = (1/N_t) * sum(w_M_list)
    return Evcy

EVCY = error_vol_CY(N_t, w_M_list)
#print(EVCY)



# To find Vol_K and hence all the error measures we need the pullback of the Kahler form wrt balanced metric

def Volume_form_builder(cont):

    volume_form_list = []
    for i in range(N_t):
        OmOmbar = 1/ ( 25 * ((abs(cont[i]))**(8)) )
        volume_form_list.append(OmOmbar)
    return volume_form_list

OmOmbar_list = Volume_form_builder(container)



def section_sympy_builder(K): #Using sympy for derivatives (I used it before and this is the best version I have so far)

    z = sp.symbols('z0 z1 z2 z3 z4')

    tuples = list(combinations_with_replacement(range(5), K))

    s_list = [sp.prod(z[idx] for idx in t) for t in tuples]  # N_k elements
    N_k = len(s_list)

    A_sym = sp.Matrix([[sp.diff(s_list[j], z[i]) for j in range(N_k)] for i in range(5)])
    A_aid = sp.lambdify(z, A_sym, 'numpy')  # returns 5xN_k array

    B_sym = sp.Array([[[sp.diff(A_sym[i, j], z[k]) for j in range(N_k)] for i in range(5)] for k in range(5)])
    B_aid = sp.lambdify(z, B_sym, 'numpy')  # returns 5x5xN_k

    return A_aid, B_aid

def derivative_section_matrix_builder(coords, K):

    A_aid, B_aid = section_sympy_builder(K)
    ds_list = []
    dds_list = []
    for coord in coords:  # coord is length-5 complex vector
        A_num = np.asarray(A_aid(*coord), dtype=np.complex128)   # (5, N_k)
        ds_list.append(A_num)
        # If not needed, skip the next two lines entirely:
        B_num = np.asarray(B_aid(*coord), dtype=np.complex128)   # (5, 5, N_k)
        dds_list.append(B_num)
    return ds_list, dds_list

ds_list, dds_list = derivative_section_matrix_builder(coordinates_for_every_p_M, K)




def K_0_builder(h, s):

    k_0_list = []

    for i in range(N_t):
        k_0 = 1 / ( np.einsum('ab, a, b', h, s[i], np.conj(s[i])) )
        k_0_list.append(k_0)
    return k_0_list

K_0_list = K_0_builder(h_new, svl)




def K_i_builder(h, s, ds):   # NBBBBBB found a way of storing s_alphas as a list of vectors really useful to generalise
    k_i_list = []

    for i in range(N_t):
        k_i = np.einsum('ab,ia,b->i', h, ds[i], np.conj(s[i]) )
        k_i_list.append(k_i)

    return k_i_list

K_i_list = K_i_builder(h_new, svl, ds_list)



def Kbar_i_builder(h, s, ds):
    Kbar_i_list = []

    for i in range(N_t):
        kbar_i = np.einsum('ab,a,ib->i', h, s[i], np.conj(ds[i]))
        Kbar_i_list.append(kbar_i)
    return Kbar_i_list

Kbar_i_list = Kbar_i_builder(h_new, svl, ds_list)



def K_ijbar_builder(h, ds):
    K_ijbar_list = []

    for i in range(N_t):
        K_ijbar = np.einsum('ab,ia,jb->ij', h, ds[i], np.conj(ds[i]) )
        K_ijbar_list.append(K_ijbar)
    return K_ijbar_list

K_ijbar_list = K_ijbar_builder(h_new, ds_list)



def RF_metric_on_P4(K, K_0, K_ijbar, K_i, Kbar_i):
    rfmop4 = []

    for i in range(N_t):
        g = (1/(np.pi * K)) * ( (K_0[i] * K_ijbar[i]) - ((K_0[i] ** 2) * np.einsum('m,n->mn',K_i[i], Kbar_i[i]) ) )
        rfmop4.append(g)

    return rfmop4

P4_Ricci_flat_metrics_list = RF_metric_on_P4(K, K_0_list, K_ijbar_list, K_i_list, Kbar_i_list)



def CY_RF_metric_builder(J, g_tilda):
    rfmoCY = []

    for i in range(N_t):
        g = np.einsum('ai,ij,bj->ab', J[i], g_tilda[i], np.conj(J[i]))
        rfmoCY.append(g)
    return rfmoCY

CY_Ricci_flat_metrics_list = CY_RF_metric_builder(Jack, P4_Ricci_flat_metrics_list)

print('RF CY metric', CY_Ricci_flat_metrics_list[0])



def determinants_CY_metric(g, N_t):
    dCYm = []

    for i in range(N_t):
        det = np.linalg.det(g[i])
        dCYm.append(det)

    return dCYm

det_CY_metric = determinants_CY_metric(CY_Ricci_flat_metrics_list, N_t)





def Error_Vol_K_builder(det, OmOmbar, w_M, N_t):
    Vol_K = (1/N_t) * np.sum([ (( det[i] / OmOmbar[i] ) * w_M[i]) for i in range(N_t)])
    return Vol_K

EVK = Error_Vol_K_builder(det_CY_metric, OmOmbar_list, w_M_list, N_t)

print('det', det_CY_metric[0])
print('OmOm', OmOmbar_list[0])
print('volumes ratio', EVK/EVCY)



def Error_sigma(EVK, EVCY, det, OmOmbar):
    sigma = (1/(N_t * EVCY)) * (np.sum([ (abs(1 - ( (det[i]/EVK) / (OmOmbar[i]/EVCY) )) * w_M_list[i]) for i in range(N_t) ]))
    return sigma

sigma = Error_sigma(EVK, EVCY, det_CY_metric, OmOmbar_list)

print(sigma)






def K_ij_builder(h, dds, s):

    K_ij_list = []

    for i in range(N_t):
        K_ij = np.einsum('ab,ija,b->ij', h, dds[i], np.conj(s[i]))
        K_ij_list.append(K_ij)
    return K_ij_list

K_ij_list = K_ij_builder(h_new, dds_list, svl)



def K_ijkbar_builder(h, dds, ds):

    K_ijkbar_list = []

    for i in range(N_t):
        K_ijkbar = np.einsum('ab,ija,kb->ijk', h, dds[i], np.conj(ds[i]))
        K_ijkbar_list.append(K_ijkbar)
    return K_ijkbar_list

K_ijkbar_list = K_ijkbar_builder(h_new, dds_list, ds_list)



def K_ikjbarlbar_builder(h, dds, N_t):
    K_ikjbarlbar_list = []

    for i in range(N_t):
        K_ikjbarlbar = np.einsum('ab,ika,jlb->ikjl', h, dds[i], np.conj(dds[i]))
        K_ikjbarlbar_list.append(K_ikjbarlbar)
    return K_ikjbarlbar_list

K_ikjbarlbar_list =  K_ikjbarlbar_builder(h_new, dds_list, N_t)



def Kbar_jlkbar_builder(h, ds, dds):
    Kbar_jlkbar_list = []

    for i in range(N_t):
        Kbar_jlkbar = np.einsum('ab,ia,jlb->ijl', h, ds[i], np.conj(dds[i]))
        Kbar_jlkbar_list.append(Kbar_jlkbar)
    return Kbar_jlkbar_list

Kbar_jlkbar_list = Kbar_jlkbar_builder(h_new, ds_list, dds_list)





def derivative_metric_p4(K_0, K_i, K_ijbar, Kbar_i, K_ij, K_ijkbar):
    dmoP4 = []

    for i in range(N_t):
        dg = (1/(np.pi * K)) * ( - ((K_0[i])**2) * ( np.einsum('i,kl->ikl', K_i[i], K_ijbar[i])
                                                     + np.einsum('k,il->ikl', K_i[i], K_ijbar[i])
                                                     + np.einsum('l,ik->ikl', Kbar_i[i], K_ij[i]) )
                                + K_0[i] * K_ijkbar[i]
                                + 2 * ((K_0[i])**3) * ( np.einsum('i,k,l->ikl', K_i[i], K_i[i], Kbar_i[i]) ) )
        dmoP4.append(dg)

    return dmoP4

P4_dg = derivative_metric_p4(K_0_list, K_i_list, K_ijbar_list, Kbar_i_list, K_ij_list, K_ijkbar_list)




def double_deriv_metric_p4(K_0, K_ikjbarlbar, K_ijbar, K_ij, Kbar_i, K_ijkbar, K_i, Kbar_jlkbar):
    ddmoP4 = []

    for i in range(N_t):
        ddg = (1/(np.pi * K)) * ((K_0[i] * K_ikjbarlbar[i])
                                 - ((K_0[i])**2) * (np.einsum('ij,kl->ijkl', K_ijbar[i], K_ijbar[i])
                                                    +np.einsum('ik,jl->ijkl', K_ij[i],np.conj(K_ij[i]))
                                                    +np.einsum('kj,il->ijkl', K_ijbar[i], K_ijbar[i]))
                                 - ((K_0[i])**2) * (np.einsum('j,ikl->ijkl', Kbar_i[i], K_ijkbar[i])
                                                    + np.einsum('l,ikj->ijkl', Kbar_i[i], K_ijkbar[i])
                                                    + np.einsum('i,kjl->ijkl', K_i[i], Kbar_jlkbar[i])
                                                    + np.einsum('k,ijl->ijkl', K_i[i], Kbar_jlkbar[i]))
                                 + 2 * ((K_0[i])**3) * (np.einsum('i,j,kl->ijkl', K_i[i], Kbar_i[i], K_ijbar[i])
                                                        +np.einsum('ij,k,l->ijkl', K_ijbar[i], K_i[i], Kbar_i[i])
                                                        +np.einsum('j,k,il->ijkl', Kbar_i[i], K_i[i], K_ijbar[i])
                                                        +np.einsum('i,kj,l->ijkl', K_i[i],K_ijbar[i], Kbar_i[i])
                                                        +np.einsum('i,k,jl->ijkl', K_i[i], K_i[i], np.conj(K_ij[i]))
                                                        +np.einsum('j,ik,l->ijkl', Kbar_i[i], K_ij[i], Kbar_i[i]))
                                 - 6 * ((K_0[i])**4) * np.einsum('i,j,k,l->ijkl', K_i[i], Kbar_i[i], K_i[i], Kbar_i[i]))

        ddmoP4.append(ddg)

    return ddmoP4

P4_ddg = double_deriv_metric_p4(K_0_list, K_ikjbarlbar_list, K_ijbar_list, K_ij_list, Kbar_i_list, K_ijkbar_list, K_i_list, Kbar_jlkbar_list)



def CY_RF_derivative_metric_builder(dJ, g, J, P4dg):
    rfdmonCY = []

    for i in range(N_t):
        dg = np.einsum('iak,km,bm->iab', dJ[i], g[i], np.conj(J[i])) + np.einsum('ak,ikm,bm->iab', J[i], P4dg[i], np.conj(J[i]))
        rfdmonCY.append(dg)
    return rfdmonCY
CYRF_dg = CY_RF_derivative_metric_builder(deriv_Jack, P4_Ricci_flat_metrics_list, Jack, P4_dg)



def CY_RF_conj_derivative_metric_builder(J, g, dg, dJ):
    rfcdmonCY = []

    for i in range(N_t):
        cdg = (np.einsum('ak,ikm,bm->iab', J[i], np.conj(dg[i]), np.conj(J[i]))
               + np.einsum('ak,km,ibm->iab', J[i], g[i], np.conj(dJ[i])))
        rfcdmonCY.append(cdg)
    return rfcdmonCY

CYRF_cdg = CY_RF_conj_derivative_metric_builder(Jack, P4_Ricci_flat_metrics_list, P4_dg, deriv_Jack)





def CY_RF_double_derivative_metric_builder(dJ, dg, J, g, ddg):
    rfddmonCY = []

    for i in range(N_t):
        ddg_CY = ( np.einsum('iak,jkm,bm->ijab', dJ[i], np.conj(dg[i]), np.conj(J[i]))
               + np.einsum('ak,ijkm,bm->ijab', J[i], ddg[i], np.conj(J[i]))
                + np.einsum('iak,km,jbm->ijab', dJ[i], g[i], np.conj(dJ[i]))
                  + np.einsum('ak,ikm,jbm->ijab', J[i], dg[i], np.conj(dJ[i])) )
        rfddmonCY.append(ddg_CY)
    return rfddmonCY

CYRF_ddg = CY_RF_double_derivative_metric_builder(deriv_Jack, P4_dg, Jack, P4_Ricci_flat_metrics_list, P4_ddg)



def Ricci_tensor_P4_wrt_CY_metrics(g, dg, cdg, ddg):
    R_ij_wrt_CYg_list =[]

    for i in range(N_t):
        R_ijbar = ( np.einsum('nm,ims,sp,jpn->ij', np.linalg.inv(g[i]), dg[i], np.linalg.inv(g[i]), cdg[i])
                     + np.einsum('nm,ijmn->ij', np.linalg.inv(g[i]), ddg[i]))
        R_ij_wrt_CYg_list.append(R_ijbar)
    return R_ij_wrt_CYg_list

R_ij_wrt_CYg = Ricci_tensor_P4_wrt_CY_metrics(CY_Ricci_flat_metrics_list, CYRF_dg, CYRF_cdg, CYRF_ddg)



def CY_Ricci_tensor(J, R_ij):
    R_abbar_list = []

    for i in range(N_t):
        R_abbar = np.einsum('ai,ij,bj->ab', J[i], R_ij[i], np.conj(J[i]))
        R_abbar_list.append(R_abbar)
    return R_abbar_list

CY_R_abbar = CY_Ricci_tensor(Jack, R_ij_wrt_CYg)

print('Ricci tensor on CY',CY_R_abbar[0])



def CY_Ricci_scalar(g, R_abbar):
    R_list = []

    for i in range(N_t):
        R = np.einsum('mn,mn', np.linalg.inv(g[i]).T, R_abbar[i])
        R_list.append(R)
    return R_list

CY_R = CY_Ricci_scalar(CY_Ricci_flat_metrics_list, CY_R_abbar)

print('Ricci scalar on CY',CY_R[0])



def Ricci_error(detg, OmOm, R, w_M):

    R = ( ((EVK) ** (1/3)) / (N_t * EVCY) ) * ( np.sum([ ( (detg[i] / OmOm[i]) * abs(R[i]) * w_M[i] ) for i in range(N_t)]) )

    return R

R_error = Ricci_error(det_CY_metric, OmOmbar_list, CY_R, w_M_list)

print(R_error)


def Manush_error():

    M = ((EVCY/EVK) ** (2/3)) * ((1/EVCY) ** (1/3)) * R_error

    return M

Manush = Manush_error()

print(Manush)


























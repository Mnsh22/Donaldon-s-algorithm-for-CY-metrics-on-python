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
    v = v / np.linalg.norm(v)
    return v

v = sample_point_C5_on_unit_sphere()

#print(np.linalg.norm(v)) # just checking if answer indeed gives me 1.

# Now need to project this point into CP^4 by removing the phase, ie: find a way such that both v and e^itheta
# end up mapping to the same coordinate. And one can do it by just multiplying v by the negative of the argument of the
# complex number of any one component. Ie:

def projected_S9_point_onto_coord_of_CP4(v):
    w = np.exp(- 1j * np.angle(v[4]))*v
    return w

# Now we use the two random points on S9 to define a line in CP^4 intersecting X, Ie: following the polynomial equation.
def find_quintic_roots():
    v1 = sample_point_C5_on_unit_sphere()
    v2 = sample_point_C5_on_unit_sphere()
    p = projected_S9_point_onto_coord_of_CP4(v1)
    q = projected_S9_point_onto_coord_of_CP4(v2)


    polynomial = np.zeros(6, dtype=complex)  # Vector each containing a term in the expansion of the polynomial
    # cause np.roots works with vectors only.

    for i in range(5):
        for k in range(6):
            polynomial[k] = polynomial[k] + math.comb(5, k) * p[i] ** (5 - k) * q[i] ** k
            #This is just storing (p+tq)^5 coefficient on

        # Now that coeff is fully built, solve for roots
    roots = np.roots(polynomial[::-1])  # np.roots wants highest degree first

    return roots, p, q

# Run the function
roots, p, q = find_quintic_roots()

# Evaluate points and check if they satisfy the quintic
for i, t in enumerate(roots):
    z = p + t * q
    z = z/np.linalg.norm(z)  # normalize to stay in CP^4
    Qz = np.sum(z ** 5)
    #print(f"Root {i + 1}: t = {t}")
    #print(f"Check: Q(z) = {Qz:.1e}  |  Abs = {np.abs(Qz):.1e}") # last line was added from help of GPT for an error I
    # didn't understand. However it worked. So if it isn't broken, don't fix it :).

# Now we need to repeat the process n-times n=p_M times.

def generate_quintic_points(p_M_points):
    points = []
    while len(points) < p_M_points:
        roots, p, q = find_quintic_roots()

        for t in roots:  # we gonna loop over the t in the roots function
            z = p + t * q  # specifically over this equation, so the loop will generate p and q values and sub them in here
            # for every t value.
            z /= np.linalg.norm(z)  #normalise it cause we on a sphere

            # Optional: check that Q(z) ≈ 0, condition to break the code if not accurate enough
            Qz = np.sum(z ** 5)
            if np.abs(Qz) > 1e-15:  # threshold
               continue

            points.append(z)

        if len(points) >= p_M_points: # limit of the while loop.
            break

    return np.array(points)  # shape: (n_points, 5)

sample = generate_quintic_points(50010) ### PUT DESIRED VALUE FOR N_p !!!!!!!!!!!!!


#print("Shape:", sample.shape)  # (1000, 5)
#print("Sample point:", sample[0]) #I am just checking if it's working alright

# samples therefore contains all the p_M generated points. Hence we can finally start constructing the T-Map




















# STEP 2: Constructing T-map


# Now we first create a fn able to find the |z_J| term of the T-map

# First like suggested by the ML 4 CY paper we need to make sure we pick the right z_J (coming into the T-map eqn)
def coordinates_picking(sample):

    Results = [] # create a list collecting the output as you loop over the sample by [] and name it results.

   # The following loop takes element of sample. Takes the zeroth element of coordinates and calculates the norm.
    # Then if the next norm, built out of the next element of the coord. Is bigger then it replace it, if it's not it
    # keeps it. And we store the such element.
    for x in sample:
        starting_i = 0
        starting_norm = abs(x[0]) # we store the value of the first one.
        for i in range(1, 5):
            current_norm = abs(x[i]) # we getting the new norm values
            if current_norm > starting_norm: # Replace the new values from the old one and since we inside the range 1
                # to 5 it will loop over everything so it will automatically update.
                starting_norm = current_norm
                starting_i = i

        Results.append(starting_i)

    return Results

fixed = coordinates_picking(sample)

#print(coordinates_picking(sample)) # print function to see we indeed get a 1000 values running 0 to 4. (the indices with
# the highest norm.)
#print("Count:", len(coordinates_picking(sample)), "\n") # checking sake that we have 1000 values




# Following function will set one of the coordinates of the array to 1 according to the function we defined above.
def coordinates_fixing(sample):
    fix = fixed
    for i in range(len(sample)): # useful trick to assign no 1 to length in order for the list
        b=fix[i] # b is the associated element for the i'th component of the list.
        sample[i] = sample[i]/sample[i][b]  # sample[i] is the i'th element (= array) of the sample list since they match lengths and order
        # anyways and so sample[i][b] is the b'th element in the array.

    return sample #really important cause this tells us that such change is stored in the sample, so when calling such
    # function the sample list in this fn has already the 1's substituted into it.

coord_fix_fn = coordinates_fixing(sample) # Need to do it if we wanna define a fn in terms of it.

#print(coordinates_fixing(sample)) # just checking if it's right
#print("Count:", len(coordinates_fixing(sample)), "\n") # again just a check




# Just like before we wanna do the same but now for |dQ/dz| as mentioned by paper, so pick coord w/ highest norm and
# then fix such coordinate.
def extra_coordinate_picking(coord_fix_fn):

    Results_extra = []

    for y in coord_fix_fn:

        starting_extra = 0
        starting_extra_norm = 5 * (y[0]) ** 4 # for every element in y we calculate the 0'th term to be the eqn on left

        # the tricky bit of this fn is actually that we wanna take the sample with 1 already in it. And not use such one
        # in out comparison. Cause otherwise since many terms are less than zero, 5 * (y[0]) ** 4 will be less than 1
        # and 1 will always win. Hence I am making sure it will lose, choosing a big -ve no.

        if y[0] == 1:
            starting_extra_norm = -500
            '''try removing index check gpt how and instead of looping over just remove and confront'''

        # Now i can start the looping process checking for each i BUT remember again make sure to filter the 1 out.

        for i in range(1, 5):
            current_extra_norm = 5 * (y[i] ** 4)

            if y[i] == 1: # filtering the 1's out.
                current_extra_norm = -500

            if current_extra_norm > starting_extra_norm: # same code as above just check and replace
                starting_extra_norm = current_extra_norm
                starting_extra = i

        Results_extra.append(starting_extra)

    return Results_extra

extras = extra_coordinate_picking(coord_fix_fn)

# wanna see the list just for eye check that nothing suspicious is popping up.
# This section was just built to check to see if the code was working, the idea is really simple (since my coding
# experience is bad) but yh idea is that I take the difference between the two lists, no zero should appear, since it
# would mean that such 1 was taken (I know i normalised so the component's module should always be less than 1) which
# corresponded to the fixed list value. Can do since they have same order,

#print(extras)
#print("Count:", len(extras), "\n") # checking sake that we have 1000 values

#diff = [a-b for a,b in zip(fixed,extras)]
#print(diff)
#print("Count:", len(diff), "\n")

#for x in diff:
    #if x == 0:
        #print("Dyakolini")
#else:
    #print('letsgoski')




# That was way more painful that it looked. Anyways now we can build the coordinates fixer.
def extra_coordinates_fixing(coord_fix_fn, extras):

    for i in range(len(extras)):

        b = extras[i]
        not_b_indices = [k for k in range(5) if k != b] # just like before but != means not equal so make a list of
        # numbers that are not the b index
        s = sum( coord_fix_fn[i][k]**5 for k in not_b_indices ) # NB: Need to equal this to a number or Int problem
        coord_fix_fn[i][b] = (-s)**(1/5)

    return coord_fix_fn

#print(extra_coordinates_fixing(coord_fix_fn))

coordinates_for_every_p_M = extra_coordinates_fixing(coord_fix_fn, extras)
#print(coordinates_for_every_p_M[1], coordinates_for_every_p_M[998])

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

container = z_J_container(extras, coord_fix_fn)

#print(z_J_container()) # compare with each no as it comes out if it's right or not. Indeed it worked :)




# Now we need to create the det function and we will have all the ingredients to make the T-map. To build such function
# we will need to first create the Jacobian matrix. Then the Kahler form over the CP^4 (FS). And finally the determinant
# of the pullback given by the determinant of two Jacobian matrices acting over the FS kahler form.

def Jacobian_matrix(extras,container):

    Jacobians = []
    for y in range(len(extras)):

        g = coordinates_for_every_p_M[y]
        c = extras[y] # i'th element of extras, ie: index we fix to -(sum...)^1/5
        d = fixed[y]  # i'th element of fixed, ie: index we fix to 1
        J = np.zeros(( 3, 5), dtype=complex)  # the main idea is take a zero x zero matrix and based
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
                h = b[q] # I define h to be the first or second or third element of b respectively w/ first,
                # second or third loop of q. Reason being is cause if i = h then that position in which h is
                # which is given by q by definition, ie: for the h'th column and q'th row then such component
                # is equal to 1.
                if i == c: # infamous conditions
                    J[q][i] = (-(g[h] ** 4) / (container[y]) ** 4)
                elif i == h:
                    J[q][i]=1 # infamous conditions

        Jacobians.append(J)

    return Jacobians

Jack = Jacobian_matrix(extras,container)
#print("First Jacobian matrix:\n", Jack[0])
#print("Second Jacobian matrix:\n", Jack[1])
#print("Shape of Jack[0]:", Jack[0].shape)




# Now onto defining the metric, for the Kahler form.
def metric_builder(fixed):
    container_of_metrics = []
    for x in range(len(fixed)):  # remember that g[i][j] means i'th row j'th column element.
        g = np.zeros((5, 5), dtype=complex)
        z = coordinates_for_every_p_M[x]
        for i in range(5):
            for j in range(5):
                if i == j:
                    g[i][j] = (1/np.pi) * ((1/(z[0]*np.conj(z[0])+z[1]*np.conj(z[1])+z[2]*np.conj(z[2])
                                              +z[3]*np.conj(z[3])+z[4]*np.conj(z[4])))
                                           - ((z[i]*np.conj(z[j]))/(z[0]*np.conj(z[0])+z[1]*np.conj(z[1])+z[2]*np.conj(z[2])
                                              +z[3]*np.conj(z[3])+z[4]*np.conj(z[4]))**2))

                else:
                    g[i][j] = (-(1/np.pi)) * ((z[j]*np.conj(z[i]))/((z[0]*np.conj(z[0])+z[1]*np.conj(z[1])
                                                                      +z[2]*np.conj(z[2])+z[3]*np.conj(z[3])
                                                                      +z[4]*np.conj(z[4])))**2)

        container_of_metrics.append(g)

    return container_of_metrics

metrics_at_each_p_M = metric_builder(fixed)

#print(metrics_at_each_p_M[1], metrics_at_each_p_M[998])




# Now we can code the determinant of the pullback of the Kahler form. det(i J^T g J)

def determinant_builder(extras):

    determinant_pullback_at_each_p_M = []

    for i in range(len(extras)):
        g = metrics_at_each_p_M[i]
        J = Jack[i]
        Jt = np.matrix_transpose(J)
        Jtbar = np.conjugate(Jt)

        pullback = np.einsum('ia,ab,bj -> ij', J,g,Jtbar)
        #print(pullback.shape)

        det = np.linalg.det(pullback)

        determinant_pullback_at_each_p_M.append( det )

    return determinant_pullback_at_each_p_M

determinant_list = determinant_builder(extras)

#print(determinant_builder())
#print("Count", len(determinant_builder()) ,'\n')


# For simplicity for the T-map is better if we define the values for the factor of the sum over N_p points of
# 1/25|z_J|^8 det(f*w)


# We first define the monomials of the map.

n = 5  # number of coordinates we are considering
k = 5  # order polynomial we are considering

#def N_k_builder():
N_k = math.comb(n + k - 1, k) #we looking at k less than 5 anyways, remember that for k>5 need to remove dof

#print(N_k)

#creating a function that generates the list of monomials combination for a given k (user's choice)

def Monomial_list_coord_value():

    Monomial_list = []
    for i in range(len(sample)):
        x = coordinates_for_every_p_M[i]
        variables = [x[0],x[1],x[2],x[3],x[4]]
        combo = combinations_with_replacement(variables, k)
        gh = list(combo)
        Monomial_list.append(gh)

    return Monomial_list

every_single_monomial_combination_tuple = Monomial_list_coord_value()
#print(every_single_monomial_combination_tuple[2])






# Now we need to add a code that creates the T-map. We divide the labour in two factors. First one Being N_k/Vol_CY and
# the second factor containing the sum over the matrix constructed by the monomials times the weight
# First factor given by:

def weight_list():
    weight_list = []
    for i in range(len(sample)):
        w = 1 / (25 * (abs(container[i]) ** 8) * (determinant_list[i]))
        weight_list.append(w)
    return weight_list

w_M_list = weight_list()

def first_factor(N_k):

    Vol_CY = (1/len(sample)) * sum(w_M_list[i] for i in range(len(sample)))
    first_fact = (N_k)/(Vol_CY)

    return first_fact

ff = first_factor(N_k)






# For the second factor we need to create first a list containing matrix that should contain s_alpha s_betabar at
# each p_M. Then construct the sum function timing by the weight given by 1/(25...determinant_list[i]).
# Starting by the numerator

def section_vector_list():

    section_vec_list = []

    for i in range(len(sample)):
        aid_list = []
        x = every_single_monomial_combination_tuple[i]
        s = np.zeros(N_k, dtype=complex)

        for j in range(N_k):
            y = list(x[j])
            prod = y[0]*y[1]*y[2]*y[3]*y[4]
            # cause y a tuple and not int
            aid_list.append(prod)


        for r in range(N_k):
            s[r] = aid_list[r] # s_alpha vector

        section_vec_list.append(s)

    return section_vec_list

svl = section_vector_list()

def section_matrix_generator():

    section_matrix_list = []


    for f in range(len(sample)):
        B = np.zeros((N_k,N_k), dtype=complex)
        s = svl[f]
        A = np.einsum('i,j->ij', s,np.matrix.conj(s))
        C = A + B
        for i in range(N_k):
            C[i][i] = (A[i][i]+np.conj(A[i][i]))/2
        section_matrix_list.append(C)

    return section_matrix_list

sfm = section_matrix_generator()
#print(len(sfm))
#print(sfm[0])
#print(sfm[0].shape)




def second_factor_weight_and_num():

    selection_iteration_term = []

    for i in range(len(sample)):
        term = w_M_list[i] * sfm[i]
        selection_iteration_term.append(term)

    return selection_iteration_term

sfwad = second_factor_weight_and_num()
#print(sfwad[1]-sfwad[0])
#print(second_factor_weight_and_denom())




# Now finally we try constructing the T-map by putting these factors together and summing them.

def sum_over_h_second_factor():

    h = np.eye(N_k, dtype=complex)
    #h = np.array([[1, 1j, -1, -1j, 0],
                        #[-1j, 1, 1j, -1, 0],
                       # [-1, -1j, 1, 1j, 0],
                        #[1j, -1, -1j, 1, 0],
                       # [0, 0, 0, 0, 1]], dtype=complex)


    factor = sum((( sfm[i] * w_M_list[i]) / (np.einsum("mn,mn",np.linalg.inv(h),sfm[i])) ) for i in range(len(sample)) )
    #print(factor)
    return factor

sohsf = sum_over_h_second_factor()

#print(sohsf)
#print(sohsf.shape)


def T_map_function():

    T_map = ff * sohsf

    for _ in range(10): # Input here how many times to iterate the T_map
        T_map =  ff * sum( (sfm[i] * w_M_list[i])/ (np.einsum("mn,mn",np.transpose(np.linalg.inv(T_map)),sfm[i])) for i in range(len(sample)) )
    return T_map

T_map = T_map_function()

#print(T_map)
#print(T_map.shape)

h_new = np.transpose(np.linalg.inv(T_map))

print(h_new)















# STEP 3: Calculate the sigma error in the code. It's nice to have a nice recap of the variables we have.
# Np = 1000
# k = 1
# Iteration times = 20

N_t = 50000

def error_vol_CY(N_t):
    # Just like above here pick the desired N_k value over which the T-map should operate.
    Evcy = (1/N_t) * sum(w_M_list[i] for i in range(N_t))
    return Evcy

EVCY = error_vol_CY(N_t)
#print(EVCY)

def Volume_form_builder():

    volume_form_list = []
    for i in range(N_t):
        OmOmbar = (5 ** (-2)) * ((abs(container[i]))**(-8))
        volume_form_list.append(OmOmbar)
    return volume_form_list

OmOmbar_list = Volume_form_builder()


# NOTE THIS IS JUST FOR k = 1. NEED TO FIND A WAY TO GENERALISE ON MATEMATICA FOR AT LEAST k=2 TOO.
# My idea is for now to build a code that would be good enough so that people can just change what can be calculated
# analytically. The idea is that to compute the metric g, one needs information about derivatives of polynomial.
# For k less 2 one can do them by hand, but more is jarring so still need to find a way.

def derivative_section_matrix_builder():
    z0 = sp.Symbol('z0')
    z1 = sp.Symbol('z1')
    z2 = sp.Symbol('z2')
    z3 = sp.Symbol('z3')
    z4 = sp.Symbol('z4')
    variables = [z0, z1, z2, z3, z4]

    # all degree-5 monomials (126 of them)
    gh = list(combinations_with_replacement(variables, 5))

    # products for each 5-tuple
    some_list = []
    for j in range(len(gh)):
        y = gh[j]
        prod = y[0] * y[1] * y[2] * y[3] * y[4] #each s_alpha element
        some_list.append(prod)

    rows = len(some_list)  # 126
    cols = len(variables)  # 5

    # derivative matrix A (symbolic)
    A = np.empty((rows, cols), dtype=object)
    for j in range(rows):
        for i in range(cols):
            A[j, i] = sp.diff(some_list[j], variables[i])



    # make a list of d_i_s_alpha for each point p_M
    ds_list = []
    for x in range(len(sample)):
        coord = coordinates_for_every_p_M[x]
        A_num = np.empty((rows, cols), dtype=float)
        for j in range(rows):
            for i in range(cols):
                expr = A[j, i]
                value = (
                    expr
                    .subs(z0, coord[0])
                    .subs(z1, coord[1])
                    .subs(z2, coord[2])
                    .subs(z3, coord[3])
                    .subs(z4, coord[4])
                )
                A_num[j, i] = float(value)
        ds_list.append(A_num)
    return ds_list

ds_list = derivative_section_matrix_builder()

print(ds_list[126])











I = np.eye(5, dtype= complex) # compute a matrix for d_i s_alpha and in this case since d_jbar s_betabar
# is the same then dw

def K_ij_builder():
    K_ij_list = []

    for i in range(N_t):
        k_ijbar = np.einsum('ia,ab,bj -> ij', I,h_new,I)
        K_ij_list.append(k_ijbar)
    return K_ij_list

K_ijbar_list = K_ij_builder()

def K_0_builder():
    k_0_list = []
    for i in range(N_t):
        k_0 = 1 / ( np.einsum("mn,mn", h_new, sfm[i]) )
        k_0_list.append(k_0)
    return k_0_list

K_0_list = K_0_builder()


def K_i_builder():   # NBBBBBB found a way of storing s_alphas as a list of vectors really useful to generalise
    k_i_list = []
    helping_list = []
    for i in range(N_t):
        aid_list = []
        x = every_single_monomial_combination_tuple[i]
        A = np.zeros(N_k, dtype=complex)

        for j in range(N_k):
            y = list(x[j])
            prod = y[0] #*y[1] if k=2. *y[1]*y[2] if k=3 and so on. Note that this is even for k=1
            # cause y a tuple and not int
            aid_list.append(prod)

        for r in range(N_k): # this is supposed to be s_alpha vectors
            A[r] = aid_list[r]

        helping_list.append(A)

    for f in range(N_t):
        k_i = np.einsum('ij,jk,k->i', h_new, I, helping_list[f])

        k_i_list.append(k_i)

    return k_i_list

K_i_list = K_i_builder()

#print(K_i_list[3].shape)



def metric_list():

    metric_list = []

    for i in range(N_t): # from notes ML 4CY
        g = (1/(k * np.pi))* ( (K_0_list[i] * K_ijbar_list[i]) - ( ((K_0_list[i])*(K_0_list[i])) * np.outer(K_i_list[i], np.matrix.conj(K_i_list[i])) ) )

        metric_list.append(g)

    return metric_list

metroboomin = metric_list()

#print(metroboomin[2]-metroboomin[499])

def actual_determinant_builder():

    determinant_pullback_list = []

    for i in range(N_t):
        g = metroboomin[i]
        J = Jack[i]
        Jt = np.matrix_transpose(J)
        Jtbar = np.conjugate(Jt)

        pullback = np.einsum('ia,ab,bj -> ij', J,g,Jtbar)

        det = np.linalg.det(pullback)

        determinant_pullback_list.append(det)

    return determinant_pullback_list

det_metroboomin_list = actual_determinant_builder()

def error_Vol_K():

    evk = (1/N_t) * sum((det_metroboomin_list[i] / (OmOmbar_list[i])) * w_M_list[i] for i in range(N_t))
    return evk

EVK = error_Vol_K()


def sigma_builder():

    sigma = (1/(N_t*EVCY)) * sum((abs(1-((det_metroboomin_list[i]/EVK)/((OmOmbar_list[i])/EVCY)))) * w_M_list[i] for i in range(N_t))
    return sigma

sigma = sigma_builder()

print(sigma)

print(abs(EVK/EVCY))

t2 = perf_counter()

print("Elapsed time:", t2, t1)

print("Elapsed time during the whole program in seconds:",t2-t1)















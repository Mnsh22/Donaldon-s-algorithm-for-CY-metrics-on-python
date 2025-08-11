# Plan for general code:
# Step 1: Generate random points on the quintic,
# Step 2: Construct T-map
import numpy as np
import math
from itertools import combinations_with_replacement



# STEP 1: Generate random points on the quintic,

# random points on unit 9-sphere corresponds to sample points in CP^4.
# Hence we generate random points on S9 via following function.

def sample_point_C5_on_unit_sphere():

    v = np.random.randn(5) + 1j*np.random.randn(5)
    v = v / np.linalg.norm(v)
    return v

v = sample_point_C5_on_unit_sphere()

print(np.linalg.norm(v)) # just checking if answer indeed gives me 1.

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
def generate_quintic_points():
    points = []
    p_M_points = 5000 ### PUT DESIRED VALUE FOR N_p !!!!!!!!!!!!!
    while len(points) < p_M_points:
        roots, p, q = find_quintic_roots()

        for t in roots:  # we gonna loop over the t in the roots function
            z = p + t * q  # specifically over this equation, so the loop will generate p and q values and sub them in here
            # for every t value.
            z /= np.linalg.norm(z)  #normalise it cause we on a sphere

            # Optional: check that Q(z) ≈ 0, condition to break the code if not accurate enough
            Qz = np.sum(z ** 5)
            if np.abs(Qz) >= 1e-8:  # threshold
               continue

            points.append(z)

        if len(points) >= p_M_points: # limit of the while loop.
            break

    return np.array(points)  # shape: (n_points, 5)

sample = generate_quintic_points()

#print("Shape:", sample.shape)  # (1000, 5)
print("Sample point:", sample[0]) #I am just checking if it's working alright

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
    g = np.zeros((5,5), dtype = complex) # remember that g[i][j] means i'th row j'th column element.

    container_of_metrics = []

    for x in range(len(fixed)):
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

#print(metrics_at_each_p_M[999])




# Now we can code the determinant of the pullback of the Kahler form. det(i J^T g J)

def determinant_builder(extras):

    determinant_pullback_at_each_p_M = []

    for i in range(len(extras)):
        g = metrics_at_each_p_M[i]
        J = Jack[i]
        Jt = np.transpose(J)
        Jtbar = np.conjugate(Jt)

        pullback = 1j * J @ g @ Jtbar

        det = np.linalg.det(pullback)

        determinant_pullback_at_each_p_M.append(det)

    return determinant_pullback_at_each_p_M

determinant_list = determinant_builder(extras)

#print(determinant_builder())
#print("Count", len(determinant_builder()) ,'\n')


# For simplicity for the T-map is better if we define the values for the factor of the sum over N_p points of
# 1/25|z_J|^8 det(f*w)


# We first define the monomials of the map.

#creating a function that generates the list of monomials combination for a given k (user's choice)

def Monomial_list_coord_value():

    Monomial_list = []
    for i in range(len(sample)):
        x = coordinates_for_every_p_M[i]
        variables = [x[0],x[1],x[2],x[3],x[4]]
        combo = combinations_with_replacement(variables, 5) ### INPUT DEGREE OF COMBINATION YOU WANNA FIND !!!!!
        gh = list(combo)
        Monomial_list.append(gh)

    return Monomial_list

every_single_monomial_combination_tuple = Monomial_list_coord_value()

n = 5  # number of coordinates we are considering
k = 2  # order polynomial we are considering

#def N_k_builder():
N_k = math.comb(n + k - 1, k)

print(N_k)




# Now we need to add a code that creates the T-map. We divide the labour in two factors. First one Being N_k/Vol_CY and
# the second factor containing the sum over the matrix constructed by the monomials times the weight
# First factor given by:
def first_factor(N_k):

    factor = 0
    # Just like above here pick the desired N_k value over which the T-map should operate.
    for i in range(len(sample)):
        factor = factor + 1/(25 * (abs(container[i]) ** 8) * (determinant_list[i]))
    Vol_CY = (1/len(sample)) * factor
    first_fact = (N_k)/(Vol_CY)

    return first_fact

ff = first_factor(N_k)






# For the second factor we need to create first a list containing matrix that should contain s_alpha s_betabar at
# each p_M. Then construct the sum function timing by the weight given by 1/(25...determinant_list[i]).
# Starting by the numerator

def second_factor_matrix():

    section_matrix_list = []
    aid_list = []

    A = np.zeros((N_k,N_k), dtype=complex)

    for i in range(len(sample)):
        x = every_single_monomial_combination_tuple[i]
        for j in range(N_k):
            y = x[j]
            prod = y[0]*y[1]*y[2]*y[3]*y[4]
            aid_list.append(prod)

        for r in range(N_k):
            for v in range(N_k):
                A[r][v] = aid_list[r] * np.conj(aid_list[v]) # section v * section r element in matrix.
                section_matrix_list.append(A)

    return section_matrix_list

sfm = second_factor_matrix()

#print(sfm[0])
#print(sfm[0].shape)




def second_factor_weight_and_num():

    selection_iteration_term = []

    for i in range(len(sample)):
        term = (1/(25 * (abs(container[i]) ** 8) * (determinant_list[i]))) * sfm[i]
        selection_iteration_term.append(term)

    return selection_iteration_term

sfwad = second_factor_weight_and_num()

#print(second_factor_weight_and_denom())




# Now finally we try constructing the T-map by putting these factors together and summing them.

def sum_over_h_second_factor():

    factor = np.zeros((N_k,N_k), dtype = complex)
    h = np.eye(N_k, dtype=complex)

    for i in range(len(sample)):
        factor = factor + (1/(np.einsum("mn,mn",h,sfm[i]))) * sfwad[i]

    return factor

sohsf = sum_over_h_second_factor()

#print(sohsf)
#print(sohsf.shape)


def T_map_function():

    T_map = ff * sohsf
    factor = 0
    for _ in range(10): # Input here how many times to iterate the T_map
        for i in range(len(sample)):
            factor = factor + ff * (1 /(np.einsum("mn,mn",T_map,sfm[i]))) * sfwad[i]
        T_map = ff * factor
    return T_map

T_map = T_map_function()
#print(T_map)
#print(T_map.shape)

# AAAAAAAHHHHHHHHH WE MAY HAVE FOUND THE T_MAP

h_new = np.transpose(np.linalg.inv(T_map))

print(h_new.shape)















# STEP 3: Calculate the error in the code. It's nice to have a nice recap of the variables we have.
# Np = 1000
# k = 5
# Iteration times = 10

N_t = 4000

def error_vol_CY(N_t, container, determinant_list):

    factor = 0
    # Just like above here pick the desired N_k value over which the T-map should operate.
    for i in range(N_t):
        factor = factor + 1/(25 * (abs(container[i]) ** 8) * (determinant_list[i]))

    Evcy = (1/N_t) * factor

    return Evcy

EVCY = error_vol_CY(N_t, container, determinant_list)
#print(EVCY)


def error_vol_K(determinant_list, container):

    factor = 0
    for i in range(N_t):

        factor = factor + ((determinant_list[i])/(1/(25*((container[i])**8))))*(1/(25 * (abs(container[i]) ** 8) * (determinant_list[i])))

    evk = (1/N_t)*(-1j/8)*factor

    return evk

EVK = error_vol_K(determinant_list, container)

def sigma_measure_error(determinant_list, container, EVCY):
    factor = 0
    for i in range(N_t):
        factor = factor + (abs( 1 - (-1j/8)*(determinant_list[i]/EVK)/((1/(25*(abs(container[i])**8)))/EVCY)))*(1/(25 * (abs(container[i]) ** 8) * (determinant_list[i])))

    sigma = (1/(N_t*EVCY)) * factor

    return sigma

sigma = sigma_measure_error(determinant_list, container, EVCY)
print(abs(sigma)*100)


using MPI
using T8code
using T8code.Libt8: sc_init, sc_free, sc_finalize, sc_array_new_data, sc_array_destroy
using T8code.Libt8: SC_LP_ESSENTIAL, SC_LP_PRODUCTION

include("t8_step3_common.jl")

# The data that we want to store for ea ch element.
struct t8_step6_data_per_element_t
  # The first three data fields are not necessary for our
  # computations in this step, but are left here for reference.

  level :: Cint

  volume :: Cdouble
  midpoint :: NTuple{3,Cdouble}

  # Element length in x- and y-direction.
  dx :: Cdouble
  dy :: Cdouble

  # `Height` which is filled according to the position of the element.
  # in the computational domain.
  height :: Cdouble

  # Storage for our finite difference computations.
  schlieren :: Cdouble
  curvature :: Cdouble
end

# In this function we first allocate a new uniformly refined forest at given
# refinement level. Then a second forest is created, where user data for the
# adaption call (cf. step 3) is registered.  The second forest inherts all
# properties of the first ("root") forest and deallocates it. The final
# adapted and commited forest is returned back to the calling scope.
function t8_step6_build_forest(comm, dim, level)
  cmesh = t8_cmesh_new_periodic(comm, dim)

  scheme = t8_scheme_new_default_cxx()

  adapt_data = t8_step3_adapt_data_t(
    (0.0, 0.0, 0.0),      # Midpoints of the sphere.
    0.5,                  # Refine if inside this radius.
    0.7                   # Coarsen if outside this radius.
  )

  # Start with a uniform forest.
  forest = t8_forest_new_uniform(cmesh, scheme, level, 0, comm)

  forest_apbg_ref = Ref(t8_forest_t())
  t8_forest_init(forest_apbg_ref)
  forest_apbg = forest_apbg_ref[]

  # Adapt, partition, balance and create ghost elements all in one go.
  # See steps 3 and 4 for more details.
  t8_forest_set_user_data(forest_apbg, Ref(adapt_data))
  t8_forest_set_adapt(forest_apbg, forest, @t8_adapt_callback(t8_step3_adapt_callback), 0)
  t8_forest_set_partition(forest_apbg, C_NULL, 0)
  t8_forest_set_balance(forest_apbg, C_NULL, 0)
  t8_forest_set_ghost(forest_apbg, 1, T8_GHOST_FACES)
  t8_forest_commit(forest_apbg)

  return forest_apbg
end

# Allocate and fill the element data array with `heights` from an arbitrary
# mathematical function. Returns a pointer to the array which is then ownded by
# the calling scope.
function t8_step6_create_element_data(forest)
  # Check that the forest is a committed.
  @T8_ASSERT(t8_forest_is_committed(forest) == 1)

  # Get the number of local elements of forest.
  num_local_elements = t8_forest_get_local_num_elements(forest)
  # Get the number of ghost elements of forest.
  num_ghost_elements = t8_forest_get_num_ghosts(forest)

  # Build an array of our data that is as long as the number of elements plus
  # the number of ghosts.
  element_data = Array{t8_step6_data_per_element_t}(undef, num_local_elements + num_ghost_elements)

  # Get the number of trees that have elements of this process.
  num_local_trees = t8_forest_get_num_local_trees(forest)

  # Compute vertex coordinates. Note: Julia has column-major.
  verts = Matrix{Cdouble}(undef,3,3)

  # Element Midpoint
  midpoint = Vector{Cdouble}(undef,3)

  # Loop over all local trees in the forest.
  current_index = 0
  for itree = 0:num_local_trees-1
    tree_class = t8_forest_get_tree_class(forest, itree)
    eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

    # Get the number of elements of this tree.
    num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

    # Loop over all local elements in the tree.
    for ielement = 0:num_elements_in_tree-1
      current_index += 1 # Note: Julia has 1-based indexing, while C/C++ starts with 0.

      element = t8_forest_get_element_in_tree(forest, itree, ielement)

      level = t8_element_level(eclass_scheme, element)
      volume = t8_forest_element_volume(forest, itree, element)

      t8_forest_element_centroid(forest, itree, element, pointer(midpoint))

      t8_element_vertex_reference_coords(eclass_scheme, element, 0, @view(verts[:,1]))
      t8_element_vertex_reference_coords(eclass_scheme, element, 1, @view(verts[:,2]))
      t8_element_vertex_reference_coords(eclass_scheme, element, 2, @view(verts[:,3]))

      dx = verts[1,2] - verts[1,1]
      dy = verts[2,3] - verts[2,1]

      # Shift x and y to the center since the domain is [0,1] x [0,1].
      x = midpoint[1] - 0.5
      y = midpoint[2] - 0.5
      r = sqrt(x * x + y * y) * 20.0      # arbitrarly scaled radius

      # Some 'interesting' height function.
      height = sin(2.0 * r) / r

      element_data[current_index] = t8_step6_data_per_element_t(
        level, volume, Tuple(midpoint), dx, dy, height, 0.0, 0.0
      )
    end
  end

  return element_data
end

# Gather the 3x3 stencil for each element and compute finite difference approximations
# for schlieren and curvature of the stored heights in the elements.
function t8_step6_compute_stencil(forest, element_data)
  # Check that forest is a committed, that is valid and usable, forest.
  @T8_ASSERT(t8_forest_is_committed(forest) == 1)

  # Get the number of trees that have elements of this process. 
  num_local_trees = t8_forest_get_num_local_trees(forest)

  stencil = Matrix{Cdouble}(undef, 3, 3)
  dx = Vector{Cdouble}(undef, 3)
  dy = Vector{Cdouble}(undef, 3)

  # Loop over all local trees in the forest. For each local tree the element
  # data (level, midpoint[3], dx, dy, volume, height, schlieren, curvature) of
  # each element is calculated and stored into the element data array.
  current_index = 0
  for itree = 0:num_local_trees-1
    tree_class = t8_forest_get_tree_class(forest, itree)
    eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

    num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

    # Loop over all local elements in the tree.
    for ielement = 0:num_elements_in_tree-1
      current_index += 1 # Note: Julia has 1-based indexing, while C/C++ starts with 0.

      element = t8_forest_get_element_in_tree(forest, itree, ielement)

      # Gather center point of the 3x3 stencil.
      stencil[2,2] = element_data[current_index].height
      dx[2] = element_data[current_index].dx
      dy[2] = element_data[current_index].dy

      # Loop over all faces of an element.
      num_faces = t8_element_num_faces(eclass_scheme, element)
      for iface = 1:num_faces
        neighids_ref = Ref{Ptr{t8_locidx_t}}()
        neighbors_ref = Ref{Ptr{Ptr{t8_element}}}()
        neigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

        dual_faces_ref = Ref{Ptr{Cint}}()
        num_neighbors_ref = Ref{Cint}()

        forest_is_balanced = Cint(1)

        t8_forest_leaf_face_neighbors(forest, itree, element,
          neighbors_ref, iface-1, dual_faces_ref, num_neighbors_ref,
          neighids_ref, neigh_scheme_ref, forest_is_balanced)

        num_neighbors = num_neighbors_ref[]
        dual_faces    = 1 .+ unsafe_wrap(Array, dual_faces_ref[], num_neighbors)
        neighids      = 1 .+ unsafe_wrap(Array, neighids_ref[], num_neighbors)
        neighbors     = unsafe_wrap(Array, neighbors_ref[], num_neighbors)
        neigh_scheme  = neigh_scheme_ref[]

        # Retrieve the `height` of the face neighbor. Account for two neighbors
        # in case of a non-conforming interface by computing the average.
        height = 0.0
        if num_neighbors > 0
          for ineigh = 1:num_neighbors
            height = height + element_data[neighids[ineigh]].height
          end
          height = height / num_neighbors
        end

        # Fill in the neighbor information of the 3x3 stencil.
        if iface == 1 # NORTH
          stencil[1,2] = height
          dx[1] = element_data[neighids[1]].dx
        elseif iface == 2 # SOUTH
          stencil[3,2] = height
          dx[3] = element_data[neighids[1]].dx
        elseif iface == 3 # WEST
          stencil[2,1] = height
          dy[1] = element_data[neighids[1]].dy
        elseif iface == 4 # EAST
          stencil[2,3] = height
          dy[3] = element_data[neighids[1]].dy
        end

        # Free allocated memory.
        sc_free(t8_get_package_id(), neighbors_ref[])
        sc_free(t8_get_package_id(), dual_faces_ref[])
        sc_free(t8_get_package_id(), neighids_ref[])
      end

      # Prepare finite difference computations. The code also accounts for non-conforming interfaces.
      xslope_m = 0.5 / (dx[1] + dx[2]) * (stencil[2,2] - stencil[1,2])
      xslope_p = 0.5 / (dx[2] + dx[3]) * (stencil[3,2] - stencil[2,2])

      yslope_m = 0.5 / (dy[1] + dy[2]) * (stencil[2,2] - stencil[2,1])
      yslope_p = 0.5 / (dy[2] + dy[3]) * (stencil[2,3] - stencil[2,2])

      xslope = 0.5 * (xslope_m + xslope_p)
      yslope = 0.5 * (yslope_m + yslope_p)

      # TODO: Probably still not optimal at non-conforming interfaces.
      xcurve = (xslope_p - xslope_m) * 4 / (dx[1] + 2.0 * dx[2] + dx[3])
      ycurve = (yslope_p - yslope_m) * 4 / (dy[1] + 2.0 * dy[2] + dy[3])

      # Compute schlieren and curvature norm.
      schlieren = sqrt(xslope * xslope + yslope * yslope)
      curvature = sqrt(xcurve * xcurve + ycurve * ycurve)

      element_data[current_index] = t8_step6_data_per_element_t(
        element_data[current_index].level, 
        element_data[current_index].volume,
        element_data[current_index].midpoint,
        element_data[current_index].dx,
        element_data[current_index].dy,
        element_data[current_index].height,
        schlieren, 
        curvature
      )
    end
  end
end

# Each process has computed the data entries for its local elements.  In order
# to get the values for the ghost elements, we use
# t8_forest_ghost_exchange_data.  Calling this function will fill all the ghost
# entries of our element data array with the value on the process that owns the
# corresponding element. */
function t8_step6_exchange_ghost_data(forest, element_data)
  # t8_forest_ghost_exchange_data expects an sc_array (of length num_local_elements + num_ghosts).
  # We wrap our data array to an sc_array.
  sc_array_wrapper = sc_array_new_data(pointer(element_data), sizeof(t8_step6_data_per_element_t), length(element_data))

  # Carry out the data exchange. The entries with indices > num_local_elements will get overwritten.
  t8_forest_ghost_exchange_data(forest, sc_array_wrapper)

  # Destroy the wrapper array. This will not free the data memory since we used sc_array_new_data.
  sc_array_destroy(sc_array_wrapper)
end

# The prefix for our output files.
prefix_forest_with_data = "t8_step6_stencil"

# The uniform refinement level of the forest.
dim = 2
level = 1

#
# Initialization.
#

# Initialize MPI. This has to happen before we initialize sc or t8code.
mpiret = MPI.Init()

# We will use MPI_COMM_WORLD as a communicator.
comm = MPI.COMM_WORLD

# Initialize the sc library, has to happen before we initialize t8code.
sc_init(comm, 1, 1, C_NULL, SC_LP_ESSENTIAL)
# Initialize t8code with log level SC_LP_PRODUCTION. See sc.h for more info on the log levels.
t8_init(SC_LP_PRODUCTION)

# Initialize an adapted forest with periodic boundaries.
forest = t8_step6_build_forest(comm, dim, level)

#
# Data handling and computation.
#

# Build data array and gather data for the local elements.
element_data = t8_step6_create_element_data(forest)

# Exchange the neighboring data at MPI process boundaries.
t8_step6_exchange_ghost_data(forest, element_data)

# Compute stencil.
# t8_step6_compute_stencil(forest, element_data)

# Check that forest is a committed, that is valid and usable, forest.
@T8_ASSERT(t8_forest_is_committed(forest) == 1)

# Get the number of trees that have elements of this process. 
num_local_trees = t8_forest_get_num_local_trees(forest)

# count total number of elements
num_elements = 0
for itree = 0:num_local_trees-1
tree_class = t8_forest_get_tree_class(forest, itree)
eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)
num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)
num_elements += num_elements_in_tree
end
faces_per_element = zeros(Int, num_elements)

# count faces per element
iface = 1
for itree = 0:num_local_trees-1
tree_class = t8_forest_get_tree_class(forest, itree)
eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)
num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)
for ielement = 0:num_elements_in_tree-1
    element = t8_forest_get_element_in_tree(forest, itree, ielement)
    num_faces = t8_element_num_faces(eclass_scheme, element)
    faces_per_element[iface] = num_faces
    iface += 1
end
end

face_offsets = cumsum(faces_per_element) .- faces_per_element[1]

num_total_faces = sum(faces_per_element)
FToF = collect(1:num_total_faces)
split_faces = Int[]

# Loop over all local trees in the forest. 
current_element = 1
for itree = 0:num_local_trees-1
    tree_class = t8_forest_get_tree_class(forest, itree)
    eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)
    num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)
    for ielement = 0:num_elements_in_tree-1
        element = t8_forest_get_element_in_tree(forest, itree, ielement)
        num_faces = t8_element_num_faces(eclass_scheme, element)
        for iface = 1:num_faces
            neighids_ref = Ref{Ptr{t8_locidx_t}}()
            neighbors_ref = Ref{Ptr{Ptr{t8_element}}}()
            neigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

            dual_faces_ref = Ref{Ptr{Cint}}()
            num_neighbors_ref = Ref{Cint}()
            forest_is_balanced = Cint(1)
            t8_forest_leaf_face_neighbors(forest, itree, element,
                neighbors_ref, iface-1, dual_faces_ref, num_neighbors_ref,
                neighids_ref, neigh_scheme_ref, forest_is_balanced)

            num_neighbors = num_neighbors_ref[]
            dual_faces    = 1 .+ unsafe_wrap(Array, dual_faces_ref[], num_neighbors)
            neighids      = 1 .+ unsafe_wrap(Array, neighids_ref[], num_neighbors)
            neighbors     = unsafe_wrap(Array, neighbors_ref[], num_neighbors)
            neigh_scheme  = neigh_scheme_ref[]        

            current_face = iface + face_offsets[current_element]
            if num_neighbors == 1
                neighbor_face = dual_faces[1] + face_offsets[neighids[1]]
                FToF[current_face] = neighbor_face
            elseif num_neighbors > 1 
                push!(split_faces, current_face)
                # neighbor_face
            end

            # Free allocated memory.
            sc_free(t8_get_package_id(), neighbors_ref[])
            sc_free(t8_get_package_id(), dual_faces_ref[])
            sc_free(t8_get_package_id(), neighids_ref[])
        end

        current_element += 1
        
    end
end

#
# Clean-up
#

# Destroy the forest.
t8_forest_unref(Ref(forest))
t8_global_productionf(" Destroyed forest.\n")

sc_finalize()


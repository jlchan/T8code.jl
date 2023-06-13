using MPI
using T8code
using T8code.Libt8: sc_init, sc_free, sc_finalize, sc_array_new_data, sc_array_destroy
using T8code.Libt8: SC_LP_ESSENTIAL, SC_LP_PRODUCTION

include("t8_step3_common.jl")

# In this function we first allocate a new uniformly refined forest at given
# refinement level. Then a second forest is created, where user data for the
# adaption call (cf. step 3) is registered.  The second forest inherts all
# properties of the first ("root") forest and deallocates it. The final
# adapted and commited forest is returned back to the calling scope.
function t8_step6_build_forest(element_type, comm, dim, level)
  
  # cmesh = t8_cmesh_new_periodic(comm, dim)
  if element_type isa StartUpDG.Tri
    cmesh = t8_cmesh_new_hypercube(T8_ECLASS_TRIANGLE, comm, 0, 0, 0)
  elseif element_type isa StartUpDG.Quad
    cmesh = t8_cmesh_new_hypercube(T8_ECLASS_QUAD, comm, 0, 0, 0)
  else
    @show element_type
  end

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


# The uniform refinement level of the forest.
dim = 2
level = 2

using StartUpDG
rd = RefElemData(Tri(), 3)

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
forest = t8_step6_build_forest(rd.element_type, comm, dim, level)

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

# coordinates
VX = [zeros(StartUpDG.num_vertices(rd.element_type)) for _ in 1:num_elements]
VY = [zeros(StartUpDG.num_vertices(rd.element_type)) for _ in 1:num_elements]

# count faces per element and get coordinates
current_element = 1
num_mortar_faces = 0
for itree = 0:num_local_trees-1
  tree_class = t8_forest_get_tree_class(forest, itree)
  eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)
  num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)
  for ielement = 0:num_elements_in_tree-1
    element = t8_forest_get_element_in_tree(forest, itree, ielement)
    num_faces = t8_element_num_faces(eclass_scheme, element)

    num_corners = t8_element_num_corners(eclass_scheme, element)
    for corner_number in 1:num_corners 
      coordinates = Vector{Cdouble}(undef, 3) 
      t8_forest_element_coordinate(forest, itree, element, corner_number-1, pointer(coordinates))
      VX[current_element][corner_number] = coordinates[1]
      VY[current_element][corner_number] = coordinates[2]
    end

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

      if num_neighbors > 1
        num_mortar_faces += num_neighbors
      end

      sc_free(t8_get_package_id(), neighbors_ref[])
      sc_free(t8_get_package_id(), dual_faces_ref[])
      sc_free(t8_get_package_id(), neighids_ref[])
    end

    faces_per_element[current_element] = num_faces
    current_element += 1
  end
end

face_offsets = cumsum(faces_per_element) .- faces_per_element[1]

num_element_faces = sum(faces_per_element)
FToF = collect(1:(num_element_faces + num_mortar_faces))
nonconforming_faces = Int[]

# Loop over all local trees in the forest. 
current_element = 1
nonconforming_face_offset = 0
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

      current_face = iface + face_offsets[current_element]
      if num_neighbors == 1 # then it's a conforming face

        neighbor_face = dual_faces[1] + face_offsets[neighids[1]]
        FToF[current_face] = neighbor_face

      elseif num_neighbors > 1 # then it's a non-conforming face

        # add the current face index to the list of non-conforming faces (to split)
        push!(nonconforming_faces, current_face)
        
        # if it's a non-conforming face with 2:1 balance, the neighboring faces
        # are conforming (e.g., non-split) faces. 
        neighbor_faces = dual_faces .+ face_offsets[neighids]

        # split faces are ordered after un-split faces, so we
        # track the total number of conforming faces. 
        split_faces_indices = 
          @. num_element_faces + nonconforming_face_offset + (1:num_neighbors)

        nonconforming_face_offset += num_neighbors 

        # make connections between mortar faces
        FToF[split_faces_indices] .= neighbor_faces
      end

      # Free allocated memory.
      sc_free(t8_get_package_id(), neighbors_ref[])
      sc_free(t8_get_package_id(), dual_faces_ref[])
      sc_free(t8_get_package_id(), neighids_ref[])
    end

    current_element += 1
    
  end
end

# Clean-up
t8_forest_unref(Ref(forest)) # Destroy the forest.
t8_global_productionf(" Destroyed forest.\n")

sc_finalize()


# permute indices for StartUpDG ordering

function permute_vertices!(::Quad, VXY)
    VX, VY = VXY
    p = [1, 3, 2, 4]
    for e in eachindex(VX, VY)
        VX[e] = VX[e][p]
        VY[e] = VY[e][p]
    end
    return VX, VY
end

function permute_vertices!(::Tri, VXY)
    VX, VY = VXY

    for e in eachindex(VX, VY)
        vx, vy = VX[e], VY[e]
        area = StartUpDG.compute_triangle_area(zip(vx, vy))
        if area < 0
            VX[e] = vx[[2, 1, 3]]
            VY[e] = vy[[2, 1, 3]]
        end
    end
    return VX, VY
end

md = MeshData(permute_vertices!(rd.element_type, (VX, VY)), FToF, nonconforming_faces, rd)

using Plots
scatter(md.xyz..., leg=false, ms=3); plot!(rd, md)
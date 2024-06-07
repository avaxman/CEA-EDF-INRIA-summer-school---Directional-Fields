import os
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import cmath


# Helper function to compute_angle_defect()---ignore
def accumarray(indices, values):
    output = np.zeros((np.max(indices) + 1), dtype=values.dtype)
    indFlat = indices.flatten()
    valFlat = values.flatten()
    # for index in range(indFlat.shape[0]):
    #     output[indFlat[index]] += valFlat[index]
    np.add.at(output, indFlat, valFlat)

    return output


# Loading an OFF file, and producing vertices and faces
def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


# input:
# |V|x3 vertices (double)
# |F|x3 faces (triangles in mesh; indices into "vertices").
# output:
# normals: |F|x3 face normals (normalized)
# faceAreas: |F| face areas
# barycenters |F|x3 face barycenters
# basisX, basisY: |F|x3 (for each), an (arbitrary) orthonormal basis per face
def compute_geometric_quantities(vertices, faces):
    face_vertices = vertices[faces]

    # Compute vectors on the face
    vectors1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
    vectors2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]

    # Compute face normals using cross product
    normals = np.cross(vectors1, vectors2)
    faceAreas = 0.5 * np.linalg.norm(normals, axis=1)

    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    basisX = vectors1
    basisX = basisX / np.linalg.norm(basisX, axis=1, keepdims=True)
    basisY = np.cross(normals, basisX)

    barycenters = (vertices[faces[:, 0], :] + vertices[faces[:, 1], :] + vertices[faces[:, 2], :]) / 3
    return normals, faceAreas, barycenters, basisX, basisY


# Helper function for compute_topological_quantities---ignore
def createEH(edges, halfedges):
    # Create dictionaries to map halfedges to their indices
    halfedges_dict = {(v1, v2): i for i, (v1, v2) in enumerate(halfedges)}

    EH = np.zeros((len(edges), 2), dtype=int)

    for i, (v1, v2) in enumerate(edges):
        # Check if the halfedge exists in the original order
        if (v1, v2) in halfedges_dict:
            EH[i, 0] = halfedges_dict[(v1, v2)]
        # Check if the halfedge exists in the reversed order
        if (v2, v1) in halfedges_dict:
            EH[i, 1] = halfedges_dict[(v2, v1)]

    return EH


# input: vertices and faces (see documentation for compute_geometric_quantities())
# output:
# halfedges: 3|F| x 2 indices to all pairs F(i), F(i+1 mod 3) of halfedges per face
# edges: |E| x2 the unique edges, with indices (source, target) for vertices in each row.
# EF: |E|x2 list of (left, right) faces to each edge
def compute_topological_quantities(vertices, faces):
    halfedges = np.empty((3 * faces.shape[0], 2))
    for face in range(faces.shape[0]):
        for j in range(3):
            halfedges[3 * face + j, :] = [faces[face, j], faces[face, (j + 1) % 3]]

    edges, firstOccurence, numOccurences = np.unique(np.sort(halfedges, axis=1), axis=0, return_index=True,
                                                     return_counts=True)
    edges = halfedges[np.sort(firstOccurence)]
    edges = edges.astype(int)
    halfedgeBoundaryMask = np.zeros(halfedges.shape[0])
    halfedgeBoundaryMask[firstOccurence] = 2 - numOccurences
    edgeBoundMask = halfedgeBoundaryMask[np.sort(firstOccurence)]

    boundEdges = edges[edgeBoundMask == 1, :]
    boundVertices = np.unique(boundEdges).flatten()

    # EH = [np.where(np.sort(halfedges, axis=1) == edge)[0] for edge in edges]
    # EF = []

    EH = createEH(edges, halfedges)
    EF = EH // 3

    return halfedges, edges, EF


# Input:
# N: symmetry order (should always be given 4 in this tutorial)
# inFaces: list of indices into faces (or here into basisX and basisY) of faces that have a vector to be converted
# extField: |inFaces|x3 (double) tangent single vector field in each of the "inFaces" (the "u" in the slides)
# basisX, basisY: each |F|x3 of the face-based tangent basis vectors
# Output:
# powerField: a complex |inFaces|x1 array of the powerfield (X=u^N)
def extrinsic_to_power(N, inFaces, extField, basisX, basisY):
    complexField = np.sum(extField * basisX[inFaces, :], axis=1) + 1j * np.sum(extField * basisY[inFaces, :], axis=1)
    powerField = np.exp(np.log(complexField.reshape(-1, 1)) * N)
    return powerField


# the inverse function to that above, with same type of parameters
def power_to_extrinsic(N, inFaces, powerField, basisX, basisY):
    extField = np.zeros([powerField.shape[0], 3 * N])
    repField = np.exp(np.log(powerField) / N)
    for i in range(0, N):
        currField = repField * cmath.exp(2 * i * cmath.pi / N * 1j)
        extField[:, 3 * i:3 * i + 3] = np.real(currField) * basisX[inFaces, :] + np.imag(currField) * basisY[inFaces, :]

    return extField


# Input: vertices and faces like the above
# Output: |V|x1 array of angle defects per vertex (discrete Gaussian curvature)
def compute_angle_defect(vertices, faces):
    vi = vertices[faces[:, 0], :]
    vj = vertices[faces[:, 1], :]
    vk = vertices[faces[:, 2], :]

    cosijk = np.sum((vi - vj) * (vk - vj), axis=1) / np.linalg.norm(vi - vj, axis=1) / np.linalg.norm(vk - vj, axis=1)
    cosjki = np.sum((vj - vk) * (vi - vk), axis=1) / np.linalg.norm(vj - vk, axis=1) / np.linalg.norm(vi - vk, axis=1)
    coskij = np.sum((vk - vi) * (vj - vi), axis=1) / np.linalg.norm(vk - vi, axis=1) / np.linalg.norm(vj - vi, axis=1)

    cornerAngles = np.arccos(np.column_stack((coskij, cosijk, cosjki)))

    angleDefect = 2 * np.pi - accumarray(faces, cornerAngles)

    return angleDefect


# Input: as explained above
# Output:
# singVertices: a |S|x1 array (|S| - number of singular vertices) of indices into "vertices" of singular vertices
# singIndices: a |S|x1 integer array of singularity indices, where the true fractional index is singIndices/N (you don't need to explicitly compute the fractional index)
def get_singularities(N, powerField, vertices, faces, edges, basisX, basisY):
    # angle defect
    K = compute_angle_defect(vertices, faces).reshape(-1, 1)

    # d0 operator
    rows = np.arange(0, edges.shape[0])
    rows = np.tile(rows[:, np.newaxis], 2)
    values = np.full((edges.shape[0], 2), [-1, 1])
    d0 = csc_matrix((values.flatten(), (rows.flatten(), edges.flatten())), shape=(edges.shape[0], vertices.shape[0]))

    # computing effort
    edgeVectors = vertices[edges[:, 1], :] - vertices[edges[:, 0], :]
    edgeVectorsFace1 = extrinsic_to_power(N,  EF[:, 0], edgeVectors, basisX, basisY)
    edgeVectorsFace2 = extrinsic_to_power(N, EF[:, 1], edgeVectors, basisX, basisY)

    expEffort = (powerField[EF[:, 1]] * np.conj(edgeVectorsFace2)) / (powerField[EF[:, 0]] * np.conj(edgeVectorsFace1))
    effort = np.imag(np.log(expEffort))

    indices = (d0.T * effort + N * K) / (2 * np.pi)
    singVertices = np.column_stack(np.nonzero(np.round(indices)))[:, 0]
    singIndices = np.round(indices)[singVertices]
    return singVertices, singIndices


# Input: (only the new parts):
#   constFaces: an array of indices into "faces" of the faces that are constrained
#   constPowerField: a |constFaces|x1 complex array of prescribed power field 1-1 corresponding to constFaces
# Output:
#   powerField: the full |F|x1 complex power field, including the constPowerField as a subset within the constFaces positions
def interpolate_power_field(N, constFaces, constPowerField, vertices, faces, edges, EF, basisX, basisY):
    # creating the connection
    edgeVectors = vertices[edges[:, 1], :] - vertices[edges[:, 0], :]
    edgeVectorsFace1 = extrinsic_to_power(N, EF[:, 0], edgeVectors, basisX, basisY)
    edgeVectorsFace2 = extrinsic_to_power(N, EF[:, 1], edgeVectors, basisX, basisY)

    # preparing constraints and linear system
    rows = np.column_stack((np.arange(0, edges.shape[0]), np.arange(0, edges.shape[0])))
    cols = EF
    values = np.column_stack([-np.conj(edgeVectorsFace1), np.conj(edgeVectorsFace2)])
    A = coo_matrix((values.flatten(), (rows.flatten(), cols.flatten())), shape=(EF.shape[0], faces.shape[0]))
    A = A.tocsr()
    AConst = A[:, constFaces]
    varFaces = [i for i in range(A.shape[1]) if i not in constFaces]
    AVar = A[:, varFaces]
    b = -AConst * constPowerField
    powerField = np.zeros([faces.shape[0], 1], dtype=np.complex128)
    powerField[constFaces] = constPowerField
    powerField[varFaces] = spsolve(np.dot(np.conj(AVar.T), AVar), np.conj(AVar.T).dot(b)).reshape(-1, 1)

    return powerField


# helper function that returns the 5-percent faces with the biggest dihedral angle to their neighbors, and the consequent edge between them
# Nothing for you to implement here
def get_biggest_dihedral(vertices, edges, EF, faces, normals):
    # Computing all dihedral angles
    cosDihedrals = np.sum(normals[EF[:, 0], :] * normals[EF[:, 1], :], axis=1)
    dihedralAngle = np.arccos(np.clip(cosDihedrals, -1, 1))

    # getting top 5% in absolute value
    top5Percent = int(np.ceil(0.05 * len(dihedralAngle)))

    threshold = np.partition(np.abs(dihedralAngle), -top5Percent)[-top5Percent]

    # Find the 5% sharpest edges and allocating the edge vectors to the adjacent faces
    constEdges = np.where(np.abs(dihedralAngle) >= threshold)[0]
    constFaces = np.reshape(EF[constEdges, :], (2 * len(constEdges), 1)).flatten()
    extField = np.reshape(np.tile(vertices[edges[constEdges, 1], :] - vertices[edges[constEdges, 0], :], (1, 2)), (2 * len(constEdges), 3))

    #checking normals are orthogonal to field
    # constNormals = normals[constFaces,:]
    # fieldNormalDot = np.sum(constNormals * extField, axis = 1)
    return constFaces, extField


if __name__ == '__main__':

    # Loading and initializing
    vertices, faces = load_off_file(os.path.join('..', 'data', 'fandisk.off'))
    halfedges, edges, EF = compute_topological_quantities(vertices, faces)
    normals, faceAreas, barycenters, basisX, basisY = compute_geometric_quantities(vertices, faces)
    N = 4  # keep this fixed! it means "cross field"

    # constraining the %5 of the faces with the highest dihedral angle
    constFaces, constExtField = get_biggest_dihedral(vertices, edges, EF, faces, normals)

    # getting the constrained power field (Xconst = uconst^4, and in complex coordinates)
    constPowerField = extrinsic_to_power(N, constFaces, constExtField, basisX, basisY)

    # Interpolating the field to all other faces
    fullPowerField = interpolate_power_field(N, constFaces, constPowerField, vertices, faces, edges, EF, basisX, basisY)

    # Getting back the extrinsic R^3 field
    extField = power_to_extrinsic(N, range(0, faces.shape[0]), fullPowerField, basisX, basisY)

    # Extracting ingularities
    singVertices, singIndices = get_singularities(4, fullPowerField, vertices, faces, edges, basisX, basisY)

    # PolyScope visualization - nothing to do here
    ps.init()
    ps_mesh = ps.register_surface_mesh("Hello World Mesh", vertices, faces)

    for i in range(0, 4):
        ps_mesh.add_vector_quantity("Full Power Field" + str(i), extField[:, 3 * i:3 * i + 3], defined_on='faces')

    ps_cloud = ps.register_point_cloud("Singularities", vertices[singVertices, :])
    ps_cloud.add_scalar_quantity("Indices", singIndices.flatten(), vminmax=(-N, N), enabled=True)

    ps_cloud = ps.register_point_cloud("Constrained Faces", barycenters[constFaces, :])
    ps_cloud.add_vector_quantity("Input Field", constExtField)

    ps.show()

import os
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import cmath
import meshio


def accumarray(indices, values):
    output = np.zeros((np.max(indices) + 1), dtype=values.dtype)
    indFlat = indices.flatten()
    valFlat = values.flatten()
    # for index in range(indFlat.shape[0]):
    #     output[indFlat[index]] += valFlat[index]
    np.add.at(output, indFlat, valFlat)

    return output


def load_tet_mesh(filename):
    mesh = meshio.read(filename)
    boundF = mesh.cells[0].data if mesh.cells[0].type == 'triangle' else mesh.cells[1].data
    for i in range(0, len(mesh.cells)):
        if (mesh.cells[i].type == 'tetra'):
            tets = mesh.cells[i].data

    vertices = mesh.points

    return vertices, boundF, tets


def compute_face_quantities(vertices, faces):
    face_vertices = vertices[faces]

    # Compute vectors on the face
    vectors1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
    vectors2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]

    # Compute face normals using cross product
    normals = np.cross(vectors1, vectors2)
    faceAreas = 0.5 * np.linalg.norm(normals, axis=1)

    normals /= (2.0 * faceAreas[:, np.newaxis])

    basisX = vectors1
    basisX = basisX / np.linalg.norm(basisX, axis=1, keepdims=True)
    basisY = np.cross(normals, basisX)
    return normals, faceAreas, basisX, basisY


def createFH(faces, halffaces):
    # Create dictionaries to map halfface to their indices
    halffaces_dict = {(v1, v2, v3): i for i, (v1, v2, v3) in enumerate(halffaces)}

    FH = np.zeros((len(faces), 2), dtype=int)

    for i, (v1, v2, v3) in enumerate(faces):
        # Check if the halfface exists in the original order
        if (v1, v2, v3) in halffaces_dict:
            FH[i, 0] = halffaces_dict[(v1, v2, v3)]
        # Check if the halfface exists in the reversed order up to cyclic permutation
        if (v2, v1, v3) in halffaces_dict:
            FH[i, 1] = halffaces_dict[(v2, v1, v3)]
        if (v3, v2, v1) in halffaces_dict:
            FH[i, 1] = halffaces_dict[(v3, v2, v1)]
        if (v1, v3, v2) in halffaces_dict:
            FH[i, 1] = halffaces_dict[(v1, v3, v2)]

    return FH


def compute_topology(vertices, tets):
    halffaces = np.empty((4 * tets.shape[0], 3))
    for tet in range(tets.shape[0]):
        halffaces[(4 * tet):(4 * tet + 4), :] = [[tets[tet, 0], tets[tet, 1], tets[tet, 2]],
                                                 [tets[tet, 2], tets[tet, 1], tets[tet, 3]],
                                                 [tets[tet, 0], tets[tet, 2], tets[tet, 3]],
                                                 [tets[tet, 1], tets[tet, 0], tets[tet, 3]]]

    faces, firstOccurence, numOccurences = np.unique(np.sort(halffaces, axis=1), axis=0, return_index=True,
                                                     return_counts=True)
    faces = halffaces[np.sort(firstOccurence)]
    faces = faces.astype(int)
    halffaceBoundaryMask = np.zeros(halffaces.shape[0])
    halffaceBoundaryMask[firstOccurence] = 2 - numOccurences
    faceBoundMask = halffaceBoundaryMask[np.sort(firstOccurence)]

    boundFaces = faces[faceBoundMask == 1, :]
    boundVertices = np.unique(boundFaces).flatten()

    FH = createFH(faces, halffaces)
    FT = np.column_stack((FH[:, 0] // 4, (FH[:, 0] + 3) % 4, FH[:, 1] // 4, (FH[:, 1] + 3) % 4))

    return halffaces, faces, faceBoundMask, boundVertices, boundFaces, FH, FT


def get_singularities(N, powerField, vertices, edges, faces, tets):
    # angle defect
    K = compute_angle_defect(vertices, faces).reshape(-1, 1)

    # d0 operator
    rows = np.arange(0, edges.shape[0])
    rows = np.tile(rows[:, np.newaxis], 2)
    values = np.full((edges.shape[0], 2), [-1, 1])
    d0 = csc_matrix((values.flatten(), (rows.flatten(), edges.flatten())), shape=(edges.shape[0], vertices.shape[0]))

    # computing effort
    edgeVectors = vertices[edges[:, 1], :] - vertices[edges[:, 0], :]
    edgeVectorsFace1 = extrinsic_to_power(N, edgeVectors, EF[:, 0], basisX, basisY)
    edgeVectorsFace2 = extrinsic_to_power(N, edgeVectors, EF[:, 2], basisX, basisY)

    expEffort = (powerField[EF[:, 2]] * np.conj(edgeVectorsFace2)) / (powerField[EF[:, 0]] * np.conj(edgeVectorsFace1))
    effort = np.imag(np.log(expEffort))

    indices = (d0.T * effort + N * K) / (2 * np.pi)
    singVertices = np.column_stack(np.nonzero(np.round(indices)))[:, 0]
    singIndices = np.round(indices)[singVertices]
    return singVertices, singIndices


def interpolate_SPH_field(constFieldSPH, constTets, vertices, edges, faces, tets, FT):
    # creating the connection
    edgeVectors = vertices[edges[:, 1], :] - vertices[edges[:, 0], :]
    edgeVectorsFace1 = extrinsic_to_power(N, edgeVectors, EF[:, 0], basisX, basisY)
    edgeVectorsFace2 = extrinsic_to_power(N, edgeVectors, EF[:, 2], basisX, basisY)

    # preparing constraints and linear system
    crossFieldConst = constField2D  # np.exp(np.log(constField2D) * 4)
    rows = np.column_stack((np.arange(0, edges.shape[0]), np.arange(0, edges.shape[0])))
    cols = EF[:, [0, 2]]
    values = np.column_stack([-np.conj(edgeVectorsFace1), np.conj(edgeVectorsFace2)])
    A = coo_matrix((values.flatten(), (rows.flatten(), cols.flatten())), shape=(EF.shape[0], faces.shape[0]))
    A = A.tocsr()
    AConst = A[:, constFaces]
    varFaces = [i for i in range(A.shape[1]) if i not in constFaces]
    AVar = A[:, varFaces]
    b = -AConst * crossFieldConst
    crossFieldFull = np.zeros([faces.shape[0], 1], dtype=np.complex128)
    crossFieldFull[constFaces] = crossFieldConst
    crossFieldFull[varFaces] = spsolve(np.dot(np.conj(AVar.T), AVar), np.conj(AVar.T).dot(b)).reshape(-1, 1)

    return crossFieldFull


def interpolate_octahedral_field(constField3D, constBoundFaces, vertices, edge, faces, tets, FT, boundFaces,
                                 boundNormals, boundBasisX, boundBasisY):
    # creating the constraints on the boundary tets
    vc = np.sqrt(5 / 12) * np.array([1, 0, 0, 0, 0, 0, 0, 0, 1]).T
    vs = np.sqrt(5 / 12) * np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]).T
    vn = np.sqrt(7 / 12) * np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]).T

    # constructing boundary tet function
    boundTets = FT[boundFaces, 0]
    u0 = np.zeros([len(boundTets), 1])
    Hc = np.zeros([9 * len(boundTets), len(boundTets)])
    Hs = np.zeros([9 * len(boundTets), len(boundTets)])
    for constIndex in range(0, len(boundTets)):
        Rn = shortest_rotation(np.identity(3), np.array(boundBasisX[constBoundFaces[constIndex], :],
                                                        boundBasisY[constBoundFace[constIndex], :],
                                                        boundNormals[constBoundFaces[constIndex], :]).T)
        WRn = wigner(Rn)
        u0[9 * constIndex:9 * constIndex + 9] = WRn @ nl
        Hc[9 * constIndex:9 * constIndex + 9, boundTets[constIndex]] = WRn @ vc
        Hs[9 * constIndex:9 * constIndex + 9, boundTets[constIndex]] = WRn @ vs

    # getting the values for the constrained faces
    cConst = np.zeros([len(constBoundFaces), 1])
    sConst = np.zeros([len(constBoundFaces), 1])
    for constIndex in range(0, len(constBoundFaces)):
        cConst[constIndex] = np.dot(constField3D[constIndex, :], boundBasisX[constBoundFaces[constIndex], :])
        sConst[constIndex] = np.dot(constField3D[constIndex, :], boundBasisY[constBoundFaces[constIndex], :])

    # Abstracting constraints
    HcVar = Hc[:, varBoundFaces]
    HcConst = Hc[:, constBoundFaces]
    HsVar = Hs[:, varBoundFaces]
    HsConst = Hs[:, constBoundFaces]
    A = np.column_stack(HcVar, HsVar);
    b = v0 + HcConst * cConst + HsConst * sConst

    # Constructing the differential operator for the dirichlet energy
    innerFaces = np.setdiff1d(range(0, faces.shape[0]), boundFaces)
    rows = np.column_stack((np.arange(0, innerFaces.shape[0]), np.arange(0, innerFaces.shape[0])))
    cols = FT[innerFaces, [0, 2]]
    values = np.full((innerFaces.shape[0], 2), [-1, 1])
    d2 = coo_matrix((values.flatten(), (rows.flatten(), cols.flatten())), shape=(innerFaces.shape[0], tets.shape[0]))

    d2Inner = d2[:, innerTets]
    d2Bound = d2[:, boundTets]
    E = 


if __name__ == '__main__':
    ps.init()

    vertices, boundFaces, tets = load_tet_mesh(os.path.join('..', 'data', 'bone_80k.mesh'))

    #

    halffaces, faces, faceBoundMask, boundVertices, boundFaces, FH, FT = compute_topology(vertices, tets)

    boundNormals, boundFaceAreas, basisX, basisY = compute_face_quantities(vertices, boundFaces)

    # N = 4
    constBoundFaces = [1, 1000, 2000, 3000]
    constField3D = vertices[faces[constFaces, 2], :] - vertices[faces[constFaces, 1], :]
    constField3D /= np.linalg.norm(constField3D, axis=1, keepdims=True)

    SPHField = interpolate_octahedral_field(constField3D, constBoundFaces, vertices, edge, faces, tets, FT, boundFaces,
                                            basisX, basisY)

    extField = SPH_to_Euclidean(SPHField)

    # in complex coordinates
    # constField2D = extrinsic_to_power(N, constField3D, constFaces, basisX, basisY)

    # crossField = interpolate_power_field(N, constField2D, constFaces, vertices, faces, edges, EF, basisX, basisY)

    # extField = power_to_extrinsic(N, crossField, range(0, faces.shape[0]), basisX, basisY)

    ps_mesh = ps.register_volume_mesh("Tet Mesh", vertices, tets)

    for i in range(0, 6):
        ps_mesh.add_vector_quantity("field" + str(i), extField[:, 3 * i:3 * i + 3], defined_on='tets')

    # singVertices, singIndices = get_singularities(4, crossField, vertices, faces, edges, basisX, basisY)

    # ps_cloud = ps.register_point_cloud("Singularities", vertices[singVertices, :])
    # ps_cloud.add_scalar_quantity("Indices", singIndices.flatten(), vminmax=(-N, N), enabled=True)

    ps.show()

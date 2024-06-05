import os
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import cmath
import meshio
from scipy.special import sph_harm


def OctaTensorGradient(q, x, y, z):
    n = len(x)
    assert n == q.shape[1]

    q = q.T

    dx = [
        2 * np.pi ** (-1 / 2) * x * (x ** 2 + y ** 2 + z ** 2),
        (-3 / 4) * (35 * np.pi ** (-1)) ** (1 / 2) * y * (-3 * x ** 2 + y ** 2),
        (9 / 2) * ((35 / 2) * np.pi ** (-1)) ** (1 / 2) * x * y * z,
        (-3 / 4) * (5 * np.pi ** (-1)) ** (1 / 2) * y * (3 * x ** 2 + y ** 2 - 6 * z ** 2),
        (-9 / 2) * ((5 / 2) * np.pi ** (-1)) ** (1 / 2) * x * y * z,
        (9 / 4) * np.pi ** (-1 / 2) * x * (x ** 2 + y ** 2 - 4 * z ** 2),
        (3 / 4) * ((5 / 2) * np.pi ** (-1)) ** (1 / 2) * z * (-9 * x ** 2 - 3 * y ** 2 + 4 * z ** 2),
        (-3 / 2) * (5 * np.pi ** (-1)) ** (1 / 2) * x * (x ** 2 - 3 * z ** 2),
        (9 / 4) * ((35 / 2) * np.pi ** (-1)) ** (1 / 2) * (x ** 2 - y ** 2) * z,
        (3 / 4) * (35 * np.pi ** (-1)) ** (1 / 2) * (x ** 3 - 3 * x * y ** 2)
    ]

    dy = [
        2 * np.pi ** (-1 / 2) * y * (x ** 2 + y ** 2 + z ** 2),
        (3 / 4) * (35 * np.pi ** (-1)) ** (1 / 2) * x * (x ** 2 - 3 * y ** 2),
        (9 / 4) * ((35 / 2) * np.pi ** (-1)) ** (1 / 2) * (x ** 2 - y ** 2) * z,
        (-3 / 4) * (5 * np.pi ** (-1)) ** (1 / 2) * x * (x ** 2 + 3 * y ** 2 - 6 * z ** 2),
        (3 / 4) * ((5 / 2) * np.pi ** (-1)) ** (1 / 2) * z * (-3 * x ** 2 - 9 * y ** 2 + 4 * z ** 2),
        (9 / 4) * np.pi ** (-1 / 2) * y * (x ** 2 + y ** 2 - 4 * z ** 2),
        (-9 / 2) * ((5 / 2) * np.pi ** (-1)) ** (1 / 2) * x * y * z,
        (3 / 2) * (5 * np.pi ** (-1)) ** (1 / 2) * y * (y ** 2 - 3 * z ** 2),
        (-9 / 2) * ((35 / 2) * np.pi ** (-1)) ** (1 / 2) * x * y * z,
        (-3 / 4) * (35 * np.pi ** (-1)) ** (1 / 2) * (3 * x ** 2 * y - y ** 3)
    ]

    dz = [
        2 * np.pi ** (-1 / 2) * z * (x ** 2 + y ** 2 + z ** 2),
        np.zeros([n, 1]),
        (-3 / 4) * ((35 / 2) * np.pi ** (-1)) ** (1 / 2) * y * (-3 * x ** 2 + y ** 2),
        9 * (5 * np.pi ** (-1)) ** (1 / 2) * x * y * z,
        (-9 / 4) * ((5 / 2) * np.pi ** (-1)) ** (1 / 2) * y * (x ** 2 + y ** 2 - 4 * z ** 2),
        np.pi ** (-1 / 2) * ((-9) * x ** 2 * z - 9 * y ** 2 * z + 6 * z ** 3),
        (-9 / 4) * ((5 / 2) * np.pi ** (-1)) ** (1 / 2) * x * (x ** 2 + y ** 2 - 4 * z ** 2),
        (9 / 2) * (5 * np.pi ** (-1)) ** (1 / 2) * (x ** 2 - y ** 2) * z,
        (3 / 4) * ((35 / 2) * np.pi ** (-1)) ** (1 / 2) * x * (x ** 2 - 3 * y ** 2),
        np.zeros([n, 1])
    ]

    dx = np.stack(dx, axis=1).squeeze()
    dy = np.stack(dy, axis=1).squeeze()
    dz = np.stack(dz, axis=1).squeeze()

    grad = np.stack((np.sum(q * dx, axis=1), np.sum(q * dy, axis=1), np.sum(q * dz, axis=1)), axis=1)

    return grad


def Octa2Frames(q):
    n = q.shape[1]

    q = np.vstack([(np.sqrt(189) / 4) * np.ones((1, n)), q / np.linalg.norm(q, ord=2, axis=0)])

    v1 = np.random.randn(n, 3)
    v2 = np.random.randn(n, 3)
    delta = 1
    eps = np.finfo(float).eps

    while delta > np.power(eps, 0.9) * np.sqrt(n):
        x = v1[:, 0][:, np.newaxis]
        y = v1[:, 1][:, np.newaxis]
        z = v1[:, 2][:, np.newaxis]

        w1 = OctaTensorGradient(q, x, y, z)

        x = v2[:, 0][:, np.newaxis]
        y = v2[:, 1][:, np.newaxis]
        z = v2[:, 2][:, np.newaxis]

        w2 = OctaTensorGradient(q, x, y, z)

        v1 = w1 / np.linalg.norm(w1, ord=2, axis=1, keepdims=True)
        v2 = w2 - np.sum(w2 * v1, axis=1, keepdims=True) * v1
        v2 = v2 / np.linalg.norm(v2, ord=2, axis=1, keepdims=True)
        delta = np.linalg.norm(v1 - v1, 'fro')

    v3 = np.cross(v1, v2)

    frames = np.stack((v1.T, v2.T, v3.T), axis=1)

    return frames


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


def shortest_rotation(R1, R2):
    return R1.T @ R2


def interpolate_octahedral_field(constField3D, constBoundFaces, faces, tets, FT, boundFaces,
                                 boundNormals, boundBasisX, boundBasisY):
    # creating the constraints on the boundary tets
    vc = np.sqrt(5 / 12) * np.array([1, 0, 0, 0, 0, 0, 0, 0, 1]).T
    vs = np.sqrt(5 / 12) * np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]).T
    vn = np.sqrt(7 / 12) * np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]).T

    # constructing boundary tet function
    boundTets = FT[boundFaces, 0]
    v0 = np.zeros([len(boundTets), 1])
    Hc = np.zeros([9 * len(boundTets), len(boundTets)])
    Hs = np.zeros([9 * len(boundTets), len(boundTets)])
    for constIndex in range(0, len(boundTets)):
        Rn = shortest_rotation(np.identity(3), np.array([boundBasisX[constIndex, :],
                                                         boundBasisY[constIndex, :],
                                                         boundNormals[constIndex, :]]).T)

        WRn = wigner_matrix(Rn, 4)
        v0[9 * constIndex:9 * constIndex + 9] = WRn @ vn.reshape(-1, 1)
        Hc[9 * constIndex:9 * constIndex + 9, boundTets[constIndex]] = WRn @ vc.reshape(-1, 1)
        Hs[9 * constIndex:9 * constIndex + 9, boundTets[constIndex]] = WRn @ vs.reshape(-1, 1)

    # getting the values for the constrained faces
    cConst = np.zeros([len(constBoundFaces), 1])
    sConst = np.zeros([len(constBoundFaces), 1])
    for constIndex in range(0, len(constBoundFaces)):
        cConst[constIndex] = np.dot(constField3D[constIndex, :], boundBasisX[constBoundFaces[constIndex], :])
        sConst[constIndex] = np.dot(constField3D[constIndex, :], boundBasisY[constBoundFaces[constIndex], :])

    # Abstracting constraints
    varBoundFaces = np.setdiff1d(range(0, len(boundTets)), constBoundFaces)
    HcVar = Hc[:, varBoundFaces]
    HcConst = Hc[:, constBoundFaces]
    HsVar = Hs[:, varBoundFaces]
    HsConst = Hs[:, constBoundFaces]
    A = np.column_stack([HcVar, HsVar])
    b = v0 + HcConst * cConst + HsConst * sConst

    # Constructing the differential operator for the dirichlet energy
    innerFaces = np.setdiff1d(range(0, faces.shape[0]), boundFaces)
    rows = np.column_stack((np.arange(0, innerFaces.shape[0]), np.arange(0, innerFaces.shape[0])))
    cols = FT[innerFaces, [0, 2]]
    values = np.full((innerFaces.shape[0], 2), [-1, 1])
    d2 = coo_matrix((values.flatten(), (rows.flatten(), cols.flatten())), shape=(innerFaces.shape[0], tets.shape[0]))

    innerTets = np.setdiff1d(range(0, tets.shape[0]), boundTets)
    d2Inner = d2[:, innerTets]
    d2Bound = d2[:, boundTets]
    E = np.column_stack([d2Inner, d2Bound @ A]);
    f = d2Bound @ b
    x = spsolve(E.T @ E, -E.T @ f)
    uInner = x[0:9 * len(innerTets)];
    cs = x[9 * len(innerTets):]
    uBound = A @ cs + b

    SPHField = np.zeros([tets.shape[0], 9])
    SPHField[innerTets, :] = uInner.reshape(len(innerTets), 9)
    SPHField[boundTets, :] = uBound.reshape(len(boundTets), 9)

    return SPHField


def SPH_to_euclidean(SPHField):
    extField = np.zeros([SPHField.shape[0], 18])
    return extField


if __name__ == '__main__':
    ps.init()

    vertices, boundFaces, tets = load_tet_mesh(os.path.join('..', 'data', 'bone_80k.mesh'))

    halffaces, faces, faceBoundMask, boundVertices, boundFaces, FH, FT = compute_topology(vertices, tets)

    boundNormals, boundFaceAreas, boundBasisX, boundBasisY = compute_face_quantities(vertices, boundFaces)

    q = np.random.randn(9, 10)
    R = Octa2Frames(q)

    # N = 4
    constBoundFaces = [1, 1000, 2000, 3000]
    constField3D = vertices[boundFaces[constBoundFaces, 2], :] - vertices[boundFaces[constBoundFaces, 1], :]
    constField3D /= np.linalg.norm(constField3D, axis=1, keepdims=True)

    SPHField = interpolate_octahedral_field(constField3D, constBoundFaces, faces, tets, FT, boundFaces,
                                            boundNormals, boundBasisX, boundBasisY)

    extField = SPH_to_euclidean(SPHField)

    ps_mesh = ps.register_volume_mesh("Tet Mesh", vertices, tets)

    for i in range(0, 6):
        ps_mesh.add_vector_quantity("field" + str(i), extField[:, 3 * i:3 * i + 3], defined_on='tets')

    # singVertices, singIndices = get_singularities(4, crossField, vertices, faces, edges, basisX, basisY)

    # ps_cloud = ps.register_point_cloud("Singularities", vertices[singVertices, :])
    # ps_cloud.add_scalar_quantity("Indices", singIndices.flatten(), vminmax=(-N, N), enabled=True)

    ps.show()

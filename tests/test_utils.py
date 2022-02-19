from materia.utils import (
    divisors,
    expand,
    lcm,
    linearly_independent,
    nearest_points,
    nontrivial_vector,
    normalize,
    orthogonal_decomp,
    periodicity,
    perpendicular_vector,
    reflection_matrix,
    rotation_matrix,
    sample_spherical_lune,
    sample_spherical_triangle,
    tetrahedron_volume,
)
import numpy as np
import unittest


class TestDivisors(unittest.TestCase):
    def test_divisors_1(self):
        self.assertEqual(divisors(1), [1])

    def test_divisors_primes(self):
        self.assertEqual(divisors(3), [1, 3])
        self.assertEqual(divisors(17), [1, 17])
        self.assertEqual(divisors(2099), [1, 2099])

    def test_divisors_composites(self):
        self.assertEqual(divisors(4), [1, 2, 4])
        self.assertEqual(divisors(60), [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60])
        self.assertEqual(
            divisors(2000),
            [
                1,
                2,
                4,
                5,
                8,
                10,
                16,
                20,
                25,
                40,
                50,
                80,
                100,
                125,
                200,
                250,
                400,
                500,
                1000,
                2000,
            ],
        )


class TestExpand(unittest.TestCase):
    def test_expand_trivial(self):
        self.assertEqual(
            expand("/home/user/whatever/whatever.png"),
            "/home/user/whatever/whatever.png",
        )

    def test_expand_trivial_dir(self):
        self.assertEqual(
            expand("user/whatever/whatever.png", dir="/home"),
            "/home/user/whatever/whatever.png",
        )


class TestLCM(unittest.TestCase):
    def test_lcm(self):
        self.assertEqual(lcm([1, 3]), 3)
        self.assertEqual(lcm([1, 3, 5]), 15)
        self.assertEqual(lcm([4, 9, 2, 6, 15]), 180)
        self.assertEqual(lcm([4, 9, 2, 9, 3, 15]), 180)


class TestLinearlyIndependent(unittest.TestCase):
    def test_three_indep_5D(self):
        X = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).T
        self.assertTrue(np.allclose(linearly_independent(X), X))

    def test_three_dep_5D(self):
        X = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ).T
        self.assertTrue(np.allclose(linearly_independent(X), X[:, 0:1]))

    def test_six_vecs_5D_one_known(self):
        X = np.array(
            [
                [0.0, 1.0, 0.0, 2.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ).T
        Y = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]).T
        self.assertTrue(
            np.allclose(
                linearly_independent(X, Y),
                np.vstack([Y[:, 0], X[:, 0], X[:, 1], X[:, 4], X[:, 5]]).T,
            )
        )


class TestNearestPoints(unittest.TestCase):
    def test_nearest_points(self):
        points = np.array(
            [[1, 0, 0], [0, 0, 1], [0, 0, 2], [0.5, 0.5, 0.5], [10, 3, 4]]
        ).T

        target = np.array([[1, 0, 0], [0.5, 0.5, 0.5]]).tolist()
        result = nearest_points(points, 2).T.tolist()

        for v in target:
            self.assertIn(v, result)

        target = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1]]).tolist()
        result = nearest_points(points, 3).T.tolist()

        for v in target:
            self.assertIn(v, result)

        target = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1], [0, 0, 2]]).tolist()
        result = nearest_points(points, 4).T.tolist()

        for v in target:
            self.assertIn(v, result)

        target = points.T.tolist()
        result = nearest_points(points, 5).T.tolist()

        for v in target:
            self.assertIn(v, result)

        points = np.array(
            [[0.5, 0, 0], [-0.5, 0, 0], [0, 0, 0], [0, 0, 2], [1, 0, 2], [10, 3, 4]]
        ).T

        target = np.array([[0.5, 0, 0], [0, 0, 0]]).tolist()
        result = nearest_points(points, 2).T.tolist()

        for v in target:
            self.assertIn(v, result)

        target = np.array([[0.5, 0, 0], [0, 0, 0], [-0.5, 0, 0]]).tolist()
        result = nearest_points(points, 3).T.tolist()

        for v in target:
            self.assertIn(v, result)

        target = np.array([[0.5, 0, 0], [0, 0, 0], [-0.5, 0, 0], [0, 0, 2]]).tolist()
        result = nearest_points(points, 4).T.tolist()

        for v in target:
            self.assertIn(v, result)

        target = np.array(
            [[0.5, 0, 0], [0, 0, 0], [-0.5, 0, 0], [0, 0, 2], [1, 0, 2]]
        ).tolist()
        result = nearest_points(points, 5).T.tolist()

        for v in target:
            self.assertIn(v, result)

        target = points.T.tolist()
        result = nearest_points(points, 6).T.tolist()

        for v in target:
            self.assertIn(v, result)


class TestNontrivialVectors(unittest.TestCase):
    def test_nontrivial_vectors(self):
        yz_swap = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        xy_plane_reflection = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

        self.assertIsNone(nontrivial_vector(np.eye(3)))
        self.assertIsNone(nontrivial_vector(-np.eye(3)))
        self.assertIsNone(nontrivial_vector(np.zeros((3, 3))))

        v = nontrivial_vector(R=yz_swap, seed=42809437)
        self.assertFalse(np.allclose(yz_swap @ v, v))

        v = nontrivial_vector(R=xy_plane_reflection, seed=42809437)
        self.assertFalse(np.allclose(xy_plane_reflection @ v, v))

        v = nontrivial_vector(R=yz_swap, seed=139856018)
        self.assertFalse(np.allclose(yz_swap @ v, v))

        v = nontrivial_vector(R=xy_plane_reflection, seed=139856018)
        self.assertFalse(np.allclose(xy_plane_reflection @ v, v))


class TestNormalize(unittest.TestCase):
    def test_normalize(self):
        a = np.array([1, 0, 0])
        self.assertTrue(np.allclose(normalize(a), a))

        a = np.array([0, 1, 0])
        self.assertTrue(np.allclose(normalize(a), a))

        a = np.array([0, 0, 1])
        self.assertTrue(np.allclose(normalize(a), a))

        a = np.array([1, 1, 1])
        b = a / np.sqrt(3)
        self.assertTrue(np.allclose(normalize(a), b))

        a = np.array([1, 2, -1])
        b = a / np.sqrt(6)
        self.assertTrue(np.allclose(normalize(a), b))

        a = np.array([0, 0, 0])
        self.assertTrue(np.allclose(normalize(a), a))


class TestOrthogonalDecomp(unittest.TestCase):
    def test_simple(self):
        u = np.array([[0.0, 0.0, 1.0]]).T
        v = np.array([[1.0, 2.0, 3.0]]).T

        proj, comp = orthogonal_decomp(v, u)
        self.assertTrue(np.allclose(proj, np.array([[0.0, 0.0, 3.0]]).T))
        self.assertTrue(np.allclose(comp, np.array([[1.0, 2.0, 0.0]]).T))


class TestPeriodicity(unittest.TestCase):
    def test_identity(self):
        X = np.eye(3)
        n = periodicity(X)

        self.assertEqual(n, 1)

    def test_rotation_180_deg(self):
        X = np.array(
            [
                [np.cos(np.pi), -np.sin(np.pi), 0],
                [np.sin(np.pi), np.cos(np.pi), 0],
                [0, 0, 1],
            ]
        )
        n = periodicity(X)

        self.assertEqual(n, 2)

    def test_rotation_60_deg(self):
        X = np.array(
            [
                [np.cos(np.pi / 3), -np.sin(np.pi / 3), 0],
                [np.sin(np.pi / 3), np.cos(np.pi / 3), 0],
                [0, 0, 1],
            ]
        )
        n = periodicity(X)

        self.assertEqual(n, 6)


class TestPerpendicularVector(unittest.TestCase):
    def test_perpendicular_vector_one_arg(self):
        a = np.array([[1, 0, 0]]).T
        c = np.array([[0, 1, 0]]).T
        self.assertTrue(np.allclose(perpendicular_vector(a), c))

    def test_perpendicular_vector_two_args(self):
        a = np.array([[1, 0, 0]]).T
        b = np.array([[0, 2, 0]]).T
        c = np.array([[0, 0, 1]]).T
        self.assertTrue(np.allclose(perpendicular_vector(a, b), c))


class TestReflectionMatrix(unittest.TestCase):
    def test_reflection_matrix(self):
        a = np.array([1, 0, 0]).reshape((3, 1))
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(reflection_matrix(a), R))

        a = np.array([0, 1, 0]).reshape((3, 1))
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.assertTrue(np.allclose(reflection_matrix(a), R))

        a = np.array([0, 0, 1]).reshape((3, 1))
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.assertTrue(np.allclose(reflection_matrix(a), R))

        a = np.array([1, 1, 1]).reshape((3, 1))
        R = np.array([[1, -2, -2], [-2, 1, -2], [-2, -2, 1]]) / 3
        self.assertTrue(np.allclose(reflection_matrix(a), R))

        a = np.array([1, 2, -1]).reshape((3, 1))
        R = np.array([[2, -2, 1], [-2, -1, 2], [1, 2, 2]]) / 3
        self.assertTrue(np.allclose(reflection_matrix(a), R))


class TestRotationMatrix(unittest.TestCase):
    def test_rotation_matrix_source_target(self):
        m = np.array([[1.0, 0.0, 0.0]]).T
        n = np.array([[1.0, 0.0, 0.0]]).T
        R = np.eye(3)
        self.assertTrue(
            np.allclose(
                rotation_matrix(m=m, n=n),
                R,
            )
        )

        m = np.array([[1.0, 0.0, 0.0]]).T
        n = np.array([[0.0, 1.0, 0.0]]).T
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(
            np.allclose(
                rotation_matrix(m=m, n=n),
                R,
            )
        )

    def test_rotation_matrix_source_target_improper(self):
        m = np.array([[1.0, 0.0, 0.0]]).T
        n = np.array([[1.0, 0.0, 0.0]]).T
        R = np.eye(3)
        self.assertTrue(
            np.allclose(
                rotation_matrix(m=m, n=n, improper=True),
                R,
            )
        )

        m = np.array([[1.0, 0.0, 0.0]]).T
        n = np.array([[0.0, 1.0, 0.0]]).T
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        self.assertTrue(
            np.allclose(
                rotation_matrix(m=m, n=n, improper=True),
                R,
            )
        )

    def test_rotation_matrix_axis_angle(self):
        axis = np.array([[1.0, 0.0, 0.0]]).T
        theta = np.pi / 3
        R = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        self.assertTrue(
            np.allclose(
                rotation_matrix(axis, theta),
                R,
            )
        )

    def test_rotation_matrix_axis_angle_improper(self):
        axis = np.array([[1.0, 0.0, 0.0]]).T
        theta = np.pi / 3
        R = np.array(
            [
                [-1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        self.assertTrue(
            np.allclose(
                rotation_matrix(axis, theta, improper=True),
                R,
            )
        )

    def test_value_error(self):
        with self.assertRaises(ValueError):
            rotation_matrix()
        with self.assertRaises(ValueError):
            rotation_matrix(theta=np.pi / 2)
        with self.assertRaises(ValueError):
            m = np.array([[1.0, 0.0, 0.0]]).T
            rotation_matrix(theta=np.pi / 2, m=m)


class TestSampleSphericalLune(unittest.TestCase):
    def test_quarter(self):
        n1 = np.array([[0.0, 0.0, 1.0]]).T
        n2 = np.array(
            [
                [
                    0.0,
                    1.0,
                    0.0,
                ]
            ]
        ).T
        x, y, z = sample_spherical_lune(n1, n2).squeeze()
        self.assertTrue(np.allclose(np.linalg.norm([x, y, z]), 1))
        self.assertTrue(x >= -1 and x <= 1)
        self.assertTrue(y >= 0 and y <= 1)
        self.assertTrue(z >= 0 and z <= 1)

    def test_30_deg(self):
        n1 = np.array([[0.0, 0.0, 1.0]]).T
        n2 = np.array(
            [
                [
                    -np.sin(np.pi / 6),
                    0.0,
                    np.cos(np.pi / 6),
                ]
            ]
        ).T
        x, y, z = sample_spherical_lune(n1, n2).squeeze()
        self.assertTrue(np.allclose(np.linalg.norm([x, y, z]), 1))
        self.assertTrue(x >= 0 and x <= 1)
        self.assertTrue(y >= -1 and y <= 1)
        self.assertTrue(z >= 0 and z <= np.sin(np.pi / 6))


class TestSampleSphericalTriangle(unittest.TestCase):
    def test_octant(self):
        A = np.array([[1.0, 0.0, 0.0]]).T
        B = np.array([[0.0, 1.0, 0.0]]).T
        C = np.array([[0.0, 0.0, 1.0]]).T
        sin_alpha = sin_beta = sin_gamma = 1
        x, y, z = sample_spherical_triangle(
            A, B, C, sin_alpha, sin_beta, sin_gamma
        ).squeeze()
        self.assertTrue(np.allclose(np.linalg.norm([x, y, z]), 1))
        self.assertTrue(x >= 0)
        self.assertTrue(y >= 0)
        self.assertTrue(z >= 0)


class TestTetrahedronVolume(unittest.TestCase):
    def test_regular(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        ).T
        self.assertEqual(tetrahedron_volume(vertices), 1 / 3)

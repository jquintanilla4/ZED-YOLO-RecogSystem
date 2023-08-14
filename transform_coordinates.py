import numpy as np

# Hard-coded transformation matrix based on previous calculations
transformation_matrix = [[6.37723734, -55.82804153, -48.21478604, -188.81357681],
[-8.71952636, 83.04295145, 68.10584278, 270.61119858],
[0., 0., 0., 0.]]  # noqa: E122, E201

transformation_matrix = np.vstack([transformation_matrix, [0, 0, 0, 1]])


def transform_coord(x: float, y: float, z: float):
    # Use the pre-calculated transformation matrix to transform the given point
    transformed_point = transformation_matrix @ np.array([x, y, z, 1])
    return transformed_point[:3]


if __name__ == '__main__':
    # test_point = (4.099073886871338, 1.2561070919036865, -4.980190277099609)
    # test_point = (-2.6746127605438232, 0.6071292161941528, -4.972853183746338)
    test_point = (0.06996123492717743, -
                  0.9377466440200806, -2.8210196495056152)
    transformed = transform_coord(*test_point)
    print(transformed)

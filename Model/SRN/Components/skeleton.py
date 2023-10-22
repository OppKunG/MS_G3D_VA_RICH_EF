import numpy as np


def joint_full_body():
    joint = np.asarray(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
        ]
    )
    return joint


def occlusion_list():
    # Skeleton parts [1-25]
    parts = dict()

    # 1. Left arm
    parts[1] = np.array([5, 6, 7, 8, 22, 23]) - 1

    # 2. Right arm
    parts[2] = np.array([9, 10, 11, 12, 24, 25]) - 1

    # 3. Two hands
    parts[3] = np.array([22, 23, 24, 25]) - 1

    # 4. Two legs
    parts[4] = np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1

    # 5. Trunk
    parts[5] = np.array([1, 2, 3, 4, 21]) - 1

    # 6. Left body
    parts[6] = np.array([5, 6, 7, 8, 13, 14, 15, 16, 22, 23]) - 1

    # 7. Right body
    parts[7] = np.array([9, 10, 11, 12, 17, 18, 19, 20, 24, 25]) - 1

    # 8-25. No occlusion
    parts[8] = np.array([], dtype=np.int32)
    parts[9] = np.array([], dtype=np.int32)
    parts[10] = np.array([], dtype=np.int32)
    parts[11] = np.array([], dtype=np.int32)
    parts[12] = np.array([], dtype=np.int32)
    parts[13] = np.array([], dtype=np.int32)
    parts[14] = np.array([], dtype=np.int32)
    parts[15] = np.array([], dtype=np.int32)
    parts[16] = np.array([], dtype=np.int32)
    parts[17] = np.array([], dtype=np.int32)
    parts[18] = np.array([], dtype=np.int32)
    parts[19] = np.array([], dtype=np.int32)
    parts[20] = np.array([], dtype=np.int32)
    parts[21] = np.array([], dtype=np.int32)
    parts[22] = np.array([], dtype=np.int32)
    parts[23] = np.array([], dtype=np.int32)
    parts[24] = np.array([], dtype=np.int32)

    return parts


def joint_list():
    joint_list = (
        np.array(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
            ]
        )
        - 1
    )

    return joint_list

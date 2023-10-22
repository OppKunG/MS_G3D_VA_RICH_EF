import numpy as np


class Graph():
    def __init__(self, max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.center = self._get_non_occluded_edge()
        self.num_node1, self.edge1, self.center1 = self._get_left_arm_occluded_edge()
        self.num_node2, self.edge2, self.center2 = self._get_right_arm_occluded_edge()
        self.num_node3, self.edge3, self.center3 = self._get_two_hands_occluded_edge()
        self.num_node4, self.edge4, self.center4 = self._get_two_legs_occluded_edge()
        self.num_node5, self.edge5, self.center5 = self._get_trunk_occluded_edge()
        self.num_node6, self.edge6, self.center6 = self._get_left_body_occluded_edge()
        self.num_node7, self.edge7, self.center7 = self._get_right_body_occluded_edge()
        self.num_node8, self.edge8, self.center8 = self._get_two_arms_occluded_edge()
        self.num_node9, self.edge9, self.center9 = self._get_right_arm_left_hands_occluded_edge()
        self.num_node10, self.edge10, self.center10 = self._get_two_legs_trunk_occluded_edge()
        self.num_node11, self.edge11, self.center11 = self._get_two_arms_trunk_occluded_edge()

        # get adjacency matrix
        self.hop_dis = self._get_hop_distance()
        self.hop_dis1 = self._get_hop_distance1()
        self.hop_dis2 = self._get_hop_distance2()
        self.hop_dis3 = self._get_hop_distance3()
        self.hop_dis4 = self._get_hop_distance4()
        self.hop_dis5 = self._get_hop_distance5()
        self.hop_dis6 = self._get_hop_distance6()
        self.hop_dis7 = self._get_hop_distance7()
        self.hop_dis8 = self._get_hop_distance8()
        self.hop_dis9 = self._get_hop_distance9()
        self.hop_dis10 = self._get_hop_distance10()
        self.hop_dis11 = self._get_hop_distance11()

        # normalization
        self.A = self._get_non_occluded_adjacency()
        self.A1 = self._get_left_arm_occluded_adjacency()
        self.A2 = self._get_right_arm_occluded_adjacency()
        self.A3 = self._get_two_hands_occluded_adjacency()
        self.A4 = self._get_two_legs_occluded_adjacency()
        self.A5 = self._get_trunk_occluded_adjacency()
        self.A6 = self._get_left_body_occluded_adjacency()
        self.A7 = self._get_right_body_occluded_adjacency()
        self.A8 = self._get_two_arms_occluded_adjacency()
        self.A9 = self._get_right_arm_left_hand_occluded_adjacency()
        self.A10 = self._get_two_legs_trunk_occluded_adjacency()
        self.A11 = self._get_two_arms_trunk_occluded_adjacency()
        # self.random_A = np.load('/home/antony/RA-GCNv2/src/train_occluded_5_A.npy')  # (900, 25, 25)

    def __str__(self):
        return self.A, self.A1, self.A2, self.A3, self.A4, self.A5, self.A6, self.A7, self.A8, self.A9, self.A10, self.random_A

    def _get_non_occluded_edge(self):
        num_node = 25
        # none occluded
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                          (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                          (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                          (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                          (22, 23), (23, 8), (24, 25), (25, 12)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_left_arm_occluded_edge(self):
        num_node = 25

        # left arm occluded
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3),
                          (9, 21), (10, 9), (11, 10), (12, 11), (24, 25), (25, 12),
                          (13, 1), (14, 13), (15, 14), (16, 15),
                          (17, 1), (18, 17), (19, 18), (20, 19)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_right_arm_occluded_edge(self):
        num_node = 25

        # right arm occluded
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3),
                          (5, 21), (6, 5), (7, 6), (8, 7), (22, 23), (23, 8),
                          (13, 1), (14, 13), (15, 14), (16, 15),
                          (17, 1), (18, 17), (19, 18), (20, 19)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_two_hands_occluded_edge(self):
        num_node = 25

        # two hands occluded
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                          (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                          (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                          (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_two_legs_occluded_edge(self):
        num_node = 25

        # two legs occluded
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3),
                          (5, 21), (6, 5), (7, 6), (8, 7), (22, 23), (23, 8),
                          (9, 21), (10, 9), (11, 10), (12, 11), (24, 25), (25, 12)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_trunk_occluded_edge(self):
        num_node = 25

        # trunk occluded
        neighbor_1base = [(6, 5), (7, 6), (8, 7), (22, 23), (23, 8),
                          (10, 9), (11, 10), (12, 11), (24, 25), (25, 12),
                          (14, 13), (15, 14), (16, 15),
                          (18, 17), (19, 18), (20, 19)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_left_body_occluded_edge(self):
        num_node = 25
        # none occluded
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3),
                          (9, 21), (10, 9), (11, 10), (12, 11), (24, 25), (25, 12),
                          (17, 1), (18, 17), (19, 18), (20, 19)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_right_body_occluded_edge(self):
        num_node = 25
        # none occluded
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3),
                          (5, 21), (6, 5), (7, 6), (8, 7), (22, 23), (23, 8),
                          (13, 1), (14, 13), (15, 14), (16, 15)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_two_arms_occluded_edge(self):
        num_node = 25
        # none occluded
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3),
                          (13, 1), (14, 13), (15, 14), (16, 15),
                          (17, 1), (18, 17), (19, 18), (20, 19)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_right_arm_left_hands_occluded_edge(self):
        num_node = 25
        # none occluded
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3),
                          (5, 21), (6, 5), (7, 6), (8, 7),
                          (13, 1), (14, 13), (15, 14), (16, 15),
                          (17, 1), (18, 17), (19, 18), (20, 19)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_two_legs_trunk_occluded_edge(self):
        num_node = 25

        # two legs occluded
        neighbor_1base = [(6, 5), (7, 6), (8, 7), (22, 23), (23, 8),
                          (9, 21), (10, 9), (11, 10), (12, 11), (24, 25), (25, 12)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)

    def _get_two_arms_trunk_occluded_edge(self):
        num_node = 25
        # none occluded
        neighbor_1base = [(13, 1), (14, 13), (15, 14), (16, 15),
                          (17, 1), (18, 17), (19, 18), (20, 19)]

        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        # print(edge)
        center = 21 - 1
        return (num_node, edge, center)


    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance1(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge1:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance2(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge2:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance3(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge3:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance4(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge4:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance5(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge5:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance6(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge6:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance7(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge7:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance8(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge8:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance9(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge9:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance10(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge10:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_hop_distance11(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge11:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis


    def _get_non_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        return A

    def _get_left_arm_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis1 == hop] = 1
        # left arm occluded
        adjacency[4, 4] = 0
        adjacency[5, 5] = 0
        adjacency[6, 6] = 0
        adjacency[7, 7] = 0
        adjacency[21, 21] = 0
        adjacency[22, 22] = 0
        # print('a', adjacency)
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis1 == hop] = normalize_adjacency[self.hop_dis1 == hop]
        return A

    def _get_right_arm_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis2 == hop] = 1

        # right arm occluded
        adjacency[8, 8] = 0
        adjacency[9, 9] = 0
        adjacency[10, 10] = 0
        adjacency[11, 11] = 0
        adjacency[23, 23] = 0
        adjacency[24, 24] = 0

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis2 == hop] = normalize_adjacency[self.hop_dis2 == hop]
        return A

    def _get_two_hands_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis3 == hop] = 1

        # two hands occluded
        adjacency[21, 21] = 0
        adjacency[22, 22] = 0
        adjacency[23, 23] = 0
        adjacency[24, 24] = 0

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis3 == hop] = normalize_adjacency[self.hop_dis3 == hop]
        return A

    def _get_two_legs_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis4 == hop] = 1

        # two legs occluded
        adjacency[12, 12] = 0
        adjacency[13, 13] = 0
        adjacency[14, 14] = 0
        adjacency[15, 15] = 0
        adjacency[16, 16] = 0
        adjacency[17, 17] = 0
        adjacency[18, 18] = 0
        adjacency[19, 19] = 0

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis4 == hop] = normalize_adjacency[self.hop_dis4 == hop]
        return A

    def _get_trunk_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis5 == hop] = 1

        # trunk occluded
        adjacency[0, 0] = 0
        adjacency[1, 1] = 0
        adjacency[2, 2] = 0
        adjacency[3, 3] = 0
        adjacency[20, 20] = 0

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis5 == hop] = normalize_adjacency[self.hop_dis5 == hop]
        return A

    def _get_left_body_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis6 == hop] = 1

        # left_body occluded
        adjacency[4, 4] = 0
        adjacency[5, 5] = 0
        adjacency[6, 6] = 0
        adjacency[7, 7] = 0
        adjacency[12, 12] = 0
        adjacency[13, 13] = 0
        adjacency[14, 14] = 0
        adjacency[15, 15] = 0
        adjacency[21, 21] = 0
        adjacency[22, 22] = 0

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis6 == hop] = normalize_adjacency[self.hop_dis6 == hop]
        return A

    def _get_right_body_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis7 == hop] = 1

        # left_body occluded
        adjacency[8, 8] = 0
        adjacency[9, 9] = 0
        adjacency[10, 10] = 0
        adjacency[11, 11] = 0
        adjacency[16, 16] = 0
        adjacency[17, 17] = 0
        adjacency[18, 18] = 0
        adjacency[19, 19] = 0
        adjacency[23, 23] = 0
        adjacency[24, 24] = 0

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis7 == hop] = normalize_adjacency[self.hop_dis7 == hop]
        return A

    def _get_two_arms_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis8 == hop] = 1

        # two arms occluded
        adjacency[4, 4] = 0
        adjacency[5, 5] = 0
        adjacency[6, 6] = 0
        adjacency[7, 7] = 0
        adjacency[8, 8] = 0
        adjacency[9, 9] = 0
        adjacency[10, 10] = 0
        adjacency[11, 11] = 0
        adjacency[21, 21] = 0
        adjacency[22, 22] = 0
        adjacency[23, 23] = 0
        adjacency[24, 24] = 0

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis8 == hop] = normalize_adjacency[self.hop_dis8 == hop]
        return A

    def _get_right_arm_left_hand_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis9 == hop] = 1

        # right arm occluded
        adjacency[8, 8] = 0
        adjacency[9, 9] = 0
        adjacency[10, 10] = 0
        adjacency[11, 11] = 0
        adjacency[21, 21] = 0
        adjacency[22, 22] = 0
        adjacency[23, 23] = 0
        adjacency[24, 24] = 0

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis9 == hop] = normalize_adjacency[self.hop_dis9 == hop]
        return A

    def _get_two_legs_trunk_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis10 == hop] = 1

        # two legs occluded
        adjacency[0, 0] = 0
        adjacency[1, 1] = 0
        adjacency[2, 2] = 0
        adjacency[3, 3] = 0

        adjacency[12, 12] = 0
        adjacency[13, 13] = 0
        adjacency[14, 14] = 0
        adjacency[15, 15] = 0
        adjacency[16, 16] = 0
        adjacency[17, 17] = 0
        adjacency[18, 18] = 0
        adjacency[19, 19] = 0
        adjacency[20, 20] = 0
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis10 == hop] = normalize_adjacency[self.hop_dis10 == hop]
        return A

    def _get_two_arms_trunk_occluded_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis11 == hop] = 1

        # two arms occluded
        adjacency[0, 0] = 0
        adjacency[1, 1] = 0
        adjacency[2, 2] = 0
        adjacency[3, 3] = 0
        adjacency[20, 20] = 0
        adjacency[4, 4] = 0
        adjacency[5, 5] = 0
        adjacency[6, 6] = 0
        adjacency[7, 7] = 0
        adjacency[8, 8] = 0
        adjacency[9, 9] = 0
        adjacency[10, 10] = 0
        adjacency[11, 11] = 0
        adjacency[21, 21] = 0
        adjacency[22, 22] = 0
        adjacency[23, 23] = 0
        adjacency[24, 24] = 0

        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis11 == hop] = normalize_adjacency[self.hop_dis11 == hop]
        return A


    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD
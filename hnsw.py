import heapq
'''
Heaps are used in HNSW to maintain sets of candidates and nearest neighbors efficiently.
Candidates are stored as a min-heap, allowing quick extraction of the closest candidate.
Nearest neighbors are also stored as a min-heap, facilitating comparison and updates as new neighbors are found.
'''
import numpy as np
import random
import math

class HNSW:
    def __init__(self, max_connections=16, max_layers=2, ef=16, efConstruction=16):
        '''
        efConstruction - The HNSW build parameter that controls the quality of the search at build time.
        maxConnections - The HNSW build parameter controls how many outgoing edges a node can have in the HNSW graph.
        '''
        self.node_id_to_vector = {}
        '''
        dictionary mapping node ids to their respective vectors, e.g.
        {
            0: [0.3, 0.2, ..., 0.6],
            1: [1.8, 0.22, ..., 0.1],
            ...
        }
        '''
        self.graph = {}
        '''
        list of layers, e.g.
        {
            0: {
                node_id: [neighbor1_id, neighbor2_id, ...],
                node_id: [neighbor1_id, neighbor2_id, ...]
            },
            1: ...
        }
        '''
        self.max_connections = max_connections # how many outgoing edges a node can have in the HNSW graph
        self.max_layers = max_layers # max depth of the hierarchy
        self.ef = ef # query time parameter for quality of search
        self.efConstruction = efConstruction # HNSW build parameter controls quality of search at build time
    
    def _distance(self, a, b):
        if type(a) == int: # distance with query_id
            a_vector = np.array(self.node_id_to_vector[a])
        else: # distance with query vector directly
            a_vector = np.array(a)

        if type(b) == int: # distance with query_id
            b_vector = np.array(self.node_id_to_vector[b])
        else: # distance with query vector directly
            b_vector = np.array(b)
        
        return np.sqrt(np.sum((a_vector - b_vector)**2))
    
    def _search_layer(self, q, ep, ef, layer_num):
        '''
        q:              tuple, query vector
        eps:            tuple, entry point (node ids)
        ef:             int, number of nearest to q elements to return
        layer_num:      int, layer number we are searching through (this is a hierarchical graph)
        '''
        visited = {ep} # set of visited elements (v in paper)
        candidates = [(-self._distance(ep, q), ep)] # set of candidates (C in paper)
        '''
        (negative distance between the entry point and query point, query point)

        We store the negative distance because the heapq module returns the smallest element from the heap,
        and we want to get the largest distance instead.

        So e.g. ep = (1,1) and q = (2,2) --> with L2 distance, we store (-sqrt(2), (1,1))
                ep = (6,6) and q = (2,2) --> with L2 distance, we store (-sqrt(32), (6,6))

        heapq.heappop(candidates) --> returns (-sqrt(32), (6,6)) because -sqrt(32) < -sqrt(2)
        '''
        dynamic_nearest_neighbors_list = [(-float('inf'), ep)] # dynamic list of `found` nearest neighbors (W in paper)
        '''
        e.g. [(-inf, (1,1)), (-inf, (6,6))]

        During the search process, as new candidate points are considered,
        -- The algorithm checks if the new point is closer than the current furtherest point in the dynamic nearest neighbor list
        -- If the candidate is closer (or len(list) < ef) the candidate is added to the list
        -- If len(list) > ef, the furthest point is removed
        -- The list always contains the closest points found so far during the search 

        So e.g. we compare (1,1) and update the list to (-sqrt(2), (1,1))
        ... then when we call heapq.heapop --> (-inf, (6,6)) will be returned
        '''
        while len(candidates) > 0:
            nearest_distance_to_query, nearest_neighbor = heapq.heappop(candidates)
            furthest_distance_to_query, furthest_neighbor = heapq.heappop(dynamic_nearest_neighbors_list) 
            heapq.heappush(dynamic_nearest_neighbors_list, (furthest_distance_to_query, furthest_neighbor))

            if -nearest_distance_to_query > -furthest_distance_to_query:
                break # all elements in the dynamic list of nearest neighbors have been evaluated

            for e in self.graph[layer_num][nearest_neighbor]:
                if e not in visited:
                    visited.add(e)
                    distance_e = self._distance(e, q)
                    furthest_distance_to_query, furthest_neighbor = heapq.heappop(dynamic_nearest_neighbors_list)
                    heapq.heappush(dynamic_nearest_neighbors_list, (furthest_distance_to_query, furthest_neighbor))
                    
                    if distance_e < -(furthest_distance_to_query) or len(dynamic_nearest_neighbors_list) < ef:
                        heapq.heappush(candidates, (-distance_e, e))
                        heapq.heappush(dynamic_nearest_neighbors_list, (-distance_e, e))
                        if len(dynamic_nearest_neighbors_list) > ef:
                            heapq.heappop(dynamic_nearest_neighbors_list)
            
        return [neighbor for dist, neighbor in sorted(dynamic_nearest_neighbors_list)]

    def _insert_node(self, q_id, q, M, M_max, efConstruction):
        '''
        q_id:           int     -- new node id when inserting query into graph
        q:              []int   -- query vector
        M:              int     -- number of established connections
        M[max]:         int     -- max number of connections for each element per layer
        efConstruction: int     -- size of the dynamic candidate list
        '''
        self.node_id_to_vector[q_id] = q # always going to be adding this node to the graph

        if not self.graph: # case when inserting first node into the graph
            self.graph[0] = {q_id: []}
            return
        
        nearest_neighbors = [] # list for the currently found nearest elements
        ep, L = self._get_entry_point() # get entry point and layer of the entry point to HNSW graph
        mL = 1 / math.log(M)
        l = math.floor(-math.log(random.uniform(0, 1)) * mL)
        print(l)

        # If the new node's level (l) is higher than the current highest layer (L),
        for i in range(L+1, l+1):
            self.graph[i] = {q_id: []}

        # Find the new entry point for layers L down to l+1
        for lc in range(L, l + 1, -1):
            nearest_neighbors = self._search_layer(q_id, ep, efConstruction, lc)
            ep = nearest_neighbors[0] # check this

        # insert the new element and update connections for layers min(L, 1) down to 0
        for lc in range(min(L, l), -1, -1):
            nearest_neighbors = self._search_layer(q_id, ep, efConstruction, lc)
            neighbors_to_add = self._select_neighbors_heuristic(q_id, nearest_neighbors, M, lc)

            # add bidrectional connections from neighbors to q at layer lc
            self._add_bidirectional_connections(q_id, neighbors_to_add, lc)

            # shrink connections if needed
            for e in neighbors_to_add:
                e_conn = self.graph[lc][e]
                if len(e_conn) > M_max:
                    e_new_conn = self._select_neighbors_heuristic(e, e_conn, M_max, lc)
                    self.graph[lc][e] = e_new_conn
            
            ep = nearest_neighbors[0] if nearest_neighbors else ep # check this
        
        # update the entry point if necessary
        if l > L:
            self.graph[l] = {q_id: []}

    def _add_bidirectional_connections(self, q_id, neighbors_to_add, lc):
        # Add q as a neighbor to each of the nodes in neighbors_to_add at level lc
        for neighbor in neighbors_to_add:
            if neighbor not in self.graph[lc]:
                self.graph[lc][neighbor] = []
            self.graph[lc][neighbor].append(q_id)

        # Add each neighbor in neighbors_to_add to q's connections at level lc
        if q_id not in self.graph[lc]:
            self.graph[lc][q_id] = []
        self.graph[lc][q_id].extend(neighbors_to_add)

    def _get_entry_point(self):
        top_layer = max(self.graph.keys())
        entry_point = random.choice(list(self.graph[top_layer].keys()))
        return entry_point, top_layer

    def _select_neighbors_simple(self, q_id, C, M):
        '''
        q_id:      int,             query id
        C:         list of tuples,  candidate elements
        M:         int,             number of neighbors to return
        '''
        # Compute the distance between q and each candidate element in C
        distances = [(self._distance(q_id, candidate), candidate) for candidate in C]

        # Sort the candidate elements by distance to q (ascending order)
        distances.sort()

        # Select the M nearest elements to q
        M_nearest_elements = [candidate for dist, candidate in distances[:M]]

        return M_nearest_elements

    def _select_neighbors_heuristic(self, q_id, C, M, lc, extendCandidates=False, keepPrunedConnections=False):
        '''
        q_id:                      int,            query vector id
        C:                      list of np.arrays, candidate elements
        M:                      int, number of neighbors to return
        lc:                     int, layer number
        extendCandidates:       bool, flag indicating whether or not to extend candidate list
        keepPrunedConnections:  bool, flag indicating whether or not to add discarded elements
        '''

        R = []  # result set
        W = list(C)  # working queue for the candidates

        # Extend candidates by their neighbors
        if extendCandidates:
            for e in C:
                for e_adj in self.graph[lc].get(tuple(e), []):
                    if e_adj not in W:
                        W.append(e_adj)

        W_discarded = []  # queue for the discarded candidates

        while W and len(R) < M:
            # Fidn the point in the working list with the smallest distance
            e = min(W, key=lambda x: self._distance(q_id, x))
            # remvoe the point as we are about to explore it
            W.remove(e)

            # If the result list R is empty OR
            # If the distance between the query and e is less than 
            if len(R) == 0:
                R.append(e)
            else:
                flag = True
                for candidate in R:
                    if (self._distance(q_id, e) > self._distance(q_id, candidate)):
                        flag = False
                        break

                if(flag):
                    R.append(e)
                else:
                    W_discarded.append(e)

        # Add some of the discarded connections from W_discarded
        if keepPrunedConnections:
            while len(W_discarded) > 0 and len(R) < M:
                e = min(W_discarded, key=lambda x: self._distance(q_id, x))
                W_discarded.remove(e)
                R.append(e)

        return R

    def _pretty_print_graph(self):
        """
        Prints the HNSW graph in a human-readable format.
        """
        print("\nHNSW Graph")
        print("----------")
        for layer in sorted(self.graph.keys()):
            print(f"Layer {layer}:")
            for node, connections in sorted(self.graph[layer].items()):
                print(f"  Node {node}: {connections}")
            print()

    
    def search(self, q, K, ef):
        '''
        q:  np.array, query element
        K:  int, number of nearest neighbors to return
        ef: int, size of the dynamic candidate list
        '''

        W = []  # set for the current nearest elements
        ep, L = self._get_entry_point()  # entry point

        # Traverse from top layer to layer 1
        for lc in range(L, 0, -1):
            W = self._search_layer(q, ep, ef=1, layer_num=lc)
            ep = min(W, key=lambda x: self._distance(x, q))

        # Search layer 0
        W = self._search_layer(q, ep, ef, layer_num=0)

        # Return K nearest elements from W to q
        return sorted(W, key=lambda x: self._distance(q, x))[:K]
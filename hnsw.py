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
    def __init__(self, max_connections=16, max_layers=16, ef=16, efConstruction=16):
        # self.nodes = {} # node id to vector dictionary, this is how this actually works, but keeping it easy for this.
        self.graph = {}
        '''
        list of layers, e.g.
        {
            0: {
                node_vector: [neighbor1_vector, neighbor2_vector, ...],
                node_vector: [neighbor1_vector, neighbor2_vector, ...]
            },
            1: ...
        }
        '''
        self.max_connections = max_connections # how many outgoing edges a node can have in the HNSW graph
        self.max_layers = max_layers # max depth of the hierarchy
        self.ef = ef # query time parameter for quality of search
        self.efConstruction = efConstruction # HNSW build parameter controls quality of search at build time
    
    def _distance(self, a, b):
        a, b = np.array(a), np.array(b)
        return np.sqrt(np.sum((a - b)**2))
    
    def _search_layer(self, q, ep, ef, layer_num):
        '''
        q:              []int, query vector
        eps:             []int, entry points (node ids)
        ef:             int, number of nearest to q elements to return
        layer_num:      int, layer number we are searching through (this is a hierarchical graph)
        '''
        visited = set(ep) # set of visited elements (v in paper)
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

            for e in self.layers[layer_num][nearest_neighbor]:
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
            
        return [neighbor for dist, neighbor in dynamic_nearest_neighbors_list]

    def _insert_node(self, q, M, M_max, efConstruction, mL):
        '''
        q:              []int   -- query vector
        M:              int     -- number of established connections
        M[max]:         int     -- max number of connections for each element per layer
        efConstruction: int     -- size of the dynamic candidate list
        mL:             int     -- normalization factor for level generation
        '''
        nearest_neighbors = [] # list for the currently found nearest elements
        ep, L = self._get_entry_point # get entry point and layer of the entry point to HNSW graph
        l = math.floor(-math.log(random.uniform(0, 1)) * mL)

        # Find the new entry point for layers L down to l+1
        for lc in range(L, l + 1, -1):
            nearest_neighbors = self._search_layer(q, ep, efConstruction, lc)
            ep = nearest_neighbors[0] # !! CONNOR LOOK HERE - need to make sure this is the case !!

        # insert the new element and update connections for layers min(L, 1) down to 0
        for lc in range(min(L, 1), -1, -1):
            nearest_neighbors = self._search_layer(q, ep, efConstruction, lc)
            neighbors = self._select_neighbors_simple(q, nearest_neighbors, M, lc)

            # add bidrectional connections from neighbors to q at layer lc
            self._add_bidirectional_connections(q, neighbors, lc)

            # shrink connection sif needed
            for e in neighbors:
                e_conn = self.graph[lc][e]
                if len(e_conn) > M_max[lc]:
                    e_new_conn = self._select_neighbors_simple(e, e_conn, M_max[lc], lc)
                    self.graph[lc][e] = e_new_conn
            
            ep = nearest_neighbors
        
        # update the entry point if necessary
        
        # !! CONNOR LOOK HERE - WHAT IS THIS? !!
        
        #if l > L:
            #self._set_entry_point(q)

  
        # Insert the node into the graph, updating connections as necessary.
        pass

    def _get_entry_point(self):
        return (1,1), 2

    def _set_entry_point(self, q):
        self.entry_point = q

    def _select_neighbors_simple(self, q, C, M):
        pass

    def _select_neighbors_heuristic(self, q, C, M, lc, extendCandidates, keepPrunedConnections):
        pass

    def _choose_max_layer(self):
        # Randomly choose the maximum layer for the new node.
        pass
    
    def _update_connections(self, node_id, vector, layer, visited_set):
        # Update connections for the new node in the given layer.
        pass
    
    def search(self, query_vector, k=1):
        # Search for the k-nearest neighbors of the query vector.
        pass

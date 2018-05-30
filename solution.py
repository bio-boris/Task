import urllib
import copy


memoize = {}

def longestPrefixSuffixMatch(s1, s2, lines):
    if (s1, s2) in memoize:
	return memoize[(s1,s2)]
	
    for x in reversed(range(len(lines[s1]) + 1)):
	if lines[s2].endswith(lines[s1][:x]):
		memoize[(s1, s2)] = x
		return x
             
    # TODO :  use assertion here to always throw error.
    # Error should never happen
    print "Should never happen" 
    memoize[(s1, s2)] = -1
    return -1


def sanitize_lines(idx_list, lines):
	prev_idx = None
	lines_copy = []
	for idx in idx_list:
		if prev_idx is not None:
			lines_copy.append(lines[idx][longestPrefixSuffixMatch(idx, prev_idx, lines):])
		else:
			lines_copy.append(lines[idx])
		prev_idx = idx
	return lines_copy

# Simple part 1 solution when the input is not shuffled
def decode(filename):
	with open(filename, "r") as f:
		lines = f.read().splitlines()
		lines = [urllib.unquote_plus(x) for x in lines] 
		print ''.join(sanitize_lines(range(len(lines)), lines))	
				

def find_sol(idx, sol, chosen, graph):
	if len(chosen) == len(graph):
		return True

	if end_ids.issubset(chosen):
		return False

        # NOTE : This could be used for really deep graphs to further optimize using 
	# level cache. For shallow graphs this over head is not required.
	#for lev in range(len(child_branch)-1, len(chosen) , -1):
	#	if  final_level[lev].issubset(chosen):
	#		return False

	for idx_n in graph[idx]:
		if idx_n in chosen or idx_n not in final_level[len(chosen)]:
			continue

		sol.append(idx_n)
		chosen.add(idx_n)
	
		if find_sol(idx_n, sol, chosen, graph):
			return True
		else :
			sol.pop()
			chosen.remove(idx_n)
	return False

# TODO :  this is the final graph generated from unshuffled data. 
# this graph is also pruned//mutated for ease of traversing.
# This should be made local rather having different methods mutating it.
child_branch = {}

def get_parent(graph, prev_parent):
	num_nodes = len(graph)
	node_ids = set([k for k in graph])
        for node_id in graph:
                for child in graph[node_id]:
                        if child in node_ids:
                                node_ids.remove(child)
	if len(node_ids) == 1:
		for id in node_ids:
			return id
	else:
		if prev_parent in child_branch:
			if len(child_branch[prev_parent]) == 1:
				for id in child_branch[prev_parent]:
					return id
		return -1

def get_last(graph, prev_last):
	parents = set()
	for node in graph:
		if len(graph[node]) == 0:
			return node
	for node in child_branch:
		if prev_last in child_branch[node]:
			parents.add(node)
	if len(parents) == 1:
		for x in parents:
			return x
	return -1

def get_start_list(graph, result, prev_parent):
	parent = get_parent(graph, prev_parent)
	if parent != -1:
		result.append(parent)
		graph_copy = copy.deepcopy(graph)
		del graph_copy[parent]
                for id in graph_copy:
                        if parent in graph_copy[id]:
                                graph_copy[id].remove(parent)
		get_start_list(graph_copy, result, parent)
		
		
def get_end_list(graph, result, prev_last):
	end = get_last(graph, prev_last)
	if end != -1:
		result.append(end)
		graph_copy = copy.deepcopy(graph)
		del graph_copy[end]
		for id in graph_copy:
			if end in graph_copy[id]:
				graph_copy[id].remove(end)
		get_end_list(graph_copy, result, end)


# TODO : all such global mutable data strucutres should instead be local and 
# passed around.
start_decided = []
end_decided = []
start_ids = set()
end_ids = set()
reverse_graph = [False]

# Method that takes a graph and removes the nodes and edges not required to 
# be traversed. Basically if we determine what are the deterministic end and 
# start nodes , then there is no need to explore those part of graph during backtracking 
# traversal (DFS)
def prune_final_graph(graph, start_list, end_list):
	if len(start_list) > 0:
		start_p = graph[start_list[len(start_list) - 1]].copy()
	else:
		start_p = set(graph.keys())
	
	end_p = set()
	if len(end_list) > 0:
		last = end_list[0]
		for id in graph:
			if last in graph[id]:
				end_p.add(id)
	else:
		end_p = set(graph.keys()) 

	for id in start_list:
		del graph[id]
	for id in end_list:
		del graph[id]
	
	temp = []
	temp.extend(start_list)
	temp.extend(end_list)
	for id in graph:
		for i in temp:
			if i in graph[id]:
				graph[id].remove(i)

	for id in end_list:
		if id in start_p:
			start_p.remove(id)

	for id in start_list:
		if id in end_p:
			end_p.remove(id)

	# TODO : clean this up. 
	# just using start_p or end_p should be good enough
	# Kept here just for safetly and ease of debugging.
	for id in start_p:
		start_ids.add(id)
	for id in end_p:
		end_ids.add(id)

	start_decided.extend(start_list)
	end_decided.extend(end_list)
	# If end nodes are smaller set then begin nodes, it is 
	# better to start traversing from bottom of graph. 
	# This reverses the graph.
	if len(end_ids) < len(start_ids):
		reverse_graph[0] = True
		#temp_ids = copy.deepcopy(start_ids)
		start_ids.clear()
		for id in end_p:
			start_ids.add(id)

		end_ids.clear()
		for id in start_p:
			end_ids.add(id)
		reverseGraph(graph)

# Method that takes a graph and generates a reverse graph,
# i.e from parent -> child nodes, it creates a child -> parent nodes.
def reverseGraph(graph):
	graph_r = copy.deepcopy(graph)
        graph.clear()
        for key in graph_r:
		graph[key] = set()
	for key in graph_r:
		for child in graph_r[key]:
			graph[child].add(key)	

# BFS : breadth first search traversal to generate
# parent -> child graph
def bfs_top(ids, level):
	if level == len(child_branch):
		return
	top_level[level] = set(ids)
	child = set()
	for id in ids:
		for c in child_branch[id]:
			child.add(c)	
	bfs_top(child, level+1)

# BFS : breadth first search traversal to generate
# child -> parent graph
def bfs_bottom(ids, level):
	if level < 0:
                return
	bottom_level[level] = set(ids)
	parent = set()
	for node in child_branch:
		for id in ids:
			if id in child_branch[node]:
				parent.add(node)
	bfs_bottom(parent, level -1)


# TODO :  make these local variables and not global shared mutable variables.
top_level = {}
bottom_level = {}
final_level = {}

def get_level_cache(graph):
	if len(start_ids) > 0:
		curr_level = 0
		bfs_top(start_ids, curr_level)
	if len(end_ids) > 0:
		curr_level = len(child_branch)-1
		bfs_bottom(end_ids, curr_level)

	for level in range(len(child_branch)):
		final_level[level] = top_level[level] & bottom_level[level] 

			

def preprocess(lines):
	for i in range(len(lines)):
		child_branch[i] = set()
		for j in range(i):
			k = longestPrefixSuffixMatch(i, j, lines)
			if k > 3:
				child_branch[j].add(i)
			m = longestPrefixSuffixMatch(j, i, lines)
			if m > 3:
				child_branch[i].add(j)

	start = get_parent(child_branch, None)
	end = get_last(child_branch, None)

	start_list = []
	get_start_list(child_branch, start_list, None)
	end_list = []
	get_end_list(child_branch, end_list, None)
	end_list = end_list[::-1]
	# Debug info after preprocessing the inital graph
	#print "preprocessing done " , " start " , start, " end " , end
	#print "start_list" , start_list, "end_list" , end_list

	# Remove unneccesary nodes and edges fromt he graph
	prune_final_graph(child_branch, start_list, end_list)
	# For optimization and geennerating level based node cache
	get_level_cache(child_branch)


def decode_shuffled(filename):
	print "reading file : ", filename
        with open(filename, "r") as f:
                lines = f.read().splitlines()
		orig_lines = list(lines)
                lines = [urllib.unquote_plus(x) for x in lines]
		preprocess(lines)
		for idx in start_ids:
			sol = []
			chosen = set()
			sol.append(idx)
			chosen.add(idx)
			if find_sol(idx, sol, chosen, child_branch):
				final_sol = []
				final_sol.extend(start_decided)
				if reverse_graph[0] == True:
					sol = sol[::-1]
				final_sol.extend(sol)
				final_sol.extend(end_decided)
				print urllib.unquote_plus(''.join(sanitize_lines(final_sol, lines)))
				return
			else:
				sol.pop()
				chosen.remove(idx)
	print "could not find solution"



### For local debugging using local files
#decode('helloworldjava')
#decode_shuffled('helloworldjava')
#decode_shuffled('helloworldjavash')
decode_shuffled('pythonshuffled')
#decode_shuffled('shakesh')
#decode_shuffled('lpsumsh')

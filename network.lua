-- neat/network.lua
-- Builds a feedforward network from a genome with proper topological sorting
local class = require("lovely-neat.modified_middleclass")

---@class Network: Object
local Network = class("Network")

-- activation function (sigmoid)
local function sigmoid(x)
	return 1 / (1 + math.exp(-4.9 * x))
end

-- Topological sort using Kahn's algorithm
local function topologicalSort(nodes, connections)
	local inDegree = {}
	local adjList = {}
	local nodeOrder = {}

	-- Initialize
	for id, _ in pairs(nodes) do
		inDegree[id] = 0
		adjList[id] = {}
	end

	-- Build adjacency list and calculate in-degrees
	for _, conn in pairs(connections) do
		if conn.enabled and nodes[conn.from] and nodes[conn.to] then
			table.insert(adjList[conn.from], conn.to)
			inDegree[conn.to] = (inDegree[conn.to] or 0) + 1
		end
	end

	-- Find all nodes with no incoming edges
	local queue = {}
	for id, degree in pairs(inDegree) do
		if degree == 0 then
			table.insert(queue, id)
		end
	end

	-- Process queue
	while #queue > 0 do
		local current = table.remove(queue, 1)
		table.insert(nodeOrder, current)

		-- For each neighbor of current node
		for _, neighbor in ipairs(adjList[current] or {}) do
			inDegree[neighbor] = inDegree[neighbor] - 1
			if inDegree[neighbor] == 0 then
				table.insert(queue, neighbor)
			end
		end
	end

	return nodeOrder
end

-- Build network from genome
---@param genome Genome
function Network.buildFromGenome(genome)
	-- Create node data structure
	local nodes = {}
	for id, node in pairs(genome.nodes) do
		nodes[id] = {
			node = node,
			incoming = {},
			activation = 0,
		}
	end

	-- Build incoming connections for each node
	for _, conn in pairs(genome.connections) do
		if conn.enabled and nodes[conn.from] and nodes[conn.to] then
			table.insert(nodes[conn.to].incoming, {
				from = conn.from,
				weight = conn.weight,
			})
		end
	end

	-- Get topological order for evaluation
	local nodeOrder = topologicalSort(genome.nodes, genome.connections)

	-- Return network object
	local net = {}

	function net:evaluate(inputValues)
		-- Reset all activations
		for id, ndata in pairs(nodes) do
			ndata.activation = 0
		end

		-- Set input and bias values
		for id, ndata in pairs(nodes) do
			if ndata.node.type == "input" then
				ndata.activation = inputValues[id] or 0
			elseif ndata.node.type == "bias" then
				ndata.activation = 1
			end
		end

		-- Process nodes in topological order
		for _, nodeId in ipairs(nodeOrder) do
			local ndata = nodes[nodeId]
			if ndata and ndata.node.type ~= "input" and ndata.node.type ~= "bias" then
				local sum = 0
				for _, incoming in ipairs(ndata.incoming) do
					local fromNode = nodes[incoming.from]
					if fromNode then
						sum = sum + fromNode.activation * incoming.weight
					end
				end
				ndata.activation = sigmoid(sum)
			end
		end

		-- Collect outputs
		local outputs = {}
		for id, ndata in pairs(nodes) do
			if ndata.node.type == "output" then
				table.insert(outputs, { id = id, value = ndata.activation })
			end
		end

		-- Sort outputs by id for consistency
		table.sort(outputs, function(a, b)
			return a.id < b.id
		end)

		return outputs
	end

	return net
end

return Network

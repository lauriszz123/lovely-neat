-- neat/network.lua
-- Builds a feedforward network from a genome with proper topological sorting
local class = require("lovely-neat.modified_middleclass")

---@class Network: Object
local Network = class("Network")

--- @param x number
local function sigmoid(x)
	return 1 / (1 + math.exp(-4.9 * x)) -- 4.9 is a common steepness factor
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

	-- Derivative of sigmoid function
	local function sigmoidDerivative(x)
		return x * (1 - x) -- Since x is already sigmoid(z), derivative is sigmoid(z) * (1 - sigmoid(z))
	end

	-- Backpropagation function
	function net:backward(targetOutputs, learningRate)
		learningRate = learningRate or 0.1

		-- Initialize gradients for all nodes
		local nodeGradients = {}
		for id, ndata in pairs(nodes) do
			nodeGradients[id] = 0
		end

		-- Calculate output layer gradients (error derivatives)
		local outputIndex = 1
		for id, ndata in pairs(nodes) do
			if ndata.node.type == "output" then
				local target = targetOutputs[outputIndex] or 0
				local output = ndata.activation
				-- Gradient = (target - output) * sigmoid'(output)
				nodeGradients[id] = (target - output) * sigmoidDerivative(output)
				outputIndex = outputIndex + 1
			end
		end

		-- Backpropagate gradients through the network (reverse topological order)
		local reverseOrder = {}
		for i = #nodeOrder, 1, -1 do
			table.insert(reverseOrder, nodeOrder[i])
		end

		for _, nodeId in ipairs(reverseOrder) do
			local ndata = nodes[nodeId]
			if ndata and ndata.node.type ~= "input" and ndata.node.type ~= "bias" then
				-- For hidden nodes, calculate gradient from downstream nodes
				if ndata.node.type == "hidden" then
					local gradient = 0
					-- Sum gradients from all outgoing connections
					for _, conn in pairs(genome.connections) do
						if conn.enabled and conn.from == nodeId and nodes[conn.to] then
							gradient = gradient + nodeGradients[conn.to] * conn.weight
						end
					end
					nodeGradients[nodeId] = gradient * sigmoidDerivative(ndata.activation)
				end

				-- Update weights for incoming connections
				for _, incoming in ipairs(ndata.incoming) do
					local fromNode = nodes[incoming.from]
					if fromNode then
						-- Calculate weight update: learningRate * gradient * input_activation
						local weightUpdate = learningRate * nodeGradients[nodeId] * fromNode.activation

						-- Find and update the corresponding connection in the genome
						for _, conn in pairs(genome.connections) do
							if conn.enabled and conn.from == incoming.from and conn.to == nodeId then
								conn.weight = conn.weight + weightUpdate
								-- Also update the network's internal weight
								incoming.weight = conn.weight
								break
							end
						end
					end
				end
			end
		end

		-- Return the total error for monitoring
		local totalError = 0
		outputIndex = 1
		for id, ndata in pairs(nodes) do
			if ndata.node.type == "output" then
				local target = targetOutputs[outputIndex] or 0
				local output = ndata.activation
				totalError = totalError + 0.5 * (target - output) * (target - output)
				outputIndex = outputIndex + 1
			end
		end

		return totalError
	end

	function net:draw(x, y)
		-- Fixed bounds for the network visualization
		local maxWidth = 250
		local maxHeight = 400

		-- Analyze network topology to determine layers
		local function calculateNodeLayers()
			local nodeLayers = {}
			local nodeDepths = {}

			-- Initialize all nodes with depth 0
			for id, ndata in pairs(nodes) do
				nodeDepths[id] = 0
			end

			-- Set input and bias nodes to layer 0
			for id, ndata in pairs(nodes) do
				if ndata.node.type == "input" or ndata.node.type == "bias" then
					nodeDepths[id] = 0
				end
			end

			-- Calculate depths through multiple passes
			local changed = true
			local maxPasses = 10
			local passes = 0

			while changed and passes < maxPasses do
				changed = false
				passes = passes + 1

				for _, conn in pairs(genome.connections) do
					if conn.enabled and nodes[conn.from] and nodes[conn.to] then
						local fromDepth = nodeDepths[conn.from]
						local requiredToDepth = fromDepth + 1

						if nodeDepths[conn.to] < requiredToDepth then
							nodeDepths[conn.to] = requiredToDepth
							changed = true
						end
					end
				end
			end

			-- Group nodes by their calculated depth
			local layerGroups = {}
			for id, depth in pairs(nodeDepths) do
				if not layerGroups[depth] then
					layerGroups[depth] = {}
				end
				table.insert(layerGroups[depth], { id = id, value = nodes[id].activation, type = nodes[id].node.type })
			end

			-- Convert to ordered array
			local orderedLayers = {}
			for depth = 0, 10 do -- Support up to 10 layers
				if layerGroups[depth] then
					table.insert(orderedLayers, { nodes = layerGroups[depth], depth = depth })
				end
			end

			return orderedLayers
		end

		-- Get layers based on topology analysis
		local activeLayers = calculateNodeLayers()

		-- Calculate spacing between layers
		local layerSpacing = #activeLayers > 1 and maxWidth / (#activeLayers - 1) or 0

		-- Calculate node positions for each layer
		for layerIndex, layer in ipairs(activeLayers) do
			local nodeCount = #layer.nodes
			local nodeRadius = math.min(6, math.max(2, math.min(maxHeight / (nodeCount * 2.5), maxWidth / 20)))

			-- Calculate vertical spacing to fit all nodes in height
			local totalNodeHeight = nodeCount * nodeRadius * 2
			local availableHeight = maxHeight - (nodeRadius * 2)
			local nodeSpacing = nodeCount > 1 and math.min(nodeRadius * 2.5, availableHeight / (nodeCount - 1)) or 0

			-- Center the column vertically
			local startY = y + (maxHeight - (nodeCount - 1) * nodeSpacing) / 2

			layer.x = x + (layerIndex - 1) * layerSpacing
			layer.nodeRadius = nodeRadius
			layer.nodeSpacing = nodeSpacing
			layer.startY = startY
		end

		-- Draw connections first (behind nodes)
		for _, conn in pairs(genome.connections) do
			if conn.enabled and nodes[conn.from] and nodes[conn.to] then
				-- Only draw connections with significant weight (threshold)
				local weightThreshold = 0.05
				if math.abs(conn.weight) < weightThreshold then
					goto continue
				end

				local fromNode = nodes[conn.from]
				local toNode = nodes[conn.to]

				-- Find layer positions
				local fromPos, toPos = nil, nil

				for layerIndex, layer in ipairs(activeLayers) do
					for nodeIndex, node in ipairs(layer.nodes) do
						if node.id == conn.from then
							fromPos = {
								x = layer.x,
								y = layer.startY + (nodeIndex - 1) * layer.nodeSpacing,
							}
						end
						if node.id == conn.to then
							toPos = {
								x = layer.x,
								y = layer.startY + (nodeIndex - 1) * layer.nodeSpacing,
							}
						end
					end
				end

				if fromPos and toPos then
					-- Color code connections: green for positive, red for negative
					if conn.weight > 0 then
						love.graphics.setColor(0, 0.6, 0, 0.6)
					else
						love.graphics.setColor(0.6, 0, 0, 0.6)
					end

					-- Line thickness based on weight strength
					love.graphics.setLineWidth(math.max(0.3, math.min(1.5, math.abs(conn.weight) * 1)))
					love.graphics.line(fromPos.x, fromPos.y, toPos.x, toPos.y)
				end

				::continue::
			end
		end

		-- Draw nodes on top of connections
		for layerIndex, layer in ipairs(activeLayers) do
			for nodeIndex, node in ipairs(layer.nodes) do
				local nodeX = layer.x
				local nodeY = layer.startY + (nodeIndex - 1) * layer.nodeSpacing

				-- Get activation value and clamp it between 0 and 1
				local activation = math.max(0, math.min(1, node.value or 0))

				-- Fill color: interpolate between black (0) and white (1)
				local fillColor = activation
				love.graphics.setColor(fillColor, fillColor, fillColor, 1)
				love.graphics.circle("fill", nodeX, nodeY, layer.nodeRadius)

				-- Outline color: interpolate between white (0) and black (1)
				local outlineColor = 1 - activation
				love.graphics.setColor(outlineColor, outlineColor, outlineColor, 1)
				love.graphics.setLineWidth(0.5)
				love.graphics.circle("line", nodeX, nodeY, layer.nodeRadius)
			end
		end

		-- Draw bounding box for reference (optional - remove if not needed)
		love.graphics.setColor(0.3, 0.3, 0.3, 0.5)
		love.graphics.setLineWidth(1)
		love.graphics.rectangle("line", x, y, maxWidth, maxHeight)
	end

	return net
end

return Network

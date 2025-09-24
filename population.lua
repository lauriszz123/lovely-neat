-- neat/population.lua
local class = require("lovely-neat.modified_middleclass")

local Innovation = require("lovely-neat.innovation")
local Genome = require("lovely-neat.genome")
local Node = require("lovely-neat.node")
local Species = require("lovely-neat.species")
local Network = require("lovely-neat.network")
local util = require("lovely-neat.util")
local Connection = require("lovely-neat.connection")

---@class Population: Object
local Population = class("Population")

-- Optimized config for better performance and evolution
local function defaultConfig()
	---@class DefaultPopulationConfig
	local config = {
		populationSize = 100, -- Increased for better diversity
		inputCount = 3,
		outputCount = 1,
		bias = true,
		-- Hidden layer configuration - now randomized per genome
		minHiddenLayers = 0, -- Minimum hidden layers (can be 0 for direct connections)
		maxHiddenLayers = 3, -- Reduced from 4 for performance
		minNodesPerLayer = 2, -- Minimum nodes per hidden layer
		maxNodesPerLayer = 6, -- Reduced from 8 for performance

		-- Sparse connectivity parameters - optimized
		sparseConnectivity = true, -- Enable sparse random connections
		connectionProbability = 0.4, -- Increased from 0.3 for better connectivity
		guaranteedOutputConnections = true, -- Ensure each output has at least one input

		compatThreshold = 3.0,
		c1 = 1.0,
		c2 = 1.0,
		c3 = 0.4,
		-- Optimized mutation rates for faster convergence
		weightPerturbRate = 0.9, -- Reduced from 0.95 for stability
		weightPerturbStrength = 2.5, -- Reduced from 3.0 for stability
		addNodeRate = 0.05, -- Reduced from 0.08 for performance
		addConnRate = 0.08, -- Reduced from 0.12 for performance
		removeConnRate = 0.015, -- Reduced from 0.02 for stability
		weightInitRange = 2.5, -- Reduced from 3 for stability
		elitism = 3, -- Increased from 2 for better preservation
		survivalThreshold = 0.25, -- Reduced from 0.3 for stronger selection
		stagnationThreshold = 12, -- Reduced from 15 for faster adaptation
		-- Evolution parameters - optimized
		crossoverRate = 0.8, -- Increased from 0.75 for more diversity
		weightMutationRate = 0.85, -- Reduced from 0.95 for stability
		uniformWeightRate = 0.1, -- Reduced from 0.15 for stability

		-- Dynamic mutation amplifiers - optimized for faster evolution
		mutationAmplifierOverGenerations = 150, -- Reduced from 200 for faster convergence
		connectionMutationAmplifierRange = { from = 3, to = 50 }, -- Shorter boom period

		-- Multiple mutation attempts - optimized
		maxMutationAttempts = 3, -- Reduced from 5 for performance
	}

	return config
end

function Population:initialize(cfg)
	---@type DefaultPopulationConfig
	cfg = cfg or {}
	local finalCfg = defaultConfig()
	for k, v in pairs(cfg) do
		finalCfg[k] = v
	end

	self.cfg = finalCfg

	---@type Innovation
	self.innovation = Innovation()

	---@type Species[]
	self.species = {}

	---@type Genome[]
	self.genomes = {}

	self.generation = 1
	---@type Genome?
	self.best = nil
	self.generationWithoutImprovement = 0
	self.bestFitnessEver = -math.huge

	-- create initial population
	for i = 1, finalCfg.populationSize do
		local g = Genome()
		g.fitness = 0
		g.adjustedFitness = 0

		-- create input nodes
		for inId = 1, finalCfg.inputCount do
			local nid = self.innovation:nextNode()
			g:addNode(Node(nid, "input"))
		end

		-- bias node
		if finalCfg.bias then
			local nid = self.innovation:nextNode()
			g:addNode(Node(nid, "bias"))
		end

		-- output nodes
		for out = 1, finalCfg.outputCount do
			local nid = self.innovation:nextNode()
			g:addNode(Node(nid, "output"))
		end

		-- Create hidden layers if specified
		local layerNodes = {} -- Track nodes by layer for connection
		local allNodeIds = {} -- Track all node IDs for sparse connectivity

		-- Layer 0: Input + bias nodes
		layerNodes[0] = {}
		for id, node in pairs(g.nodes) do
			if node.type == "input" or node.type == "bias" then
				table.insert(layerNodes[0], id)
				table.insert(allNodeIds, id)
			end
		end

		-- Create random number of hidden layers with random sizes
		local numHiddenLayers = math.random(finalCfg.minHiddenLayers, finalCfg.maxHiddenLayers)

		if _G.DEBUG then
			if i <= 5 then -- Debug output for first 5 genomes
				print(string.format("Genome %d: %d hidden layers", i, numHiddenLayers))
			end
		end

		for layerIndex = 1, numHiddenLayers do
			layerNodes[layerIndex] = {}
			local layerSize = math.random(finalCfg.minNodesPerLayer, finalCfg.maxNodesPerLayer)

			if _G.DEBUG then
				if i <= 5 then -- Debug output for first 5 genomes
					print(string.format("  Layer %d: %d nodes", layerIndex, layerSize))
				end
			end

			for nodeIdx = 1, layerSize do
				local nid = self.innovation:nextNode()
				g:addNode(Node(nid, "hidden"))
				table.insert(layerNodes[layerIndex], nid)
				table.insert(allNodeIds, nid)
			end
		end -- Final layer: Output nodes
		local finalLayerIndex = numHiddenLayers + 1
		layerNodes[finalLayerIndex] = {}
		for id, node in pairs(g.nodes) do
			if node.type == "output" then
				table.insert(layerNodes[finalLayerIndex], id)
				table.insert(allNodeIds, id)
			end
		end

		-- Create sparse random connections instead of fully connected layers
		if finalCfg.sparseConnectivity then
			-- Sparse connectivity: random connections between compatible nodes
			local connectedOutputs = {} -- Track which outputs have connections
			local totalConnections = 0 -- Count total connections for debug

			for fromLayerIdx = 0, finalLayerIndex - 1 do
				local fromLayer = layerNodes[fromLayerIdx]

				-- Connect to all subsequent layers (allowing skip connections)
				for toLayerIdx = fromLayerIdx + 1, finalLayerIndex do
					local toLayer = layerNodes[toLayerIdx]

					for _, fromNodeId in ipairs(fromLayer) do
						for _, toNodeId in ipairs(toLayer) do
							-- Random chance of connection
							if math.random() < finalCfg.connectionProbability then
								local innovId = self.innovation:nextConnId(fromNodeId, toNodeId)
								local weight = (math.random() * 4 - 2) -- Range [-2, 2]
								g:addConnection(Connection(fromNodeId, toNodeId, weight, true, innovId))
								totalConnections = totalConnections + 1

								-- Track that this output is connected
								if toLayerIdx == finalLayerIndex then
									connectedOutputs[toNodeId] = true
								end
							end
						end
					end
				end
			end

			if _G.DEBUG then
				if i <= 5 then -- Debug output for first 5 genomes
					print(string.format("  Sparse connections: %d total", totalConnections))
				end -- Ensure each output has at least one connection if guaranteed
			end
			if finalCfg.guaranteedOutputConnections then
				for _, outputId in ipairs(layerNodes[finalLayerIndex]) do
					if not connectedOutputs[outputId] then
						-- Connect to a random node from a previous layer
						local possibleSources = {}
						for layerIdx = 0, finalLayerIndex - 1 do
							for _, nodeId in ipairs(layerNodes[layerIdx]) do
								table.insert(possibleSources, nodeId)
							end
						end

						if #possibleSources > 0 then
							local sourceId = possibleSources[math.random(#possibleSources)]
							local innovId = self.innovation:nextConnId(sourceId, outputId)
							local weight = (math.random() * 4 - 2)
							g:addConnection(Connection(sourceId, outputId, weight, true, innovId))
						end
					end
				end
			end
		else
			-- Fallback: Connect layers sequentially (feedforward) - dense connectivity
			for layerIdx = 0, finalLayerIndex - 1 do
				local fromLayer = layerNodes[layerIdx]
				local toLayer = layerNodes[layerIdx + 1]

				for _, fromNodeId in ipairs(fromLayer) do
					for _, toNodeId in ipairs(toLayer) do
						local innovId = self.innovation:nextConnId(fromNodeId, toNodeId)
						local weight = (math.random() * 4 - 2) -- Range [-2, 2]
						g:addConnection(Connection(fromNodeId, toNodeId, weight, true, innovId))
					end
				end
			end
		end
		table.insert(self.genomes, g)
	end
end

-- Enhanced speciate with dynamic threshold
function Population:speciate()
	-- Adjust compatibility threshold based on species count
	local targetSpecies = math.max(5, math.min(20, self.cfg.populationSize / 10))
	if #self.species > targetSpecies then
		self.cfg.compatThreshold = self.cfg.compatThreshold * 1.05
	elseif #self.species < targetSpecies then
		self.cfg.compatThreshold = self.cfg.compatThreshold * 0.95
	end
	self.cfg.compatThreshold = math.max(0.5, math.min(5.0, self.cfg.compatThreshold))

	-- clear species members
	for _, s in ipairs(self.species) do
		s:clear()
	end

	for _, g in ipairs(self.genomes) do
		local placed = false
		for _, s in ipairs(self.species) do
			local d = g:compatibility(s.representative, { c1 = self.cfg.c1, c2 = self.cfg.c2, c3 = self.cfg.c3 })
			if d < self.cfg.compatThreshold then
				s:addMember(g)
				placed = true
				break
			end
		end
		if not placed then
			---@type Species
			local s = Species(g)
			s:addMember(g)
			table.insert(self.species, s)
		end
	end

	-- remove empty species and update representatives
	local newSpecies = {}
	for _, s in ipairs(self.species) do
		if #s.members > 0 then
			-- Update representative to best member
			table.sort(s.members, function(a, b)
				return a.fitness > b.fitness
			end)
			s.representative = s.members[1]
			table.insert(newSpecies, s)
		end
	end
	self.species = newSpecies
end

-- produce networks for evaluation
function Population:buildNetworks()
	local out = {}
	for _, g in ipairs(self.genomes) do
		local net = Network.buildFromGenome(g)
		table.insert(out, { genome = g, network = net })
	end
	return out
end

-- Calculate dynamic mutation rates based on current generation
function Population:getDynamicMutationRates()
	local currentGen = self.generation or 1
	local rates = {
		addNodeRate = self.cfg.addNodeRate,
		addConnRate = self.cfg.addConnRate,
		removeConnRate = self.cfg.removeConnRate,
	}

	-- Node addition amplifier: starts VERY HIGH, decreases to low over time
	if self.cfg.mutationAmplifierOverGenerations and currentGen <= self.cfg.mutationAmplifierOverGenerations then
		-- Calculate amplifier: starts at 25x (MASSIVE), decreases to 1x (normal) over generations
		local progress = currentGen / self.cfg.mutationAmplifierOverGenerations
		local amplifier = 25.0 * (1.0 - progress) + 1.0 * progress -- Linear interpolation from 25 to 1
		rates.addNodeRate = self.cfg.addNodeRate * amplifier
	end

	-- Connection mutation amplifier: boom period between specific generations
	local connRange = self.cfg.connectionMutationAmplifierRange
	if connRange and currentGen >= connRange.from and currentGen <= connRange.to then
		-- During boom period: 5x amplifier for both add and remove
		rates.addConnRate = self.cfg.addConnRate * 5.0
		rates.removeConnRate = self.cfg.removeConnRate * 5.0
	end

	return rates
end

-- Enhanced mutation function with multiple complete attempts
function Population:mutateGenome(genome)
	local mutated = false
	local dynamicRates = self:getDynamicMutationRates()
	local maxAttempts = self.cfg.maxMutationAttempts or 1

	-- Perform multiple complete mutation attempts
	for attempt = 1, maxAttempts do
		-- Weight mutations
		if math.random() < self.cfg.weightMutationRate then
			for _, c in pairs(genome.connections) do
				if math.random() < self.cfg.weightPerturbRate then
					-- Perturb existing weight
					c.weight = c.weight + (math.random() * 2 - 1) * self.cfg.weightPerturbStrength
				elseif math.random() < self.cfg.uniformWeightRate then
					-- Completely new weight
					c.weight = (math.random() * 4 - 2) -- Range [-2, 2]
				end
			end
			mutated = true
		end

		-- Add connection
		if math.random() < dynamicRates.addConnRate then
			if genome:mutateAddConnection(self.innovation) then
				mutated = true
			end
		end

		-- Remove connection
		if math.random() < dynamicRates.removeConnRate then
			if genome:mutateRemoveConnection(self.innovation) then
				mutated = true
			end
		end

		-- Add node
		if math.random() < dynamicRates.addNodeRate then
			if genome:mutateAddNode(self.innovation) then
				mutated = true
			end
		end
	end

	return mutated
end

-- Enhanced evolution
function Population:epoch()
	-- sort genomes by fitness
	table.sort(self.genomes, function(a, b)
		return a.fitness > b.fitness
	end)

	-- Track best fitness
	local currentBest = self.genomes[1].fitness
	if currentBest > self.bestFitnessEver then
		self.bestFitnessEver = currentBest
		self.best = util.copy(self.genomes[1])
		self.generationWithoutImprovement = 0
	else
		self.generationWithoutImprovement = self.generationWithoutImprovement + 1
	end

	-- Increase mutation rates if stagnating
	if self.generationWithoutImprovement > 5 then
		self.cfg.addNodeRate = math.min(0.2, self.cfg.addNodeRate * 1.1)
		self.cfg.addConnRate = math.min(0.3, self.cfg.addConnRate * 1.1)
		self.cfg.removeConnRate = math.min(0.3, self.cfg.removeConnRate * 1.1)
		self.cfg.weightPerturbStrength = math.min(3.0, self.cfg.weightPerturbStrength * 1.1)
	end

	-- speciate
	self:speciate()

	-- Remove stagnant species (except if they contain the best genome)
	local activeSpecies = {}
	for _, s in ipairs(self.species) do
		---@cast s Species
		s:updateStagnation()
		if s.stale < self.cfg.stagnationThreshold or s:containsBest(self.best) then
			table.insert(activeSpecies, s)
		end
	end
	self.species = activeSpecies

	-- compute adjusted fitnesses
	local totalAdjusted = 0
	for _, s in ipairs(self.species) do
		s:computeAdjustedFitnesses()
		local sum = 0
		for _, g in ipairs(s.members) do
			sum = sum + (g.adjustedFitness or 0)
		end
		s.average = sum / math.max(1, #s.members)
		totalAdjusted = totalAdjusted + sum
	end

	-- create next generation
	local newGen = {}

	-- Elite selection - keep best from each species
	for _, s in ipairs(self.species) do
		table.sort(s.members, function(a, b)
			return a.fitness > b.fitness
		end)
		for i = 1, math.min(self.cfg.elitism, #s.members) do
			table.insert(newGen, util.copy(s.members[i]))
		end
	end

	-- Calculate offspring per species
	local offspring = {}
	for _, s in ipairs(self.species) do
		local share = 0
		for _, g in ipairs(s.members) do
			share = share + (g.adjustedFitness or 0)
		end
		local nChildren =
			math.floor((share / math.max(1e-8, totalAdjusted)) * (self.cfg.populationSize - #newGen) + 0.5)
		offspring[s.id] = nChildren
	end

	-- Reproduce
	for _, s in ipairs(self.species) do
		local n = offspring[s.id] or 0
		if n > 0 then
			table.sort(s.members, function(a, b)
				return a.fitness > b.fitness
			end)

			for i = 1, n do
				local child

				-- Cross over or clone
				if math.random() < self.cfg.crossoverRate and #s.members > 1 then
					-- Tournament selection
					local survivorCount = math.max(1, math.floor(#s.members * self.cfg.survivalThreshold))
					local parentA = s.members[math.random(1, survivorCount)]
					local parentB = s.members[math.random(1, survivorCount)]

					if parentA.fitness >= parentB.fitness then
						child = parentA:crossover(parentB)
					else
						child = parentB:crossover(parentA)
					end
				else
					-- Clone best
					child = util.copy(s.members[1])
				end

				-- Mutate - ALWAYS mutate all offspring for rapid evolution
				self:mutateGenome(child)

				table.insert(newGen, child)
				if #newGen >= self.cfg.populationSize then
					break
				end
			end
		end
		if #newGen >= self.cfg.populationSize then
			break
		end
	end

	-- Fill remaining slots with heavily mutated copies for rapid exploration
	while #newGen < self.cfg.populationSize do
		local parent = self.genomes[math.random(1, math.min(10, #self.genomes))] -- Bias toward better genomes
		local copy = util.copy(parent)
		-- Apply extra mutations for rapid exploration
		self:mutateGenome(copy)
		self:mutateGenome(copy) -- Double mutation for extra diversity
		table.insert(newGen, copy)
	end

	self.genomes = newGen
	self.generation = (self.generation or 1) + 1

	-- Reset fitness for the new generation so they start fresh
	for _, g in ipairs(self.genomes) do
		g.fitness = 0
		g.adjustedFitness = 0
	end
end

---@return Genome?
function Population:getBest()
	return self.best
end

function Population:getStats()
	return {
		generation = self.generation,
		bestFitness = self.bestFitnessEver,
		species = #self.species,
		stagnation = self.generationWithoutImprovement,
		compatThreshold = self.cfg.compatThreshold,
	}
end

return Population

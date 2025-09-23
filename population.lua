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

-- Better config for Flappy Bird
local function defaultConfig()
	---@class DefaultPopulationConfig
	local config = {
		populationSize = 150,
		inputCount = 3,
		outputCount = 1,
		bias = true,
		-- Hidden layer configuration
		hiddenLayers = { 4, 3 }, -- Array of hidden layer sizes: [4 nodes, 3 nodes]
		compatThreshold = 3.0,
		c1 = 1.0,
		c2 = 1.0,
		c3 = 0.4,
		-- Reduced mutation rates for multiple mutations per generation
		weightPerturbRate = 0.9,
		weightPerturbStrength = 2.0,
		addNodeRate = 0.03, -- Reduced from 0.1 for multiple attempts
		addConnRate = 0.05, -- Reduced from 0.15 for multiple attempts
		removeConnRate = 0.005, -- Reduced from 0.01 for multiple attempts
		weightInitRange = 2, -- Increased range
		elitism = 2, -- Keep top 2 per species
		survivalThreshold = 0.3, -- Increased from 0.2
		stagnationThreshold = 15,
		-- New parameters for better evolution
		crossoverRate = 0.75,
		weightMutationRate = 0.8,
		uniformWeightRate = 0.1, -- Chance to completely randomize weight
		modInnovSeed = nil,

		-- Dynamic mutation amplifiers over generations
		mutationAmplifierOverGenerations = 500, -- Over 1000 generations, node addition decreases
		connectionMutationAmplifierRange = { from = 250, to = 850 }, -- Connection mutation boom from gen 100-200

		-- Multiple mutation attempts per generation
		maxMutationAttempts = 3, -- Allow up to 3 mutation attempts per genome per generation
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

		-- Layer 0: Input + bias nodes
		layerNodes[0] = {}
		for id, node in pairs(g.nodes) do
			if node.type == "input" or node.type == "bias" then
				table.insert(layerNodes[0], id)
			end
		end

		-- Create hidden layers
		if finalCfg.hiddenLayers and #finalCfg.hiddenLayers > 0 then
			for layerIndex, layerSize in ipairs(finalCfg.hiddenLayers) do
				layerNodes[layerIndex] = {}
				for nodeIdx = 1, layerSize do
					local nid = self.innovation:nextNode()
					g:addNode(Node(nid, "hidden"))
					table.insert(layerNodes[layerIndex], nid)
				end
			end
		end

		-- Final layer: Output nodes
		local finalLayerIndex = (finalCfg.hiddenLayers and #finalCfg.hiddenLayers or 0) + 1
		layerNodes[finalLayerIndex] = {}
		for id, node in pairs(g.nodes) do
			if node.type == "output" then
				table.insert(layerNodes[finalLayerIndex], id)
			end
		end

		-- Connect layers sequentially (feedforward)
		for layerIdx = 0, finalLayerIndex - 1 do
			local fromLayer = layerNodes[layerIdx]
			local toLayer = layerNodes[layerIdx + 1]

			for _, fromNodeId in ipairs(fromLayer) do
				for _, toNodeId in ipairs(toLayer) do
					local innovId = self.innovation:nextConnId(fromNodeId, toNodeId)
					-- Use larger initial weight range for better diversity
					local weight = (math.random() * 4 - 2) -- Range [-2, 2]
					g:addConnection(Connection(fromNodeId, toNodeId, weight, true, innovId))
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
		-- Calculate amplifier: starts at 10x (BIG), decreases to 0.5x (low) over generations
		local progress = currentGen / self.cfg.mutationAmplifierOverGenerations
		local amplifier = 20.0 * (1.0 - progress) + 0.5 * progress -- Linear interpolation from 10 to 0.5
		rates.addNodeRate = self.cfg.addNodeRate * amplifier
	end

	-- Connection mutation amplifier: boom period between specific generations
	local connRange = self.cfg.connectionMutationAmplifierRange
	if connRange and currentGen >= connRange.from and currentGen <= connRange.to then
		-- During boom period: 3x amplifier for both add and remove
		rates.addConnRate = self.cfg.addConnRate * 3.0
		rates.removeConnRate = self.cfg.removeConnRate * 3.0
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

				-- Mutate
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

	-- Fill remaining slots with mutated copies
	while #newGen < self.cfg.populationSize do
		local parent = self.genomes[math.random(1, math.min(10, #self.genomes))] -- Bias toward better genomes
		local copy = util.copy(parent)
		self:mutateGenome(copy)
		table.insert(newGen, copy)
	end

	self.genomes = newGen
	self.generation = (self.generation or 1) + 1
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

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
		compatThreshold = 3.0,
		c1 = 1.0,
		c2 = 1.0,
		c3 = 0.4,
		-- Increased mutation rates for better exploration
		weightPerturbRate = 0.9,
		weightPerturbStrength = 2.0,
		addNodeRate = 0.1, -- Increased from 0.03
		addConnRate = 0.15, -- Increased from 0.05
		weightInitRange = 2, -- Increased range
		elitism = 2, -- Keep top 2 per species
		survivalThreshold = 0.3, -- Increased from 0.2
		stagnationThreshold = 15,
		-- New parameters for better evolution
		crossoverRate = 0.75,
		weightMutationRate = 0.8,
		uniformWeightRate = 0.1, -- Chance to completely randomize weight
		modInnovSeed = nil,
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

		-- fully connect inputs + bias to outputs with diverse weights
		local inputIds = {}
		for id, node in pairs(g.nodes) do
			if node.type == "input" or node.type == "bias" then
				table.insert(inputIds, id)
			end
		end

		for _, inId in ipairs(inputIds) do
			for id, node in pairs(g.nodes) do
				if node.type == "output" then
					local innovId = self.innovation:nextConnId(inId, node.id)
					-- Use larger initial weight range for better diversity
					local weight = (math.random() * 4 - 2) -- Range [-2, 2]
					g:addConnection(Connection.new(inId, node.id, weight, true, innovId))
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

-- Enhanced mutation function
function Population:mutateGenome(genome)
	local mutated = false

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
	if math.random() < self.cfg.addConnRate then
		if genome:mutateAddConnection(self.innovation) then
			mutated = true
		end
	end

	-- Add node
	if math.random() < self.cfg.addNodeRate then
		if genome:mutateAddNode(self.innovation) then
			mutated = true
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
	if self.generationWithoutImprovement > 10 then
		self.cfg.addNodeRate = math.min(0.2, self.cfg.addNodeRate * 1.1)
		self.cfg.addConnRate = math.min(0.3, self.cfg.addConnRate * 1.1)
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

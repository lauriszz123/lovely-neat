-- neat/species.lua
local class = require("lovely-neat.modified_middleclass")

---@class Species: Object
local Species = class("Species")

function Species:initialize(representative, id)
	self.id = id or math.random(1000000)
	self.representative = representative -- a genome
	self.members = {}
	self.bestFitness = -math.huge
	self.stale = 0
	self.average = 0
	self.lastImprovement = 0
end

function Species:addMember(genome)
	table.insert(self.members, genome)
end

function Species:clear()
	self.members = {}
end

function Species:computeAdjustedFitnesses()
	-- Fitness sharing - divide by species size
	local speciesSize = math.max(1, #self.members)
	for _, g in ipairs(self.members) do
		g.adjustedFitness = g.fitness / speciesSize
	end
end

function Species:updateStagnation()
	if #self.members == 0 then
		return
	end

	-- Sort members by fitness to find current best
	table.sort(self.members, function(a, b)
		return a.fitness > b.fitness
	end)
	local currentBest = self.members[1].fitness

	if currentBest > self.bestFitness then
		self.bestFitness = currentBest
		self.stale = 0
		self.lastImprovement = 0
	else
		self.stale = self.stale + 1
		self.lastImprovement = self.lastImprovement + 1
	end
end

function Species:containsBest(bestGenome)
	if not bestGenome then
		return false
	end

	for _, member in ipairs(self.members) do
		if member.fitness >= bestGenome.fitness then
			return true
		end
	end
	return false
end

return Species

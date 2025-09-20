-- neat/species.lua
local Species = {}
Species.__index = Species

function Species.new(representative, id)
    return setmetatable({
        id = id or math.random(1000000),
        representative = representative, -- a genome
        members = {},
        bestFitness = -1/0,
        stale = 0,
    }, Species)
end

function Species:addMember(genome)
    table.insert(self.members, genome)
end

function Species:clear()
    self.members = {}
end

function Species:computeAdjustedFitnesses()
    -- each genome should have fitness set already
    for _, g in ipairs(self.members) do
        g.adjustedFitness = g.fitness / (#self.members)
    end
end

return Species

-- neat/population.lua
local Innovation = require("neat.innovation")
local Genome = require("neat.genome")
local Node = require("neat.node")
local Species = require("neat.species")
local Network = require("neat.network")
local util = require("neat.util")

local Population = {}
Population.__index = Population

-- config with defaults
local function defaultConfig()
    return {
        populationSize = 150,
        inputCount = 3,
        outputCount = 1,
        bias = true,
        compatThreshold = 3.0,
        c1 = 1.0, c2 = 1.0, c3 = 0.4,
        weightPerturbRate = 0.8,
        weightPerturbStrength = 0.5,
        addNodeRate = 0.03,
        addConnRate = 0.05,
        weightInitRange = 1,
        elitism = 1, -- keep top per species
        survivalThreshold = 0.2,
        stagnationThreshold = 15,
        modInnovSeed = nil,
    }
end

function Population.new(cfg)
    cfg = cfg or {}
    local finalCfg = defaultConfig()
    for k,v in pairs(cfg) do finalCfg[k] = v end

    local innov = Innovation.new()
    local pop = setmetatable({
        cfg = finalCfg,
        innovation = innov,
        genomes = {},
        species = {},
        generation = 1,
        best = nil,
    }, Population)

    -- create initial population
    -- create nodes (inputs, optional bias, outputs)
    for i = 1, finalCfg.populationSize do
        local g = Genome.new()
        g.fitness = 0
        g.adjustedFitness = 0
        -- create input nodes
        for inId = 1, finalCfg.inputCount do
            local nid = innov:nextNode()
            g:addNode(Node.new(nid, "input"))
        end
        if finalCfg.bias then
            local nid = innov:nextNode()
            g:addNode(Node.new(nid, "bias"))
        end
        -- output nodes
        for out = 1, finalCfg.outputCount do
            local nid = innov:nextNode()
            g:addNode(Node.new(nid, "output"))
        end
        -- fully connect inputs + bias to outputs
        local inputIds = {}
        for id,node in pairs(g.nodes) do
            if node.type == "input" or node.type == "bias" then table.insert(inputIds, id) end
        end
        for _, inId in ipairs(inputIds) do
            for id,node in pairs(g.nodes) do
                if node.type == "output" then
                    local innovId = innov:nextConnId(inId, node.id)
                    g:addConnection(require("neat.connection").new(inId, node.id, (math.random()*2-1), true, innovId))
                end
            end
        end
        table.insert(pop.genomes, g)
    end

    return pop
end

-- speciate genomes
function Population:speciate()
    -- clear species members
    for _, s in ipairs(self.species) do s:clear() end
    for _, g in ipairs(self.genomes) do
        local placed = false
        for _, s in ipairs(self.species) do
            local d = g:compatibility(s.representative, {c1=self.cfg.c1, c2=self.cfg.c2, c3=self.cfg.c3})
            if d < self.cfg.compatThreshold then
                s:addMember(g)
                placed = true
                break
            end
        end
        if not placed then
            local s = Species.new(g)
            s:addMember(g)
            table.insert(self.species, s)
        end
    end
    -- remove empty species
    local newSpecies = {}
    for _, s in ipairs(self.species) do
        if #s.members > 0 then
            table.insert(newSpecies, s)
        end
    end
    self.species = newSpecies
end

-- produce networks for evaluation: returns list of {genome, network}
function Population:buildNetworks()
    local out = {}
    for _, g in ipairs(self.genomes) do
        local net = Network.buildFromGenome(g)
        table.insert(out, {genome = g, network = net})
    end
    return out
end

-- evolve using fitnesses (you must set genome.fitness for each genome before calling epoch)
function Population:epoch()
    -- sort genomes by fitness
    table.sort(self.genomes, function(a,b) return a.fitness > b.fitness end)
    if not self.best or self.genomes[1].fitness > self.best.fitness then
        self.best = self.genomes[1]
    end

    -- speciate
    self:speciate()

    -- compute average fitness per species and total
    local totalAdjusted = 0
    for _, s in ipairs(self.species) do
        s:computeAdjustedFitnesses()
        local sum = 0
        for _, g in ipairs(s.members) do sum = sum + (g.adjustedFitness or 0) end
        s.average = sum / math.max(1, #s.members)
        totalAdjusted = totalAdjusted + sum
    end

    -- create next generation
    local newGen = {}

    -- elitism: keep the top genomes
    for _, s in ipairs(self.species) do
        table.sort(s.members, function(a,b) return a.fitness > b.fitness end)
        for i = 1, math.min(self.cfg.elitism, #s.members) do
            table.insert(newGen, util.copy(s.members[i]))
        end
    end

    -- calculate number of children per species proportional to adjusted fitness
    local children = {}
    for _, s in ipairs(self.species) do
        local share = 0
        for _, g in ipairs(s.members) do share = share + (g.adjustedFitness or 0) end
        local nChildren = math.floor((share / math.max(1e-8, totalAdjusted)) * (self.cfg.populationSize - #newGen) + 0.5)
        children[s.id] = nChildren
    end

    -- reproduce
    for _, s in ipairs(self.species) do
        local n = children[s.id] or 0
        if n > 0 then
            -- sort members by fitness
            table.sort(s.members, function(a,b) return a.fitness > b.fitness end)
            -- keep the top X as parents
            for i = 1, n do
                -- tournament or roulette selection inside species
                local a = s.members[math.random(1, math.max(1, math.floor(#s.members * self.cfg.survivalThreshold)))]
                local b = s.members[math.random(1, #s.members)]
                local child
                if a.fitness >= b.fitness then
                    child = a:crossover(b)
                else
                    child = b:crossover(a)
                end
                -- mutate child
                if math.random() < self.cfg.weightPerturbRate then child:mutateWeights(self.cfg) end
                if math.random() < self.cfg.addConnRate then child:mutateAddConnection(self.innovation) end
                if math.random() < self.cfg.addNodeRate then child:mutateAddNode(self.innovation) end
                table.insert(newGen, child)
                if #newGen >= self.cfg.populationSize then break end
            end
        end
        if #newGen >= self.cfg.populationSize then break end
    end

    -- fill with mutated copies if still small
    while #newGen < self.cfg.populationSize do
        local parent = self.genomes[math.random(1,#self.genomes)]
        local copy = util.copy(parent)
        if math.random() < self.cfg.weightPerturbRate then copy:mutateWeights(self.cfg) end
        if math.random() < self.cfg.addConnRate then copy:mutateAddConnection(self.innovation) end
        if math.random() < self.cfg.addNodeRate then copy:mutateAddNode(self.innovation) end
        table.insert(newGen, copy)
    end

    self.genomes = newGen
    self.generation = (self.generation or 1) + 1
end

return Population

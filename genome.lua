-- neat/genome.lua
local Node = require("neat.node")
local Connection = require("neat.connection")
local util = require("neat.util")

local Genome = {}
Genome.__index = Genome

-- genome holds nodes map (id -> Node) and connections map (innovation -> Connection)
function Genome.new()
    return setmetatable({
        nodes = {}, -- id -> Node
        connections = {}, -- innovation -> Connection
        maxNodeId = 0,
    }, Genome)
end

-- convenience to add node
function Genome:addNode(node)
    self.nodes[node.id] = node
    if node.id > (self.maxNodeId or 0) then self.maxNodeId = node.id end
end

function Genome:addConnection(conn)
    self.connections[conn.innovation] = conn
end

-- find connection by from,to
function Genome:hasConnection(from, to)
    for _, c in pairs(self.connections) do
        if c.from == from and c.to == to then return true end
    end
    return false
end

-- mutate weights
function Genome:mutateWeights(cfg)
    for _, c in pairs(self.connections) do
        if math.random() < (cfg.weightPerturbRate or 0.9) then
            -- perturb
            c.weight = c.weight + (math.random() * 2 - 1) * (cfg.weightPerturbStrength or 0.5)
        else
            -- assign new
            c.weight = (math.random() * 2 - 1) * (cfg.weightInitRange or 1)
        end
    end
end

-- add connection mutation
function Genome:mutateAddConnection(innovation, maxAttempts)
    maxAttempts = maxAttempts or 20
    local nodeIds = {}
    for id, _ in pairs(self.nodes) do table.insert(nodeIds, id) end
    for _ = 1, maxAttempts do
        local a = nodeIds[math.random(#nodeIds)]
        local b = nodeIds[math.random(#nodeIds)]
        if a ~= b then
            -- do not connect into input nodes
            if self.nodes[a].type == "output" and self.nodes[b].type == "input" then
                -- avoid backward trivial
            else
                -- check existing
                if not self:hasConnection(a, b) then
                    local innov = innovation:nextConnId(a, b)
                    local conn = require("neat.connection").new(a, b, (math.random()*2-1), true, innov)
                    self:addConnection(conn)
                    return true
                end
            end
        end
    end
    return false
end

-- add node (split connection)
function Genome:mutateAddNode(innovation)
    -- pick a random enabled connection
    local candidates = {}
    for _, c in pairs(self.connections) do
        if c.enabled then table.insert(candidates, c) end
    end
    if #candidates == 0 then return false end
    local c = candidates[math.random(#candidates)]
    c.enabled = false
    local newNodeId = innovation:nextNode()
    local node = require("neat.node").new(newNodeId, "hidden")
    self:addNode(node)
    -- create two connections: from->new, new->to
    local innov1 = innovation:nextConnId(c.from, newNodeId)
    local innov2 = innovation:nextConnId(newNodeId, c.to)
    local Conn = require("neat.connection")
    local conn1 = Conn.new(c.from, newNodeId, 1, true, innov1)
    local conn2 = Conn.new(newNodeId, c.to, c.weight, true, innov2)
    self:addConnection(conn1)
    self:addConnection(conn2)
    return true
end

-- crossover (assumes self.fitness >= other.fitness when using uniform)
function Genome:crossover(other)
    local child = Genome.new()
    -- copy nodes
    for id,node in pairs(self.nodes) do
        child:addNode(require("neat.node").new(node.id, node.type))
    end
    -- combine connections by innovation
    for innov, c1 in pairs(self.connections) do
        local c2 = other.connections[innov]
        if c2 then
            -- matching: pick randomly
            local chosen = (math.random() < 0.5) and c1 or c2
            child:addConnection(require("neat.connection").new(chosen.from, chosen.to, chosen.weight, chosen.enabled, chosen.innovation))
        else
            -- disjoint or excess: inherited from more fit parent (self)
            child:addConnection(require("neat.connection").new(c1.from, c1.to, c1.weight, c1.enabled, c1.innovation))
        end
    end
    return child
end

-- compatibility distance
function Genome:compatibility(other, cfg)
    cfg = cfg or {}
    local c1 = cfg.c1 or 1.0
    local c2 = cfg.c2 or 1.0
    local c3 = cfg.c3 or 0.4
    local n = math.max(#(self.connections), #(other.connections))
    if n < 1 then n = 1 end
    -- build innovation->connection maps
    local innovA = {}
    for k,v in pairs(self.connections) do innovA[k] = v end
    local innovB = {}
    for k,v in pairs(other.connections) do innovB[k] = v end
    local excess = 0
    local disjoint = 0
    local weightDiff = 0
    local matching = 0
    -- find max innovation IDs
    local maxA, maxB = 0,0
    for innov,_ in pairs(innovA) do if innov > maxA then maxA = innov end end
    for innov,_ in pairs(innovB) do if innov > maxB then maxB = innov end end
    local allInnov = {}
    for innov,_ in pairs(innovA) do allInnov[innov]=true end
    for innov,_ in pairs(innovB) do allInnov[innov]=true end
    for innov,_ in pairs(allInnov) do
        local a = innovA[innov]
        local b = innovB[innov]
        if a and b then
            matching = matching + 1
            weightDiff = weightDiff + math.abs(a.weight - b.weight)
        else
            if innov > maxA or innov > maxB then
                excess = excess + 1
            else
                disjoint = disjoint + 1
            end
        end
    end
    local wd = matching > 0 and (weightDiff / matching) or 0
    local distance = (c1 * excess / n) + (c2 * disjoint / n) + (c3 * wd)
    return distance
end

return Genome

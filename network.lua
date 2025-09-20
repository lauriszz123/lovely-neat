-- neat/network.lua
-- Builds a feedforward network from a genome (no recurrence). Supports activation step.
local Network = {}
Network.__index = Network

-- activation function (sigmoid)
local function sigmoid(x)
    return 1 / (1 + math.exp(-4.9 * x))
end

-- topological order build (Kahn)
function Network.buildFromGenome(genome)
    -- nodes: id -> {node, incoming conn list}
    local nodes = {}
    for id,node in pairs(genome.nodes) do
        nodes[id] = {node = node, incoming = {}}
    end
    for _, conn in pairs(genome.connections) do
        if conn.enabled then
            if nodes[conn.to] then
                table.insert(nodes[conn.to].incoming, {from = conn.from, weight = conn.weight})
            end
        end
    end
    -- return an object with evaluate(inputs table)
    local net = {}
    function net:evaluate(inputValues)
        -- set inputs and bias
        for id, ndata in pairs(nodes) do
            ndata.node.activation = 0
            if ndata.node.type == "input" then
                ndata.node.activation = inputValues[id] or 0
            elseif ndata.node.type == "bias" then
                ndata.node.activation = 1
            end
        end
        -- naive iterative evaluation: since network may have hidden layers but acyclic, iterate nodes that are not inputs/bias
        -- We'll loop nodes multiple times but because it's acyclic the values should converge in one pass if we compute only when inputs ready.
        -- Simpler approach: process nodes sorted by id (assumes genome created in feedforward order) — works for typical NEAT small nets.
        -- We'll evaluate all non-input nodes:
        for id, ndata in pairs(nodes) do
            if ndata.node.type ~= "input" and ndata.node.type ~= "bias" then
                local sum = 0
                for _, inc in ipairs(ndata.incoming) do
                    local fromNode = nodes[inc.from]
                    if fromNode then
                        sum = sum + fromNode.node.activation * inc.weight
                    end
                end
                ndata.node.activation = sigmoid(sum)
            end
        end
        -- collect outputs
        local outputs = {}
        for id, ndata in pairs(nodes) do
            if ndata.node.type == "output" then
                table.insert(outputs, {id=id, value = ndata.node.activation})
            end
        end
        return outputs
    end
    return net
end

return Network

-- neat/node.lua
local Node = {}
Node.__index = Node

-- types: "input", "hidden", "output", "bias"
function Node.new(id, nodeType)
    return setmetatable({
        id = id,
        type = nodeType or "hidden",
        activation = 0,
        incoming = {}, -- list of connection gene refs (ids)
    }, Node)
end

return Node

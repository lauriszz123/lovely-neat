-- neat/connection.lua
local Connection = {}
Connection.__index = Connection

function Connection.new(from, to, weight, enabled, innovation)
    return setmetatable({
        from = from,
        to = to,
        weight = weight or (math.random() * 2 - 1),
        enabled = enabled == nil and true or enabled,
        innovation = innovation,
    }, Connection)
end

return Connection

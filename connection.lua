-- neat/connection.lua
local class = require("lovely-neat.modified_middleclass")

---@class Connection: Object
local Connection = class("Connection")

function Connection:initialize(from, to, weight, enabled, innovation)
	self.from = from
	self.to = to
	self.weight = weight or (math.random() * 2 - 1)
	self.enabled = enabled == nil and true or enabled
	self.innovation = innovation
end

return Connection

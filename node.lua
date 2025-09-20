-- neat/node.lua
local class = require("lovely-neat.modified_middleclass")

---@class Node: Object
local Node = class("Node")

-- A node in the neural network
---@param id number
---@param nodeType string @"input" | "hidden" | "output" | "bias"
function Node:initialize(id, nodeType)
	self.id = id
	self.type = nodeType or "hidden"
	self.activation = 0
	self.incoming = {} -- list of connection gene refs (ids)
end

return Node

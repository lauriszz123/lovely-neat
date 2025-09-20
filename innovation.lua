-- neat/innovation.lua
-- Simple global innovation number tracker (module returns an object you should keep).
local class = require("lovely-neat.modified_middleclass")

---@class Innovation: Object
local Innovation = class("Innovation")

function Innovation:initialize()
	self.nextInnovation = 1
	self.connMap = {} -- key: from..":"..to => innovation id
	self.nextNodeId = 1 -- unique node ids
end

function Innovation:nextConnId(from, to)
	local key = tostring(from) .. ":" .. tostring(to)
	if not self.connMap[key] then
		self.connMap[key] = self.nextInnovation
		self.nextInnovation = self.nextInnovation + 1
	end
	return self.connMap[key]
end

function Innovation:nextNode()
	local id = self.nextNodeId
	self.nextNodeId = self.nextNodeId + 1
	return id
end

return Innovation

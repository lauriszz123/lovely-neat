-- neat/innovation.lua
-- Simple global innovation number tracker (module returns an object you should keep).
local Innovation = {}
Innovation.__index = Innovation

function Innovation.new()
	return setmetatable({
		nextInnovation = 1,
		connMap = {}, -- key: from..":"..to => innovation id
		nextNodeId = 1, -- unique node ids
	}, Innovation)
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

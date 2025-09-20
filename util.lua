-- neat/util.lua
local util = {}

function util.shuffle(t)
	for i = #t, 2, -1 do
		local j = math.random(i)
		t[i], t[j] = t[j], t[i]
	end
end

function util.copy(t, seen)
	-- Handle non-table values
	if type(t) ~= "table" then
		return t
	end

	-- Initialize seen table on first call
	seen = seen or {}

	-- Check if we've already copied this table (circular reference)
	if seen[t] then
		return seen[t]
	end

	local o = {}
	-- Mark this table as seen before recursing
	seen[t] = o

	for k, v in pairs(t) do
		if type(v) == "table" then
			o[k] = util.copy(v, seen)
		else
			o[k] = v
		end
	end

	-- Copy metatable if it exists
	local mt = getmetatable(t)
	if mt then
		setmetatable(o, mt)
	end

	return o
end

return util

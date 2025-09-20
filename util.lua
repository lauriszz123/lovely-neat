-- neat/util.lua
local util = {}

function util.shuffle(t)
    for i = #t, 2, -1 do
        local j = math.random(i)
        t[i], t[j] = t[j], t[i]
    end
end

function util.copy(t)
    if type(t) ~= "table" then return t end
    local o = {}
    for k,v in pairs(t) do o[k] = util.copy(v) end
    return o
end

return util

require 'nn'
local autograd = require 'autograd'
local gradcheck = require 'autograd.gradcheck' {randomizeInput = true}
local totem = require 'totem'
local tester = totem.Tester()

------------------------------------------------------------------------------
--  Training Options
------------------------------------------------------------------------------

torch.manualSeed(1)
opt = {}
opt.under = true      -- Set model to under predict.
opt.biasWeight = 0.05 -- controls under / over prediction amount.
d = 3

------------------------------------------------------------------------------
--  Model (Normal NN)
------------------------------------------------------------------------------


local numhid1 = 9
local n_outputs = 1

-- The checkGrad function returns false if the following 4 lines are commented.
-- If they are commented the gradCheck returns -> error=0.12893175662625 false
local model = nn.Sequential()           
model:add(nn.Linear(d, numhid1)) 
model:add(nn.Sigmoid())
model:add(nn.Linear(numhid1, n_outputs))

------------------------------------------------------------------------------
--  AutoGrad MSE Criterion Loss Function designed to over/under predict.
------------------------------------------------------------------------------

local autoBiasedMSECriterion = function(inputs)

  local delta = inputs.T - inputs.x 

  -- Add the following check because the last call to this function
  -- is different to the rest and the code below errors.
  if type(inputs.T)~='table' then
     if opt.under then
        delta[torch.gt(delta,0)] = delta[torch.gt(delta,0)]:mul(opt.biasWeight)
     elseif opt.over then
        delta[torch.lt(delta,0)] = delta[torch.lt(delta,0)]:mul(opt.biasWeight)
     end 
  end   

    loss = torch.sum( torch.cmul(delta, delta) ) / (inputs.x:dim() == 2 and inputs.x:size(1)*inputs.x:size(2) or inputs.x:size(1))

  return loss
end  

criterion = autograd.nn.AutoCriterion('AutoMSE')(autoBiasedMSECriterion)  

------------------------------------------------------------------------------
--  Data
------------------------------------------------------------------------------

targets = torch.Tensor({{1},{8}})
preds = torch.Tensor({{3},{5}}) 

------------------------------------------------------------------------------
--  Train (Not required for gradCheck)
------------------------------------------------------------------------------

-- inputs = torch.Tensor({{1,2,3},{8,4,5}})
-- local outputs = model:forward(inputs)

 
------------------------------------------------------------------------------
--  Grad Check
------------------------------------------------------------------------------ 

-- the following line errors.
-- tester:assert(gradcheck(autoBiasedMSECriterion, {x=preds,T=targets}), 'incorrect gradients')

print(gradcheck(autoBiasedMSECriterion, {x=preds,T=targets}))
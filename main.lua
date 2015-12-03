require 'torch'
require 'optim'
require 'math'
require 'nn'
autograd = require 'autograd'
local train = require 'train'
local gradcheck = require 'autograd.gradcheck' {randomizeInput = true}

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Trains a Regression model that can be biased to under or over predict.')
cmd:text('Example:')
cmd:text('$> th main.lua -under')
cmd:text('Options:')
cmd:option('-under', false, 'train model to under predict')
cmd:option('-over', false, 'train model to over predict')
cmd:text()
local opt = cmd:parse(arg or {})

if not(opt.under or opt.over) 
  then error('-over or -under option has not been passed as a command line argument.') 
end  

torch.manualSeed(1)

------------------------------------------------------------------------------
-- Training Options & Hyperparams
------------------------------------------------------------------------------
opt.model_name = 'model'
opt.optimization = 'sgd'
opt.print_training_loss = 1000
opt.test_model_iteration = 20000 -- how often to print the training & test loss.

local optimState       
local optimMethod      

optimState = {
  learningRate = 1e-1,
  momentum = 0.4,
  learningRateDecay = 1e-4
}
opt.batch_size = 500
opt.epochs = 20000
optimMethod = optim.sgd

-- The lower the bias weight the more the model will under / over predict.
opt.biasWeight = 0.05

------------------------------------------------------------------------------
-- Data
------------------------------------------------------------------------------
data = torch.load('data/boston.t7')

-- Normalize data
mean = data.xe:mean()
std = data.xe:std()

-- train
data.xr:add(-mean)
data.xr:div(std)

-- test
data.xe:add(-mean)
data.xe:div(std)

data.train_data = data.xr
data.train_targets = data.yr
data.test_data = data.xe
data.test_targets = data.ye

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------

local n_inputs = data.train_data:size(2)
local numhid1 = 9
local n_outputs = 1

local model = nn.Sequential()           

model:add(nn.Linear(n_inputs, numhid1)) 
model:add(nn.Sigmoid())
model:add(nn.Linear(numhid1, n_outputs))

------------------------------------------------------------------------------
-- Custom AutoGrad MSE Criterion Loss Function designed to over/under predict.
------------------------------------------------------------------------------
local autoBiasedMSECriterion = function(input, target)

  local delta = target - input 
  if opt.under then
    delta.value[torch.gt(delta.value,0)] = delta.value[torch.gt(delta.value,0)]:mul(opt.biasWeight)
  elseif opt.over then
    delta.value[torch.lt(delta.value,0)] = delta.value[torch.lt(delta.value,0)]:mul(opt.biasWeight)
  end  

  return torch.sum( torch.cmul(delta, delta) ) / (input:dim() == 2 and input:size(1)*input:size(2) or input:size(1))
end  


-- Create an autograd criterion using the loss function above.
criterion = autograd.nn.AutoCriterion('AutoMSE')(autoBiasedMSECriterion)  

------------------------------------------------------------------------------
-- TRAIN
------------------------------------------------------------------------------
train(opt,optimMethod,optimState, data, model, criterion)

------------------------------------------------------------------------------
-- TEST
------------------------------------------------------------------------------
y_hat = model:forward(data.test_data)
diff = y_hat - data.test_targets

print('\n#   prediction     actual      diff')
for i = 1,20 do
    print(string.format("%2d    %6.2f      %6.2f     %6.2f", i,  y_hat[i][1],data.test_targets[i],diff[i][1]))
end
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'util.OneHot'
require 'util.misc'

local batchLoader = require 'util.MinibatchLoader'
local loader = batchLoader.create(100)

local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'

local opt={}
opt.rnn_size = 128
opt.num_layers = 2
opt.seq_length = 100
opt.batch_size = 100
opt.dropout = 0

protos = {}
protos.rnn = RNN.rnn(1, opt.rnn_size, opt.num_layers, opt.dropout)
protos.criterion = nn.ClassNLLCriterion()

init_state={}
for L=1,opt.num_layers do
  local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
  table.insert(init_state, h_init:clone())
end

params, grad_params = model_utils.combine_all_parameters(protos.rnn)

params:uniform(-0.08, 0.08)

clones={}
for name, proto in pairs(protos) do
  print('cloning '..name)
  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

local init_state_global = clone_list(init_state)

function feval(w)
  if w ~= params then
    params:copy(w)
  end
  grad_params:zero()

  local x,y = loader:next_batch()

    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        print(t)
        local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end



local optim_state = {learningRate = 0.002, alpha = 0.95}

for i = 1, 1000 do
  local _, loss = optim.rmsprop(feval, params, optim_state)
  print(i,loss)
end

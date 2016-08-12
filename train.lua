require 'torch'
require 'nn'
require 'optim'
require 'csvigo'
require 'xlua'

function options()
	local cmd = torch.CmdLine()
	cmd:text('Training options:')
	cmd:option('-modelPath', './model/default/', 'Path of nn model')
	cmd:option('-dataPath', './data/', 'Path of data and label')
	cmd:option('-dict', './LM/dict', 'Path and filename of dictionary')
	cmd:option('-langMdl', './LM/', 'Path and filename of word2vec model')
	cmd:option('-logFile', 'log.txt', 'Path and filename of log')
	cmd:option('-vectorSize', 50, 'Size of input vectors')
	cmd:option('-windowSize', 3, 'Window size')
	cmd:option('-batchSize', 5, 'Batch size')
	cmd:option('-nEpochs', 20, 'Number of epochs')
	cmd:option('-learningRate', 1e-3, 'Learning rate')
	cmd:option('-weightDecay', 0, 'Weight decay')
	cmd:option('-momentum', 0, 'Momentum')
	cmd:option('-learningRateDecay', 1e-7, 'Learning rate decay')
	local opt = cmd:parse(arg or {})
	return opt
end

-- check if two tensors equal
function tensorEqual(a, b)
	local result = true
	for i = 1, opt.vectorSize do
		if a[i] ~= b[i] then
			result = false
		end
	end
	return result
end

function load_data(datafile, labelfile, window)
	if window == 3 then
		return load_data_3(datafile, labelfile)
	elseif window == 5 then
		return load_data_5(datafile, labelfile)
	else
		return {}
	end
end

-- core algorithm of converting a file with sequence of vector into tensors
-- then into tables in which input matrices match labels
function load_data_3(datafile, labelfile)
	local label_index = 0
	local front = 1
	local rear = 2
	local text = csvigo.load{ path = datafile, mode = 'raw' }
	local label = csvigo.load{ path = labelfile, mode = 'raw' }
	local txt = torch.Tensor(text)
	local lbl = torch.Tensor(label)
	local dataset = {}
	-- while rear pointer did not reach end of file
	while rear < #text do
		-- if detected zero padding at rear position
		if tensorEqual( txt[rear] , pivot ) then
			-- for each group of 3 vectors
			for i = front, rear-2 do
				-- slice tensor 'txt' and get input
				local input = txt[{ { i , i + 2 } , {} }]
				-- increment of index
				label_index = label_index + 1
				-- get corresponding target
				local target = lbl[label_index]
				-- construct one training example in dataset
				dataset[label_index] = { input , target }
			end
			-- move front pointer to the zero-padding rear pointer is pointing at
			front = rear
			-- move rear pointer forward, since overlapping pointers could not form input matrix
			rear = rear + 1
		end
		-- move rear pointer forward until it detects zero-padding
		rear = rear + 1
	end
	return dataset
end

function load_data_5(datafile, labelfile)
	local label_index = 0
	local front = 1
	local rear = 3
	local text = csvigo.load{ path = datafile, mode = 'raw' }
	local label = csvigo.load{ path = labelfile, mode = 'raw' }
	local txt = torch.Tensor(text)
	local lbl = torch.Tensor(label)
	local dataset = {}
	while rear < #text do
		if tensorEqual( txt[rear] , pivot ) then
			rear = rear + 1
			for i = front, rear-4 do
				local input = txt[{ { i , i + 4 } , {} }]
				label_index = label_index + 1
				local target = lbl[label_index]
				dataset[label_index] = { input , target }
			end
			front = rear - 1
			rear = rear + 1
		end
		rear = rear + 1
	end
	return dataset
end

function model_init()
	os.execute("mkdir "..opt.modelPath)
	local cnn = nn.Sequential()
	cnn:add(nn.TemporalConvolution(opt.vectorSize, 1, opt.windowSize, 1))
	cnn:add(nn.Sigmoid())
	torch.save(opt.modelPath.."0_mdl.t7", cnn)
end

function train(epoch)
	-- log:write("Online epoch: "..epoch.."\n")
	print("Online epoch: "..epoch)
	-- print('Loading dataset ... ')
	src_txt = opt.dataPath..epoch.."_source.txt"
	print("Loading source text: "..src_txt)
	-- log:write("Loading source text: "..src_txt.."\n")
	-- log:flush()
	-- call src2vec.py to generate input vector file
	dataGenCmd = "python src2vec.py "..opt.vectorSize.." "..opt.windowSize.." "..opt.dict.." "..opt.langMdl.." "..src_txt.." "..opt.dataPath.." "..opt.logFile
	os.execute(dataGenCmd)
	-- log:write("Loading label file: "..opt.dataPath..epoch.."_label.csv\n")
	print("Loading label file: "..opt.dataPath..epoch.."_label.csv")
	-- load input vectors and labels
	local dataset = load_data(opt.dataPath..'tmp_data.csv', opt.dataPath..epoch..'_label.csv', opt.windowSize)
	local dataSize = table.getn(dataset)
	print('Loading model ... ')
	-- load model
	local model = torch.load(opt.modelPath..(epoch-1).."_mdl.t7")
	-- get model parameters
	local parameters,gradParameters = model:getParameters()
	-- set loss function
	local loss = nn.BCECriterion()
	print('Training ... ')
	-- log:write("Training data size: "..dataSize.."\n")
	print("Training data size: "..dataSize)
	-- iterate over dataset, with step set by batch size
	for t = 1,dataSize,opt.batchSize do
		xlua.progress(t,dataSize)
		-- construct inputs table and targets table that contain the whole minibatch
		local inputs = {}
		local targets = {}
		-- math.min: just in case the last mini-batch is smaller than designated batch size
		for i = t, math.min(t + opt.batchSize - 1, dataSize) do
			local input = dataset[i][1]
			local target = dataset[i][2]
			table.insert(inputs, input)
        	table.insert(targets, target)
        end
        -- first parameter of sgd: a function to calculate error
        local feval = function(x)
        	model:zeroGradParameters()
        	local f = 0
        	for i = 1, #inputs do
        		-- forward pass
        		local output = model:forward(inputs[i])
        		local err = loss:forward(output,targets[i])
        		f = f + err
        		-- backward pass
        		local grad = loss:backward(output,targets[i])
        		model:backward(inputs[i],grad)
        	end
        	gradParameters:div(#inputs)
        	f = f/#inputs
        	return f, gradParameters
        end
        -- update model parameters
        optim.sgd(feval, parameters, optimState)
    end
    print("Epoch "..epoch.." training finished.")
    -- save intermediate models
    torch.save(opt.modelPath..epoch.."_mdl.t7", model)
    -- remove temp files and collect garbage
    model:clearState()
    --os.execute("rm "..opt.dataPath..'tmp_data.csv')
    collectgarbage()
end

-- train
opt = options()
log = io.open(opt.logFile, "a+")
log:write("------ Training ------\n")
for k, v in pairs(opt) do
   log:write(k..": "..v.."\n")
end
log:flush()
optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	learningRateDecay = opt.learningRateDecay
}
pivot = torch.Tensor(opt.vectorSize):zero()
model_init()
for epoch = 1, opt.nEpochs do
	print('Online epoch: '..epoch)
	train(epoch)
end
log:write("------ End Training ------\n")
log:close()
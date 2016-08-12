require 'nn';
require 'csvigo';

function options()
	local cmd = torch.CmdLine()
	cmd:text('Testing options:')
	cmd:option('-modelPath', './model/default/', 'Path of nn model')
	cmd:option('-dataPath', './data/', 'Path of data and label')
	cmd:option('-dict', './LM/dict', 'Path and filename of dictionary')
	cmd:option('-langMdl', './LM/', 'Path and filename of word2vec model')
	cmd:option('-logFile', 'log.txt', 'Path and filename of log')
	cmd:option('-dataNo', 1, 'NO. of testing dataset')
	cmd:option('-modelNo', 1, 'NO. of model to be tested')
	cmd:option('-vectorSize', 50, 'Size of input vectors')
	cmd:option('-windowSize', 3, 'Window size')
	local opt = cmd:parse(arg or {})
	return opt
end

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

function load_data_3(datafile, labelfile)
	print("Loading dataset ... ")
	local label_index = 0
	local front = 1
	local rear = 2
	local text = csvigo.load{ path = datafile, mode = 'raw' }
	local label = csvigo.load{ path = labelfile, mode = 'raw' }
	local txt = torch.Tensor(text)
	local lbl = torch.Tensor(label)
	local dataset = {}
	while rear < #text do
		if tensorEqual( txt[rear] , pivot ) then
			for i = front, rear-2 do
				local input = txt[{ { i , i + 2 } , {} }]
				label_index = label_index + 1
				local target = lbl[label_index]
				dataset[label_index] = { input , target }
			end
			front = rear
			rear = rear + 1
		end
		rear = rear + 1
	end
	return dataset
end

function load_data_5(datafile, labelfile)
	print("Loading dataset ... ")
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

function measure(model, dataset)
	gold_std = 0	-- number of errors in golden standard
	recall_count = 0	-- number of errors in golden standard hit by model
	precision_count = 0	-- number of errors reported by model correctly
	err_report = 0	-- number of errors reported by model
	local dataSize = table.getn(dataset)

	for i = 1, dataSize do	
		local output = model:forward(dataset[i][1])
		if dataset[i][2][1] == 0 then
			gold_std = gold_std + 1
			if output[1][1] < 0.5 then
				recall_count = recall_count + 1
			end
		end
		if output[1][1] < 0.5 then
			err_report = err_report + 1
			if dataset[i][2][1] == 0 then
				precision_count = precision_count + 1
			end
		end
	end

	recall = recall_count / gold_std
	precision = precision_count / err_report
	f_0_5 = ( ( 1 + 0.25 ) * recall * precision ) / ( recall + 0.25 * precision )

	log:write("recall = "..recall.."\n")
	log:write("precision = "..precision.."\n")
	log:write("F_0.5 = "..f_0_5.."\n")
	log:flush()

	print(gold_std)
	print(err_report)

	print("\nrecall = ", recall)
	print("\nprecision = ", precision)
	print("\nF_0.5 = ", f_0_5)
end

function test()
	print('Loading test dataset ... ')
	src_txt = opt.dataPath..opt.dataNo.."_source.txt"
	dataGenCmd = "python src2vec.py "..opt.vectorSize.." "..opt.windowSize.." "..opt.dict.." "..opt.langMdl.." "..src_txt.." "..opt.dataPath.." "..opt.logFile
	-- call src2vec.py to generate input vector file
	os.execute(dataGenCmd)
	-- load dataset
	local dataset = load_data(opt.dataPath..'tmp_data.csv', opt.dataPath..opt.dataNo..'_label.csv', opt.windowSize)	
	print('Loading model ... ')
	-- load model
	local model = torch.load(opt.modelPath..opt.modelNo.."_mdl.t7")
	-- do test and calculate F_0.5
	measure(model,dataset)
	model:clearState()
    os.execute("rm "..opt.dataPath..'tmp_data.csv')
    collectgarbage()
end

-- test
opt = options()
log = io.open(opt.logFile, "a+")
log:write("------ Testing ------\n")
for k, v in pairs(opt) do
   log:write(k..": "..v.."\n")
end
log:flush()
pivot = torch.Tensor(opt.vectorSize):zero()
test()
log:write("------ End Testing ------\n")
log:close()
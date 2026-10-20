function model = createDataset(model)
% This function creates or loads one of the datasets and assigns them to X
% and Y matrices. The synthetic datasets accept a noise level assigned to
% them.

if  strcmp(model.datasetName, "concrete")
    opts = delimitedTextImportOptions("NumVariables", 9);

    % Specify range and delimiter
    opts.DataLines = [2, Inf];
    opts.Delimiter = ",";

    % Specify column names and types
    opts.VariableNames = ["cement", "slag", "flyash", "water", "superplasticizer", "coarseaggregate", "fineaggregate", "age", "csMPa"];
    opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    % Import the data
    ConcreteDataYeh = readtable("Concrete_Data_Yeh.csv", opts);

    %% Convert to output type
    ConcreteDataYeh = table2array(ConcreteDataYeh);

    %% Clear temporary variables
    clear opts

    model.x = ConcreteDataYeh(:, 1:end-1);
    model.y = ConcreteDataYeh(:, end);
elseif strcmp(model.datasetName, "synthetic-circles-classification")
    
    currentFile     = mfilename('fullpath');
    [pathstr, ~, ~] = fileparts(currentFile);
    paths           = fullfile(pathstr, 'datasets/synthetic');
    addpath(paths)
    load('data.mat');
    model.ycode           = Y;
    model.y               = dummyvar(Y);
    model.x               = X;

end
end
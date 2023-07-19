%This file passes a high pass filter on certain columns in the test data

%Get directories
all_folders = dir('Unfiltered data')
clear file_name
clear read_dir
clear folder
clear CSVs

for i = 3:4
    folder=getfield(all_folders, {i}, 'name'); %index starts from 3
    read_dir=string(folder)+'/ES';
    CSVs = {dir('Unfiltered data/'+read_dir).name};
    iter = max(size(CSVs));
    for j = 3:iter
        file_name=CSVs(j);
        [a, b, ext] = fileparts(file_name);
        file_name=read_dir+'/'+file_name;
        if strcmp(ext, '.csv')
            apply_filter(file_name);
        end
    end
end

function apply_filter(file_name)
    %Read in data from CSVs (Ignoring AE file for now)
    data = csvread('Unfiltered data/' + file_name, 4, 0);

    %Apply high pass filters
    %Columns wanted are 2,4,6,8
    fs=3000;
    fpass=10;
    encoder_data = highpass(data(:,2),fpass,fs);
    S1_data = highpass(data(:,4),fpass,fs);
    S2_data = highpass(data(:,6),fpass,fs);
    vib_data = highpass(data(:,8),fpass,fs);
    %Rewrite data to new CSV file with headers

    write_dir = 'Filtered data/' + file_name;
    write_data=[encoder_data S1_data S2_data vib_data];
    writematrix(write_data, write_dir);
end




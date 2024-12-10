addpath('meanshift');
addpath('libsvm/')
addpath('./libsvm/matlab/');
addpath('./etract_feature/');
addpath('./utils/');
addpath('./meanshift_edison_matlab_interface-master/');

% fid=fopen('UIUC_file.txt');
% i=0;
% while ~feof(fid)
%     aline=fgetl(fid);
%     
%     annotate(aline);
%     i = i + 1
% end

% if i < 2598
%         i = i + 1;
%         continue;
%     end

%url = 'data/_MG_5885.jpg';
% 取得 data 資料夾內所有 JPG 圖片的路徑
image_files = dir('data/*.jpg'); 

% 遍歷每個圖片進行處理
for k = 1:length(image_files)
    % 取得圖片檔案的完整路徑
    input_filename = fullfile(image_files(k).folder, image_files(k).name);
    
    % 取得圖片的基礎名稱（不含副檔名）
    [~, base_name, ~] = fileparts(image_files(k).name);
    
    % 呼叫 detect 函數，取得處理所需的參數
    [seg, segnum, between, near, centroids, label, grad, texthist] = detect(input_filename);
    
    % 呼叫 removal 函數，並根據輸入圖片名稱輸出結果
    output_filename = fullfile('output', [base_name, '.jpg']); % 在 output 資料夾中存儲
    removal(seg, segnum, between, label, near, centroids, input_filename, grad, texthist, output_filename);
    
    disp(['Processed and saved: ', output_filename]);
end

% url = 'data/9.jpg';
% [seg, segnum, between, near, centroids, label, grad, texthist] = detect(url);
% removal(seg, segnum, between, label, near, centroids, url, grad, texthist);
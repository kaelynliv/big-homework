function main
    % 初始化全局变量
    noisyImg = []; % 用于存储添加噪声后的图像

    % 创建图形化界面
    hFig = figure('Name', '图像处理GUI', 'NumberTitle', 'off', 'MenuBar', 'none', ...
                  'Position', [300, 200, 1000, 600]);

    % 添加控件
    uicontrol('Style', 'pushbutton', 'String', '打开图像', ...
              'Position', [50, 520, 100, 30], 'Callback', @openImage);
    uicontrol('Style', 'pushbutton', 'String', '直方图操作', ...
              'Position', [50, 470, 100, 30], 'Callback', @histogramOperations);
    uicontrol('Style', 'pushbutton', 'String', '对比度增强', ...
              'Position', [50, 420, 100, 30], 'Callback', @contrastEnhancement);
    uicontrol('Style', 'pushbutton', 'String', '几何变换', ...
              'Position', [50, 370, 100, 30], 'Callback', @geometricTransform);
    uicontrol('Style', 'pushbutton', 'String', '添加噪声', ...
              'Position', [50, 320, 100, 30], 'Callback', @addNoise);
    uicontrol('Style', 'pushbutton', 'String', '滤波处理', ...
              'Position', [50, 270, 100, 30], 'Callback', @applyFilter);
    uicontrol('Style', 'pushbutton', 'String', '边缘提取', ...
              'Position', [50, 220, 100, 30], 'Callback', @edgeDetection);
    uicontrol('Style', 'pushbutton', 'String', '目标提取', ...
              'Position', [50, 170, 100, 30], 'Callback', @objectExtraction);
    uicontrol('Style', 'pushbutton', 'String', '特征提取', ...
              'Position', [50, 120, 100, 30], 'Callback', @featureExtraction);

    % 创建显示区
    axes('Units', 'pixels', 'Position', [200, 150, 700, 400]);
    setappdata(hFig, 'Image', []);
    setappdata(hFig, 'NoisyImage', []); % 用于存储噪声图像

    %% 嵌套函数定义

    % 打开图像
    function openImage(~, ~)
        [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', '图像文件 (*.jpg, *.png, *.bmp)'});
        if isequal(filename, 0)
            return;
        end
        img = imread(fullfile(pathname, filename));
        setappdata(hFig, 'Image', img);
        imshow(img, []);
    end

    % 直方图操作
    function histogramOperations(~, ~)
        img = getappdata(hFig, 'Image');
        if isempty(img)
            msgbox('请先打开图像！', '错误', 'error');
            return;
        end

        choice = questdlg('选择直方图操作：', '直方图操作', ...
                          '显示灰度直方图', '直方图均衡化', '直方图匹配', '显示灰度直方图');

        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end

        switch choice
            case '显示灰度直方图'
                figure, imhist(grayImg);
                title('灰度直方图');

            case '直方图均衡化'
                eqImg = histeq(grayImg);
                figure, imshow(eqImg, []);
                title('直方图均衡化');

            case '直方图匹配'
                [refFilename, refPathname] = uigetfile({'*.jpg;*.png;*.bmp', '图像文件 (*.jpg, *.png, *.bmp)'});
                if isequal(refFilename, 0)
                    msgbox('未选择参考图像！', '错误', 'error');
                    return;
                end
                refImg = imread(fullfile(refPathname, refFilename));
                if size(refImg, 3) == 3
                    refGray = rgb2gray(refImg);
                else
                    refGray = refImg;
                end
                matchedImg = imhistmatch(grayImg, refGray);
                figure, imshow(matchedImg, []);
                title('直方图匹配');
        end
    end

    % 对比度增强
    function contrastEnhancement(~, ~)
        img = getappdata(hFig, 'Image');
        if isempty(img)
            msgbox('请先打开图像！', '错误', 'error');
            return;
        end

        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end

        choice = questdlg('选择对比度增强方法：', '对比度增强', ...
                          '线性变换', '对数变换', '指数变换', '线性变换');

        switch choice
            case '线性变换'
                enhancedImg = imadjust(grayImg, stretchlim(grayImg), []);
                figure, imshow(enhancedImg, []);
                title('线性变换增强');

            case '对数变换'
                c = 255 / log(1 + double(max(grayImg(:))));
                logImg = c * log(1 + double(grayImg));
                logImg = uint8(logImg);
                figure, imshow(logImg, []);
                title('对数变换增强');

            case '指数变换'
                gamma = 2;
                expImg = imadjust(grayImg, [], [], gamma);
                figure, imshow(expImg, []);
                title('指数变换增强');
        end
    end

    % 几何变换（缩放和旋转）
    function geometricTransform(~, ~)
        img = getappdata(hFig, 'Image');
        if isempty(img)
            msgbox('请先打开图像！', '错误', 'error');
            return;
        end

        choice = questdlg('选择几何变换类型：', '几何变换', ...
                          '缩放', '旋转', '取消', '缩放');
        switch choice
            case '缩放'
                scaleFactor = inputdlg('输入缩放比例（例如：0.5 表示缩小为一半）：');
                if isempty(scaleFactor)
                    return;
                end
                scaleFactor = str2double(scaleFactor{1});
                if isnan(scaleFactor) || scaleFactor <= 0
                    msgbox('缩放比例必须是正数！', '错误', 'error');
                    return;
                end
                scaledImg = imresize(img, scaleFactor);
                figure, imshow(scaledImg);
                title(['缩放变换，比例：' num2str(scaleFactor)]);
            case '旋转'
                angle = inputdlg('输入旋转角度（例如：45）：');
                if isempty(angle)
                    return;
                end
                angle = str2double(angle{1});
                if isnan(angle)
                    msgbox('旋转角度必须是数字！', '错误', 'error');
                    return;
                end
                rotatedImg = imrotate(img, angle);
                figure, imshow(rotatedImg);
                title(['旋转变换，角度：' num2str(angle)]);
        end
    end

    % 添加噪声
    function addNoise(~, ~)
        img = getappdata(hFig, 'Image');
        if isempty(img)
            msgbox('请先打开图像！', '错误', 'error');
            return;
        end

        % 用户选择噪声类型
        noiseType = questdlg('选择噪声类型：', '噪声类型', '高斯噪声', '椒盐噪声', '高斯噪声');
        if isempty(noiseType)
            return;
        end

        % 根据噪声类型获取用户输入的参数
        switch noiseType
            case '高斯噪声'
                prompt = {'请输入高斯噪声的均值（默认：0）：', '请输入高斯噪声的方差（默认：0.01）：'};
                dlgTitle = '高斯噪声参数';
                defaultVals = {'0', '0.01'};
                userInput = inputdlg(prompt, dlgTitle, [1 50], defaultVals);
                if isempty(userInput)
                    return;
                end
                meanVal = str2double(userInput{1});
                varVal = str2double(userInput{2});
                if isnan(meanVal) || isnan(varVal) || varVal <= 0
                    msgbox('均值或方差无效，请输入有效数值！', '错误', 'error');
                    return;
                end
                noisyImg = imnoise(img, 'gaussian', meanVal, varVal);

            case '椒盐噪声'
                prompt = {'请输入椒盐噪声的密度（默认：0.05，范围：[0, 1]）：'};
                dlgTitle = '椒盐噪声参数';
                defaultVal = {'0.05'};
                userInput = inputdlg(prompt, dlgTitle, [1 50], defaultVal);
                if isempty(userInput)
                    return;
                end
                density = str2double(userInput{1});
                if isnan(density) || density <= 0 || density > 1
                    msgbox('噪声密度无效，请输入范围在 (0, 1] 的有效数值！', '错误', 'error');
                    return;
                end
                noisyImg = imnoise(img, 'salt & pepper', density);
        end

        % 显示带噪声的图像
        figure, imshow(noisyImg);
        title(['添加噪声后的图像（', noiseType, '）']);
        setappdata(hFig, 'NoisyImage', noisyImg);

        % 自动引导用户进行滤波处理
        choice = questdlg('是否对噪声图像进行滤波处理？', '滤波处理', '是', '否', '是');
        if strcmp(choice, '是')
            applyFilter();
        end
    end

    % 滤波处理
    function applyFilter(~, ~)
        hFig = gcf; % 获取当前图形窗口句柄
        noisyImg = getappdata(hFig, 'NoisyImage');

        if isempty(noisyImg)
            msgbox('噪声图像不存在，请先添加噪声！', '错误', 'error');
            return;
        end

        % 选择滤波方式
        filterType = questdlg('选择滤波方式：', '滤波处理', ...
                              '空域滤波', '频域滤波', '取消', '空域滤波');
        if isempty(filterType) || strcmp(filterType, '取消')
            return;
        end

        % 空域和频域滤波处理
        switch filterType
            case '空域滤波'
                spatialFilter = questdlg('选择空域滤波方法：', '空域滤波', ...
                                         '均值滤波', '中值滤波', '均值滤波');
                switch spatialFilter
                    case '均值滤波'
                        h = fspecial('average', [3 3]);
                        filteredImg = imfilter(noisyImg, h);
                        figure, imshow(filteredImg);
                        title('均值滤波处理后的图像');
                    case '中值滤波'
                        filteredImg = medfilt2(rgb2gray(noisyImg));
                        figure, imshow(filteredImg);
                        title('中值滤波处理后的图像');
                end

            case '频域滤波'
                grayImg = rgb2gray(noisyImg);
                fftImg = fft2(double(grayImg));
                fftShift = fftshift(fftImg);
                magnitude = log(1 + abs(fftShift));
                figure, imshow(magnitude, []);
                title('频域图像（傅里叶频谱）');

                freqFilter = questdlg('选择频域滤波方法：', '频域滤波', ...
                                      '低通滤波', '高通滤波', '低通滤波');
                if isempty(freqFilter)
                    return;
                end

                [M, N] = size(grayImg);
                D0 = 50;
                [U, V] = meshgrid(1:N, 1:M);
                D = sqrt((U - N/2).^2 + (V - M/2).^2);

                switch freqFilter
                    case '低通滤波'
                        H = double(D <= D0);
                    case '高通滤波'
                        H = double(D > D0);
                end

                filteredFFT = fftShift .* H;
                filteredImg = ifft2(ifftshift(filteredFFT));
                filteredImg = abs(filteredImg);

                figure, imshow(uint8(filteredImg), []);
                title(['频域滤波处理后的图像（', freqFilter, '）']);
        end
    end

  % 边缘提取
function edgeDetection(~, ~)
    img = getappdata(hFig, 'Image');
    if isempty(img)
        msgbox('请先打开图像！', '错误', 'error');
        return;
    end
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    % 使用 listdlg 提供多选项支持
    options = {'Roberts', 'Prewitt', 'Sobel', 'Laplacian'};
    [selection, ok] = listdlg('PromptString', '选择边缘检测算子：', ...
                              'SelectionMode', 'single', ...
                              'ListString', options);

    if ~ok
        msgbox('未选择任何边缘检测算子！', '提示', 'warn');
        return;
    end

    % 根据用户选择的算子进行边缘检测
    edgeType = options{selection};
    switch edgeType
        case 'Roberts'
            edges = edge(grayImg, 'Roberts');
        case 'Prewitt'
            edges = edge(grayImg, 'Prewitt');
        case 'Sobel'
            edges = edge(grayImg, 'Sobel');
        case 'Laplacian'
            edges = edge(grayImg, 'log'); % Laplacian of Gaussian
    end

    % 显示边缘提取结果
    figure, imshow(edges);
    title(['使用' edgeType '算子检测的边缘']);
end

 % 目标提取
    function objectExtraction(~, ~)
        img = getappdata(hFig, 'Image');
        if isempty(img)
            msgbox('请先打开图像！', '错误', 'error');
            return;
        end
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end
        threshold = graythresh(grayImg);
        binaryImg = imbinarize(grayImg, threshold);
        figure, imshow(binaryImg);
        title('目标提取后的二值图像');
    end

 % 特征提取
function featureExtraction(~, ~)
    img = getappdata(hFig, 'Image');
    if isempty(img)
        msgbox('请先打开图像！', '错误', 'error');
        return;
    end
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    %% 1. 原始图像特征提取
    % LBP 特征提取并可视化
    lbpOriginal = extractLBPFeatures(grayImg, 'Upright', false);
    lbpImgOriginal = extractLBPImage(grayImg); % 自定义函数，生成 LBP 可视化图像

    % HOG 特征提取并可视化
    [hogOriginal, visOriginal] = extractHOGFeatures(grayImg);

    % 显示原始图像的 LBP 和 HOG 可视化
    figure, subplot(1, 2, 1), imshow(lbpImgOriginal, []);
    title('原始图像的 LBP 可视化');
    subplot(1, 2, 2), imshow(grayImg);
    hold on, plot(visOriginal);
    title('原始图像的 HOG 可视化');

    %% 2. 目标提取后的特征提取
    % 目标提取后的二值图像
    threshold = graythresh(grayImg);
    binaryImg = imbinarize(grayImg, threshold);

    % LBP 特征提取并可视化
    lbpTarget = extractLBPFeatures(binaryImg, 'Upright', false);
    lbpImgTarget = extractLBPImage(binaryImg); % 自定义函数，生成 LBP 可视化图像

    % HOG 特征提取并可视化
    [hogTarget, visTarget] = extractHOGFeatures(binaryImg);

    % 显示目标提取后的 LBP 和 HOG 可视化
    figure, subplot(1, 2, 1), imshow(lbpImgTarget, []);
    title('目标提取后图像的 LBP 可视化');
    subplot(1, 2, 2), imshow(binaryImg);
    hold on, plot(visTarget);
    title('目标提取后图像的 HOG 可视化');

    %% 3. 显示特征维数信息
    msgbox({['原始图像 LBP 特征维数: ', num2str(length(lbpOriginal))], ...
            ['原始图像 HOG 特征维数: ', num2str(length(hogOriginal))], ...
            ['目标提取后 LBP 特征维数: ', num2str(length(lbpTarget))], ...
            ['目标提取后 HOG 特征维数: ', num2str(length(hogTarget))]}, ...
           '特征提取结果');
end

%% 辅助函数：生成 LBP 可视化图像
function lbpImg = extractLBPImage(grayImg)
    % 检查输入图像是否为灰度图
    if size(grayImg, 3) == 3
        grayImg = rgb2gray(grayImg);
    end

    % 初始化 LBP 图像
    lbpImg = zeros(size(grayImg));

    % 定义邻域和 LBP 半径
    radius = 1;
    neighbors = 8;

    % 对每个像素计算 LBP
    for i = 1 + radius : size(grayImg, 1) - radius
        for j = 1 + radius : size(grayImg, 2) - radius
            % 获取邻域像素值
            patch = grayImg(i - radius:i + radius, j - radius:j + radius);

            % 计算中心像素和邻域的差值
            center = grayImg(i, j);
            binaryPattern = patch >= center;

            % 二进制编码
            binaryPattern = binaryPattern(:)';
            binaryPattern(5) = []; % 移除中心像素
            lbpValue = bi2de(binaryPattern);

            % 设置 LBP 图像值
            lbpImg(i, j) = lbpValue;
        end
    end

    % 归一化到 [0, 255]
    lbpImg = uint8(255 * (lbpImg / max(lbpImg(:))));
end

end
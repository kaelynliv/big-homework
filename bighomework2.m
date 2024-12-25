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
    uicontrol('Style', 'pushbutton', 'String', '噪声与滤波', ...
              'Position', [50, 320, 100, 30], 'Callback', @addNoiseAndFilter); % 合并后的控件
    uicontrol('Style', 'pushbutton', 'String', '边缘提取', ...
              'Position', [50, 270, 100, 30], 'Callback', @edgeDetection);
    uicontrol('Style', 'pushbutton', 'String', '目标提取', ...
              'Position', [50, 220, 100, 30], 'Callback', @objectExtraction);
    uicontrol('Style', 'pushbutton', 'String', '特征提取', ...
              'Position', [50, 170, 100, 30], 'Callback', @featureExtraction);

    % 创建显示区
    axes('Units', 'pixels', 'Position', [200, 150, 700, 400]);
    setappdata(hFig, 'Image', []);

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

    % 几何变换
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
    % 噪声与滤波（改进后的功能）
    function addNoiseAndFilter(~, ~)
        img = getappdata(hFig, 'Image');
        if isempty(img)
            msgbox('请先打开图像！', '错误', 'error');
            return;
        end

        % 步骤 1：用户选择噪声类型
        noiseType = questdlg('选择噪声类型：', '噪声类型', '高斯噪声', '椒盐噪声', '取消');
        if isempty(noiseType) || strcmp(noiseType, '取消')
            % 如果用户选择取消或关闭对话框，直接返回
            return;
        end

        % 根据噪声类型输入参数
        switch noiseType
            case '高斯噪声'
                prompt = {'请输入高斯噪声的均值（默认：0）：', '请输入高斯噪声的方差（默认：0.01）：'};
                dlgTitle = '高斯噪声参数';
                defaultVals = {'0', '0.01'};
                userInput = inputdlg(prompt, dlgTitle, [1 50], defaultVals);
                if isempty(userInput) % 如果用户关闭输入框
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
                prompt = {'请输入椒盐噪声的密度（默认：0.05）：'};
                dlgTitle = '椒盐噪声参数';
                defaultVal = {'0.05'};
                userInput = inputdlg(prompt, dlgTitle, [1 50], defaultVal);
                if isempty(userInput) % 如果用户关闭输入框
                    return;
                end
                density = str2double(userInput{1});
                if isnan(density) || density <= 0 || density > 1
                    msgbox('密度无效，请输入范围在 (0, 1] 的有效数值！', '错误', 'error');
                    return;
                end
                noisyImg = imnoise(img, 'salt & pepper', density);
        end

        % 显示带噪声的图像
        setappdata(hFig, 'Image', noisyImg);
        imshow(noisyImg, []);
        title(['添加噪声后的图像（', noiseType, '）']);

        % 步骤 2：用户选择滤波类型
        filterCategory = questdlg('选择滤波类型：', '滤波类型', '空域滤波', '频域滤波', '取消');
        if isempty(filterCategory) || strcmp(filterCategory, '取消')
            % 如果用户选择取消或关闭对话框，直接返回
            return;
        end

        switch filterCategory
            case '空域滤波'
                % 选择具体空域滤波方法
                filterType = questdlg('选择空域滤波方法：', '空域滤波', '均值滤波', '中值滤波', '取消');
                if isempty(filterType) || strcmp(filterType, '取消')
                    return;
                end
                switch filterType
                    case '均值滤波'
                        h = fspecial('average', [3 3]);
                        filteredImg = imfilter(noisyImg, h);
                        figure, imshow(filteredImg, []);
                        title('均值滤波处理后的图像');
                    case '中值滤波'
                        if size(noisyImg, 3) == 3
                            noisyImg = rgb2gray(noisyImg);
                        end
                        filteredImg = medfilt2(noisyImg);
                        figure, imshow(filteredImg, []);
                        title('中值滤波处理后的图像');
                end

            case '频域滤波'
                % 转换为灰度图
                if size(noisyImg, 3) == 3
                    noisyImg = rgb2gray(noisyImg);
                end
                % 傅里叶变换
                fftImg = fft2(double(noisyImg));
                fftShift = fftshift(fftImg);
                magnitude = log(1 + abs(fftShift));
                figure, imshow(magnitude, []);
                title('频域图像（傅里叶频谱）');

                % 选择具体频域滤波方法
                freqFilter = questdlg('选择频域滤波方法：', '频域滤波', '低通滤波', '高通滤波', '取消');
                if isempty(freqFilter) || strcmp(freqFilter, '取消')
                    return;
                end

                % 设置截止频率
                prompt = {'请输入截止频率（默认值：50）：'};
                dlgTitle = '频域滤波参数';
                defaultVal = {'50'};
                userInput = inputdlg(prompt, dlgTitle, [1 50], defaultVal);
                if isempty(userInput)
                    return;
                end
                D0 = str2double(userInput{1});
                if isnan(D0) || D0 <= 0
                    msgbox('截止频率无效，请输入有效数值！', '错误', 'error');
                    return;
                end

                [M, N] = size(noisyImg);
                [U, V] = meshgrid(1:N, 1:M);
                D = sqrt((U - N/2).^2 + (V - M/2).^2);

                switch freqFilter
                    case '低通滤波'
                        H = double(D <= D0);
                    case '高通滤波'
                        H = double(D > D0);
                end

                % 应用频域滤波
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
        edgeType = questdlg('选择边缘检测算子：', '边缘检测', 'Sobel', 'Prewitt', 'Laplacian', 'Sobel');
        switch edgeType
            case 'Sobel'
                edges = edge(grayImg, 'Sobel');
            case 'Prewitt'
                edges = edge(grayImg, 'Prewitt');
            case 'Laplacian'
                edges = edge(grayImg, 'log');
        end
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

        lbpFeatures = extractLBPFeatures(grayImg);
        [hogFeatures, visualization] = extractHOGFeatures(grayImg);

        figure, imshow(grayImg);
        hold on;
        plot(visualization);
        title('HOG 特征可视化');

        msgbox(['LBP 特征维数: ' num2str(length(lbpFeatures)) ', HOG 特征维数: ' num2str(length(hogFeatures))], ...
               '特征提取结果');
    end
end 
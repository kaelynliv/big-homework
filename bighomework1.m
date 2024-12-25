function main
    % 创建图形化界面
    hFig = figure('Name', '图像处理GUI', 'NumberTitle', 'off', 'MenuBar', 'none', ...
                  'Position', [300, 200, 1000, 600]);

    % 添加控件
    uicontrol('Style', 'pushbutton', 'String', '打开图像', ...
              'Position', [50, 520, 100, 30], 'Callback', @openImage);
    uicontrol('Style', 'pushbutton', 'String', '灰度直方图', ...
              'Position', [50, 470, 100, 30], 'Callback', @showHistogram);
    uicontrol('Style', 'pushbutton', 'String', '直方图均衡化', ...
              'Position', [50, 420, 100, 30], 'Callback', @histEqualization);
    uicontrol('Style', 'pushbutton', 'String', '直方图匹配', ...
              'Position', [50, 370, 100, 30], 'Callback', @histMatching);
    uicontrol('Style', 'pushbutton', 'String', '图像缩放', ...
              'Position', [50, 320, 100, 30], 'Callback', @scaleImage);
    uicontrol('Style', 'pushbutton', 'String', '图像旋转', ...
              'Position', [50, 270, 100, 30], 'Callback', @rotateImage);
    uicontrol('Style', 'pushbutton', 'String', '添加噪声', ...
              'Position', [50, 220, 100, 30], 'Callback', @addNoise);
    uicontrol('Style', 'pushbutton', 'String', '边缘提取', ...
              'Position', [50, 170, 100, 30], 'Callback', @edgeDetection);
    uicontrol('Style', 'pushbutton', 'String', '目标提取', ...
              'Position', [50, 120, 100, 30], 'Callback', @objectExtraction);
    uicontrol('Style', 'pushbutton', 'String', '特征提取', ...
              'Position', [50, 70, 100, 30], 'Callback', @featureExtraction);

    % 创建显示区
    axes('Units', 'pixels', 'Position', [200, 150, 700, 400]);
    setappdata(hFig, 'Image', []);

    % 嵌套函数定义
    function openImage(~, ~)
        [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', '图像文件 (*.jpg, *.png, *.bmp)'});
        if isequal(filename, 0)
            return;
        end
        img = imread(fullfile(pathname, filename));
        setappdata(hFig, 'Image', img);
        imshow(img, []);
    end

    function showHistogram(~, ~)
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
        figure, imhist(grayImg);
        title('灰度直方图');
    end

    function histEqualization(~, ~)
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
        eqImg = histeq(grayImg);
        figure, imshow(eqImg, []);
        title('直方图均衡化');
    end

    function histMatching(~, ~)
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
        refImg = imread('reference.jpg'); % 需要用户准备参考图像
        if size(refImg, 3) == 3
            refGray = rgb2gray(refImg);
        else
            refGray = refImg;
        end
        matchedImg = imhistmatch(grayImg, refGray);
        figure, imshow(matchedImg, []);
        title('直方图匹配');
    end

    function scaleImage(~, ~)
        img = getappdata(hFig, 'Image');
        if isempty(img)
            msgbox('请先打开图像！', '错误', 'error');
            return;
        end
        scaleFactor = inputdlg('输入缩放比例（例如：0.5 表示缩小为一半）：');
        scaleFactor = str2double(scaleFactor{1});
        scaledImg = imresize(img, scaleFactor);
        figure, imshow(scaledImg);
        title('缩放后的图像');
    end

    function rotateImage(~, ~)
        img = getappdata(hFig, 'Image');
        if isempty(img)
            msgbox('请先打开图像！', '错误', 'error');
            return;
        end
        angle = inputdlg('输入旋转角度（例如：45）：');
        angle = str2double(angle{1});
        rotatedImg = imrotate(img, angle);
        figure, imshow(rotatedImg);
        title('旋转后的图像');
    end

    function addNoise(~, ~)
        img = getappdata(hFig, 'Image');
        if isempty(img)
            msgbox('请先打开图像！', '错误', 'error');
            return;
        end
        noiseType = questdlg('选择噪声类型：', '噪声类型', '高斯噪声', '椒盐噪声', '高斯噪声');
        if strcmp(noiseType, '高斯噪声')
            noisyImg = imnoise(img, 'gaussian');
        else
            noisyImg = imnoise(img, 'salt & pepper');
        end
        figure, imshow(noisyImg);
        title('添加噪声后的图像');
    end

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
                edges = edge(grayImg, 'log'); % 拉普拉斯近似
        end
        figure, imshow(edges);
        title(['使用' edgeType '算子检测的边缘']);
    end

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

        % LBP 特征提取
        lbpFeatures = extractLBPFeatures(grayImg);

        % HOG 特征提取
        [hogFeatures, visualization] = extractHOGFeatures(grayImg);

        % 显示 HOG 可视化
        figure, imshow(grayImg);
        hold on;
        plot(visualization);
        title('HOG 特征可视化');

        % 显示特征维数
        msgbox(['LBP 特征维数: ' num2str(length(lbpFeatures)) ', HOG 特征维数: ' num2str(length(hogFeatures))], ...
               '特征提取结果');
    end
end

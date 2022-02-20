% Na 2014
clear; close all;

% num_img = [0, 1, 5, 8, 17, 18];
num_img = [13];

for ni = num_img
    dir_imgs = '../data/testimgs/';
    try
        img = imread(sprintf('%s%d.png', dir_imgs, 100000 + ni));
    catch
        img = imread(sprintf('%s%d.jpg', dir_imgs, 100000 + ni));
    end

    % 2.1 전처리 (G - (R - G))
    % img1 = 2 * img[..., 1] - img[..., 0]
    I = img(:, :, 2) - (img(:, :, 1) - img(:, :, 2));

    % 2.2 치아 영역 추축 , 3개로 나눔
    [m, n] = size(I);

    A = I > 25;
    bbox = regionprops(double(A), 'Area', 'BoundingBox');
    bb = bbox.BoundingBox;

    coord = [ceil(bb(1)), ceil(bb(1)) + bb(3) - 1, ceil(bb(2)), ceil(bb(2)) + bb(4) - 1];

%     B = imcrop(A ,[bb(1), bb(2), bb(3), bb(4)]);
%     J = imcrop(double(I) .* A, [bb(1), bb(2), bb(3), bb(4)]);

    B = A(coord(3):coord(4), coord(1):coord(2));
    J = double(I) .* A;
    J = J(coord(3):coord(4), coord(1):coord(2));

    [h, w] = size(B);
    tu = w / 20;

    l{1} = [1, round(tu*2), round(tu*18)+1, round(tu*20)];
    l{2} = [round(tu*2)+1, round(tu*5), round(tu*15)+1, round(tu*18)];
    l{3} = [round(tu*5)+1, round(tu*15)];
    
    beta = 1; 
    d1 = l{2}(2) - l{1}(1) + 1;
    d2 = l{1}(4) - l{2}(3) + 1;
    alpha1 = beta + ( d1:-1:1 ) / d1;
    alpha2 = beta + ( 1:d2 ) / d2;

    K = double(J) / 255;
    K(:, l{1}(1):l{2}(2)) = K(:, l{1}(1):l{2}(2)) .* alpha1;
    K(:, l{2}(3):l{1}(4)) = K(:, l{2}(3):l{1}(4)) .* alpha2;


    L1 = K(:, [l{1}(1):l{1}(2)]);
    L2 = K(:, [l{1}(3):l{1}(4)]);
    L3 = K(:, [l{2}(1):l{2}(2)]);
    L4 = K(:, [l{2}(3):l{2}(4)]);
    L5 = K(:, [l{3}(1):l{3}(2)]);

    th3 = graythresh(L5);
    th2 = th3 / 2;
    th1 = 25 / 255;

    U1 = L1 .* (L1 > th1);
    U2 = L2 .* (L2 > th1);
    U3 = L3 .* (L3 > th2);
    U4 = L4 .* (L4 > th2);
    U5 = L5 .* (L5 > th3);

    U = [U1, U3, U5, U4, U2];

%     temp = imregionalmin(-U, 18);
%     temp2 = imopen(temp, ones(3));
%     figure; imshow(temp2)

    bdr = bwskel(boundarymask(U ~= 0));
%     seed = imregionalmin(-U);
    
    I = -U;
    se = strel('disk', 20);
    % Io = imopen(I,se);
    % imshow(Io)
    % title('Opening')

    Ie = imerode(I,se);
    Iobr = imreconstruct(Ie,I);
%     imshow(Iobr)
%     title('Opening-by-Reconstruction')

    Iobrd = imdilate(Iobr,se);
    Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
    Iobrcbr = imcomplement(Iobrcbr);
%     imshow(Iobrcbr)
%     title('Opening-Closing by Reconstruction')

    % 2.3 국부 최대값 검출 시드 설정
    fgm = imregionalmin(Iobrcbr);
%     imshow(fgm)
%     title('Regional Maxima of Opening-Closing by Reconstruction')

    se2 = strel(ones(5,5));
    fgm2 = imclose(fgm,se2);
    fgm3 = imerode(fgm2,se2);

    fgm4 = bwareaopen(fgm3,20);
    I3 = labeloverlay(I,fgm4);
%     imshow(I3)
%     title('Modified Regional Maxima Superimposed on Original Image')

    % 2.4 에지 검출
    gmag = imgradient(I);

    % 2.5 마커 이용 워터쉬드 변환

    % figure; imagesc(uimg); mesh(uimg)
    bw = imbinarize(Iobrcbr);
%     imshow(bw)
%     title('Thresholded Opening-Closing by Reconstruction')

    D = bwdist(U~=0);
    DL = watershed(D);
    bgm = DL == 0;
%     imshow(bgm)


%     seed = lee2010seed(-U);
    seed = fgm4 .* (U ~= 0);

    use = imimposemin(imgaussfilt(double(bdr), 1), seed);

%     use = bdr*100;
%     use(seed > .5) = -inf;
%     use(bgm) = -inf;

    res = double(watershed(use));
    res(U == 0) = -1;
%     figure; imagesc(res)

    

    save(sprintf("Na2014_MWA/%d.mat", 400000+ni), "img", "res", "coord")
end

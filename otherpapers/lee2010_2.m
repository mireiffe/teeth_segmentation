% lee 2010_2

num_img = [0, 1, 5, 8, 17, 18];
% num_img = [0];

for ni = num_img
    dir_imgs = '../data/testimgs/';
    try
        img = imread(sprintf('%s%d.png', dir_imgs, 100000 + ni));
    catch
        img = imread(sprintf('%s%d.jpg', dir_imgs, 100000 + ni));
    end
    [m, n, ~] = size(img);

    I = img;
    I(:, :, 2) = 255 - img(:, :, 2);
    I(:, :, 3) = 255 - img(:, :, 3);

    indic = I(:, :, 1) - I(:, :, 2);
    sreg = indic > mean(indic) + 1.5*std(double(indic));

%     figure; imshowpair(indic > mean(indic) + 1*std(double(indic)), img)

    nI = double(I) / 255;

    % 2.2 경면 반사 검출
    w = [0, 0, 0]; b = 0; alpha = 1;
    nI_ = reshape(nI, [], 3);
    sreg_ = sign(double(reshape(sreg, [], 1)) - .5);

    ep = 0;
    while 1
        ep = ep + 1;
        spl = randi(m * n, 1);

        t = sreg_(spl);
        x = nI_(spl, :);
        yin = x * w' + b;
    
        if yin >= 0
            y = 1;
        else
            y = -1;
        end

        if y ~= t
            w = w + alpha * t * x;
            b = b + alpha * t;
        else
            if ep > 1
                break
            end
        end
    end
    w = [2.377, -6.106, -2.694];
    b = 1;

    nnI = sum(nI .* reshape(w, 1, 1, []), 3) + b;

    figure; imagesc(nnI > 0)

    ker = ones(3) / 3^2;
    
    J = double(img) / 255;
    for k = 1:1000
        for i = 1:3
            J(:, :, i) = conv2(J(:, :, i), ker, 'same');
        end
        J = J .* double(nnI > 0) + double(img) / 255 .* double(nnI <= 0);
%         J(J == 0) = img(J == 0);
    end
%     figure; imshow(nnI > 0)
%     figure; imshow(J, [])
%%

    % 2.2.1 전처리 (G - (R - G))
    % img1 = 2 * img[..., 1] - img[..., 0]
    I = J(:, :, 2) - (J(:, :, 1) - J(:, :, 2));

    % 2.2.2 모폴로지 영상처리
    se = strel('disk',20);
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
    fgm = imregionalmax(Iobrcbr);
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

    D = bwdist(bw);
    DL = watershed(D);
    bgm = DL == 0;
%     imshow(bgm)
%     title('Watershed Ridge Lines')

    gmag2 = imimposemin(gmag, bgm|fgm4);

    % uimg = ones(size(I));
    % uimg(fgm4 & (gmag <= 70)) = -inf;
    % uimg = uimg + gmag;
    % figure; imagesc(uimg)
    % % uimg(gmag > .2) = 1;

    L = watershed(gmag2);

    figure(ni+1); imagesc(L)

    % figure; imshow(img); hold on; contour(L, [-eps, 0], 'g')

    save(sprintf("Lee2010_MCWA/%d.mat", 300000+ni), "img", "J", "L")
    

end
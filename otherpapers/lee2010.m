% lee 2010

num_img = [0];

for ni = num_img
    dir_imgs = '../data/testimgs/';
    try
        img = imread(sprintf('%s%d.png', dir_imgs, 100000 + ni));
    catch
        img = imread(sprintf('%s%d.jpg', dir_imgs, 100000 + ni));
    end

    % 2.2.1 전처리 (G - (R - G))
    % img1 = 2 * img[..., 1] - img[..., 0]
    I = img(:, :, 2) - (img(:, :, 1) - img(:, :, 2));

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

    save(sprintf("forMCWA/%d.mat", 100000+ni), "img", "L")
end
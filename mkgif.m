
filename = sprintf('backfore.gif');

dir_load = sprintf('ppt/siamis22');
for i = 0:6:350
    fprintf('.');
                    
    img_i = imread(sprintf('%s/back/back%04d.png', dir_load, i));

    img_i = imresize(img_i, .4);            
    [imind, cm] = rgb2ind(img_i, 256);

    dlt = 1/64;
    if i == 0
        % n 회 반복 + 1/24초의 딜레이를 가지는 gif 생성. 무한 반복은 inf로 함
        imwrite(imind,cm,filename,'gif','Loopcount',1,'DelayTime', dlt);
    else
        % 똑같은 파일에 추가를 할 것이므로 append로 함
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime', dlt);
    end
    close all;
end

for i = 0:3:63
    fprintf('.');
                    
    img_i = imread(sprintf('%s/fore/fore%04d.png', dir_load, i));

    img_i = imresize(img_i, .4);
    [imind, cm] = rgb2ind(img_i, 256);

    dlt = 1/16;

        % 똑같은 파일에 추가를 할 것이므로 append로 함
        imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime', dlt);
    close all;
end
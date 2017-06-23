%max_image = 427;
max_image = 100;
start = 0;
distances = zeros(max_image-start, 1);
thresholding = false;

for i = start:(max_image-1)
    j = i + 1
    index = i - start + 1;
    img1 = imread(strcat('STATE', num2str(i), '.png'));
    img2 = imread(strcat('STATE', num2str(j), '.png'));
    img1red = img1(:,:,1);

    [width,height] = size(img1red);
    img2red = img2(:,:,1);
    if thresholding
        img1red = img1red >= 127;
        img2red = img2red >= 127;
    end
    u_limit = width;
    v_limit = height;
    difference = abs(img1red - img2red);
    distances(index) = sum(sum(difference)) / (width*height);
end
figure()
plot((1+start):max_image, distances)
if thresholding
    img1red = img1red * 255;
end
image(img1red);
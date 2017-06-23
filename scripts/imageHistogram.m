%max_image = 427;
max_image = 47;
start = 2;
edit_distances = zeros(max_image-start, 1);
block_size = 8;
use_equality_test = true;
equality_distance = 0.1;
use_log = false;
thresholding = true;

for i = start:(max_image-1)
    j = i + 1
    index = (i - start) + 1
    img1 = imread(strcat('Screenshot (', num2str(i), ').png'));
    img2 = imread(strcat('Screenshot (', num2str(j), ').png'));
    out = imhist((img1(:,:,1)+img1(:,:,2)+img1(:,:,3))/3, 127);
    out2 = imhist((img2(:,:,1)+img2(:,:,2)+img2(:,:,3))/3, 127);
    difference = sum(abs(out - out2));
    edit_distances(index) = difference;
end
figure()
plot((1+start):max_image, edit_distances/max(max(edit_distances)))
    
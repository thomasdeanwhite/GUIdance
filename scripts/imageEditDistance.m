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
    index = i - start + 1;
    img1 = imread(strcat('Screenshot (', num2str(i), ').png'));
    img2 = imread(strcat('Screenshot (', num2str(j), ').png'));
    img1red = img1(:,:,1);
    [width,height] = size(img1red);
    img2red = img2(:,:,1);
    if thresholding
        img1red = img1red >= 127;
        img2red = img2red >= 127;
    end
    u_limit = (width/block_size) -1;
    v_limit = (height/block_size) -1;
    unmatched_blocks = ones(u_limit, v_limit);
    for u = 0:u_limit
       for v = 0:v_limit
           bx = 1 + block_size*u;
           by = 1 + block_size*v;
           block = img1red(bx:bx+(block_size-1), by:by+(block_size-1));
           max_match = block_size * block_size;
           if use_equality_test
               % 255 is highest difference
               max_match = max_match * 255 * equality_distance; 
           end
           if use_log
               max_match = log(max_match);
           end
           
           match = max_match;

           best_match = zeros(2, 1);
           for x = 1:block_size:(width-block_size)
              for y = 1:block_size:(height-block_size)
                  if (unmatched_blocks(1 + round(x/block_size), 1 + round(y/block_size)) == 0)
                     continue; 
                  end
                  block2 = img2red(x:x+(block_size-1), y:y+(block_size-1));
                  block_difference = 1;
                  if use_equality_test
                      block_difference = block ~= block2;
                  else
                      block_difference = abs(block - block2);
                  end
                  differences = sum(sum(block_difference));
                  if use_log
                      difference = log(differences);
                  end
                  if differences < match
                      match = differences;
                      best_match(1) = x;
                      best_match(2) = y;
                  end
                  
                  if match == 0
                     break;
                  end
              end
              if match == 0
                 break;
              end
           end
           if match < max_match
               unmatched_blocks(1 + round(best_match(1)/block_size), 1 + round(best_match(2)/block_size)) = 0;
           end
           edit_distances(index) = edit_distances(index) + match;
       end
    end
    edit_distances(index) = edit_distances(index) + (block_size * block_size * sum(sum(unmatched_blocks)));
end
figure()
plot((1+start):max_image, edit_distances/max(max(edit_distances)))
    
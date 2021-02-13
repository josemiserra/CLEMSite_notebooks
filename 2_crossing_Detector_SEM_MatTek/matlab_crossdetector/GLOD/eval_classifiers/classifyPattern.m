function [namesList,probabilities] = classifyPattern(clabels,image,imdir)
  namesList = {};   
  coef = Inf;
  iedges = image;
  [nx,ny,dim] = size(iedges);
  if(dim>1)
     iedges = rgb2gray(iedges);
  end    
  if(nx~=256 || ny~=256)
     iedges = imresize(iedges,[256 256]);
  end;
   
    iedges(1:20,:)= 0;
    iedges(:,1:20)= 0;
    iedges(236:256,:)= 0;
    iedges(:,236:256)= 0;
  %  se = strel('diamond',1);
  %  iedges1 = (imerode(iedges1,se));
    iedges = (iedges>0.8);
    m_dist1 = bwdist(iedges, 'chessboard');
  %    se = strel('diamond',8);
  %    iedgesd = (imdilate(iedges1,se));
    
    for j=1:length(clabels)
         filec = strcat(clabels(j),'.jpg');
         ndir = strcat(imdir,filec);
         currentimage = imread(ndir{1});
         [nx,ny,dim] = size(currentimage);
         if (dim>1)
            currentimage = rgb2gray(currentimage);
         end    
        [currentimage,Gdir] = imgradient(currentimage);
         currentimage = currentimage/max(max(currentimage));
         currentimage = (currentimage>0.8);
        [nx,ny,dim] = size(currentimage);
        if(nx~=256 || ny~=256)
             currentimage = imresize(currentimage,[256 256]);
        end;
       % if(rot>0) % Rotate if not at 0 degrees
       %  currentimage = imrotate(currentimage,-rot);
       % end
        currentimage = wiener2(currentimage);
        % se = strel('diamond',2);
        % currentimage = (imdilate(currentimage,se));
     
        [y_offset, x_offset] = m_translation_offset(currentimage,iedges);
        transIm = circshift(currentimage, [y_offset x_offset]);

        
        r = sum(sum(m_dist1(find(transIm>0))));
        coef(j,1)= r;
        coef(j,2)= j;
    end
           % normalize max min
        ly = log10(coef(:,1));
        coef(:,1) = (ly-min(ly))/(max(ly)-min(ly));
        coef(:,1) = 1-coef(:,1);
        coef(:,1) = softmax(coef(:,1)*10);
        cList = sortrows(coef,-1);
        probabilities = cList(:,1);
        namesList = clabels(cList(:,2)); % candidates
end
    
 function [y_prob] = softmax(y)  
     s = exp(y);
     y_prob = s/sum(s);
 end
   
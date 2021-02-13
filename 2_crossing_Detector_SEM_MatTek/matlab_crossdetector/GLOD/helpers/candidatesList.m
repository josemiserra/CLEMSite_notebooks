function [namesList] = candidatesList(clabels,image,imdir,rot)
  namesList = {};   
  coef = Inf;
  edges = image;
  [nx,ny,dim] = size(edges);
  if(dim>1)
     edges = rgb2gray(edges);
  end    
  if(nx~=256 || ny~=256)
     edges = imresize(edges,[256 256]);
  end;
   
    edges(1:20,:)= 0;
    edges(:,1:20)= 0;
    edges(236:256,:)= 0;
    edges(:,236:256)= 0;
  %  se = strel('diamond',1);
  %  edges1 = (imerode(edges1,se));
    m_dist1 = bwdist(edges, 'chessboard');
  %    se = strel('diamond',8);
  %    edgesd = (imdilate(edges1,se));
    
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
        if(rot>0) % Rotate if not at 0 degrees
         currentimage = imrotate(currentimage,rot);
        end
    
        % se = strel('diamond',2);
        % currentimage = (imdilate(currentimage,se));
     
        [y_offset, x_offset] = m_translation_offset(currentimage,edges);
        transIm = circshift(currentimage, [y_offset x_offset]);
    %   m_dist = bwdist(transIm, 'chessboard');
    %    se = strel('diamond',8);
    %    cim = (imdilate(currentimage,se));
        r = sum(sum(m_dist1(find(transIm>0))));
        coef(j,1)= r;
        coef(j,2)= j;
    end
        cList = sortrows(coef,1);
        namesList = clabels(cList(:,2)); % candidates
  end
    


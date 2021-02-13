function [ swt ] = cleanswt2(tos,edges)        
          
         swt = tos;
         mask = (swt>0);
         CC = bwconncomp(mask,8);
         numPixels = cellfun(@numel,CC.PixelIdxList);
         [w,h] = size(tos);
         max_pix = (0.05*w);
         for i = 1:length(numPixels)
            if(numPixels(i)< max_pix)
                swt(CC.PixelIdxList{i}) = 0;
            end
         end
         
         swt(edges>0) = max(max(tos));
         
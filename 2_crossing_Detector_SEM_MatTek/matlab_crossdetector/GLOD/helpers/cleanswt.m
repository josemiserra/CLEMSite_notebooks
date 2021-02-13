function [ swt ] = cleanswt(tos,edges)
  
  tos_m =round(median(median(tos(tos>0)))*0.5+0.5);
  if(isnan(tos_m))
     swt = tos;
     fprintf('SWT not performed.\n');
     return;
  end;
  SE = strel('rectangle',[tos_m tos_m]);
  im2 = imopen(tos,SE);
  [rows,cols]=size(im2);
  % and we label the components
  final_edge = tos+edges;
  CC = bwlabel(final_edge,4);
  % Only the components that remain after the opening are kept without 
  % modifications
  A = [];
  for i = 1:rows
       for j = 1:cols
          if(CC(i,j)>0 && im2(i,j)>0)
               A = [A CC(i,j)];
          end
       end
  end
 A = unique(A);
 swt = zeros(rows,cols);
 for i = 1:length(A)
    swt(CC ==A(i))=1;
 end
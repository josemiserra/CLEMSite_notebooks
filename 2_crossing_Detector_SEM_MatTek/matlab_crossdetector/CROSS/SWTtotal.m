function [tos] = SWTtotal(input,edge,orientation,stroke_width,angle)

if ~exist('max_angle','var'), angle = pi/6; end
inputinv = imcomplement(input);  %% needed for shadowing

swtim = SWT(input,edge,orientation,stroke_width,angle);  %% one image
swtimcomp = SWT(inputinv,edge,orientation,stroke_width,angle);  %% the inverse

% show(swtim)
% show(swtimcomp)

tos = zeros(size(input));
[rows,cols]=size(input);
for i = 1:rows
       for j = 1:cols
          if(swtim(i,j)>-1 && swtimcomp(i,j) == -1)
            tos(i,j) = swtim(i,j);
          else
            if((swtimcomp(i,j)>-1) && swtim(i,j) == -1)
                tos(i,j) = swtimcomp(i,j);
            end
          end
       end
end
tos((tos==-1))=0;
% tos = normalise(tos);
% figure,show(tos);
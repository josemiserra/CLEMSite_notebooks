function [swt_end] = SWTtotal(input,edge,orientation,stroke_width,angle)

if ~exist('max_angle','var'), angle = pi/3; end
inputinv = imcomplement(input);  %% needed for shadowing

swtim = SWT(input,edge,orientation,stroke_width,angle);  %% one image
swtimcomp = SWT(inputinv,edge,orientation,stroke_width,angle);  %% the inverse

% show(swtim)
% show(swtimcomp)

swtim(swtim<0)=0;
swtimcomp(swtimcomp<0)=0;
swt_end = swtim;
indexes = find(swtim==0);
swt_end(indexes) = swtimcomp(indexes);

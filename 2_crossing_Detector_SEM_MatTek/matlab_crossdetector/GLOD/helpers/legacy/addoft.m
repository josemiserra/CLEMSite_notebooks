function [O] = addoft(M,K,L,inc)
%% R = addoft(FSWT,12,ORIENTIM,1);
     [rows,cols]=size(M);
     O = zeros(rows,cols);
     % -90 and 90 are the same, we must remove 90
     THETA = (-90:inc:89);
     th =(-90:inc:89)*pi/180;
%      THETA = (0:inc:180);
%      th =(0:inc:180)*pi/180;
     n = length(THETA);
     
    % if(deg<0) deg = 2*pi/2-deg; end
     for i = 1:rows
            for j = 1:cols
                if(M(i,j)>0)
                   max_value = 0;
                   for tau = 1:n
                      deg = th(tau);
                      current_val = 0;
                      for k = -K/2:2:K/2
                        ni = round(i+k*round(cos(deg)));
                        nj = round(j+k*round(sin(deg)));
                        if(ni>0 && ni<rows) && (nj>0 && nj<cols)
                             val = cos(2*(L(ni,nj)-deg));
                             current_val = val + current_val;    
                        end
                      end
                      current_val = current_val/(K*0.5+1);
                      if(max_value<current_val)
                          O(i,j) = deg;
                          max_value = current_val;
                      end
                   end
                end
            end
    end
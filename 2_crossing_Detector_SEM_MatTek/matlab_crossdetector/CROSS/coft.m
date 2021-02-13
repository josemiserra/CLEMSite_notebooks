function [O] = coft(M,K,L,deg)    
     kernel = zeros(K);
     v_cos = cos(deg);
     v_sin = sin(deg);
     Mval = cos(2*(L-deg));
     
     count = 0;
     for k = -K/2-1:1:K/2+2
        ni = round(K/2+k*v_cos);
        nj = round(K/2+k*v_sin);
        if(ni>0 && ni<K) && (nj>0 && nj<K)  
            kernel(ni,nj)=1;
            count = count +1;
        end
     end
     kernel = kernel/count;
     % kernel = fliplr(kernel);
     cO = conv2(Mval,kernel,'same');
     [rows,cols]=size(M);
     O = zeros(rows,cols);
     O(M>0) = cO(M>0);
 
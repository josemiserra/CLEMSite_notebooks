function [bwedge] = autocanny(nm,canthresh)
    med= double(median(median(nm(nm>0))));
    max_factor = 0.8*max(max(nm));
    factor_a = max_factor;
    factor_b = 0.4;
    
    lenm = size(nm);
    bwedge = [];
    value = 0;
    msize = lenm(1)*lenm(2);
    while(value<canthresh) 
        bwedge =  hysthresh(nm, factor_a*med,factor_b*med);
        value = sum(sum(bwedge))/msize;
        factor_a = factor_a*0.9;
    end     
     
    while(value>canthresh) 
        factor_a = factor_a+0.01;
        bwedge =  hysthresh(nm, factor_a*med,factor_b*med);
        value = sum(sum(bwedge))/msize;       
    end     
        
fprintf('Automatic Canny done \n');
fprintf('Percentage of pixels threshold reached at: %f \n',value);
fprintf('Lower Thresh at: %f \n',factor_b);
fprintf('Higher Thresh at: %f \n',factor_a);
end
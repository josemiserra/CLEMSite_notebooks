function [coef] = corrProfiles(pr1,pr2)
    total = 0;
    [np,~]=size(pr1);
    for i=1:np
        [acor,lag] = xcorr(pr1(i,:),pr2(i,:),'coeff');
        [a,I] = max(abs(acor));
        total(i) = a;
    end
    coef = mean(total);
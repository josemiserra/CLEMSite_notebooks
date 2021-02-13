function [a, b,value] = eval_classifier()
fnames = dir('I:\eval\*.tif');
for i=1:length(fnames)
         file_ev = fnames(i).name;
         letter = file_ev(1:2);
         cneighs = calculateNeighs(letter,2);
         filec = strcat('I:\eval\',file_ev);
         currentimage = imread(filec);
         [nm,pl]=classifyPattern2(cneighs,currentimage,'\codebook\');
         value(i).letter = nm(1);
         value(i).real_letter = letter;
         value(i).prob   = pl(1);
         value(i).com = strcmp(letter,char(nm(1)));
         
end
 % True positives
 a = sum(cat(1,value(:).com))/length(fnames);
 b = sum(cat(1,value(:).prob))/length(fnames);
end
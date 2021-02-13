%% Import data from text file.
% Script for importing data from the following text file:
%
%    C:\Users\JMS\Documents\msite\MSite\user.pref
%

%% Initialize variables.
function JSONstring = importJSON(fname)

fid = fopen(fname,'r');
if fid == -1
 fprintf('File %s doesnt exist. \n',fname);
end

raw = fread(fid,inf);
str= char(raw');
JSONstring = JSON.parse(str);
fclose(fid);

end
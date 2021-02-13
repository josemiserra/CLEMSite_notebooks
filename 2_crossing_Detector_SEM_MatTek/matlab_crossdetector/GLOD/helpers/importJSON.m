%% Import data from text file.
% Script for importing data from the following text file:
%
%    C:\Users\JMS\Documents\msite\MSite\user.pref
%

%% Initialize variables.
function JSONstring = importJSON(fname)

fid = fopen(fname,'r');
if fid == -1
  msgID = 'importJSON:fileNotFound';
  msg = sprintf('File %s doesnt exist. \n',fname);
  fileException = MException(msgID,msg);
  throw(fileException);
end
raw = fread(fid,inf);
    
str= char(raw');
JSONstring = JSON.parse(str);
fclose(fid);

end
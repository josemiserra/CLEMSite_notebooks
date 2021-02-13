function profile = generateProfiles(imdir)

    ndir = strcat(imdir,'\*.jpg');
    imagefiles = dir(ndir);      
    nfiles = length(imagefiles);    % Number of files found
    
    for ii=1:nfiles
     
     currentfilename = imagefiles(ii).name;
     currentfilename = strcat(imdir,currentfilename);
     currentimage = imread(currentfilename);
     [nx,ny,dim] = size(currentimage);
      
     if (dim>1)
        currentimage = rgb2gray(currentimage);
     end    
     [currentimage,Gdir] = imgradient(currentimage);
     currentimage = currentimage/max(max(currentimage));
     currentimage = (currentimage>0.8);
     se = strel('diamond',3);
     currentimage = (imdilate(currentimage,se));
     profcurrent = getLetterProfile(currentimage,currentimage);
     profile(ii).value= profcurrent;
     profile(ii).names= imagefiles(ii).name(1:2);
    end
     
    
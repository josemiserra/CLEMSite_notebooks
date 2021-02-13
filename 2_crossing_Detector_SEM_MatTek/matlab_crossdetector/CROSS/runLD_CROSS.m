function runLD_CROSS(input_im,output_folder,tag,image_metadata,igridsize)
    % Read preferences file and extract parameters
    % Read file
    gridsize = str2double(igridsize);
    dataIm = importJSON(image_metadata);
    gridsize = gridsize/dataIm.PixelSize;
    softParams.dispim = true;
    softParams.sigma = 2;
    softParams.clahe = 1;
    softParams.canthresh = 0.075;
    softParams.strokeWidth = 30; 
    softParams.dispim = true;
    softParams.K = 12;
    mtime = clock;

    mtime = mtime(4)*100+mtime(5);
    now_date = strcat(date,'-');
    now_date = strcat(now_date,num2str(mtime));

    folder = strcat('/cross_det_',tag);    
    folder = strcat(folder,now_date);
    folder = strcat(output_folder,folder);
    mkdir(folder);
    folder = strcat(folder,'/');
    img = imread(input_im);
    
    [images,fpoints,info,error] = lineDetection(img,softParams,gridsize,folder,tag);
  
    if(error>0)
        filename = strcat(tag,'ERROR');
        filename = strcat(folder,filename);
        csvwrite(filename,error);
        return
    end
    
    

 %% Extract points from the lines detected
    filename = strcat(tag,'_fpoints_pixels.csv');
    filename = strcat(folder,filename);
    csvwrite(filename,fpoints);
     
    
    tpoint(1,1) =  dataIm.PositionX; % Remember, microscope is inverted
    tpoint(1,2) =  dataIm.PositionY; 
    inpoints = imageToStageCoordinates_SEM(fpoints,tpoint,img,dataIm.PixelSize);

    filename = strcat(tag,'_fpoints.csv');
    filename = strcat(folder,filename);
    csvwrite(filename,inpoints);
     
end
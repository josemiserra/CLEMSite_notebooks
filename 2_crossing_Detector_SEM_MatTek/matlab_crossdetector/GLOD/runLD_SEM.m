function runLD_SEM(input_image,output_folder,parameters_filename,image_metadata,scripts_folder,true_letter)
    %   runLD(INPUT_IMAGE,OUTPUT_FOLDER,PARAMETERS,IMAGE_METADATA,ADDITIONAL_FOLDER,TRUE_LETTER) 
    %   Given an input image, computes the detection of a grid and writes
    %   the resulting data coordinates to a text file.  
    %   
    %   PARAMETERS contains information given by the user that affects the
    %   performance of the line detection during the preprocessing part.
    %           - Gaussian blur for noise averaging 
    %           - CLAHE if necessary because uneven illummination
    %           - CANTHRESH it is the amount of white pixels recovered for
    %           the canny edges histeresis thresholding algorithm. 
    %           This value is related to the amount of clutter being present in the image. If the
    %           amount of clutter is substantial this parameter can help to
    %           reduce it, but needs to be balanced with the amount of
    %           features from the grid that can be seen (too much removal can make it disappear them). 
    %               - Values are between 0.05 to 0.3. For SEM images 0.06 is
    %               recommended, for SEM 0.075 is nice parameter.
    %               
    %   It also provides information about the grid to be detected.
    %           - gridsize : this algorithm is designed for 2 spacing
    %           grids, but can be easily modified for any desired grid
    %           pattern. 
    %   IMAGE_DATA must contain at least 4 essential parameters:
    %       - PixelSize IN MICROMETERS
    %       - Coordinate X,Y from the center.
    %       - Predicted letter in the center (this letter will be evaluated
    %       by Image analysis algorithms and changed to one of the
    %       neighbors in case of a bad prediction).
    %       - If the value true_letter is true, then no image analysis
    %       checking is done
    %       - Orientation of the grid. Necessary to reduce the amount of
    %       comparisons performed by the letter matching algorithm. It is
    %       not absolutely necessary but simplifies and avoids unnecessary
    %       errors coming from a bad prediction in the image analysis side.
    %       This orientation can be done by extra algorithms on the client
    %       side, or simply asked to the user.
    %       - TAG - image sample name
    %   
    try
        data = importJSON(parameters_filename);
        matlab_folder = scripts_folder;
        addpath(genpath(matlab_folder));
    
        softParams.sigma = str2double(data.preferences.line_detection{1}.SEM{1}.gaussian);
        softParams.clahe = str2double(data.preferences.line_detection{1}.SEM{1}.clahe);
        softParams.canthresh = str2double(data.preferences.line_detection{1}.SEM{1}.canny_thresh);
        softParams.dispim = true;
        softParams.K = 12;
        softParams.wiener = [2 2];
        softParams.gridSize = [str2double(data.preferences.grid{1}.div1); str2double(data.preferences.grid{1}.div2)];
        softParams.strokeWidth = str2double(data.preferences.line_detection{1}.SEM{1}.stroke);
        if(~isempty(image_metadata))
            dataIm = importJSON(image_metadata);
            softParams.gridSize = softParams.gridSize;
            softParams.PixelSize = dataIm.PixelSize;
            tag = dataIm.tag;
            gridSize = softParams.gridSize/softParams.PixelSize;
        else
            printf('ERROR: Image parameters not available')
            return
        end
    
        mtime = clock;

        mtime = mtime(4)*100+mtime(5);
        now_date = strcat(date,'-');
        now_date = strcat(now_date,num2str(mtime));

        folder = strcat('/ld_',now_date);
        folder = strcat(output_folder,folder);
        mkdir(folder);
        folder = strcat(folder,'/');
        img = imread(input_image);
    
        p_angles = [45 -45];
        [images,fpoints,info,error] = glod(img,softParams,folder,tag,p_angles);
    
        if(error>0)
            filename = strcat(tag,'_ERROR.csv');
            filename = strcat(folder,filename);
            csvwrite(filename,error);
            fprintf('ERROR');
            return
        end    
    
        
        %% Crop patterns embedded in the grid   
        lettersf = strcat('/letters_',date);
        lfolder = strcat(folder,lettersf);
        mkdir(lfolder);
        lfolder = strcat(lfolder,'/');
     
        lettersfo = strcat('/letters_orig_',date);
        lfoldero = strcat(lfolder,lettersfo);
        mkdir(lfoldero);
        lfoldero = strcat(lfoldero,'/');
        % save originals, no names, just the crops of the letters    
        ang = min(abs(info.angneg),info.angpos); 
        if(ang== abs(info.angneg))
            ang = -ang;
        end
        
        if(length(gridSize)==2)
            distsq = max(gridSize)+min(gridSize)*0.5;
        else
            distsq = max(gridSize);
        end
        crop_letters(img,info.cutpoints,ang,distsq,lfoldero);
 
        [iLength, iWidth] = size(img);
        tpoint(1,1) = round(iWidth/2);
        tpoint(1,2) = round(iLength/2);
        tpoint(1,3) = 0;
        lnames = [];
        hint = num2str(randi([1 5000000],1)); % in case that no letter is identified
        hint = strcat('__',hint);
        % Select only points inside the square and give them the proper letter name  
         %  The points are translated to stage coordinates given the pixel size   
        if(~isempty(image_metadata))        
           % X and Y are inverted in this microscope (LEICA SP5)
           tpoint(1,1) =  dataIm.posx; 
           tpoint(1,2) =  dataIm.posy;  
           orientation = dataIm.orientation;
           % templates for the recognition
           cbdir = strcat(scripts_folder,'/codebook/');
           
           [flist,hint,prob_patt] = identify_pattern_SEM(images{5},info.cutpoints,distsq,lfolder,orientation,info.centrePointIndex,dataIm.letter_center,cbdir,true_letter);
           
           if(isempty(flist))
               fprintf('ERROR detecting square images.');
               return
           end
           % Take out points that are outside the image (they are predicted
           % and less precisse)
           % fprintf('Original estimation \n');
           % struct2table(flist)
           % Select only points that are INSIDE the image
           points = cat(1,flist(:).point);
           [inpoints,sorderin] = getInsidePoints(points,img);
           flist = flist(sorderin);
           struct2table(flist)
           lnames = cat(1,flist(:).letter);
           inpoints = imageToStageCoordinates_SEM(inpoints,tpoint,img,dataIm.PixelSize);           
    else
         [inpoints,~] = getInsidePoints(fpoints,img);
    end
         
    
    filename = strcat(tag,'_impar.csv');
    filename = strcat(folder,filename);
    together = [info.centrePointIndex; info.angneg; info.angpos; length(inpoints); prob_patt];
    csvwrite(filename,together);
    dlmwrite(filename,dataIm.orientation,'delimiter','','-append');
    dlmwrite(filename,hint,'delimiter','','-append');
    %% Finally write the XML file
    tag = strcat(tag,'_');
    filename = strcat(tag,'_fcoordinates_calibrate.xml');
    filename = strcat('/',filename);
    filename = strcat(output_folder,filename);
    mappoints = getMapCoordinates(lnames);
    saveCoordinatesXML(inpoints,mappoints,lnames,filename);
    
catch ME
      getReport(ME)
end
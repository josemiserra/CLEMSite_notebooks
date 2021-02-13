function profile = getLetterProfile(img,edges)
    windowSize = 3;
    [nx,ny] = size(img);
    if(nx~=256 || ny~=256)
        img = imresize(img,[256 256]);
        edges = imresize(edges,[256 256]);
    end
    [grad,or]=canny(img,2);
    [orientim, reliability] = ridgeorient(grad, 1, 5, 5);
       
    img(1:20,:)= 0;
    img(:,1:20)= 0;
    img(236:256,:)= 0;
    img(:,236:256)= 0;
    
    edges(1:20,:)= 0;
    edges(:,1:20)= 0;
    edges(236:256,:)= 0;
    edges(:,236:256)= 0;

    %show(edges)
    %show(img)
    b = (1/windowSize)*ones(1,windowSize);
    a = 1;
    angles = [ -90 -60 -45 -30  0.001  30 45 60  ];
    anglesrad = (angles)*(pi/180);
    PRf = [];
    for i=1:length(angles)
        imgSft = soft(edges,12,orientim,anglesrad(i));      
        %show(imgSft);
        PR1 = projectionsTwo(imgSft,angles(i));
        PR1 = filter(b,a,PR1);  
        PRf(i,:) = PR1;
    end
    PRf = PRf/max(max(PRf));
    PRf(find(PRf)>0.1)=0.0;
    profile = PRf;
    
    
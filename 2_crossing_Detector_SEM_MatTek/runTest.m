    pathSCAN = '.\\data\\crosses\\TEST_curated\\' %% Don't forget the bar 
    scans = dir(pathSCAN);
    scans(ismember( {scans.name}, {'.', '..'})) = []; 
    for k = 1 : length(scans)
        scan_folder = strcat(pathSCAN,scans(k).name);
        scanxfiles = dir(scan_folder);
        for m = 1 : length(scanxfiles)
            if (startsWith(scanxfiles(m).name,'ref') && (scanxfiles(m).isdir == 0)) 
                name_f = scanxfiles(m).name;
                sname_f = strcat(scan_folder,'\',name_f);
                imname = strcat(scanxfiles(m).folder,'\',scanxfiles(m).name);
                runLD_CROSS(imname, scanxfiles(m).folder, name_f, 'mat_info.txt','35');
            end
            close all force
        end
    end
function [SWT] = SWT(input,edgeImage,orientim,stroke_width,max_angle)

if ~exist('max_angle','var'), max_angle = pi/6; end
if ~exist('stroke_width','var'), stroke_width = 20; end

 %h = fspecial('average');
 %g = imfilter(input, h);

 % The orientation image
 
im = gaussfilt(input,1);   
[grad_x, grad_y] = derivative5(im,'x','y');
g_mag =sqrt(grad_x.^2+grad_y.^2);
cres = 0;
prec = 0.3;
G_x = grad_x ./ g_mag;
G_y = grad_y ./ g_mag;


[rows,cols]=size(input);
 SWT = ones(rows,cols);
 SWT = -SWT;
 count = 1;
 h_stroke = stroke_width*0.5;
 for i = 1:rows
       for j = 1:cols
                if(edgeImage(i,j)>0)
                    count = 1;
                    points_x(count) =  j;
                    points_y(count) = i;
                    count = count+1;
                   
                    curX = double(j)+0.5;
                    curY = double(i)+0.5;
                    cres = 0;
                    while cres<stroke_width                
                        curX = curX + G_x(i,j)*prec; % find directionality increments x or y
                        curY = curY + G_y(i,j)*prec; 
                        cres = cres +1;
                        curPixX = floor(curX);
                        curPixY = floor(curY);
                        if(curPixX<1 || curPixX > cols || curPixY <1 || curPixY>rows)
                           break; 
                        end
                    
                       points_x(count) =  curPixX;
                       points_y(count) =  curPixY;
                       count = count+1;

                    if(edgeImage(curPixY,curPixX)>0 && count> h_stroke)%% && count<20)
                         ang_plus =  orientim(i,j)+max_angle;
                         if(ang_plus>pi)
                             ang_plus = pi;
                         end
                         ang_minus = orientim(i,j)-max_angle;
                         if(ang_minus<0)
                             ang_minus = 0;
                         end
                         if((orientim(curPixY,curPixX)<ang_plus) && (orientim(curPixY,curPixX)>ang_minus))
                             dist= sqrt((curPixX - j)^2 + (curPixY-i)^2);
                             for k =1:count-1
                                    if((SWT(points_y(k),points_x(k))<0))
                                        SWT(points_y(k),points_x(k))=dist;
                                    else
                                        SWT(points_y(k),points_x(k))= min(dist,SWT(points_y(k),points_x(k)));
                                    end
                             end 
                         end
                         if(count>stroke_width-1)
                             break;
                         end
                    end         
                    end
                
                end
                
                
       end
 end
     
     
 


                    
                    
                    
clc; clear all

cam = webcam();
cam.resolution = '1280x720';
video_frame = snapshot(cam);

video_player = vision.VideoPlayer('Position',[100 100 1280 720]);
face_detector = vision.CascadeObjectDetector();
point_tracker = vision.PointTracker('MaxBidirectionalError',2);


run_loop = true;
no_of_points = 0;
frame_count = 0;

while run_loop && frame_count < 40000
   video_frame = snapshot(cam);
   gray_frame = rgb2gray(video_frame);
   frame_count = frame_count +1;
   
   if no_of_points <10
       face_rectangle = face_detector.step(gray_frame);
       
       if ~isempty(face_rectangle)
           points = detectMinEigenFeatures(gray_frame,'ROI',face_rectangle(1,:));
           
           xy_points = points.Location;
           no_of_points = size(xy_points ,1);
           release(point_tracker);
           initialize(point_tracker , xy_points , gray_frame);
           
           previous_points = xy_points ;
           
           rectangle = bbox2points(face_rectangle(1 ,:));
           face_polygon = reshape(rectangle' , 1, []);
           
           video_frame = insertShape(video_frame , 'Polygon' , face_polygon , 'linewidth', 3);
           video_frame = insertMarker(video_frame , xy_points ,'o', 'Color', 'white');
       end
       
       
   else
       
       [xy_points , isfound]= step(point_tracker , gray_frame);
       new_points = xy_points(isfound ,:);
       old_points = previous_points(isfound, :);
       
       no_of_points = size(new_points , 1);
       
       if no_of_points >=10
           [xform , old_points , new_points]= estimateGeometricTransform(...
               old_points , new_points, 'similarity', 'MaxDistance', 4);
           
           rectangle = transformPointsForward(xform, rectangle);
           
           face_polygon = reshape(rectangle' , 1 , []);
           
           video_frame = insertShape(video_frame, 'Polygon', face_polygon , 'linewidth', 3);
           video_frame = insertMarker(video_frame , new_points , 'o', 'Color', 'white');
           
           previous_points = new_points;
           setPoints(point_tracker, previous_points);
           
       end
   end
   
   step(video_player , video_frame);
   run_loop = isOpen(video_player);
         
end
    


clear cam;
release(video_player);
release(point_tracker);
release(face_detector);










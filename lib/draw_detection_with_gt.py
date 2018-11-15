import cv2
import sys

def draw_detections(file_name, path_save):
    f = open(file_name, 'r')
    lines = f.readlines()
    for line in lines:
        test_gt = line.split('*')
        detection = test_gt[0].split('   ')
        im_name = detection[0].split('/')[-1]
        if(len(test_gt)==1):
            gt=[]
        else:
            gt = test_gt[-1].split('   ')
        #print 'img is', detection[0]
        try:
            im = cv2.imread(detection[0].rstrip('\n'))
            #print 'ims shape is ', im.shape
        except:
            im = cv2.imread(detection[0].rstrip('\n'))
            #print 'img is', im, detection
        for bbox_ind in range(1, len(detection)):
            bbox = detection[bbox_ind]
            bbox = bbox.split(' ')
            for i in range(1,5):
                bbox[i]=bbox[i].split('\n')[0]
            class_info = bbox[0]
            cv2.putText(im, class_info, (int(float(bbox[1])), int(float(bbox[2]))), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 4)
            cv2.rectangle(im, 
                    (int(float(bbox[1])), int(float(bbox[2]))), 
                    (int(float(bbox[3])), int(float(bbox[4]))), (0, 255, 0), 4) 
        for gt_ind in range(len(gt)):
            gt_box = gt[gt_ind]
            if(len(gt_box)<2):
                continue
            gt_box = gt_box.split(' ')
            for i in range(len(gt_box)):
                gt_box[i]=gt_box[i].split('\n')[0]
            #print 'gt is ', gt_box
            cv2.rectangle(im,
                    (int(float(gt_box[0])), int(float(gt_box[1]))),
                    (int(float(gt_box[2])), int(float(gt_box[3]))), (255,255,255), 4)
        
        cv2.imwrite(path_save + '/' + im_name, im)
    return


if __name__ == '__main__':
    save_path = "/home/ayogg/USB/USB2018_Dataset/test_jpg"
    filename = '/home/ayogg/USB/py-faster-rcnn-new-new/output/faster_rcnn_alt_opt/usb_test/2_ZF.txt'
    draw_detections(filename, save_path)

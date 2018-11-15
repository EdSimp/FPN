#include "algorithm"
#include <math.h>
#include <cmath>
#include "caffe/layers/as_rois.hpp"
#include <vector>
namespace caffe{

	template <typename Dtype>
	void AsRoiLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	  batch_rois_ = this->layer_param_.asroi_param().post_nms_topn(); //1000 fpn
		top[0]->Reshape(1,5,1,1);
		top[1]->Reshape(1,5,1,1);
		top[2]->Reshape(1,5,1,1);
		top[3]->Reshape(1,5,1,1);
		top[4]->Reshape(1,5,1,1);
	}

	template <typename Dtype>
	void AsRoiLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
    int k0=4;
    vector<Point4f<Dtype> > rois_;
    vector<Dtype > layer_indexs_;
    vector<Point4f<Dtype> > new_rois_(k0);
    

		for (int i = 0; i < bottom[0]->num(); i++) {//get bottom rois data

    		rois_.push_back(Point4f<Dtype>(
        	bottom[0]->data_at(i,1,0,0),
        	bottom[0]->data_at(i,2,0,0),
        	bottom[0]->data_at(i,3,0,0),
        	bottom[0]->data_at(i,4,0,0)));
    		CHECK_EQ(bottom[0]->data_at(i,0,0,0), 0) << "Only single item batches are supported";
  		} //rois = bottom[0].data
  		
  		vector<vector<Point4f<Dtype> > > level_rois(k0);

  		for(size_t i = 0;i < rois_.size(); i++){ //s is used to cal the level
        Dtype w = rois_[i][2] - rois_[i][0];
  			Dtype h = rois_[i][3] - rois_[i][1];
  			Dtype s = w * h;
  			if(s <= 0){s = 1e-6;}
  		  	s = sqrt(s)/224;
  			s = log2(s) + k0;
  			s = floor(s);
  			if(s < 2){s = 2;}
  			if(s > 5){s = 5;}
  			layer_indexs_.push_back(s);//s

  			int level_idx = s - 2;
  			level_rois[level_idx].push_back(rois_[i]);

  		}
  		Point4f<Dtype> pad_roi;//empty
      //vector<Point4f<Dtype> > this_rois_;

      Dtype *top_data0 = top[0]->mutable_cpu_data();
      //top[0]->Reshape(4*batch_rois_, 5, 1, 1); 
      top[0]->Reshape(rois_.size(), 5, 1, 1);

      for(size_t level_idx = 0;level_idx < k0; level_idx++){
         if(level_rois[level_idx].size()==0){
            Dtype *top_data = top[level_idx + 1]->mutable_cpu_data();
            top[1 + level_idx]->Reshape(1, 5, 1, 1);
            top_data[0] = 0;
              top_data[1] = 0;
              top_data[2] = 0;
              top_data[3] = 0;
              top_data[4] = 0;
              top_data += top[level_idx + 1]->offset(1);
              top_data0[0] = 0;
              top_data0[1] = 0;
              top_data0[2] = 0;
              top_data0[3] = 0;
              top_data0[4] = 0;
              top_data0 += top[0]->offset(1);
              //std::cout << "this roi is 0 0 0 0" <<std::endl;
          continue;
         }
         Dtype *top_data = top[level_idx + 1]->mutable_cpu_data();
         top[1 + level_idx]->Reshape(level_rois[level_idx].size(), 5, 1, 1);
         //top[1 + level_idx]->Reshape(batch_rois_, 5, 1, 1);
         
         //for(size_t i = 0; i < batch_rois_; i++){
         for(size_t i = 0; i < level_rois[level_idx].size(); i++){
          if(i < level_rois[level_idx].size()){
              Point4f<Dtype> &box = level_rois[level_idx][i]; //每个box的四个坐标
              top_data[0] = 0;
              top_data[1] = box[0];
              top_data[2] = box[1];
              top_data[3] = box[2];
              top_data[4] = box[3];
              top_data += top[level_idx + 1]->offset(1);
              top_data0[0] = 0;
              top_data0[1] = box[0];
              top_data0[2] = box[1];
              top_data0[3] = box[2];
              top_data0[4] = box[3];
              top_data0 += top[0]->offset(1);
              //std::cout << "this roi is " << box[0] << box[1] << box[2] <<box[3] << std::endl;
          }/*else{
              top_data[0] = 0;
              top_data[1] = 0;
              top_data[2] = 0;
              top_data[3] = 0;
              top_data[4] = 0;
              top_data += top[level_idx + 1]->offset(1);
              top_data0[0] = 0;
              top_data0[1] = 0;
              top_data0[2] = 0;
              top_data0[3] = 0;
              top_data0[4] = 0;
              top_data0 += top[0]->offset(1);
          }*/
         }
      }

  		LOG(INFO) << "AsRoiLayer::Forward_cpu End";
	}

#ifdef CPU_ONLY
	STUB_GPU(AsRoiLayer);
#endif
	INSTANTIATE_CLASS(AsRoiLayer);
	REGISTER_LAYER_CLASS(AsRoi);


}








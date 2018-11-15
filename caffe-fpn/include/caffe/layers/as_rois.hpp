#ifndef CAFFE_AS_ROIS_LAYER_HPP_
#define CAFFE_AS_ROIS_LAYER_HPP_
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	template <typename Dtype>
		class Point4f {
		public:
  			Dtype Point[4]; // x1 y1 x2 y2
  			Point4f(Dtype x1 = 0, Dtype y1 = 0, Dtype x2 = 0, Dtype y2 = 0) {
    			Point[0] = x1; Point[1] = y1;
    			Point[2] = x2; Point[3] = y2;
  			}
  			Point4f(const float data[4]) {
    			for (int i=0;i<4;i++) Point[i] = data[i]; 
  			}
  			Point4f(const double data[4]) {
    			for (int i=0;i<4;i++) Point[i] = data[i]; 
  			}
  			Point4f(const Point4f &other) { memcpy(Point, other.Point, sizeof(Point)); }
  			Dtype& operator[](const unsigned int id) { return Point[id]; }
  			const Dtype& operator[](const unsigned int id) const { return Point[id]; }

		};

	template <typename Dtype>
	class AsRoiLayer : public Layer<Dtype> {
	public:
		explicit AsRoiLayer(const LayerParameter& param)
		:Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top){};
		//virtual inline const char* type() const { return "ReLU";}

	protected:
		
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom){};
		int batch_rois_;
	
	};



}//namespace caffe

#endif //CAFFE_AS_ROIS_LAYER_HPP_
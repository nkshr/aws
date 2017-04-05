#ifndef _F_CAFFE_H_
#define _F_CAFFE_H_

#include <caffe/caffe.hpp>

#include <opencv2/ximgproc/segmentation.hpp>
using cv::ximgproc::segmentation::GraphSegmentation;

#include "../util/selective_search.h"

#include "f_base.h"

class f_binary_classifier: public f_base
{
 private:
  bool m_verb;
  bool m_gpu;

  char m_fname_model_txt[1024];
  char m_fname_model_bin[1024];
  char m_fname_mean[1024];
  
  Size m_sz_input;
  Mat m_mean_img;
  
  caffe::Net<float> * m_net;
  ch_image_ref * m_pin;
  ch_image * m_pimg;
  ch_obj * m_pout;
  SelectiveSearch m_ss;
  VideoCapture m_cap;

  void transfer_img_to_input_layer(const Mat  &img);
 public:
  f_binary_classifier(const char *fname);
  virtual bool init_run();
  virtual void destroy_run();
  virtual bool proc();
};
#endif

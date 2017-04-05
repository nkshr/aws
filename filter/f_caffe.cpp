#include "stdafx.h"
#ifdef DNN
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include <google/protobuf/message_lite.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
using namespace google::protobuf;
using namespace google::protobuf::io;

#include "../util/aws_stdlib.h"

#include "../util/c_clock.h"
#include "../channel/ch_image.h"
#include "../channel/ch_obj.h"
#include "f_caffe.h"
using namespace caffe;

f_binary_classifier::f_binary_classifier(const char *fname): f_base(fname), m_verb(true), m_gpu(true){
  register_fpar("model_txt", m_fname_model_txt, 1024, "prototxt file defining net structure");
  register_fpar("model_bin", m_fname_model_bin, 1024, "caffemodel file trained by caffe");
  register_fpar("mean", m_fname_mean, 1024, "binaryproto file of mean image");
  register_fpar("img", (ch_base**)&m_pin, typeid(ch_image_ref).name(), "image for deep neural network.");
  register_fpar("obj", (ch_base**)&m_pout, typeid(ch_obj).name(), "object detected");
  register_fpar("img2", (ch_base**)&m_pimg, typeid(ch_image).name(),"");
  m_ss.setColSimWeight(1.f);
  m_ss.setTexSimWeight(1.f);
  m_ss.setSizeSimWeight(1.f);
  m_ss.setFillSimWeight(1.f);
  m_ss.setMaxNumVertexs(10000);
}



bool read_protobinary(const char *fname, BlobProto &blob_proto){
  //int fd =open(fname, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  int fd = open(fname, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found : " << fname << endl;
  ZeroCopyInputStream * raw_input = new FileInputStream(fd);
  CodedInputStream * coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(INT_MAX, 536870912);
  bool success = blob_proto.ParseFromCodedStream(coded_input);
  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void convert_protobinary_to_image(const BlobProto &blob_proto, Mat &img){
  vector<int> shape;
  shape.resize(blob_proto.shape().dim_size());
  cout << blob_proto.shape().dim_size() << endl;
  for(int i = 0; i < blob_proto.shape().dim_size(); ++i){
    shape[i] = blob_proto.shape().dim(i);
    cout << shape[i] << endl;
  }

  cout << shape[2] << ", " << shape[3] << endl;
  img.create(shape[2], shape[3], CV_32FC3);
  float * pimg = img.ptr<float>(0);
  const int total = img.total();
  const int total2 = total * 2;
  for(int i = 0; i < blob_proto.double_data_size(); ++i){
    pimg[i] = blob_proto.double_data(i);
    pimg[i+1] = blob_proto.double_data(i+total);
    pimg[i+2] = blob_proto.double_data(i+total2); 
  }
}

bool f_binary_classifier::init_run(){
  aws_scope_show ass("init_run");
  if(m_gpu)
    Caffe::set_mode(Caffe::GPU);
  else
    Caffe::set_mode(Caffe::CPU);
  

  if(m_verb){
    cout << "Loading prototxt " << m_fname_model_txt << endl;
    cout << "Loading binaryproto " << m_fname_model_bin << endl;
  }

  m_net = new Net<float>(m_fname_model_txt, TEST);
  m_net-> CopyTrainedLayersFrom(m_fname_model_bin);
  if(m_net->num_inputs() != 1){
    cerr << m_net->num_inputs() 
	 << " is invalid number of inputs of network." 
	 << endl;
    return false;
  }

  if(m_net->num_outputs() != 1){
    cerr << m_net->num_outputs()
	 << " is invalid number of outputs of network."
	 << endl;
    return false;
  }

  if(m_verb){
    cout << "Reshaping net" << endl;
  }
  Blob<float> * input_layer = m_net->input_blobs()[0];
  const int num_channels = input_layer->channels();
  if(num_channels != 3){
    cerr << "Error : binary classifier requires bgr image" << endl;
    return false;
  }
  
  m_sz_input = Size(input_layer->height(), input_layer->width());

  input_layer->Reshape(1, num_channels,
		       m_sz_input.height, m_sz_input.width);
  m_net->Reshape();

  if(m_verb){
    cout << "Loading mean image " << m_fname_mean << endl;
  }
  BlobProto blob_proto;
  bool success;
  ReadProtoFromBinaryFileOrDie(m_fname_mean, &blob_proto);
  
  // if(success){
  //   cerr << "Warning : Following files don't exist" << endl;
  //   return true;
  // }
  // bool success = read_protobinary(m_fname_mean, blob_proto);
  // if(!success){
  //   cerr << "Warning: Can't load followin protobinary." << endl;
  //   cerr << "protobinary : "<< m_fname_mean << endl;
  //   return true;
  // }
  
  if(m_verb){
    cout << "Converting protobinary to image" << endl;
  }

  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  const float * data = mean_blob.mutable_cpu_data();
  m_mean_img.create(mean_blob.height(), mean_blob.width(), CV_32FC3);
  if(m_verb){
    cout << "converting " << "proto to image " << m_mean_img.size() << endl;
  }

  float * pmean_img = m_mean_img.ptr<float>(0);
  const int total = m_mean_img.total();
  const int total2 = total * 2;
  for(int i = 0; i < total; ++i){
    pmean_img[i] = data[i];
    pmean_img[i+1] = data[i+total];
    pmean_img[i+2] = data[i+total2];
  }
  
  Mat tmp;
  resize(m_mean_img, tmp, m_sz_input);
  m_mean_img = tmp;
  
  return true;
}

void f_binary_classifier::destroy_run(){
  
}

bool f_binary_classifier::proc(){
  aws_scope_show ass("proc");

  long long timg;
  Mat img =m_pin->get_img(timg).clone();
  //Mat img = imread("/home/ubuntu/lena.png");
  if(img.empty()){
    return true;
  }
  
  cout << "start : SelectiveSearch" << endl;
  m_ss.processImage(img);
  vector<Rect> regions;
  m_ss.getRegions(regions);
  cout << "end : SelectiveSearch" << endl;

  Mat img_float;
  img.convertTo(img_float, CV_32FC3);
    
  const int num_objs = 2;
  c_vobj * pvobjs[num_objs];
  int iregions = 0;
  for(vector<Rect>::iterator it = regions.begin();
      it != regions.end(); ++it, ++iregions){
    if(iregions >= num_objs){
      break;
    }
    Mat roi = Mat(img_float, *it).clone();

    Mat roi_resized;
    if(roi.size() != m_sz_input){
      cout << "resizeing " << roi.size() << " to " << m_sz_input << endl;
      resize(roi, roi_resized, m_sz_input);
    }else{
      roi_resized = roi;
    }

    Mat roi_normalized;
    cout << "substracting" << endl;
    subtract(roi_resized, m_mean_img, roi_normalized);

    cout << "transfering" << endl;
    transfer_img_to_input_layer(roi_normalized);

    m_net->Forward();

    const Blob<float> * output_layer = m_net->output_blobs()[0];
    const float * begin = output_layer->cpu_data();
    const float * end = begin + output_layer->channels();
    vector<float> output = vector<float>(begin, end);

    pvobjs[iregions]=NULL;
    if(output[0] > output[1]){
      c_vobj * vobj = new c_vobj(timg, *it, roi, this, false);
      if(m_verb){
	cout << "detected at " << *it << endl;
	cout << output[0] << ", " << output[1] << endl;
      }
      rectangle(img, *it, Scalar(255, 0, 0));
      pvobjs[iregions] = vobj;
    }

  }
  imwrite("disp.png", img);
  m_pout->lock();
  m_pout->begin();
  while(!m_pout->is_end()){
    delete m_pout->cur();
    m_pout->ers();
    m_pout->next();
  }

  m_pout->begin();
  for(int i = 0; i < num_objs; ++i){
    if(pvobjs[i]){
      m_pout->ins(pvobjs[i]);
      c_vobj * vobj = (c_vobj*)pvobjs[i];//m_pout->cur();
      long long t;
      Rect rc;
      Mat tmp;
      const f_base * pfsrc;
      bool manual;
      vobj->get(t, rc, tmp, pfsrc, manual);
    }
  }
  m_pout->unlock();
  return true;
}

void f_binary_classifier::transfer_img_to_input_layer(const Mat &img){
  Blob<float>  * input_layer = m_net->input_blobs()[0];
  float * input_data = input_layer->mutable_cpu_data();
  const float * pimg = img.ptr<float>(0);
  const int total = img.total();
  const int total2 = total * 2;
  for(int i = 0; i < total; ++i){
    input_data[i] = pimg[i];
    input_data[i+total] = pimg[i+1];
    input_data[i+total2] = pimg[i+2]; 
  }
}
#endif

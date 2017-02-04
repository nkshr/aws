#include <vector>
#include <map>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
using namespace std;
using namespace cv;
using namespace cv::ximgproc::segmentation;

#include "selective_search.h"


SelectiveSearch::Vertex::Vertex(const int _index, const int tstart, SelectiveSearch * ss): size(0), index(_index), top_left(INT_MAX, INT_MAX), bottom_right(0, 0), ss(ss), /*base0(NULL), base1(NULL),*/ tsurv(tstart, INT_MAX){
  sub_vertexs.push_back(index);
}

void SelectiveSearch::Vertex::init(){
  calcColHist();
  calcTexHist();
  size = points.size();
  calcRegion();
  points_set.push_back(&points);
}

void SelectiveSearch::Vertex::calcColHist(){
  DMsg dmsg("caclColHist");
  col_hist.resize(3);
  for(int i = 0; i < col_hist.size(); ++i)
    col_hist[i].resize(25, 0.0);

  float istep = 25.f / 256.f;
  for(int i = 0; i < points.size(); ++i){
    for(int j = 0; j < 3; ++j){
      const uchar * p = ss->planes[j].ptr<uchar>(0);
      int bin = floor(p[points[i]] * istep);
      if(!(bin < 25)){
	cout << "bin : " <<  bin << endl;
	exit(1);
      }
      col_hist[j][bin] += 1.0;
    }
  }

  //normalize color histgram
  float isize = 1.f/(float)(points.size()*3.f);
  for(int i = 0; i < col_hist.size(); ++i){
    for(int j = 0; j < col_hist[i].size(); ++j){
      col_hist[i][j] *= isize;
      CV_Assert(col_hist[i][j] <= 1.0);
    }
  }
}

void SelectiveSearch::Vertex::calcTexHist()
{
  DMsg dmsg("calcTexHist");
  tex_hist.resize(3);
  for(int i = 0; i < tex_hist.size(); ++i){
    tex_hist[i].resize(8);
    for(int j = 0; j < tex_hist[i].size(); ++j){
      tex_hist[i][j].resize(10, 0);
    }
  }
 
  float istep_col = 10.f / 256.f;
  float istep_ori = (4.f * 2.f ) / CV_PI;
  for(int i = 0; i < points.size(); ++i){
    int pt = points[i];
    
    for(int j = 0; j < 3; ++j){
      const uchar * pimg = ss->planes[j].ptr<uchar>(0);
      const float * pori_img = ss->ori_imgs[j].ptr<float>(0);
      float tmp = pori_img[pt] * istep_ori;
      int ori_bin;
      if(tmp < 0)
	ori_bin = 8 + floor(tmp);
      else
	ori_bin = floor(tmp);
      if(!(ori_bin < 8)){
	cout << "ori_bin : " << ori_bin << endl;
	exit(1);
      }
      
      int col_bin = floor(pimg[pt] * istep_col);
      if(!(col_bin < 10)){
	cout << "col_bin : " << col_bin << endl;
	exit(1);
      }

      tex_hist[j][ori_bin][col_bin] += 1.0;
    }
  }
 
  float isize = 1.f/((float)points.size()*3);
  for(int i = 0; i < tex_hist.size(); ++i){
    for(int j = 0; j < tex_hist[i].size(); ++j){
      for(int k = 0; k < tex_hist[i][j].size(); ++k){
	tex_hist[i][j][k] *= isize;
	CV_Assert(tex_hist[i][j][k] <= 1.f);
      }
    }
  }
  CV_Assert(tex_hist.size() < 100000);
}

int SelectiveSearch::Vertex::assignOri(float val){
  const float istep = 8.0 / CV_PI;
  float ori = val * istep;
  if(ori < 0)
    ori += 4;
  return floor(ori);
}

int SelectiveSearch::Vertex::assignCol(int val){
  const float istep = 25.f / 255.f;
  int ret = floor((float)val * istep);
  return ret;
}

void SelectiveSearch::Vertex::calcRegion(){
  DMsg dmsg("calcRegion");
  region = Rect(top_left, Point(bottom_right.x + 1, bottom_right.y + 1));
}

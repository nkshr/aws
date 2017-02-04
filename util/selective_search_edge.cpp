#include <vector>
#include <map>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
using namespace std;
using namespace cv;
using namespace cv::ximgproc::segmentation;

#include "selective_search.h"

SelectiveSearch::Edge::Edge(SelectiveSearch * ss): sim(0), col_sim(0), tex_sim(0), size_sim(0), fill_sim(0), ss(ss){

}

void SelectiveSearch::Edge::calcTexSim(){
  TexHist &h0 = from->tex_hist;
  TexHist &h1 = to->tex_hist;
  CV_Assert(h0 != h1);
  tex_sim = 0;
  for(int i = 0; i < h0.size(); ++i){
    for(int j = 0; j < h0[i].size(); ++j){
      for(int k = 0; k < h0[i][j].size(); ++k){
	tex_sim += min(h0[i][j][k], h1[i][j][k]);
      }
    }
  }
}

void SelectiveSearch::Edge::calcFillSim(){
  Rect mrect = from->region | to->region;
  fill_sim = 1 - (mrect.area() - from->size - to->size)/(double)ss->planes[0].total();
  if(fill_sim > 1){
    cout << "mrect.area : " << mrect.area() << endl;
    cout << "from->size : " << from->size << endl;
    cout << "to->size : " << to->size << endl;
    exit(1);
  }
}

void SelectiveSearch::Edge::calcSizeSim(){
  CV_Assert(from->size != 0 || to->size != 0);
  size_sim = 1 - (from->size + to->size)/(float)ss->planes[0].total();
}

void SelectiveSearch::Edge::calcColSim(){
  ColHist &h0 = from->col_hist;
  ColHist &h1 = to->col_hist;

  col_sim = 0;
  for(int i = 0; i < h0.size(); ++i){
    for(int j = 0; j < h0[i].size(); ++j){
      col_sim += min(h0[i][j], h1[i][j]);
    }
  }
};

void SelectiveSearch::Edge::calcSim(){
  calcTexSim();
  calcFillSim();
  calcSizeSim();
  calcColSim();

  sim = ss->tex_sim_weight * tex_sim + ss->fill_sim_weight * fill_sim
    + ss->size_sim_weight * size_sim + ss->col_sim_weight * col_sim;
}

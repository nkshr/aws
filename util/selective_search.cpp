#include <vector>
#include <map>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
using namespace std;
using namespace cv;
using namespace cv::ximgproc::segmentation;

#include "selective_search.h"

Scalar hsv_to_rgb(Scalar c) {
    Mat in(1, 1, CV_32FC3);
    Mat out(1, 1, CV_32FC3);

    float * p = in.ptr<float>(0);

    p[0] = (float)c[0] * 360.0f;
    p[1] = (float)c[1];
    p[2] = (float)c[2];

    cvtColor(in, out, COLOR_HSV2RGB);

    Scalar t;

    Vec3f p2 = out.at<Vec3f>(0, 0);

    t[0] = (int)(p2[0] * 255);
    t[1] = (int)(p2[1] * 255);
    t[2] = (int)(p2[2] * 255);

    return t;
}

void SelectiveSearch::calcInitSims(){
  
}

void SelectiveSearch::propagateColHist(const Vertex &v0,
				       const Vertex &v1,
				       Vertex &v){
  const ColHist &h0 = v0.col_hist;
  const ColHist &h1 = v1.col_hist;
  ColHist &h = v.col_hist;
  h.resize(3);
  for(int i = 0; i < h.size(); ++i){
    h[i].resize(25, 0);
    for(int j = 0; j < h[i].size(); ++j){
      h[i][j] = (v0.size*h0[i][j] + v1.size*h1[i][j])/(v0.size + v1.size);
    }
  }
}

void SelectiveSearch::propagateTexHist(const Vertex &v0,
				       const Vertex &v1,
				       Vertex &v){
  const TexHist &h0 = v0.tex_hist;
  const TexHist &h1 = v1.tex_hist;
  TexHist &h = v.tex_hist;
  h.resize(3);
  for(int i = 0; i < h.size(); ++i){
    h[i].resize(8);
    for(int j = 0; j < h[i].size(); ++j){
      h[i][j].resize(10);
      for(int k = 0; k < h[i][j].size(); ++k){
	h[i][j][k] = (v0.size*h0[i][j][k] + v1.size*h1[i][j][k])/(v0.size + v1.size);
      }
    }
  } 
}

//void SelectiveSearch::propagate
Scalar color_mapping(int segment_id) {
  double base = (double)(segment_id) * 0.618033988749895 + 0.24443434;
  return hsv_to_rgb(Scalar(fmod(base, 1.2), 0.95, 0.80));
}

SelectiveSearch::SelectiveSearch(const float k,
				 const int min_size,
				 const double sigma):  k(k), min_size(min_size),
						       sigma(sigma), col_sim_weight(1.f),
						       fill_sim_weight(1.f),
						       size_sim_weight(1.f),
						       tex_sim_weight(1.f),
						       time(0), adj_table_size(100), vertexs_size(100), max_overlap(0.8){
  init();
  //regions.resize(1000);
}

SelectiveSearch::SelectiveSearch(): k(300), min_size(100), sigma(0.5),
				    col_sim_weight(1.f),
				    fill_sim_weight(1.f),
				    size_sim_weight(1.f),
				    tex_sim_weight(1.f),
				    time(0), adj_table_size(100),
				    vertexs_size(100),
				    max_overlap(0.8){
  init();
}

void SelectiveSearch::init(){
  gs = createGraphSegmentation();
  gs->setK(k);
  gs->setMinSize(min_size);
  gs->setSigma(sigma);
}

void SelectiveSearch::hierarGrouping(const Mat &img){
  //initialize vertexs
#ifdef TIME_SS
  clock_t tstart = clock();
#endif  
  gs->processImage(img, gs_img);
  double min, max;
  minMaxLoc(gs_img, &min, &max);

  int num_segs = (int)max + 1;
  vector<int> regions;
  vertexs.reserve(vertexs_size);
  vertexs.resize(0);
  
#ifdef DEBUG_SS
  cout << "num_segs : " << num_segs << endl;
#endif

  adj_table.reserve(adj_table_size);
  adj_table.resize(0);
  
  for(int i = 0; i < num_segs; ++i){
    vector<bool> buf;
    //buf.reserve(adj_table_size);
    buf.resize(num_segs, false);
    adj_table.push_back(buf);
  }

  time = 0;
  last_vindex = 0;
  for(int i = 0; i < num_segs; ++i, ++last_vindex){
    vertexs.push_back(Vertex(last_vindex, time, this));
    //vertexs[i].points_set.push_back(&points);
  }

  ////create adjacent table
  
#ifdef DEBUG_SS
  cout << "creating adjacent table" << endl;
#endif 
  
  for(int i = 0; i < gs_img.rows; ++i){
    int * pgs_img = gs_img.ptr<int>(i);
    for(int j = 0; j < gs_img.cols; ++j){
      if(j != 0 && pgs_img[j - 1] != pgs_img[j])
	adj_table[pgs_img[j]][pgs_img[j - 1]] = 1;

      if(j != gs_img.cols - 1 && pgs_img[j + 1] != pgs_img[j])
	adj_table[pgs_img[j]][pgs_img[j + 1]] = 1;

      if(i != 0 && pgs_img[j - gs_img.cols] != pgs_img[j])
	adj_table[pgs_img[j]][pgs_img[j - gs_img.cols]] = 1;

      if(i != gs_img.rows -1  && pgs_img[j + gs_img.cols] != pgs_img[j])
	adj_table[pgs_img[j]][pgs_img[j + gs_img.cols]] = 1;

      Vertex &vertex = vertexs[pgs_img[j]];
      vertex.points.push_back(i*gs_img.cols+ j);
      
      if(vertex.top_left.x > j)
	vertex.top_left.x = j;
      else if(vertex.bottom_right.x < j)
	vertex.bottom_right.x = j;

      if(vertex.top_left.y > i)
	vertex.top_left.y = i;
      else if(vertex.bottom_right.y < i)
	vertex.bottom_right.y = i;
    }
  }

#ifdef DEBUG_SS
  char buf[1024];
  sprintf(buf, "adj_table_%03d.txt", time);
  writeAdjTable(buf);

  sprintf(buf, "vertexs_%03d.txt", time);
  writeVertexs(buf);
#endif

  ////calc orientation
  split(img, this->planes);
  for(int i = 0; i < 3; ++i){
    Sobel(planes[i], dx_imgs[i], CV_32F, 1, 0);
    Sobel(planes[i], dy_imgs[i], CV_32F, 0, 1);
  }

  for(int i = 0; i < 3; ++i)
    ori_imgs[i].create(dx_imgs[i].size(), CV_32F);

  for(int k = 0; k < 3; ++k){
    for(int i = 0; i < dx_imgs[k].rows; ++i){
      float * pdx_img = dx_imgs[k].ptr<float>(i);
      float * pdy_img = dy_imgs[k].ptr<float>(i);
      float * pori_img = ori_imgs[k].ptr<float>(i);
      for(int j = 0; j < dx_imgs[k].cols; ++j){
	float at = (float)atan(pdy_img[j]/pdx_img[j]);
	if(isnan(at))
	   at = 0;
	pori_img[j] = at;
      }
    }
  }

#ifdef TIME_SS
  clock_t tend = clock();
  time_ofs << "initializing vertex : "
	   << (float)(tend - tstart)/CLOCKS_PER_SEC << endl;

  tstart = clock();
#endif

  for(int i = 0; i < vertexs.size(); ++i){
    vertexs[i].init();
  }
  
  //initialize edges
  for(int i = 0; i < adj_table.size(); ++i){
    for(int j = i + 1; j < adj_table.size(); ++j){
      if(adj_table[i][j]){
	Edge edge(this);
	edge.from = &vertexs[i];
	edge.to = &vertexs[j];
	edge.calcSim();
	//printf("edge.from %p\n", (void*)edge.from);
	edges.push_back(edge);
      }
    }
  }

  edges.sort(SelectiveSearch::compSims);

#ifdef TIME_SS
  tend = clock();
  time_ofs << "initializing edge : "
	   << (float)(tend - tstart)/CLOCKS_PER_SEC << endl;

  tstart = clock();
#endif

#ifdef DEBUG_SS
  Mat vis_img;
  visSeg(gs_img, vis_img);

  sprintf(buf, "vis_img_%03d.png", time);
  imwrite(buf, vis_img);

  sprintf(buf, "edges_%03d.txt", time);
  writeEdges(buf);
#endif  
  time++;
  while(edges.size() != 0){
#ifdef DEBUG_SS
    cout << "num edges : " << edges.size() << endl;
#endif 

    list<Edge>::iterator it = edges.begin();
    Edge edge = edges.front();;
    
    Vertex mv;
    mergeVertexs(*(edge.from), *(edge.to), mv);
    removeEdges(edge);

    vertexs.push_back(mv);
    
    updateAdjTable(mv);
    addEdges(mv);
    edges.sort(SelectiveSearch::compSims);

#ifdef DEBUG_SS
    sprintf(buf, "adj_table_%03d.txt", time);
    writeAdjTable(buf);

    Mat seg_img;
    createSegImg(seg_img);
    visSeg(seg_img, vis_img);
    rectangle(vis_img, mv.region, Scalar(0, 0, 0));
    sprintf(buf, "vis_img_%03d.png", time);
    imwrite(buf, vis_img);

    sprintf(buf, "edges_%03d.txt", time);
    writeEdges(buf);

    sprintf(buf, "vertexs_%03d.txt", time);
    writeVertexs(buf);
#endif
    time++;
  }

#ifdef TIME_SS
  tend = clock();
  time_ofs << "optimizing edge : "
	   << (float)(tend - tstart)/CLOCKS_PER_SEC << endl;
#endif

}

void SelectiveSearch::processImage(const Mat &img)
{
#ifdef TIME_SS
  time_ofs.open("time.txt");
  if(!time_ofs.good()){
    cerr << "time.txt is not well." << endl;
    exit(1);
  }
  time_ofs << "measuring selective search algorithm" << endl;
#endif  
  hierarGrouping(img);
  createRegions();
  edges.clear();

#ifdef TIME_SS
  time_ofs.close();
#endif
}

bool SelectiveSearch::compSims(const Edge &e0, const Edge &e1)
{
  if(e0.sim < e1.sim)
    return false;
  else
    return true;
}

bool SelectiveSearch::compPosVals(const pair<float, Rect> &r0, const pair<float, Rect> &r1){
  if(r0.first < r1.first)
    return false;
  else
    return true;
}

Mat &SelectiveSearch::getGSImage(){
  return gs_img;
}

void SelectiveSearch::mergeVertexs(Vertex &v0, Vertex &v1, Vertex &v){
#ifdef DEBUG_SS
  DMsg dmsg("Entering mergeVertexs");
  cout << v0.index << "th and " << v1.index << "th are merged." << endl;
#endif 
  
  v =  Vertex(last_vindex++, time, this);

#ifdef DEBUG_SS
  cout << "vertex " << v.index << " was constructed at time " << time << endl;
#endif 
  
  for(int i = 1; i < v0.sub_vertexs.size(); ++i){
    v.sub_vertexs.push_back(v0.sub_vertexs[i]);
  }

  for(int i = 1; i < v1.sub_vertexs.size(); ++i){
    v.sub_vertexs.push_back(v1.sub_vertexs[i]);
  }

  v.sub_vertexs.push_back(v0.sub_vertexs[0]);
  v.sub_vertexs.push_back(v1.sub_vertexs[0]);

  for(int i = 0; i < v0.points_set.size(); ++i){
    v.points_set.push_back(v0.points_set[i]);
  }

  for(int i = 0; i < v1.points_set.size(); ++i){
    v.points_set.push_back(v1.points_set[i]);
  }

  v0.tsurv.tend = time;

  v1.tsurv.tend = time;

  v.size = v0.size + v1.size;
  v.region = v0.region | v1.region;
  propagateTexHist(v0, v1, v);
 
  propagateColHist(v0, v1, v);
}

vector<Rect> SelectiveSearch::getRegions(const int lvl){
  vector<Rect> regions;
  return regions;
}

void SelectiveSearch::getRegions(vector<Rect> &_regions){
  DMsg dmsg("getRegions");
  _regions.reserve(regions.size());
  _regions.resize(0);
  list<pair<float, Rect> >::iterator it = regions.begin();
  for(; it != regions.end(); ++it){
    _regions.push_back(it->second);
  }
}

void visSeg(const Mat &seg_img,
			   Mat &vis_img){
  DMsg("Entering visSeg");
  vis_img.create(seg_img.size(), CV_8UC3);
  for(int i = 0; i < seg_img.rows; ++i){
    const int * pseg_img = seg_img.ptr<int>(i);
    uchar * pvis_img = vis_img.ptr<uchar>(i);
    for(int j = 0; j < seg_img.cols; ++j){
      Scalar color = color_mapping(pseg_img[j]);
      pvis_img[j*3] = (uchar)color[0];
      pvis_img[j*3+1] = (uchar)color[1];
      pvis_img[j*3+2] = (uchar)color[2];
    }
  }
  drawBoundaries(seg_img, Scalar(255, 255, 255), vis_img);
}

void visOneSeg(const Mat &seg_img, const int index, Mat &vis_img)
{
  Scalar red(0, 0, 255);
  vis_img.create(seg_img.size(), CV_8UC3);
  for(int i = 0; i < seg_img.rows; ++i){
    const int * pseg_img = seg_img.ptr<int>(i);
    uchar * pvis_img = vis_img.ptr<uchar>(i);
    for(int j = 0; j < seg_img.cols; ++j){
      if(index == pseg_img[j]){
	pvis_img[j*3] = (uchar)red[0];
	pvis_img[j*3+1] = (uchar)red[1];
	pvis_img[j*3+2] = (uchar)red[2];
      }
      else{
	pvis_img[j*3] = 0;
	pvis_img[j*3+1] = 0;
	pvis_img[j*3+2] = 0;
      
      }
    }
  }
}

bool writeMat(const Mat &m, char * fname){
  ofstream ofs(fname);
  if(!ofs.is_open()){
    cerr << "Error : Couldn't open " << fname << endl;
    return false;
  }
  ofs << m << endl;
  ofs.close();
  return true;  
}

void visAdjTable(const Mat &adj_table, const Mat &seg_img, const int index, Mat &vis_img){
  Scalar red(0, 0, 255);
  Scalar blue(255, 0, 0);
  Scalar green(0, 255, 0);
  vis_img = Mat::zeros(seg_img.size(), CV_8UC3);
  vector<int> adj_regions;
  const uchar * padj_table = adj_table.ptr<uchar>(index);
  for(int i = 0; i < adj_table.cols; ++i){
    if(index == i)
      continue;
    
    if(padj_table[i] == 1){
      adj_regions.push_back(i);
    }
  }
  
  for(int i = 0; i < seg_img.rows; ++i){
    const int * pseg_img = seg_img.ptr<int>(i);
    uchar * pvis_img = vis_img.ptr<uchar>(i);
    for(int j = 0; j < seg_img.cols; ++j){
      if(index == pseg_img[j]){
	pvis_img[j*3] = (uchar)red[0];
	pvis_img[j*3+1] = (uchar)red[1];
	pvis_img[j*3+2] = (uchar)red[2];
      }

      for(int k = 0; k < adj_regions.size(); ++k){
	if(pseg_img[j] == adj_regions[k]){
	  pvis_img[j*3] = (uchar)blue[0];
	  pvis_img[j*3+1] = (uchar)blue[1];
	  pvis_img[j*3+2] = (uchar)blue[2];
	  break;
	}
      }
    }
  }
  drawBoundaries(seg_img, green, vis_img);
}

vector<Rect> extractAdjRegions(const Mat &adj_table, const int index, const vector<Rect> regions){
  const uchar * padj_table = adj_table.ptr<uchar>(index);
  vector<Rect> adj_regions;
  for(int i = 0; i < adj_table.cols; ++i){
    if(padj_table[i] == 1){
      adj_regions.push_back(regions[i]);
    }
  }

  return adj_regions;
};

void drawBoundaries(const Mat &seg_img, const Scalar &bcolor, Mat &img)
{
  for(int i = 1; i < seg_img.rows - 1; ++i){
    const int * pseg_img = seg_img.ptr<int>(i);
    uchar * pimg = img.ptr<uchar>(i);
    for(int j = 1; j < seg_img.cols - 1; ++j){
      if(pseg_img[j-1-seg_img.cols] != pseg_img[j]){
	pimg[j*3] = bcolor[0];
        pimg[j*3+1] = bcolor[1];
	pimg[j*3+2] = bcolor[2];
	continue;
      }

      if(pseg_img[j-seg_img.cols] != pseg_img[j]){
	pimg[j*3] = bcolor[0];
	pimg[j*3+1] = bcolor[1];
	pimg[j*3+2] = bcolor[2];
	continue;
      }

      if(pseg_img[j + 1 -seg_img.cols] != pseg_img[j]){
	pimg[j*3] = bcolor[0];
	pimg[j*3+1] = bcolor[1];
	pimg[j*3+2] = bcolor[2];
	continue;
      }

      if(pseg_img[j-1] != pseg_img[j]){
	pimg[j*3] = bcolor[0];
	pimg[j*3+1] = bcolor[1];
	pimg[j*3+2] = bcolor[2];
	continue;
      }

      if(pseg_img[j+1] != pseg_img[j]){
	pimg[j*3] = bcolor[0];
	pimg[j*3+1] = bcolor[1];
	pimg[j*3+2] = bcolor[2];
	continue;
      }

      if(pseg_img[j - 1 + seg_img.cols] != pseg_img[j]){
	pimg[j*3] = bcolor[0];
	pimg[j*3+1] = bcolor[1];
	pimg[j*3+2] = bcolor[2];
	continue;
      }

      if(pseg_img[j + seg_img.cols] != pseg_img[j]){
	pimg[j*3] = bcolor[0];
	pimg[j*3+1] = bcolor[1];
	pimg[j*3+2] = bcolor[2];
	continue;
      }

      if(pseg_img[j + 1 + seg_img.cols] != pseg_img[j]){
	pimg[j*3] = bcolor[0];
	pimg[j*3+1] = bcolor[1];
	pimg[j*3+2] = bcolor[2];
	continue;
      }

    }
  }
}

void SelectiveSearch::createSegImg(Mat &seg_img){
  DMsg dmsg("Entering createSegImg");
  seg_img.create(gs_img.size(), CV_32SC1);
  int * pseg_img = seg_img.ptr<int>(0);
  int count = 0;
  for(vector<Vertex>::iterator it = vertexs.begin(); it != vertexs.end(); ++it){
    if(!isAlive(*it)){
      continue;
    }
    
    for(int i = 0; i < it->points_set.size(); ++i){
      vector<int> points = (*it->points_set[i]);
      for(int j = 0; j < points.size(); ++j){
    	pseg_img[points[j]] = it->index;
      }
    }
    count++;
  }

#ifdef DEBUG_SS
  cout << "num segs : " << count << endl;
#endif 
}

bool SelectiveSearch::isAlive(const Vertex &v){
  if(v.tsurv.tstart <= time && time < v.tsurv.tend){
    return true;
  }
  else{
    return false;
  }
}

void SelectiveSearch::updateAdjTable(const Vertex &v){
  DMsg dmsg("Entering updateAdjTable");
  vector<int>::const_reverse_iterator rit = v.sub_vertexs.rbegin();
  const int vindex0 = *(rit);
  const int vindex1 = *(++rit);
  vector<bool> buf;
  buf.reserve(adj_table_size);
  buf.resize(v.index + 1, false);
  adj_table.push_back(buf);
  adj_table.back().resize(v.index + 1, false);
  for(int i = 0; i < adj_table.size() - 1; ++i){
    bool enbl = false;
    bool vsub = false;
    
    
    for(int j = 0; j < v.sub_vertexs.size(); ++j){
	if(v.sub_vertexs[j] == i){
	  vsub = true;
	}
    }

    // if(!vsub)
    //   cout << i << "th vertex is vsub" << endl;
    if((adj_table[vindex0][i] || adj_table[vindex1][i]) && !vsub && isAlive(vertexs[i])){
#ifdef DEBUG_SS
      cout << i << "th vertex is accepted." << endl;
#endif
      enbl = true;
    }
    else{
#ifdef DEBUG_SS
      cout << i  << "th vertex is rejected." << endl;
#endif
    }

    adj_table[i].push_back(enbl);
    adj_table[v.index][i] = enbl;

    CV_Assert(i < adj_table.size());
    CV_Assert(v.index < adj_table.size());
    CV_Assert(vindex0 < adj_table.size());
    CV_Assert(vindex1 < adj_table.size());
    CV_Assert(i < adj_table[v.index].size());
    CV_Assert(i < adj_table[vindex0].size());
    CV_Assert(i < adj_table[vindex1].size());
  }
}

void SelectiveSearch::addEdges(Vertex &v){
  DMsg dmsg("Entering addEdges");
  int count = 0;
  for(int i = 0; i < adj_table[v.index].size(); ++i){
    if(adj_table[v.index][i] && v.index != i){
      Edge edge(this);
      edge.from = &vertexs[v.index];
      edge.to = &vertexs[i];
      edge.calcSim();
      edges.push_back(edge);
      count++;
#ifdef DEBUG_SS
      cout << "edge(" << edge.from->index
	   << ", " << edge.to->index << ")"
	   << "is added." << endl;
#endif
    }
  }
#ifdef DEBUG_SS
  cout << count << " edges added." << endl;
#endif
}

bool SelectiveSearch::writeAdjTable(char * fname){
  ofstream ofs(fname);
  if(!ofs.is_open()){
    cerr << "Error : Couldn't open " << fname << endl;
    return false;
  }
  for(int i = 0; i < adj_table.size(); ++i){
    for(int j = 0; j < adj_table[i].size(); ++j){
      ofs << adj_table[i][j] << ", ";
    }
    ofs << endl;
  }
  ofs.close();
  return true;  
}

bool SelectiveSearch::writeEdges(char * fname){
  ofstream ofs(fname);
  if(!ofs.is_open()){
    cerr << "Error : Couldn't open " << fname << endl;
    return false;
  }

  list<Edge>::iterator it;
  int i = 0;
  for(it = edges.begin(); it != edges.end(); ++it){
    ofs << "edge : " << i++ << endl;
    ofs << "\tfrom : " << it->from->index << endl;
    ofs << "\tto : " << it->to->index << endl;
    ofs << "\tsim : " << it->sim << endl;
    ofs << "\tcol_sim : " << it->col_sim << endl;
    ofs << "\ttex_sim : " << it->tex_sim << endl;
    ofs << "\tsize_sim : " << it->size_sim << endl;
    ofs << "\tfill_sim : " << (*it).fill_sim << endl;
  }
  
  ofs.close();
  return true;
}


void SelectiveSearch::removeEdges(Edge &e){
  int count = 0;
  SelectiveSearch::Vertex * from = e.from;
  SelectiveSearch::Vertex * to = e.to;
  //cout << e.to->index << endl;
  for(list<Edge>::iterator it = edges.begin(); it != edges.end();){      
    if(it->from == from ||
       it->from->index == to->index ||
       it->to == from ||
       it->to->index == to->index){
#ifdef DEBUG_SS
      cout << "edge(" << it->from->index
	   << ", " << it->to->index << ")"
	   << "is removed." << endl;
#endif
      it = edges.erase(it);
      count++;
    }else{
      ++it;
    }
  }
#ifdef DEBUG_SS
  cout << count << " edges removed." << endl;
#endif
}

bool SelectiveSearch::writeVertexs(char * fname){
  ofstream ofs(fname);
  if(!ofs.is_open()){
    cerr << "Error : Couldn't open " << fname << endl;
    return false;
  }

  vector <Vertex>::iterator it;
  for(it = vertexs.begin(); it != vertexs.end(); ++it){
    ofs << "vertex : " << it->index << endl;
    for(int i = 0; i < it->sub_vertexs.size(); ++i){
      ofs << "\tsub vertex : " << it->sub_vertexs[i] << endl;
    }
  }
  ofs.close();
  return true;
}

//setter
void SelectiveSearch::setColSimWeight(const float w){
  col_sim_weight = w;
}

void SelectiveSearch::setFillSimWeight(const float w){
  fill_sim_weight = w;
}

void SelectiveSearch::setSizeSimWeight(const float w){
  size_sim_weight = w;
}

void SelectiveSearch::setTexSimWeight(const float w){
  tex_sim_weight = w;
}

void SelectiveSearch::createRegions(){
#ifdef TIME_SS
  clock_t tstart = clock();
#endif

  DMsg dmsg("createRegions");
  int vsz = vertexs.size();
  for(int i = 0; i < vsz; ++i){
    Vertex &v = vertexs[i];
    float pos_val = (v.tsurv.tstart - vsz) * rand()/(float)RAND_MAX;
    regions.push_back(pair<float, Rect>(pos_val, v.region));
  }

  regions.sort(compPosVals);
  vector<list<pair<float, Rect> >::iterator > over_regions;
  over_regions.reserve(100);
  for(list<pair<float, Rect> >::iterator it0 = regions.begin(); it0 != regions.end(); ++it0){
    list<pair<float, Rect> >::iterator it1 = it0;
    ++it1;
    for(; it1 != regions.end(); ++it1){
      if(calcOverlap(it0->second, it1->second) > max_overlap){
    	over_regions.push_back(it1);
      }
    }

    for(int i = 0; i < over_regions.size(); ++i){
#ifdef DEBUG_SS
      cout << "num over : " << over_regions.size() << endl;
#endif
      regions.erase(over_regions[i]);
    }
    over_regions.resize(0);
  }

#ifdef TIME_SS
  clock_t tend = clock();
  time_ofs << "createRegion : "
	   << (float)(tend - tstart)/CLOCKS_PER_SEC << endl;
#endif

}

float SelectiveSearch::calcOverlap(const Rect &r0, const Rect &r1){
  Rect rand = r0 & r1;
  Rect ror = r0 | r1;
  return rand.area()/(float)ror.area();
}

void SelectiveSearch::setMaxNumVertexs(const int num){
  vertexs_size = num;
}

void SelectiveSearch::setMaxAdjTableSize(const int num){
  adj_table_size = num;
}

void SelectiveSearch::getVertexs(vector<Vertex> &vtxs){
  vtxs = vertexs;
}

void debug_print(char * str){
#ifdef DEBUG_SS
  cout << str << endl;
#endif
}

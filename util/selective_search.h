#ifndef SELECTIVE_SEARCH
#define SELECTIVE_SEARCH

Scalar color_mapping(int segment_id);
Scalar hsv_to_rgb(Scalar c);

typedef vector<vector<double> > ColHist;
typedef vector<vector<vector<double > > >TexHist;

class SelectiveSearch{
 public:
  class Vertex{
  public:
    //int id;
    int size;
    int index;
    
    Point top_left;
    Point bottom_right;
    Rect region;
    
    vector<int> points;
    vector<vector<int>* > points_set;
    vector<int> sub_vertexs;
    ColHist col_hist;//color channel, color
    TexHist tex_hist;//color channel, orientation
    SelectiveSearch * ss;
    Vertex(){};
    Vertex(const int index, const int tstart, SelectiveSearch * ss);

    struct tsurvival{
    tsurvival() : tstart(INT_MAX), tend(INT_MAX){};
    tsurvival(const int tstart, const int tend) : tstart(tstart),
	tend(tend){};
      int tstart;
      int tend;
    }tsurv;
    
    void init();
    void calcColHist();
    void calcTexHist();
    
    int assignCol(int val);
    int assignOri(float val);
    void calcRegion();
    Rect getRegion()
    {
      return region;
    }
  };

  class Edge{
  public:
    SelectiveSearch::Vertex * from;
    SelectiveSearch::Vertex * to;
    //int id;
    double sim;
    double col_sim;
    double tex_sim;
    double size_sim;
    double fill_sim;
    SelectiveSearch *ss;
    
    Edge(SelectiveSearch * ss);
    
    void calcSim();
    void calcColSim();
    void calcFillSim();
    void calcSizeSim();
    void calcTexSim();
  };

  SelectiveSearch();
  SelectiveSearch(const float k, const int min_size, const double sigma);

  void processImage(const Mat &img);

  void setColSimWeight(const float w);
  void setFillSimWeight(const float w);
  void setSizeSimWeight(const float w);
  void setTexSimWeight(const float w);
  void setMaxNumVertexs(const int num);
  void setMaxAdjTableSize(const int num);
  
  Mat& getGSImage();
  vector<Rect> getRegions(const int lvl);
  void getRegions(vector<Rect> &regions);
  void getVertexs(vector<Vertex> &vtxs);
  
 private:
  //parameter for GraphSegmentation
  float k;
  int min_size;
  double sigma;

  //parameter for selective search algorithm
  float col_sim_weight;
  float fill_sim_weight;
  float size_sim_weight;
  float tex_sim_weight;
  
  int time;
  int last_vindex;
  int adj_table_size;
  int vertexs_size;

  float max_overlap;

  ofstream time_ofs;
  
  Mat planes[3];
  Mat dx_imgs[3];
  Mat dy_imgs[3];
  Mat ori_imgs[3];
  Mat gs_img;
  vector<Vertex> vertexs;
  vector<vector<bool> > adj_table;
  list<Edge> edges;
  Ptr<GraphSegmentation> gs;

  void init();
  
  void calcInitSims();
  
  void propagateColHist(const Vertex &v0, const Vertex &v1, Vertex &v);
  void propagateTexHist(const Vertex &v0, const Vertex &v1, Vertex &v);
 
  void hierarGrouping(const Mat &img);

  void mergeVertexs(Vertex &v0, Vertex &v1, Vertex &v);

  void updateAdjTable(const Vertex &v);
  
  bool isAlive(const Vertex &v);
  
  void createSegImg(Mat &seg_img);

  void addEdges(Vertex &v);

  void removeEdges(Edge &e);

  void createRegions();
  
  bool writeAdjTable(char * fname);
  bool writeEdges(char * fname);
  bool writeVertexs(char * fname);

  float calcOverlap(const Rect &r0, const Rect &r1);
  
  static bool compSims(const Edge &e0, const Edge &e1);
  static bool compPosVals(const pair<float, Rect> &r0, const pair<float, Rect> &r1);
  //void propagateSizeHist();

  list<pair<float, Rect> > regions;

};

class DMsg{
 public:
  string buf;
    DMsg(string buf): buf(buf){
#ifdef DEBUG_SS
    cout << "Entering " << buf << endl;
#endif 
  }
    ~DMsg(){
#ifdef DEBUG_SS
    cout << "Exiting " << buf << endl;
#endif
  }
};

void visSeg(const Mat &seg_img, Mat &vis_img);

void visOneSeg(const Mat &seg_img, const int index, Mat &vis_img);

bool writeMat(const Mat &m, char * fname);

void visAdjTable(const Mat &adj_table, const Mat &seg_img, const int index, Mat &vis_img);

vector<Rect> extractAdjRegions(const Mat &adj_table, const int index, const vector<Rect> regions);

void drawBoundaries(const Mat &seg_img, const Scalar &bcolor, Mat &img);

void debug_print(char * str);
#endif

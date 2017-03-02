#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <thread>
#include <mutex>

using namespace std;

#include "../util/aws_stdlib.h"
#include "../util/aws_thread.h"
#include "../util/c_clock.h"

#include <opencv2/opencv.hpp>
using namespace cv;

#include "f_aws1_ctrl.h"

#include "../channel/ch_aws1_ctrl.h"
#include "../channel/ch_state.h"
#include "f_aws1_sim.h"

s_aws1_mod::s_aws1_state::s_aws1_state(): ang_vel(0), sog(0), ot(0){
  
}

s_aws1_mod::s_aws1_mod(): res_meng(256), res_rud(256){
}

s_aws1_mod::~s_aws1_mod(){
  for(int i = 0; i < res_meng; ++i){
    delete [] table[i];
  }
  delete [] table;
}

s_aws1_mod::s_aws1_state** s_aws1_mod::allocate2d(){
  s_aws1_state ** tmp = new s_aws1_state*[res_meng];
  for(int i = 0; i < res_meng; ++i){
    tmp[i] = new s_aws1_state[res_rud];
  }
  return tmp;
}

void s_aws1_mod::init(){
  table = new s_aws1_state*[res_meng];
  for(int i = 0 ; i < res_meng; ++i){
    table[i] = new s_aws1_state[res_rud];
  }
}


void s_aws1_mod::update(const unsigned char meng,  const unsigned char rud,
	    const float ang_vel, const float sog, const long long ot){
  const int mapped_meng = (int)round(((float)meng / 256.f)*(float)(res_meng));
  const int mapped_rud = (int)round(((float)rud / 256.f)*(float)(res_rud));
  s_aws1_state &state = table[mapped_meng][mapped_rud];
  float s0 = (float)ot / (float)(ot + state.ot);
  float s1 = 1.f - s0;
  state.ang_vel = s1 * state.ang_vel + s0 * ang_vel;
  state.sog = s1 * state.sog + s0 * sog;
  state.ot += ot;
  cout << "table[" << mapped_meng << "][" << mapped_rud << "]" << " is updated to ";
  cout << "ang_vel " << state.ang_vel << " sog " << state.sog << " ot " << state.ot;
  cout << " by s0 " << s0 << " s1 " << s1 << endl;
}

void s_aws1_mod::interpolate_table(){
  aws_scope_show ass("interpolate_table");
  for(int i = 0; i < res_meng; ++i){
    for(int j = 0; j < res_rud; ++j){
      s_aws1_state * state = &table[i][j];
      if(state->ot == 0){
	const s_aws1_state nstate = get_nearest_cell(i, j);
	state->ang_vel = nstate.ang_vel;
	state->sog = nstate.sog;
      }
      
    }
  }
}

// void s_aws1_mod::reshape(){
//   const int prev_res_meng = sizeof(table) / sizeof(s_aws1_mod::s_aws1_state*);
//   const int prev_res_rud = sizeof(&table[0]) / sizeof(s_aws1_mod::s_aws1_state);
//   cout << prev_res_meng << ", " << prev_res_rud << " to " << res_meng << ", " << res_rud << endl;
//   float fx = (float)res_meng/(float)prev_res_meng;
//   float fy = (float)res_rud/(float)prev_res_rud;

//   s_aws1_state  ** tmp = new s_aws1_state*[res_meng];
//   for(int i = 0 ; i < res_meng; ++i){
//     tmp[i] = new s_aws1_state[res_rud];
//   }

//   for(int i = 0; i < res_meng; ++i){
//     const int imeng = (int)fx * i;
//     for(int j = 0; j < res_rud; ++j){
//       const int irud = j * fy;
//       tmp[i][j] = table[imeng][irud];
//     }
//   }  

//   for(int i = 0; i < res_meng; ++i)
//     delete [] table[i];
//   delete [] table;

//   table = tmp;
// }

void s_aws1_mod::destroy(){
  for(int i = 0; i < res_meng; ++i){
    delete [] table[i];
  }
  delete [] table;
}

s_aws1_mod::s_aws1_state& s_aws1_mod::get_nearest_cell(const int meng, const int rud){
  //aws_scope_show ass(__PRETTY_FUNCTION__);
  float dmin = FLT_MAX; //maximum distance
  int min_meng = 0;
  int min_rud = 0;
  for(int i = 0; i < res_meng; ++i){
    for(int j = 0; j < res_rud; ++j){
      s_aws1_state * state = &table[i][j];
      if(state->ot){
	float d = (float)sqrt(pow(i - meng, 2.0) + pow(j - rud, 2.0));
	if(d < dmin){
	  min_meng = i;
	  min_rud = j;
	  dmin = d;
	}
      }
    }
  }
  cout << meng << ", " << rud << " is nearest to " << min_meng << ", " <<  min_rud << endl;
  return table[min_meng][min_rud];
}

bool s_aws1_mod::read(const char * fname){
  FILE * pf = fopen(fname, "rb");
  if(!pf)
    return false;

  int sz = 0;
  for(int i = 0; i < res_meng; ++i){
    for(int j = 0; j < res_rud; ++j){
      s_aws1_state * state = &table[i][j];
      sz += fread(&(state->ang_vel), sizeof(float), 1, pf);
      sz += fread(&(state->sog), sizeof(float), 1, pf);
      sz += fread(&(state->ot), sizeof(long long), 1, pf);      
    }
  }
  fclose(pf);
}

bool s_aws1_mod::write(const char * fname){
  FILE * pf = fopen(fname, "wb");
  if(!pf)
    return false;
  
  int sz = 0;
  for(int i = 0; i < res_meng; ++i){
    for(int j = 0; j < res_rud; ++j){
      s_aws1_state * state = &table[i][j];
      sz += (fwrite(&(state->ang_vel), sizeof(float), 1, pf)) * sizeof(float);
      sz += fwrite(&(state->sog), sizeof(float), 1, pf) * sizeof(float);
      sz += fwrite(&(state->ot), sizeof(long long), 1, pf) * sizeof(long long);      
    }
  }

  fclose(pf);
  const int esz = (sizeof(float)+ sizeof(float) + sizeof(long long)) * res_meng * res_rud;
  if(sz != esz){
    cerr << __PRETTY_FUNCTION__ << "\n";
    cerr << "Error : total size of model written is " << sz << ", but expected size is " << esz << endl;
    return false;
  }
  return true;
}

bool s_aws1_mod::write_sog_csv(const char * fname){
  ofstream ofs(fname);
  if(!ofs.good()){
    return false;
  }

  for(int i = 0; i < res_meng; ++i){
    for(int j = 0; j < res_rud; ++j){
      ofs << table[i][j].sog << ",";
    }
    ofs << endl;
  }
  ofs.close();
  return true;
}

bool s_aws1_mod::write_ang_vel_csv(const char * fname){
  ofstream ofs(fname);
  if(!ofs.good()){
    return false;
  }

  for(int i = 0; i < res_meng; ++i){
    for(int j = 0; j < res_rud; ++j){
      ofs << table[i][j].ang_vel << ",";
    }
    ofs << endl;
  }
  ofs.close();
  return true;
}

bool s_aws1_mod::write_ot_csv(const char * fname){
  ofstream ofs(fname);
  if(!ofs.good()){
    return false;
  }

  for(int i = 0; i < res_meng; ++i){
    for(int j = 0; j < res_rud; ++j){
      ofs << table[i][j].ot << ",";
    }
    ofs << endl;
  }
  ofs.close();
  return true;
}

s_aws1_mod::s_aws1_state s_aws1_mod::get_state(const unsigned char  meng, const unsigned char rud){
  const float smeng = 256.f / (float)res_meng;
  const float srud = 256.f / (float)res_rud;
  const int imeng = (int)round(smeng * meng);
  const int irud = (int)round(srud * rud);
}

void f_aws1_sim::get_inst()
{

  s_aws1_ctrl_inst inst;
  if(m_ch_ctrl_ui){
    m_ch_ctrl_ui->get(inst);
  }

  m_ctrl_stat.tcur = inst.tcur;
  m_ctrl_stat.ctrl_src = inst.ctrl_src;

  switch(m_ctrl_stat.ctrl_src){
  case ACS_UI:
    m_ctrl_stat.rud_aws = inst.rud_aws;
    m_ctrl_stat.meng_aws = inst.meng_aws;
    m_ctrl_stat.seng_aws = inst.seng_aws;
    break;
  case ACS_AP1:
    if(m_ch_ctrl_ap1){
      m_ch_ctrl_ap1->get(inst);
      m_ctrl_stat.rud_aws = inst.rud_aws;
      m_ctrl_stat.meng_aws = inst.meng_aws;
      m_ctrl_stat.seng_aws = inst.seng_aws;
    }else{
      cerr << "In " << m_name << ", ";
      cerr << "No autopilot channel 1 is connected" << endl;
    }
    break;
  case ACS_AP2:
    if(m_ch_ctrl_ap2){
      m_ch_ctrl_ap2->get(inst);
      m_ctrl_stat.rud_aws = inst.rud_aws;
      m_ctrl_stat.meng_aws = inst.meng_aws;
      m_ctrl_stat.seng_aws = inst.seng_aws;
    }else{
      cerr << "In " << m_name << ", ";
      cerr << "No autopilot channel 2 is connected" << endl;
    }
    break;
  }
}

void f_aws1_sim::map_stat()
{
  m_ctrl_stat.rud = map_oval(m_ctrl_stat.rud_aws, 
			   0xff, 0x7f, 0x00, 
			   m_ctrl_stat.rud_max, m_ctrl_stat.rud_nut, m_ctrl_stat.rud_min);
  m_ctrl_stat.meng = map_oval(m_ctrl_stat.meng_aws, 
			    0xff, 0x7f + 0x19, 0x7f, 0x7f - 0x19, 0x00,
			    m_ctrl_stat.meng_max, m_ctrl_stat.meng_nuf, m_ctrl_stat.meng_nut, 
			    m_ctrl_stat.meng_nub, m_ctrl_stat.meng_min);  
  m_ctrl_stat.seng = map_oval(m_ctrl_stat.seng_aws, 
			    0xff, 0x7f + 0x19, 0x7f, 0x7f - 0x19, 0x00,
			    m_ctrl_stat.seng_max, m_ctrl_stat.seng_nuf, m_ctrl_stat.seng_nut, 
			    m_ctrl_stat.seng_nub, m_ctrl_stat.seng_min);
}

f_aws1_sim::f_aws1_sim(const char * fname): f_base(fname), m_bwrite_ang_vel_csv(false), m_bwrite_sog_csv(false), m_bwrite_ot_csv(false), m_binterpolate(false), m_lat(0.f), m_lon(0.f), m_alt(100.f), m_galt(100.f){
  m_Xorg = Mat::zeros(3, 1, CV_32F);
  float * pm_Xorg = m_Xorg.ptr<float>(0);
  register_fpar("write_ang_vel_csv", &m_bwrite_ang_vel_csv, "Write angular velocity csv.");
  register_fpar("write_sog_csv", &m_bwrite_sog_csv, "Write sog csv.");
  register_fpar("write_ot_csv", &m_bwrite_ot_csv, "Write observation time csv.");

  register_fpar("mod", m_fmod, 1023, "Model file.");
  register_fpar("sog_csv", m_fsog_csv, 1023, "csv file of sog.");
  register_fpar("ang_vel_csv", m_fang_vel_csv, 1023, "csv file of angular velocity.");
  register_fpar("ot_csv", m_fot_csv, 1023, "csv file of observation time."); 

  register_fpar("lat", &m_lat, "Initial latitude (default 0)");
  register_fpar("lon", &m_lon, "Initial longitude (default 0)");
  register_fpar("alt", &m_alt, "Initial altitude (default 100)");
  register_fpar("galt", &m_galt, "Initial altitude (default 100)");
  register_fpar("cog", &m_cog, "Initial cog");
}

bool f_aws1_sim::init_run(){
  //read a model specified by m_fmod
  if(!m_mod.read(m_fmod)){
    cerr << __PRETTY_FUNCTION__ << endl;
    cerr << "Error : Couldn't read " << m_fmod << "." << endl;
    return false;
  }

  float * pm_Xorg = m_Xorg.ptr<float>(0);
  bihtoecef(m_lat, m_lon, m_alt, pm_Xorg[0], pm_Xorg[1], pm_Xorg[2]);
  getwrldrot(m_lat, m_lon, m_Rwrld);
  m_prev_t = get_time();
  return true;
}

bool f_aws1_sim::proc(){
  if(!m_ch_ctrl_stat){
    cerr << __PRETTY_FUNCTION__ << endl;
    cerr << "Error : ch_state is not specified." << endl;
    return false;
  }

  if(m_bwrite_sog_csv){
    cout << __PRETTY_FUNCTION__ << "\n";
    cout << "Writing sog to " << m_fsog_csv << endl;
    bool success = m_mod.write_sog_csv(m_fsog_csv);
    if(!success){
      cerr << __PRETTY_FUNCTION__ << "\n";
      cerr << "Error : Couldn't write " << m_fsog_csv << endl;
      return false;
    }
    else{
      cout << m_fsog_csv << " was written." << endl;
    }

    m_bwrite_sog_csv = false;
  }
  
  if(m_bwrite_ang_vel_csv){
    cout << __PRETTY_FUNCTION__ << "\n";
    cout << "Writing angular velocity to " << m_fang_vel_csv << endl;    
    bool success = m_mod.write_ang_vel_csv(m_fang_vel_csv);
    if(!success){
      cerr << __PRETTY_FUNCTION__ << "\n";
      cerr << "Error : Couldn't write " << m_fang_vel_csv << endl;
      return false;
    }
    else{
      cout << m_fang_vel_csv << " was written." << endl;
    }

    m_bwrite_ang_vel_csv = false;
  }

  if(m_bwrite_ot_csv){
    cout << __PRETTY_FUNCTION__ << "\n";
    cout << "Writing observation time to " << m_fsog_csv << endl;    
    bool success = m_mod.write_ot_csv(m_fot_csv);
    if(!success){
      cerr << __PRETTY_FUNCTION__ << "\n";
      cerr << "Error : Couldn't write " << m_fot_csv << endl;
      return false;
    }
    else{
      cout << m_fot_csv << " was written." << endl;
    }

    m_bwrite_ot_csv = false;
  }

  
  get_inst();
  map_stat();
  s_aws1_mod::s_aws1_state  state = m_mod.get_state(m_ctrl_stat.meng_aws, m_ctrl_stat.rud_aws);
  
  long long t = get_time();
  long long tdiff = t - m_prev_t;
  m_prev_t = t;
  m_cog += state.ang_vel * (tdiff / 10000000.f);
  if(m_cog > 360.f)
    m_cog -= 360.f;
  
  if(m_cog < 0.f)
    m_cog += 360.f;
  
  m_ch_state->set_velocity(m_cog, state.sog, t);
  
  Mat Xwrld(3, 1, CV_32F);
  float * pXwrld = Xwrld.ptr<float>(0);
  pXwrld[0] = state.sog * 1852.f / (float)(t / 10000000.f);
  pXwrld[1] = pXwrld[0] * sin(m_cog);
  pXwrld[0] *= cos(m_cog);
  
  Mat Xbin(3, 1, CV_32F);
  wrldtobih(m_Xorg, m_Rwrld, Xwrld, Xbin);
  float * pXbin = Xbin.ptr<float>(0);
  m_ch_state->set_position(t, pXbin[0], pXbin[1], pXbin[2], pXbin[3]);
  return true;
}

void f_aws1_sim::destroy_run(){
}

void f_aws1_mod::reshape_mod(){
  aws_scope_show(__PRETTY_FUNCTION__);
  s_aws1_mod tmp;
  tmp.res_meng = m_mod.res_meng;
  tmp.res_rud = m_mod.res_rud;
  tmp.table = m_mod.table;

  m_mod.res_meng  = m_res_meng;
  m_mod.res_rud = m_res_rud;
  m_mod.init();
  const float smeng = (float)tmp.res_meng/(float)m_res_meng;
  const float srud = (float)tmp.res_rud/(float)m_res_rud;
  for(int i = 0; i < m_res_meng; ++i){
    const int imeng = (int)floor(smeng * (float)i);
    for(int j = 0; j < m_res_rud; ++j){
      const int irud = (int)floor(srud * (float)j);
      cout << imeng << ", " << irud << " is assigned to " << i << ", " << j << endl;
      m_mod.table[i][j] = tmp.table[imeng][irud];
    }
  }
  //tmp.destroy();
}

f_aws1_mod::f_aws1_mod(const char * fname): f_base(fname), m_bwrite_mod(false), m_bwrite_ang_vel_csv(false), m_bwrite_sog_csv(false),  m_bwrite_ot_csv(false), m_binterpolate(false), m_breshape_mod(false), m_bverb(false), m_bupdate(false), m_res_meng(10), m_res_rud(10), m_max_dsog(1.f), m_max_dang_vel(1.f), m_ref_sog(-0.f), m_prev_cog(1.f),  m_ost(LLONG_MAX), m_min_ot(10){
  register_fpar("write_mod", &m_bwrite_mod, "Write a model.");
  register_fpar("write_ang_vel_csv", &m_bwrite_ang_vel_csv, "Write angular velocity csv.");
  register_fpar("write_sog_csv", &m_bwrite_sog_csv, "Write sog csv.");
  register_fpar("write_ot_csv", &m_bwrite_ot_csv, "Write observation time csv.");
  register_fpar("interpolate", &m_binterpolate, "Interpolate a table of model.");
  register_fpar("reshape_mod", &m_breshape_mod, "Reshape a model.");
  register_fpar("verb", &m_bverb, "Enable verbose (default n).");
  register_fpar("update", &m_bupdate, "Enable updating a model (default n).");

  register_fpar("mod", m_fmod, 1023, "Model file.");
  register_fpar("sog_csv", m_fsog_csv, 1023, "csv file of sog.");
  register_fpar("ang_vel_csv", m_fang_vel_csv, 1023, "csv file of angular velocity.");
  register_fpar("ot_csv", m_fot_csv, 1023, "csv file of observation time."); 
  register_fpar("res_meng", &m_res_meng, "Resolution fo main engine (default 1.f).");
  register_fpar("res_rud", &m_res_rud, "Resolution of rudder (default 1.f).");
 
  register_fpar("max_dang_vel", &m_max_dang_vel, "Maximum differece of angular velocitiy (default 1.f).");
  register_fpar("max_dsog", &m_max_dsog, "Maximum difference of sog (default 1.f).");
 
  register_fpar("min_ot", &m_min_ot, "Minimum ovservation time for accepting data.");

  register_fpar("ch_ctrl_stat", (ch_base**)&m_ch_ctrl_stat, typeid(ch_aws1_ctrl_stat).name(), "Control state channel.");
  register_fpar("ch_state", (ch_base**)&m_ch_state, typeid(ch_state).name(), "State channel");
  m_max_dstat_rud = 256 / m_res_meng;
  m_max_dstat_meng = 256 / m_res_rud;
}

bool f_aws1_mod::init_run(){
  //read model from a file specified by m_fmod.
  if(m_mod.read(m_fmod)){
    cerr << "model is read from  " << m_fmod << "." << endl;
    m_res_meng = m_mod.res_meng;
    m_res_rud = m_mod.res_rud;
  }
  else{
    m_mod.res_meng = m_res_meng;
    m_mod.res_rud = m_res_rud;
    m_mod.init();
  }
  m_max_dstat_rud = 256 / m_res_meng;
  m_max_dstat_meng = 256 / m_res_rud;
  cout << m_max_dstat_rud << endl;
  cout << m_max_dstat_meng << endl;
  return true;
}

bool f_aws1_mod::proc(){
  //get control state from channel
  if(!m_ch_ctrl_stat){
    cerr << "f_aws1_mod::proc\n";
    cerr <<"Error : chnanel of control status is not specified." << endl;
    return false;
  }
  
  s_aws1_ctrl_stat stat;
  m_ch_ctrl_stat->get(stat);
  
  if(m_binterpolate){
    m_mod.interpolate_table();
    m_binterpolate = false;
  }

  if(m_breshape_mod){
    reshape_mod();
    m_max_dstat_rud = 256 / m_res_meng;
    m_max_dstat_meng = 256 / m_res_rud;
    m_breshape_mod = false;
  }

  //write out a model
  if(m_bwrite_mod){
    cout << __PRETTY_FUNCTION__ << "\n";
    cout << "Writing model to " << m_fmod << endl;
    bool success = m_mod.write(m_fmod);
    
    if(!success){
      cerr << __PRETTY_FUNCTION__ << "\n";
      cerr << "Error : Couldn't write " << m_fmod << endl;
      return false;
    }
    else{
      cout << m_fmod << " was written." << endl;
    }

    m_bwrite_mod = false;
  }

  if(m_bwrite_sog_csv){
    cout << __PRETTY_FUNCTION__ << "\n";
    cout << "Writing sog to " << m_fsog_csv << endl;
    bool success = m_mod.write_sog_csv(m_fsog_csv);
    if(!success){
      cerr << __PRETTY_FUNCTION__ << "\n";
      cerr << "Error : Couldn't write " << m_fsog_csv << endl;
      return false;
    }
    else{
      cout << m_fsog_csv << " was written." << endl;
    }

    m_bwrite_sog_csv = false;
  }
  
  if(m_bwrite_ang_vel_csv){
    cout << __PRETTY_FUNCTION__ << "\n";
    cout << "Writing angular velocity to " << m_fang_vel_csv << endl;    
    bool success = m_mod.write_ang_vel_csv(m_fang_vel_csv);
    if(!success){
      cerr << __PRETTY_FUNCTION__ << "\n";
      cerr << "Error : Couldn't write " << m_fang_vel_csv << endl;
      return false;
    }
    else{
      cout << m_fang_vel_csv << " was written." << endl;
    }

    m_bwrite_ang_vel_csv = false;
  }

  if(m_bwrite_ot_csv){
    cout << __PRETTY_FUNCTION__ << "\n";
    cout << "Writing observation time to " << m_fsog_csv << endl;    
    bool success = m_mod.write_ot_csv(m_fot_csv);
    if(!success){
      cerr << __PRETTY_FUNCTION__ << "\n";
      cerr << "Error : Couldn't write " << m_fot_csv << endl;
      return false;
    }
    else{
      cout << m_fot_csv << " was written." << endl;
    }

    m_bwrite_ot_csv = false;
  }
  
  //A observation period is increased while control instruction and status are almost the same. Otherwise observation start time is set to 0;
  long long t;
  float cog, sog;
  m_ch_state->get_velocity(t, cog, sog);
  if(t == m_prev_t){
    return true;
  }

  float ang_vel = ((cog - m_prev_cog) /(float)((t-m_prev_t)/10000000.f));
  if(ang_vel > 180.f){
    ang_vel = cog + 360.f - m_prev_cog;
  }
  if(m_bverb){
    cout <<  "meng " << (int)stat.meng << " rud " << (int)stat.rud << " ang_vel " << ang_vel << " sog " << sog << endl;
  }

  if((abs(m_ref_stat.rud - stat.rud)  > m_max_dstat_rud ||
     abs(m_ref_stat.meng - stat.meng) > m_max_dstat_meng || 
     abs(m_ref_ang_vel - ang_vel) > m_max_dang_vel || 
     abs(m_ref_sog - sog) > m_max_dsog)){
  //update a model by a current status, if observation period is longer than a certain threshold.
    long long diff = t - m_ost;
    if(m_bverb){
      cout << "ot " <<  diff << endl;
    }
    if(diff > m_min_ot){
      if(m_bupdate)
	m_mod.update(stat.meng, stat.rud, m_ref_ang_vel, m_ref_sog, diff);
    }

    m_ost = t;
    m_ref_stat = stat;
    m_ref_sog = sog;
    m_ref_ang_vel = ang_vel;
  }
  
  m_prev_t = t;
  m_prev_cog = cog;
  return true;
}

void f_aws1_mod::destroy_run(){
}

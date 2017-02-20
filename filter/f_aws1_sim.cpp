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

s_aws1_mod::s_aws1_state::s_aws1_state(): cog(0), sog(0), t(0){
  
}

const int s_aws1_mod::res = 256;
const int s_aws1_mod::num_cells = res*res*res;
s_aws1_mod::s_aws1_mod(){
}

s_aws1_mod::~s_aws1_mod(){
  delete [] table;
}

void s_aws1_mod::get_state(const unsigned char meng, const unsigned char seng, const unsigned char rud){
  
}

void s_aws1_mod::init(){
  table = new s_aws1_state**[num_cells];
  for(int i = 0; i < res; ++i){
    table[i] = new s_aws1_state*[num_cells];
    for(int j = 0; j < res; ++j){
      table[i][j] = new s_aws1_state[res];
    }
  }
}


void s_aws1_mod::update(const unsigned char meng, const unsigned char seng, const unsigned char rud,
	    const float cog, const float sog, const long long t){
  s_aws1_state &state = table[meng][seng][rud];
  float s0 = (float)t / (float)(t + state.t);
  float s1 = 1.f - s0;
  state.cog = s0 * state.cog + s1 * cog;
  state.sog = s0 * state.sog + s1 * sog;
  state.t += t;
}

void s_aws1_mod::interpolate_table(){
  for(int i = 0; i < res; ++i){
    for(int j = 0; j < res; ++j){
      for(int k = 0; k < res; ++k){
	s_aws1_state * state = &table[i][j][k];
	if(state->t == 0){
	  const s_aws1_state nstate = get_nearest_cell(i, j, k);
	  state->cog = nstate.cog;
	  state->sog = nstate.sog;
	}
      }
    }
  }
}

s_aws1_mod::s_aws1_state& s_aws1_mod::get_nearest_cell(const int meng, const int seng, const int rud){
  float dmin = FLT_MAX; //maximum distance
  int min_meng = 0;
  int min_seng = 0;
  int min_rud = 0;
  const int res2 = res * res;
  for(int i = 0; i < res; ++i){
    for(int j = 0; j < res; ++j){
      for(int k = 0; k < res; ++k){
	s_aws1_state * state = &table[i][j][k];
	float d = FLT_MAX;
	if(state->t)
	  d = (float)sqrt(pow(i - meng, 2.0) + pow(j - seng, 2.0) + pow(k - rud, 2.0));
	if(d < dmin ){
	  min_meng = i;
	  min_seng = j;
	  min_rud = k;
	}
      }
    }
  }
  return table[min_meng][min_seng][min_rud];
}

bool s_aws1_mod::read(const char * fname){
  FILE * pf = fopen(fname, "rb");
  if(!pf)
    return false;

  int sz = 0;
  for(int i = 0; i < res; ++i){
    for(int j = 0; j < res; ++j){
      for(int k = 0; k < res; ++k){
	s_aws1_state * state = &table[i][j][k];
	sz += fread(&(state->cog), sizeof(float), 1, pf);
	sz += fread(&(state->sog), sizeof(float), 1, pf);
	sz += fread(&(state->t), sizeof(long long), 1, pf);      
      }
    }
  }
  fclose(pf);
}

bool s_aws1_mod::write(const char * fname){
  interpolate_table();

  FILE * pf = fopen(fname, "wb");
  if(!pf)
    return false;
  
  int sz = 0;
  for(int i = 0; i < res; ++i){
    for(int j = 0; j < res; ++j){
      for(int k = 0; k < res; ++k){
	s_aws1_state * state = &table[i][j][k];
	sz += fwrite(&(state->cog), sizeof(float), 1, pf);
	sz += fwrite(&(state->sog), sizeof(float), 1, pf);
	sz += fwrite(&(state->t), sizeof(long long), 1, pf);      
      }
    }
  }

  if(sz != sizeof(s_aws1_state)*pow(res, 3.0))
    return false;
  fclose(pf);
  return true;
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

f_aws1_sim::f_aws1_sim(const char * fname): f_base(fname){
}

bool f_aws1_sim::init_run(){
  //read a model specified by m_fmod
  if(!m_mod.read(m_fmod)){
    cerr << __PRETTY_FUNCTION__ << endl;
    cerr << "Error : Couldn't read " << m_fmod << "." << endl;
    return false;
  }
  return true;
}

bool f_aws1_sim::proc(){
  if(!m_ch_ctrl_stat){
    cerr << __PRETTY_FUNCTION__ << endl;
    cerr << "Error : ch_state is not specified." << endl;
    return false;
  }
  
  get_inst();
  map_stat();
  s_aws1_mod::s_aws1_state * state = &m_mod.table[m_ctrl_stat.meng_aws][m_ctrl_stat.seng_aws][m_ctrl_stat.rud_aws];

  
  long long t = get_time();
  float cog = state->cog;
  float sog = state->sog;

  m_ch_state->set_velocity(cog, sog, t);
  
  return true;
}

void f_aws1_sim::destroy_run(){
}

f_aws1_mod::f_aws1_mod(const char * fname): f_aws1_mod(fname){
}

bool f_aws1_mod::init_run(){
  //read model from a file specified by m_fmod.
  if(!m_mod.read(m_fmod)){
    cerr << __PRETTY_FUNCTION__ << endl;
    cerr << "Error : Couldn't read " << m_fmod << "." << endl;
    return false;
  }
  return true;
}

bool f_aws1_mod::proc(){
  //get control instruction from channel
  // if(!m_ch_ctrl_inst){
  //   cerr << "f_aws1_mod::proc\n";
  //   cerr << "Error : channel of control instruction is not specified." << endl;
  //   return false;
  // }
  
  // s_aws1_ctrl_inst inst;
  // m_ch_ctrl_inst->get(inst);

  //get control state from channel
  if(!m_ch_ctrl_stat){
    cerr << "f_aws1_mod::proc\n";
    cerr <<"Error : chnanel of control status is not specified." << endl;
    return false;
  }
  
  s_aws1_ctrl_stat stat;
  m_ch_ctrl_stat->get(stat);
  
  //write out a model
  if(m_bwrite){
    cout << "f_aws1_mod_proc" << endl;
    cout << "Writing model to " << m_fmod << endl;
    const bool success = m_mod.write(m_fmod);
    if(!success){
      cerr << "f_aws1_mod::proc" << endl;
      cerr << "Error : Couldn't write " << m_fmod << endl;
    }
    else{
      cout << m_fmod << " was written." << endl;
    }
    m_bwrite = false;
  }
  
  //A observation period is increased while control instruction and status are almost the same. Otherwise observation start time is set to 0;
  long long t;
  float cog, sog;
  m_ch_state->get_velocity(t, cog, sog);

  if(abs(m_ref_ctrl_stat.rud - stat.rud) ||
     abs(m_ref_ctrl_stat.meng - stat.meng) || 
     abs(m_ref_ctrl_stat.seng - stat.seng) ||
     abs(m_ref_cog - cog) > m_max_dcog || 
     abs(m_ref_sog - sog) > m_max_dsog){
  //update a model by a current status, if observation period is longer than a certain threshold.
    if(t - m_ost > m_th_op){
      m_mod.update(stat.meng, stat.seng, stat.rud, cog, sog, t);
    }
    m_ost = t;
    m_ref_ctrl_stat = stat;
  }
  return true;
}

void f_aws1_mod::destroy_run(){
}

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

#include "../channel/ch_aws1_ctrl.h"
#include "../channel/ch_state.h"
#include "f_aws1_sim.h"

s_aws1_mod::s_aws1_state::s_aws1_state(): t(0), cog(0), sog(0){
  
}

s_aws1_mod::s_aws1_mod(): res(256){
}

s_aws1_mod::~s_aws1_mod(){
  for(int i = 0; i < res; ++i){
    for(int j = 0; j < res; ++j){
      delete [] table[i][j];
    }
    delete [] table[i];
  }
  delete [] table;
}

void s_aws1_mod::get_state(const unsigned char meng, const unsigned char seng, const unsigned char rud){
  
}

void s_aws1_mod::init(){
  const int sz = 0xff;
  table = new s_aws1_state**[sz];
  for(int i = 0; i < sz; ++i){
    table[i] = new s_aws1_state*[sz]; 
    for(int j = 0; j < sz; ++j){
      table[i][j] = new s_aws1_state[sz];
    }
  }
}


void s_aws1_mod::update(const unsigned char meng, const unsigned char seng, const unsigned char rud,
	    const float cog, const float sog, const long long t){
  
}

bool s_aws1_mod::read(const char * fname){
}

bool s_aws1_mod::write(const char * fname){
  FILE * pf = fopen(fname, "wb");
  if(!pf)
    return false;
  
  int sz = 0;
  for(int i = 0; i < res; ++i){
    for(int j = 0; j < res; ++j){
      sz += fwrite(table[i][j], sizeof(s_aws1_state), sizeof(table[i][j]), pf);
    }
  }
  
  //Above code may be good, but if it is not so, following code should be recmommended.
  // for(int i = 0; i < res; ++i){
  //   for(int j = 0; j < res; ++j){
  //     for(int k = 0; k < res; ++k){
  // 	s_aws1_state * aws1_state = &table[i][j][k];
  // 	sz += fwrite(&(aws1_state->t), sizeof(long long), 1, pf);
  // 	sz += fwrite(&(aws1_state->cog), sizeof(float), 1, pf);
  // 	sz += fwrite(&(aws1_state->sog), sizeof(float), 1, pf);
  //     }
  //   }
  // }
  if(sz != sizeof(s_aws1_state)*256*256*256)
    return false;
  fclose(pf);
  return true;
}

f_aws1_sim::f_aws1_sim(const char * fname): f_base(fname){

}

bool f_aws1_sim::init_run(){
  //read a model specified by m_fmod
  
  return true;
}

bool f_aws1_sim::proc(){
  
  return true;
}

void f_aws1_sim::destroy_run(){
}

f_aws1_mod::f_aws1_mod(const char * fname): f_aws1_mod(fname){
}

bool f_aws1_mod::init_run(){
  //read model from a file specified by m_fmod.
  
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

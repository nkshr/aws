// Copyright(c) 2014 Yohei Matsumoto, Tokyo University of Marine
// Science and Technology, All right reserved. 

// f_sample.cpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Publica License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// f_sample.cpp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with f_sample.cpp.  If not, see <http://www.gnu.org/licenses/>. 

#ifndef _F_AWS1_SIM_H_
#define _F_AWS1_SIM_H_
#include "f_base.h"

struct s_aws1_mod{
  int imid;
  int imnut;
  int isnut;
  const static int res; //table resolution along each axises 
  const static int num_cells; //number of celss in a table

  struct s_aws1_state{
    float cog;  //course over ground
    float sog; //spped over ground
    long long t; //observation time

    s_aws1_state();
    //s_aws1_state(const s_aws1_state &state) = delete;
  };

  s_aws1_state ***table;

  s_aws1_mod();
  s_aws1_mod(const s_aws1_mod &mod) = delete;
  ~s_aws1_mod();

  void init();

  void get_state(const unsigned char meng, const unsigned char seng, const unsigned char rud);
  void update(const unsigned char meng, const unsigned char seng, const unsigned char rud,
	      const float cog, const float sog, const long long t);

  void interpolate_table();
  s_aws1_state& get_nearest_cell(const int meng, const int seng, const int rud);

  bool write(const char * fname);
  bool read(const char * fname);
};

class f_aws1_sim: public f_base 
{
 private:
  char * m_fmod;
  
  s_aws1_mod m_mod;
  s_aws1_ctrl_stat m_ctrl_stat;

  ch_aws1_ctrl_inst * m_ch_ctrl_ui;
  ch_aws1_ctrl_inst * m_ch_ctrl_ap1;
  ch_aws1_ctrl_inst * m_ch_ctrl_ap2;
  
  ch_aws1_ctrl_stat * m_ch_ctrl_stat; 
  ch_state * m_ch_state;

  void get_inst();

  void map_stat();
 public:
  f_aws1_sim(const char * fname);
  
  virtual bool init_run();
  
  virtual void destroy_run();
  
  virtual bool proc();
};

class f_aws1_mod: public f_base 
{
 private:
  bool m_bwrite;

  char * m_fmod;

  //cog and sog are supposed to be constant with respect to following parameters.
  float m_max_dcog;
  float m_max_dsog;
  float m_ref_cog;
  float m_ref_sog;
  
  long long m_ost; //observation start time
  long long m_th_op; //threshold for a observation period

  s_aws1_mod m_mod;

  //s_aws1_ctrl_inst m_ref_ctrl_inst;
  s_aws1_ctrl_stat m_ref_ctrl_stat;

  //ch_aws1_ctrl_inst * m_ch_ctrl_inst;
  ch_aws1_ctrl_stat * m_ch_ctrl_stat;
  ch_state * m_ch_state;
  
 public:
  f_aws1_mod(const char * fname);
  
  virtual bool init_run();
  
  virtual void destroy_run();
  
  virtual bool proc();
    
};

#endif
